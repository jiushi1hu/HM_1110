"""Double-precision water balance and exact current-day chain feasibility."""
import numpy as np
from harl.envs.cascade_reservoir.data_loader import HEAD_MODELS
from harl.envs.cascade_reservoir.action_correction import (
    action_correction_settings, action_correction_fields, project_release,
    FLOW_TOLERANCE, INTERVAL_TOLERANCE, validate_release_mapping_mode, validated_proposal_bounds)


class InfeasibleCascade(RuntimeError):
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = {} if diagnostics is None else dict(diagnostics)
        self.failure_stage = self.diagnostics.get('failure_stage')
        self.reservoir_index = self.diagnostics.get('reservoir_index')


class ReservoirPhysics:
    def __init__(self, data, config):
        self.data = data
        self.config = dict(config)
        self.correction_settings = action_correction_settings(config)
        self.release_mapping_mode = validate_release_mapping_mode(config)
        if config.get("release_mapping", "linear") != "linear":
            raise ValueError("This experiment uses linear normalized-action mapping")
        self.dt = float(config["timestep_seconds"])
        if self.dt != 86400.0:
            raise ValueError("This model is daily")
        eta = float(config["efficiency"])
        if not 0 < eta <= 1:
            raise ValueError("Efficiency must lie in (0, 1]")
        density = float(config["water_density_kg_m3"])
        gravity = float(config["gravity_m_s2"])
        if not np.isfinite([density, gravity]).all() or density <= 0 or gravity <= 0:
            raise ValueError("Water density and gravity must be finite and positive")
        self.head_model = config.get("head_model", "fixed")
        if self.head_model not in HEAD_MODELS:
            raise ValueError("Unknown head model: " + str(self.head_model))
        self.rho_g_eta = density * gravity * eta
        self.head_data_status = "supplied_typical_head"
        if self.head_model == "fixed":
            self.coefficient = self.rho_g_eta * data.head / 1e6
            self.effective_turbine_cap = np.minimum(data.turbine_cap, data.power_cap / self.coefficient)
        else:
            if getattr(data, "head_sensitivity", None) is None:
                raise ValueError("Load the sensitivity data before selecting storage_sensitivity")
            if not np.allclose(data.head_sensitivity["efficiency"], eta, rtol=0., atol=1e-12):
                raise ValueError("Head sensitivity efficiency must match the physical configuration")
            self.head_data_status = "assumed_sensitivity_only"
        self.reference_energy = float(24 * data.power_cap.sum())

    def _forebay_levels(self, storage):
        storage = np.asarray(storage, dtype=np.float64)
        if storage.shape != (5,) or not np.isfinite(storage).all():
            raise ValueError("Generation requires a finite five-vector of storage")
        tolerance = np.maximum(1., 1e-10 * np.abs(storage))
        if ((storage < self.data.vmin - tolerance) | (storage > self.data.vmax + tolerance)).any():
            raise InfeasibleCascade("Generation storage violates operating bounds")
        # Absorb tolerated roundoff only in the elevation lookup. Water balance
        # and the physical storage vectors are never clipped or changed.
        return self.data.levels(np.clip(storage, self.data.vmin, self.data.vmax))

    def generation(self, release, storage_start=None, storage_end=None):
        """Daily mean-head generation shared by transitions and the reference.

        The sensitivity scenario has constant efficiency and a constant combined
        tailwater/loss offset, so no flow/head root solve is required. Installed
        capacity is enforced at the modeled daily mean head.
        """
        release = np.asarray(release, dtype=np.float64)
        if (release.ndim < 1 or release.shape[-1] != 5 or
                not np.isfinite(release).all() or (release < 0).any()):
            raise ValueError("Generation releases must be finite nonnegative five-vectors")
        result = {}
        if self.head_model == "fixed":
            head = self.data.head.copy()
            coefficient, cap = self.coefficient, self.effective_turbine_cap
        else:
            start_level = self._forebay_levels(storage_start)
            end_level = self._forebay_levels(storage_end)
            mean_level = (start_level + end_level) / 2.
            scenario = self.data.head_sensitivity
            head = (scenario["reference_net_head_m"] + scenario["head_slope_m_per_m"] *
                    (mean_level - scenario["assumed_reference_forebay_level_m"]))
            if not np.isfinite(head).all() or (head <= 0).any():
                raise ValueError("Calculated net head must be finite and positive")
            coefficient = self.rho_g_eta * head / 1e6
            cap = np.minimum(self.data.turbine_cap, self.data.power_cap / coefficient)
            result["forebay_level_mean_m"] = mean_level
        turbine = np.minimum(release, cap)
        power = turbine * coefficient
        result.update(net_head_m=head, effective_turbine_cap_m3s=cap.copy(),
                      turbine=turbine, spill=release-turbine, power=power, energy=24.*power)
        return result

    def _storage_bounds(self, storage_lower, storage_upper):
        """Validate END-of-day bounds without changing the global curve domain."""
        if storage_lower is None and storage_upper is None:
            return self.data.vmin, self.data.vmax
        if storage_lower is None or storage_upper is None:
            raise ValueError("Supply both daily storage_lower and storage_upper")
        lower, upper = [np.asarray(x, dtype=np.float64) for x in (storage_lower, storage_upper)]
        if (lower.shape != (5,) or upper.shape != (5,) or
                not np.isfinite(lower).all() or not np.isfinite(upper).all() or (lower > upper).any()):
            raise ValueError("Daily storage bounds must be finite ordered five-vectors")
        if (lower < self.data.vmin).any() or (upper > self.data.vmax).any():
            raise ValueError("Daily storage bounds must lie within global operating bounds")
        return lower, upper

    def downstream_intervals(self, storage, external, storage_lower=None, storage_upper=None):
        storage_min, storage_max = self._storage_bounds(storage_lower, storage_upper)
        d = self.data
        lo, hi = np.zeros(5), d.release_cap.copy()
        for i in range(3, -1, -1):
            lo[i] = max(0., lo[i + 1] - external[i + 1] + (storage_min[i + 1] - storage[i + 1]) / self.dt)
            hi[i] = min(d.release_cap[i], hi[i + 1] - external[i + 1] + (storage_max[i + 1] - storage[i + 1]) / self.dt)
            if lo[i] > hi[i] + 1e-8:
                raise self._interval_error("Empty downstream-compatible release interval",
                                           storage, external, i, lo[i], hi[i])
        return lo, hi

    def _interval_error(self, message, storage, external, reservoir=None, low=None, high=None):
        details = dict(failure_stage='action_execution', no_feasible_executed_solution=True,
            storage_start_m3=np.asarray(storage).tolist(), external_inflow_m3s=np.asarray(external).tolist())
        if reservoir is not None:
            details.update(reservoir_index=int(reservoir), reservoir_id=self.data_reservoir_id(reservoir))
        if low is not None:
            details['release_low_m3s'] = float(low)
        if high is not None:
            details['release_high_m3s'] = float(high)
        return InfeasibleCascade(message, diagnostics=details)

    @staticmethod
    def data_reservoir_id(index):
        from harl.envs.cascade_reservoir.data_loader import IDS
        return IDS[index]

    def _release_reason(self, i, target, executed, low, high, inflow, storage, storage_min, storage_max):
        if executed == target:
            return 'none'
        if high-low <= INTERVAL_TOLERANCE:
            return 'forced_dispatch'
        water_low = inflow+(storage[i]-storage_max[i])/self.dt
        water_high = inflow+(storage[i]-storage_min[i])/self.dt
        if target < low:
            if low == 0.:
                return 'outflow_nonnegative'
            if np.isclose(low, water_low, rtol=0., atol=FLOW_TOLERANCE):
                return 'next_day_storage_max'
            return 'downstream_compatibility'
        if np.isclose(high, water_high, rtol=0., atol=FLOW_TOLERANCE):
            return 'next_day_storage_min'
        if np.isclose(high, self.data.release_cap[i], rtol=0., atol=FLOW_TOLERANCE):
            return 'outflow_capacity'
        return 'downstream_compatibility'

    def correct_releases(self, storage, external, target_release, storage_lower=None,
                         storage_upper=None, *, normalized_actions=None,
                         mapping_low=None, mapping_high=None):
        """Apply Pi to an explicit target-release vector using existing constraints.

        Each upstream release is the nearest point in its downstream-compatible
        interval; downstream intervals are recomputed using the corrected inflow.
        The ordering is lexicographic upstream priority, not Euclidean distance
        minimization across the five-vector. No service threshold is made hard.

        Direct callers without sampled actions receive invalid normalized-action
        inverse masks: a physical repair is measurable, but Eq. (16) is undefined.
        transition() uses this same scalar projection stage when enabled.
        """
        storage, external, target = [np.asarray(v, dtype=np.float64)
                                     for v in (storage, external, target_release)]
        if any(v.shape != (5,) or not np.isfinite(v).all() for v in (storage, external, target)):
            raise ValueError('Correction requires finite five-vectors')
        if (external < 0.).any():
            raise ValueError('External inflow must be nonnegative')
        if ((storage < self.data.vmin-1.) | (storage > self.data.vmax+1.)).any():
            raise self._interval_error('Starting storage violates operating bounds', storage, external)
        storage_min, storage_max = self._storage_bounds(storage_lower, storage_upper)
        try:
            lower, upper = self.downstream_intervals(storage, external, storage_min, storage_max)
        except InfeasibleCascade as exc:
            if exc.diagnostics:
                raise
            raise self._interval_error(str(exc), storage, external) from exc
        inflow, release, low, high = [np.empty(5) for _ in range(4)]
        controls = np.ones(5, dtype=np.float32)
        reasons = np.full(5, 'none', dtype='<U96')
        for i in range(5):
            inflow[i] = external[i]+(release[i-1] if i else 0.)
            low[i] = max(lower[i], inflow[i]+(storage[i]-storage_max[i])/self.dt)
            high[i] = min(upper[i], inflow[i]+(storage[i]-storage_min[i])/self.dt)
            if low[i] > high[i]+INTERVAL_TOLERANCE:
                raise self._interval_error('Empty feasible correction interval', storage, external,
                                           i, low[i], high[i])
            controls[i] = float(high[i]-low[i] > INTERVAL_TOLERANCE)
            release[i] = float(project_release(target[i], low[i], high[i], self.data.release_cap[i]))
            reasons[i] = self._release_reason(i, target[i], release[i], low[i], high[i],
                inflow[i], storage, storage_min, storage_max)
        available = normalized_actions is not None
        original = np.zeros(5) if not available else np.asarray(normalized_actions,dtype=float)
        if (mapping_low is None) != (mapping_high is None):
            raise ValueError('Supply both original mapping bounds')
        if available:
            map_low,map_high = validated_proposal_bounds(mapping_low,mapping_high,(5,))
            if original.shape != (5,) or not np.isfinite(original).all() or ((original<0)|(original>1)).any():
                raise ValueError('Original normalized actions must be finite in [0,1]')
            implied = map_low+original*(map_high-map_low)
            tolerance = np.maximum(FLOW_TOLERANCE,32.*np.finfo(float).eps*np.maximum(1.,np.maximum(np.abs(target),np.abs(implied))))
            if (np.abs(target-implied)>tolerance).any():
                raise ValueError('Target release is inconsistent with original normalized action and mapping bounds')
        elif mapping_low is not None:
            map_low,map_high = validated_proposal_bounds(mapping_low,mapping_high,(5,))
        else:
            map_low,map_high = low,high
        legacy = original.copy()
        legacy[controls <= 0.] = .5
        result = self._finish_transition(storage, external, inflow, release, low, high,
            controls, legacy, storage_min, storage_max, storage_lower is not None)
        result.update(action_correction_fields(original, target, release, map_low, map_high,
            controls, self.correction_settings, reasons,physical_low=low,physical_high=high))
        # Preserve the bounds used for the proposal/inverse audit when an
        # external caller supplies a different original mapping interval.
        result.update(execution_low=low.copy(), execution_high=high.copy(),
                      low=map_low.copy(), high=map_high.copy())
        result['actions'][result['controls'] <= 0.] = .5
        result['original_action_available'] = np.full(5, available, dtype=bool)
        if not available and mapping_low is None:
            result['scaled_release_correction_valid'][:] = False
            result['scaled_release_correction'][:] = 0.
        if not available:
            result['inverse_action_valid'][:] = False
            for key in ('realized_normalized_actions', 'normalized_action_correction',
                        'normalized_action_correction_magnitude', 'correction_deficit'):
                result[key][:] = 0.
            result['correction_threshold_exceeded'][:] = False
        return result

    def transition(self, storage, external, actions, storage_lower=None, storage_upper=None,
                   proposal_low=None, proposal_high=None):
        """Map actions, optionally apply Pi, then execute the conserved transition.

        Mapping and correction enforce the same existing physical constraints.
        Therefore feasible mapped targets pass through Pi unchanged. The scalar
        projection is performed before downstream inflows or storage are updated.
        """
        storage, external, actions = [np.asarray(x, dtype=np.float64) for x in (storage, external, actions)]
        if any(x.shape != (5,) or not np.isfinite(x).all() for x in (storage, external, actions)):
            raise ValueError("Expected three finite five-vectors")
        if ((actions < 0) | (actions > 1)).any() or (external < 0).any():
            raise ValueError("Invalid normalized action or external flow")
        if self.release_mapping_mode == 'csv_proposal':
            if not self.correction_settings['enabled']:
                raise ValueError('csv_proposal requires environment-side correction enabled')
            map_low,map_high = validated_proposal_bounds(proposal_low,proposal_high,(5,))
            target = map_low+actions*(map_high-map_low)
            return self.correct_releases(storage,external,target,storage_lower,storage_upper,
                normalized_actions=actions,mapping_low=map_low,mapping_high=map_high)
        d = self.data
        storage_min, storage_max = self._storage_bounds(storage_lower, storage_upper)
        if ((storage < d.vmin - 1) | (storage > d.vmax + 1)).any():
            raise self._interval_error("Starting storage violates operating bounds", storage, external)
        try:
            lower, upper = self.downstream_intervals(storage, external, storage_min, storage_max)
        except InfeasibleCascade as exc:
            if exc.diagnostics:
                raise
            raise self._interval_error(str(exc), storage, external) from exc
        inflow, release, low, high, target = [np.empty(5) for _ in range(5)]
        controls, executed_actions = np.ones(5, dtype=np.float32), actions.copy()
        reasons = np.full(5, 'none', dtype='<U96')
        for i in range(5):
            inflow[i] = external[i] + (release[i - 1] if i else 0.)
            low[i] = max(lower[i], inflow[i] + (storage[i] - storage_max[i]) / self.dt)
            high[i] = min(upper[i], inflow[i] + (storage[i] - storage_min[i]) / self.dt)
            if low[i] > high[i] + INTERVAL_TOLERANCE:
                raise self._interval_error("Empty feasible release interval at reservoir " + str(i),
                                           storage, external, i, low[i], high[i])
            target[i] = low[i] + actions[i]*(high[i]-low[i])
            if high[i] - low[i] <= INTERVAL_TOLERANCE:
                controls[i] = 0
                executed_actions[i] = 0.5
                # Preserve the baseline's unique/tolerance-sized forced dispatch.
                release[i] = min(d.release_cap[i], max(0., 0.5 * (low[i] + high[i])))
            elif self.correction_settings['enabled']:
                release[i] = float(project_release(target[i], low[i], high[i], d.release_cap[i]))
            else:
                release[i] = target[i]
            reasons[i] = self._release_reason(i, target[i], release[i], low[i], high[i],
                inflow[i], storage, storage_min, storage_max)
        result = self._finish_transition(storage, external, inflow, release, low, high,
            controls, executed_actions, storage_min, storage_max, storage_lower is not None)
        result.update(action_correction_fields(actions, target, release, low, high,
            controls, self.correction_settings, reasons))
        result['original_action_available'] = np.ones(5, dtype=bool)
        return result

    def _finish_transition(self, storage, external, inflow, release, low, high, controls,
                           executed_actions, storage_min, storage_max, supplied_bounds):
        next_storage = storage + (inflow - release) * self.dt
        residual = next_storage - storage - (inflow - release) * self.dt
        tolerance = np.maximum(1., 1e-10 * np.abs(storage))
        if ((next_storage < storage_min - tolerance) | (next_storage > storage_max + tolerance)).any():
            raise self._interval_error("Storage bound violation after execution", storage, external)
        if (release < -1e-8).any() or (release > self.data.release_cap + 1e-8).any():
            raise self._interval_error("Release capacity violation", storage, external)
        generated = self.generation(release, storage, next_storage)
        power, energy = generated["power"], generated["energy"]
        if (np.abs(residual) > tolerance).any() or (power > self.data.power_cap + 1e-7).any():
            raise AssertionError("Water/energy calculation failed")
        system_residual = next_storage.sum() - storage.sum() - (external.sum() - release[-1]) * self.dt
        if abs(system_residual) > tolerance.sum():
            raise AssertionError("Cascade water conservation failed")
        result = dict(storage_start=storage.copy(), storage_end=next_storage, inflow=inflow,
                    external=external.copy(), actions=executed_actions, release=release,
                    low=low, high=high, controls=controls, residual=residual,
                    reward=float(energy.sum() / self.reference_energy),
                    system_residual=float(system_residual), **generated)
        if supplied_bounds:
            result.update(storage_lower_m3=storage_min.copy(), storage_upper_m3=storage_max.copy())
        return result
