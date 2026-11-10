"""Vectorized scalar-equivalent physics for independent counterfactual rows.

The live environment still uses ``ReservoirPhysics.transition``. This kernel
adds only a leading batch dimension and never advances an environment or
changes its inputs. The backward compatibility pass, forward execution,
generation formulas and numerical tolerances match the scalar implementation.
"""
import numpy as np

from harl.envs.cascade_reservoir.data_loader import IDS
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade
from harl.envs.cascade_reservoir.action_correction import (
    action_correction_fields, project_release, FLOW_TOLERANCE, validated_proposal_bounds)


def _infeasible(message, storage, external, failing, reservoir=None, low=None, high=None):
    rows = np.flatnonzero(failing)
    first = int(rows[0]) if len(rows) else 0
    diagnostics = dict(failure_stage='action_execution', no_feasible_executed_solution=True,
        failed_batch_rows=rows.tolist(), failure_row_indices=rows.tolist(),
        storage_start_m3=storage[first].tolist(),
        external_inflow_m3s=external[first].tolist())
    if reservoir is not None:
        diagnostics.update(reservoir_index=int(reservoir), reservoir_id=IDS[reservoir])
    if low is not None:
        diagnostics['release_low_m3s'] = float(low[first])
    if high is not None:
        diagnostics['release_high_m3s'] = float(high[first])
    return InfeasibleCascade(message, diagnostics=diagnostics)


def _matrix(value, name, shape=None):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != len(IDS) or not np.isfinite(value).all():
        raise ValueError(name+' must be a finite batch of five-vectors')
    if shape is not None and value.shape != shape:
        raise ValueError('All transition inputs must have the same batch shape')
    return value


def _bounds(physics, shape, storage_lower, storage_upper):
    if storage_lower is None and storage_upper is None:
        return (np.broadcast_to(physics.data.vmin, shape),
                np.broadcast_to(physics.data.vmax, shape))
    if storage_lower is None or storage_upper is None:
        raise ValueError('Supply both daily storage_lower and storage_upper')
    values = []
    for value in (storage_lower, storage_upper):
        value = np.asarray(value, dtype=np.float64)
        if value.shape not in ((len(IDS),), shape) or not np.isfinite(value).all():
            raise ValueError('Daily storage bounds must be finite five-vectors or matching batches')
        values.append(np.broadcast_to(value, shape))
    lower, upper = values
    if (lower > upper).any():
        raise ValueError('Daily storage bounds must be ordered')
    if (lower < physics.data.vmin).any() or (upper > physics.data.vmax).any():
        raise ValueError('Daily storage bounds must lie within global operating bounds')
    return lower, upper


def _forebay_levels(physics, storage):
    tolerance = np.maximum(1., 1e-10*np.abs(storage))
    data = physics.data
    if ((storage < data.vmin-tolerance) | (storage > data.vmax+tolerance)).any():
        raise InfeasibleCascade('Generation storage violates operating bounds')
    # Clip only the level lookup within accepted roundoff, never physical water.
    lookup = np.clip(storage, data.vmin, data.vmax)
    levels = np.empty_like(lookup)
    for i, name in enumerate(IDS):
        points = data.anchors[name]
        levels[:, i] = np.interp(lookup[:, i], points[:, 1], points[:, 0])
    return levels


def batched_transition(physics, storage, external, actions,
                       storage_lower=None, storage_upper=None, proposal_low=None, proposal_high=None):
    """Evaluate normalized-command transitions with input shape ``(batch, 5)``.

    Optional end-state bounds have that shape or are broadcast five-vectors.
    Result vectors have shape ``(batch, 5)``; generation rewards and cascade
    residuals have shape ``(batch,)``. A failed row raises for the entire batch.
    Empty batches are allowed. As in scalar physics, the returned ``reward`` is
    unpenalized generation; the caller applies operation-reward settings.
    """
    storage = _matrix(storage, 'Storage')
    shape = storage.shape
    external = _matrix(external, 'External flow', shape)
    actions = _matrix(actions, 'Normalized actions', shape)
    if ((actions < 0.) | (actions > 1.)).any() or (external < 0.).any():
        raise ValueError('Invalid normalized action or external flow')
    csv_mode = physics.release_mapping_mode == 'csv_proposal'
    if csv_mode and not physics.correction_settings['enabled']:
        raise ValueError('csv_proposal requires environment-side correction enabled')
    map_low,map_high = (validated_proposal_bounds(proposal_low,proposal_high,shape) if csv_mode else (None,None))
    data, dt = physics.data, physics.dt
    storage_min, storage_max = _bounds(physics, shape, storage_lower, storage_upper)
    if ((storage < data.vmin-1.) | (storage > data.vmax+1.)).any():
        raise _infeasible('Starting storage violates operating bounds', storage, external,
            ((storage < data.vmin-1.) | (storage > data.vmax+1.)).any(axis=1))

    lower = np.zeros(shape, dtype=np.float64)
    upper = np.broadcast_to(data.release_cap, shape).copy()
    for i in range(len(IDS)-2, -1, -1):
        lower[:, i] = np.maximum(0., lower[:, i+1]-external[:, i+1]+
            (storage_min[:, i+1]-storage[:, i+1])/dt)
        upper[:, i] = np.minimum(data.release_cap[i], upper[:, i+1]-external[:, i+1]+
            (storage_max[:, i+1]-storage[:, i+1])/dt)
        if (lower[:, i] > upper[:, i]+1e-8).any():
            raise _infeasible('Empty downstream-compatible release interval', storage, external,
                lower[:, i] > upper[:, i]+1e-8, i, lower[:, i], upper[:, i])

    inflow, release, low, high, target = [np.empty(shape, dtype=np.float64) for _ in range(5)]
    reasons = np.full(shape, 'none', dtype='<U96')
    controls = np.ones(shape, dtype=np.float32)
    executed_actions = actions.copy()
    for i in range(len(IDS)):
        inflow[:, i] = external[:, i]+(release[:, i-1] if i else 0.)
        low[:, i] = np.maximum(lower[:, i], inflow[:, i]+(storage[:, i]-storage_max[:, i])/dt)
        high[:, i] = np.minimum(upper[:, i], inflow[:, i]+(storage[:, i]-storage_min[:, i])/dt)
        if (low[:, i] > high[:, i]+1e-8).any():
            raise _infeasible('Empty feasible release interval at reservoir '+str(i), storage, external,
                low[:, i] > high[:, i]+1e-8, i, low[:, i], high[:, i])
        forced = high[:, i]-low[:, i] <= 1e-8
        controls[forced, i] = 0.
        executed_actions[forced, i] = .5
        midpoint = np.minimum(data.release_cap[i], np.maximum(0., .5*(low[:, i]+high[:, i])))
        target[:, i] = (map_low[:,i]+actions[:,i]*(map_high[:,i]-map_low[:,i]) if csv_mode else
                        low[:, i]+actions[:, i]*(high[:, i]-low[:, i]))
        if physics.correction_settings['enabled']:
            release[:, i] = project_release(target[:, i], low[:, i], high[:, i], data.release_cap[i])
        else:
            release[:, i] = np.where(forced, midpoint, target[:, i])
        changed = release[:, i] != target[:, i]
        water_low = inflow[:, i]+(storage[:, i]-storage_max[:, i])/dt
        water_high = inflow[:, i]+(storage[:, i]-storage_min[:, i])/dt
        below_reason = np.where(low[:, i] == 0., 'outflow_nonnegative',
            np.where(np.isclose(low[:, i], water_low, rtol=0., atol=FLOW_TOLERANCE),
                     'next_day_storage_max', 'downstream_compatibility'))
        above_reason = np.where(
            np.isclose(high[:, i], water_high, rtol=0., atol=FLOW_TOLERANCE),
            'next_day_storage_min',
            np.where(np.isclose(high[:, i], data.release_cap[i], rtol=0., atol=FLOW_TOLERANCE),
                     'outflow_capacity', 'downstream_compatibility'))
        reasons[:, i] = np.where(changed,
            np.where(forced, 'forced_dispatch',
                     np.where(target[:, i] < low[:, i], below_reason, above_reason)), 'none')

    next_storage = storage+(inflow-release)*dt
    residual = next_storage-storage-(inflow-release)*dt
    tolerance = np.maximum(1., 1e-10*np.abs(storage))
    if ((next_storage < storage_min-tolerance) | (next_storage > storage_max+tolerance)).any():
        raise _infeasible('Storage bound violation after execution', storage, external,
            ((next_storage < storage_min-tolerance) | (next_storage > storage_max+tolerance)).any(axis=1))
    if (release < -1e-8).any() or (release > data.release_cap+1e-8).any():
        raise _infeasible('Release capacity violation', storage, external,
            ((release < -1e-8) | (release > data.release_cap+1e-8)).any(axis=1))

    result = {}
    if physics.head_model == 'fixed':
        head = np.broadcast_to(data.head, shape).copy()
        coefficient = physics.coefficient
        cap = np.broadcast_to(physics.effective_turbine_cap, shape).copy()
    elif physics.head_model == 'storage_sensitivity':
        start_level = _forebay_levels(physics, storage)
        end_level = _forebay_levels(physics, next_storage)
        mean_level = (start_level+end_level)/2.
        scenario = data.head_sensitivity
        head = (scenario['reference_net_head_m']+scenario['head_slope_m_per_m']*
                (mean_level-scenario['assumed_reference_forebay_level_m']))
        if not np.isfinite(head).all() or (head <= 0.).any():
            raise ValueError('Calculated net head must be finite and positive')
        coefficient = physics.rho_g_eta*head/1e6
        cap = np.minimum(data.turbine_cap, data.power_cap/coefficient)
        result['forebay_level_mean_m'] = mean_level
    else:
        raise ValueError('Unknown head model: '+str(physics.head_model))
    turbine = np.minimum(release, cap)
    power = turbine*coefficient
    energy = 24.*power
    if (np.abs(residual) > tolerance).any() or (power > data.power_cap+1e-7).any():
        raise AssertionError('Water/energy calculation failed')
    system_residual = (next_storage.sum(axis=1)-storage.sum(axis=1)-
                       (external.sum(axis=1)-release[:, -1])*dt)
    if (np.abs(system_residual) > tolerance.sum(axis=1)).any():
        raise AssertionError('Cascade water conservation failed')
    result.update(storage_start=storage.copy(), storage_end=next_storage,
        external=external.copy(), inflow=inflow, actions=executed_actions,
        release=release, low=low, high=high, controls=controls, residual=residual,
        net_head_m=head, effective_turbine_cap_m3s=cap.copy(), turbine=turbine,
        spill=release-turbine, power=power, energy=energy,
        reward=energy.sum(axis=1)/physics.reference_energy, system_residual=system_residual)
    proposal_l,proposal_h = (map_low,map_high) if csv_mode else (low,high)
    result.update(action_correction_fields(actions,target,release,proposal_l,proposal_h,
        controls,physics.correction_settings,reasons,physical_low=low,physical_high=high))
    if csv_mode:
        result.update(low=map_low.copy(),high=map_high.copy(),execution_low=low.copy(),execution_high=high.copy())
    result['actions'][result['controls'] <= 0.] = .5
    result['original_action_available'] = np.ones(shape, dtype=bool)
    if storage_lower is not None:
        result.update(storage_lower_m3=storage_min.copy(), storage_upper_m3=storage_max.copy())
    return result
