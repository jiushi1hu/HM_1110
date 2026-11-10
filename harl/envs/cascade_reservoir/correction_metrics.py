"""Undiscounted physical interventions, action corrections and forcing audits."""
import numpy as np
from harl.envs.cascade_reservoir.data_loader import IDS
from harl.envs.cascade_reservoir.action_correction import (
    action_correction_settings, correction_threshold_exceeded)

COUNT_FIELDS = ('observed_reservoir_days', 'physical_nonforced_reservoir_days',
    'valid_controllable_reservoir_days', 'controllable_reservoir_days',
    'correction_violation_count', 'genuine_correction_events',
    'normalized_correction_event_count', 'forced_dispatch_events',
    'degenerate_proposal_events', 'proposal_physical_disjoint_events',
    'scaled_release_correction_valid_days', 'numerical_adjustment_events',
    'outside_mapping_interval_events', 'outside_correction_events', 'invalid_inverse_events',
    'unobserved_reservoir_days')
SUM_FIELDS = ('discharge_correction_abs_sum_m3s', 'discharge_correction_signed_sum_m3s',
    'normalized_correction_abs_sum', 'scaled_release_correction_sum',
    'all_release_difference_abs_sum_m3s')
MAX_FIELDS = ('discharge_correction_abs_max_m3s', 'normalized_correction_abs_max',
    'scaled_release_correction_max')


def _array(value, name, binary=False):
    value = np.asarray(value)
    if value.shape != (5,) or not np.isfinite(value).all():
        raise ValueError('Correction metrics require finite five-vector '+name)
    if binary and not np.isin(value, (0, 1)).all():
        raise ValueError('Correction audit flags must be binary: '+name)
    return value


def _rates_and_means(row):
    """Recompute derived fields from pooled counts, never from mean rates."""
    valid = row['valid_controllable_reservoir_days']
    physical = row['physical_nonforced_reservoir_days']
    observed = row['observed_reservoir_days']
    events = row['genuine_correction_events']
    normalized_events = row['normalized_correction_event_count']
    scaled_valid = row['scaled_release_correction_valid_days']
    row.update(correction_violation_rate=row['correction_violation_count']/valid if valid else None,
        physical_intervention_count=events,
        physical_intervention_rate=events/physical if physical else None,
        outside_correction_fraction=row['outside_correction_events']/events if events else None,
        forced_dispatch_frequency=row['forced_dispatch_events']/observed if observed else None,
        discharge_correction_abs_mean_m3s=row['discharge_correction_abs_sum_m3s']/events if events else None,
        normalized_correction_abs_mean=row['normalized_correction_abs_sum']/normalized_events if normalized_events else None,
        scaled_release_correction_mean=row['scaled_release_correction_sum']/scaled_valid if scaled_valid else None,
        all_release_difference_abs_mean_m3s=row['all_release_difference_abs_sum_m3s']/observed if observed else None)


class ActionCorrectionMetrics:
    """Separate physical intervention frequency from the valid-inverse Eq.16 rate.

    Physical events include repairs to an uncontrollable degenerate proposal when
    the physical interval itself is nondegenerate. Forced physical dispatch is
    separate. Scaled release differences are descriptive, never a penalty.
    """
    def __init__(self, settings=None):
        self.tau = None if settings is None else action_correction_settings(
            {'action_correction': settings})['tau']
        self.counts = {k: np.zeros(5, dtype=np.int64) for k in COUNT_FIELDS}
        self.sums = {k: np.zeros(5) for k in SUM_FIELDS}
        self.maxima = {k: np.zeros(5) for k in MAX_FIELDS}
        self.no_feasible_execution_events = 0
        self.failed_by_reservoir = np.zeros(5, dtype=np.int64)
        self.failed_unknown_reservoir_events = 0

    def _set_tau(self, value):
        value = np.asarray(value, dtype=float)
        if not value.size or not np.isfinite(value).all() or not np.all(value == value.flat[0]):
            raise ValueError('Correction tolerance must be a finite scalar or uniform array')
        tau = action_correction_settings({'action_correction': {'tau': float(value.flat[0])}})['tau']
        if self.tau is not None and self.tau != tau:
            raise ValueError('Cannot pool correction metrics with different tolerances')
        self.tau = tau

    def update(self, transition):
        if 'correction_applied' not in transition:
            self.counts['unobserved_reservoir_days'] += 1
            return
        self._set_tau(transition.get('action_correction_tau', .03 if self.tau is None else self.tau))
        flag_keys = ('controls', 'inverse_action_valid', 'correction_applied',
            'forced_dispatch', 'numerical_adjustment', 'executed_outside_mapping_interval')
        keys = flag_keys+('normalized_action_correction_magnitude', 'release_correction_m3s')
        if any(k not in transition for k in keys):
            raise ValueError('Incomplete action-correction metrics audit')
        a = {k: _array(transition[k], k, k in flag_keys) for k in keys}
        forced = a['forced_dispatch'].astype(bool)
        numeric = a['numerical_adjustment'].astype(bool)
        outside = a['executed_outside_mapping_interval'].astype(bool)
        # Missing new fields are supported for the previous dynamic-interval
        # audit, where physical and effective controllability were identical.
        physical_controls = _array(transition.get('physical_controls', a['controls']),
                                   'physical_controls', True).astype(bool)
        degenerate = _array(transition.get('degenerate_proposal', np.zeros(5)),
                            'degenerate_proposal', True).astype(bool)
        disjoint = _array(transition.get('proposal_physical_disjoint', np.zeros(5)),
                         'proposal_physical_disjoint', True).astype(bool)
        physical_active = physical_controls & ~forced
        active = (a['controls'] > 0) & physical_active & ~degenerate
        valid = active & a['inverse_action_valid'].astype(bool) & ~outside
        genuine = physical_active & a['correction_applied'].astype(bool) & ~numeric
        magnitude = a['normalized_action_correction_magnitude'].astype(float)
        if (magnitude < 0).any():
            raise ValueError('Normalized correction magnitude must be nonnegative')
        violated = genuine & valid & correction_threshold_exceeded(magnitude, self.tau)
        scaled_keys = ('scaled_release_correction', 'scaled_release_correction_valid')
        if any(k in transition for k in scaled_keys) and not all(k in transition for k in scaled_keys):
            raise ValueError('Incomplete scaled release-correction audit')
        scaled = _array(transition.get('scaled_release_correction', np.zeros(5)),
                        'scaled_release_correction').astype(float)
        scaled_valid = _array(transition.get('scaled_release_correction_valid', np.zeros(5)),
                              'scaled_release_correction_valid', True).astype(bool)
        if (scaled < 0).any():
            raise ValueError('Scaled release correction must be nonnegative')
        normalized_events = valid & genuine
        changes = dict(observed_reservoir_days=np.ones(5, dtype=bool),
            physical_nonforced_reservoir_days=physical_active,
            valid_controllable_reservoir_days=valid, controllable_reservoir_days=active,
            correction_violation_count=violated, genuine_correction_events=genuine,
            normalized_correction_event_count=normalized_events,
            forced_dispatch_events=forced, degenerate_proposal_events=degenerate,
            proposal_physical_disjoint_events=disjoint,
            scaled_release_correction_valid_days=scaled_valid,
            numerical_adjustment_events=numeric & physical_active,
            outside_mapping_interval_events=outside & physical_active,
            outside_correction_events=outside & genuine,
            invalid_inverse_events=physical_active & ~valid)
        for k, value in changes.items():
            self.counts[k] += value
        delta = a['release_correction_m3s'].astype(float)
        flow = np.where(genuine, np.abs(delta), 0.)
        normalized = np.where(normalized_events, magnitude, 0.)
        scaled = np.where(scaled_valid, scaled, 0.)
        self.sums['discharge_correction_abs_sum_m3s'] += flow
        self.sums['discharge_correction_signed_sum_m3s'] += np.where(genuine, delta, 0.)
        self.sums['normalized_correction_abs_sum'] += normalized
        self.sums['scaled_release_correction_sum'] += scaled
        self.sums['all_release_difference_abs_sum_m3s'] += np.abs(delta)
        self.maxima['discharge_correction_abs_max_m3s'] = np.maximum(self.maxima['discharge_correction_abs_max_m3s'], flow)
        self.maxima['normalized_correction_abs_max'] = np.maximum(self.maxima['normalized_correction_abs_max'], normalized)
        self.maxima['scaled_release_correction_max'] = np.maximum(self.maxima['scaled_release_correction_max'], scaled)

    def record_failure(self, reservoir_index=None):
        if reservoir_index is not None and (isinstance(reservoir_index, (bool, np.bool_)) or
                not isinstance(reservoir_index, (int, np.integer)) or not 0 <= reservoir_index < 5):
            raise ValueError('Failed reservoir index must be in [0,4]')
        self.no_feasible_execution_events += 1
        if reservoir_index is None:
            self.failed_unknown_reservoir_events += 1
        else:
            self.failed_by_reservoir[int(reservoir_index)] += 1

    def merge(self, summary):
        self._set_tau(summary['correction_tolerance_tau'])
        for i, name in enumerate(IDS):
            row = summary['by_reservoir'][name]
            legacy = dict(observed_reservoir_days=row.get('controllable_reservoir_days', 0)+row.get('forced_dispatch_events', 0),
                          physical_nonforced_reservoir_days=row.get('controllable_reservoir_days', 0))
            for k in self.counts:
                self.counts[k][i] += int(row.get(k, legacy.get(k, 0)))
            for k in self.sums:
                self.sums[k][i] += float(row.get(k, 0.))
            for k in self.maxima:
                self.maxima[k][i] = max(self.maxima[k][i], float(row.get(k, 0.)))
            self.failed_by_reservoir[i] += int(row['no_feasible_execution_events'])
        self.no_feasible_execution_events += int(summary['no_feasible_execution_events'])
        self.failed_unknown_reservoir_events += int(summary['failed_unknown_reservoir_events'])

    def summary(self):
        by = {}
        for i, name in enumerate(IDS):
            row = {k: int(v[i]) for k, v in self.counts.items()}
            row.update({k: float(v[i]) for k, v in self.sums.items()})
            row.update({k: float(v[i]) for k, v in self.maxima.items()})
            row['no_feasible_execution_events'] = int(self.failed_by_reservoir[i])
            _rates_and_means(row)
            by[name] = row
        result = {k: int(v.sum()) for k, v in self.counts.items()}
        result.update({k: float(v.sum()) for k, v in self.sums.items()})
        result.update({k: float(v.max()) for k, v in self.maxima.items()})
        _rates_and_means(result)
        result.update(reservoir_ids=list(IDS), correction_tolerance_tau=.03 if self.tau is None else self.tau,
            no_feasible_execution_events=self.no_feasible_execution_events,
            failed_unknown_reservoir_events=self.failed_unknown_reservoir_events, by_reservoir=by,
            rate_definition='genuine corrections with abs(u_original-u_realized)>tau / valid controllable reservoir-days; threshold roundoff excluded',
            denominator_exclusions='physical forced dispatch, degenerate proposal, invalid/outside-interval inverse, unobserved legacy rows; infeasible attempts counted separately',
            physical_intervention_definition='genuine physical release changes / physical nonforced reservoir-days; includes degenerate-proposal repairs',
            forced_dispatch_definition='physically degenerate feasible interval; frequency denominator is all observed executed reservoir-days',
            scaled_release_correction_definition='abs(executed-target)/proposal_width for nondegenerate proposals; descriptive only; mean over all valid-width rows including zero changes and physical forced dispatch',
            magnitude_scope='discharge mean/max/sum over genuine events; normalized mean/max/sum over genuine valid-inverse events; all_release_difference statistics also retain forced/numerical differences',
            infeasibility_definition='failed execution attempts; an unknown station failure is not duplicated across five reservoirs')
        return result


def aggregate_action_correction_summaries(summaries):
    """Pool counts/sums, take maxima and recompute rates rather than averaging."""
    metrics = ActionCorrectionMetrics()
    for summary in summaries:
        metrics.merge(summary)
    return metrics.summary()


def action_correction_metric_fields(summary):
    return dict(action_correction_violation_rate=summary['correction_violation_rate'],
        action_correction_violation_count=summary['correction_violation_count'],
        action_correction_active_reservoir_days=summary['valid_controllable_reservoir_days'],
        physical_intervention_rate=summary['physical_intervention_rate'],
        physical_intervention_count=summary['physical_intervention_count'],
        physical_nonforced_reservoir_days=summary['physical_nonforced_reservoir_days'],
        outside_correction_events=summary['outside_correction_events'],
        action_correction_summary=summary)
