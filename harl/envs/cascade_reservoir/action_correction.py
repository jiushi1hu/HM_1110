"""Explicit action correction settings and finite-safe transition diagnostics.

The paper defines a correction operator but not its numerical implementation.
Here the existing physical constraints are unchanged. Corrections are distinct
from forced dispatch and from roundoff-sized release adjustments.
"""
import numpy as np

FLOW_TOLERANCE = 1e-8
INTERVAL_TOLERANCE = 1e-8


def action_correction_settings(config):
    raw = config.get('action_correction', {})
    if not isinstance(raw, dict):
        raise ValueError('action_correction must be a mapping')
    defaults = dict(enabled=False, tau=.03, penalty_enabled=False, correction_weight=1.)
    if set(raw)-set(defaults):
        raise ValueError('Unknown action_correction settings: '+str(sorted(set(raw)-set(defaults))))
    result = dict(defaults, **raw)
    for name in ('enabled', 'penalty_enabled'):
        if not isinstance(result[name], (bool, np.bool_)):
            raise ValueError('action_correction.'+name+' must be a boolean')
        result[name] = bool(result[name])
    for name in ('tau', 'correction_weight'):
        value = result[name]
        if isinstance(value, (bool, np.bool_)):
            raise ValueError('action_correction.'+name+' must be finite and nonnegative')
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('action_correction.'+name+' must be finite and nonnegative') from exc
        if not np.isfinite(value) or value < 0.:
            raise ValueError('action_correction.'+name+' must be finite and nonnegative')
        result[name] = value
    return result


def correction_threshold_exceeded(magnitude, tau):
    magnitude = np.asarray(magnitude, dtype=np.float64)
    tolerance = 8.*np.finfo(np.float64).eps*np.maximum(np.maximum(1., magnitude), abs(tau))
    return magnitude-float(tau) > tolerance


def project_release(target, low, high, capacity):
    """Nearest scalar release conditional on already executed upstream flow.

    The caller has proved downstream viability by backward interval propagation.
    This stage is a lexicographic upstream-priority projection, not a global
    Euclidean joint projection. Only the existing physical interval is enforced.
    """
    target, low, high, capacity = np.broadcast_arrays(target, low, high, capacity)
    forced = high-low <= INTERVAL_TOLERANCE
    midpoint = np.minimum(capacity, np.maximum(0., .5*(low+high)))
    # np.where preserves the original bits of already feasible proposals.
    return np.where(forced, midpoint,
                    np.where(target < low, low, np.where(target > high, high, target)))


def validate_release_mapping_mode(config):
    mode = config.get('release_mapping_mode', 'joint_feasible')
    if mode not in ('joint_feasible', 'csv_proposal'):
        raise ValueError('release_mapping_mode must be joint_feasible or csv_proposal')
    if mode == 'csv_proposal' and not action_correction_settings(config)['enabled']:
        raise ValueError('csv_proposal requires environment-side action correction enabled')
    return mode


def validated_proposal_bounds(low, high, shape):
    if low is None or high is None:
        raise ValueError('CSV/original proposal mapping requires both proposal bounds')
    result = []
    for value in (low, high):
        value = np.asarray(value, dtype=np.float64)
        if value.shape not in ((5,), shape) or not np.isfinite(value).all() or (value < 0.).any():
            raise ValueError('Proposal bounds must be finite nonnegative matching five-vectors')
        result.append(np.broadcast_to(value, shape).copy())
    if (result[0] > result[1]).any():
        raise ValueError('Proposal bounds must be ordered')
    return result


def action_correction_fields(normalized_actions, target_release, executed_release,
                             low, high, controls, settings, reasons=None,
                             physical_low=None, physical_high=None):
    """Separate original proposal geometry from state-dependent execution bounds.

    Invalid inverse/scaled values have finite zero sentinels plus validity masks.
    Physical forcing is independent of proposal degeneracy. No outside inverse
    clipping or outside-domain Eq. (16) penalty is introduced.
    """
    values = [np.asarray(v, dtype=np.float64) for v in
              (normalized_actions, target_release, executed_release, low, high, controls)]
    shape = values[0].shape
    if not shape or shape[-1] != 5 or any(v.shape != shape or not np.isfinite(v).all() for v in values):
        raise ValueError('Action-correction fields require finite matching (...,5) arrays')
    original, target, executed, lower, upper, physical_controls = values
    if ((original < 0.) | (original > 1.)).any() or not np.isin(physical_controls, (0.,1.)).all():
        raise ValueError('Normalized actions must be in [0,1] and controls binary')
    if (lower > upper+INTERVAL_TOLERANCE).any():
        raise ValueError('Original mapping bounds must be ordered')
    plow = lower if physical_low is None else np.asarray(physical_low,dtype=float)
    phigh = upper if physical_high is None else np.asarray(physical_high,dtype=float)
    if any(v.shape != shape or not np.isfinite(v).all() for v in (plow,phigh)):
        raise ValueError('Physical bounds must match proposal shape')
    forced = (physical_controls <= 0.) | (phigh-plow <= INTERVAL_TOLERANCE)
    degenerate = upper-lower <= INTERVAL_TOLERANCE
    controllable = (~forced) & ~degenerate
    delta = executed-target
    flow_tolerance = np.maximum(FLOW_TOLERANCE,
        32.*np.finfo(float).eps*np.maximum.reduce([np.ones(shape),np.abs(target),np.abs(executed)]))
    genuine = (~forced) & (np.abs(delta) > flow_tolerance)
    numerical = (~forced) & (delta != 0.) & ~genuine
    outside = (executed < lower) | (executed > upper)
    valid = controllable & ~outside
    realized = np.zeros(shape)
    np.divide(executed-lower,upper-lower,out=realized,where=valid)
    signed = np.where(valid,realized-original,0.)
    magnitude = np.abs(signed)
    exceeded = genuine & valid & correction_threshold_exceeded(magnitude,settings['tau'])
    deficit = np.where(exceeded,np.maximum(magnitude-settings['tau'],0.),0.)
    scaled = np.zeros(shape)
    np.divide(np.abs(delta),upper-lower,out=scaled,where=~degenerate)
    reason = np.full(shape,'none',dtype='<U96')
    if reasons is not None:
        supplied = np.asarray(reasons,dtype='<U96')
        if supplied.shape != shape:
            raise ValueError('Correction reasons must match transition shape')
        reason[genuine] = supplied[genuine]
    reason[genuine & (reason == 'none')] = 'execution_feasibility'
    reason[numerical] = 'numerical_roundoff'
    reason[forced] = 'forced_dispatch'
    return dict(normalized_actions=original.copy(),target_release=target.copy(),executed_release=executed.copy(),
        proposal_low=lower.copy(),proposal_high=upper.copy(),physical_low=plow.copy(),physical_high=phigh.copy(),
        physical_controls=(~forced).astype(np.float32),controls=controllable.astype(np.float32),
        release_correction_m3s=delta,correction_applied=genuine,correction_reason=reason,
        forced_dispatch=forced,degenerate_proposal=degenerate,numerical_adjustment=numerical,
        proposal_physical_disjoint=(upper < plow) | (lower > phigh),
        realized_normalized_actions=realized,inverse_action_valid=valid,
        executed_outside_mapping_interval=outside,normalized_action_correction=signed,
        normalized_action_correction_magnitude=magnitude,correction_deficit=deficit,
        correction_threshold_exceeded=exceeded,scaled_release_correction=scaled,
        scaled_release_correction_valid=~degenerate,
        action_correction_tau=np.full(shape,settings['tau']))
