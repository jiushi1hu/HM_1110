"""Eq. (16), local/global accounting, counterfactual parity and pooled rates."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harl.envs.cascade_reservoir.action_correction import (
    action_correction_fields, action_correction_settings)
from harl.envs.cascade_reservoir.correction_metrics import (
    ActionCorrectionMetrics, aggregate_action_correction_summaries)
from harl.envs.cascade_reservoir.counterfactual_credit import system_rewards_from_dispatch
from harl.envs.cascade_reservoir.data_loader import IDS
from harl.envs.cascade_reservoir.reward import (
    OperationReward, EpisodeRewardMetrics, reward_settings, action_correction_penalty_values)
from test_xld_dispatch_diagnostics import synthetic_dispatch


def settings(**overrides):
    values = dict(enabled=True, penalty_enabled=True, tau=.03, correction_weight=1.)
    values.update(overrides)
    return action_correction_settings({'action_correction': values})


def audit(original=.2, executed=40., controls=None, low=None, high=None, tau=.03):
    controls = np.ones(5) if controls is None else np.asarray(controls, dtype=float)
    low = np.zeros(5) if low is None else np.asarray(low, dtype=float)
    high = np.full(5, 100.) if high is None else np.asarray(high, dtype=float)
    original = np.broadcast_to(original, (5,)).copy()
    executed = np.broadcast_to(executed, (5,)).copy()
    result = action_correction_fields(original, low+original*(high-low), executed,
        low, high, controls, settings(tau=tau))
    result['controls'] = controls
    return result


class CorrectionRewardTest(unittest.TestCase):
    def test_tolerance_exact_boundary_and_penalty_sign(self):
        for executed, expected in ((20., 0.), (23., 0.), (23.01, .0001), (10., .07)):
            with self.subTest(executed=executed):
                deficit, cost = action_correction_penalty_values(settings(), audit(executed=executed), (5,))
                np.testing.assert_allclose(deficit, expected, rtol=0., atol=1e-15)
                np.testing.assert_allclose(cost, deficit)
                self.assertTrue((-cost <= 0.).all())

    def test_local_mixed_system_accounting_and_global_counted_once(self):
        env, source = synthetic_dispatch(action=100)
        tr = copy.deepcopy(source)
        tr.update(audit())
        base_settings = dict(env.operation_reward.settings)
        baseline = OperationReward(env.data, base_settings)
        enabled = OperationReward(env.data, dict(base_settings, action_correction=settings(correction_weight=2.)))
        old, new = copy.deepcopy(tr), copy.deepcopy(tr)
        baseline.apply(old)
        enabled.apply(new)
        cost = np.full(5, .34)
        np.testing.assert_allclose(new['local_rewards'], old['local_rewards']-cost)
        np.testing.assert_allclose(new['mixed_rewards'], old['mixed_rewards']-.5*cost)
        self.assertAlmostEqual(new['system_reward'], old['system_reward']-.5*cost.sum())
        self.assertEqual(new['global_reward'], old['global_reward'])
        self.assertEqual(new['reward'], old['reward'])
        for term in ('guaranteed_output', 'ecological_flow'):
            np.testing.assert_array_equal(new['service_shortfalls'][term], old['service_shortfalls'][term])
            np.testing.assert_array_equal(new['local_reward_penalty_components'][term],
                                          old['local_reward_penalty_components'][term])
        np.testing.assert_allclose(new['r_corr'], -cost)
        metrics = EpisodeRewardMetrics(.9)
        metrics.update(new)
        metrics.update(new)
        summary = metrics.learning_summary()
        np.testing.assert_allclose(summary['episode_local_action_correction_penalties'], 1.9*cost)
        self.assertAlmostEqual(summary['episode_system_action_correction_penalty'], .5*1.9*cost.sum())

    def test_penalty_independent_from_service_switch(self):
        env, source = synthetic_dispatch(action=100)
        tr = copy.deepcopy(source)
        tr.update(audit())
        disabled_services = dict(env.operation_reward.settings, enabled=False, action_correction=settings())
        OperationReward(env.data, disabled_services).apply(tr)
        np.testing.assert_allclose(tr['local_reward_penalties'], .17)
        np.testing.assert_allclose(tr['local_reward_penalty_components']['ecological_flow'], 0.)
        np.testing.assert_allclose(tr['local_reward_penalty_components']['guaranteed_output'], 0.)

    def test_forced_roundoff_and_outside_inverse_are_not_false_penalties(self):
        row = audit(executed=[40., 20., 20.+1e-10, 120., 23.01], controls=[1, 0, 1, 1, 1])
        deficit, cost = action_correction_penalty_values(settings(), row, (5,))
        np.testing.assert_allclose(deficit, [.17, 0., 0., 0., .0001], atol=1e-15)
        np.testing.assert_allclose(cost, deficit)
        self.assertTrue(row['forced_dispatch'][1])
        self.assertTrue(row['numerical_adjustment'][2])
        self.assertTrue(row['executed_outside_mapping_interval'][3])
        self.assertFalse(row['inverse_action_valid'][3])
        self.assertEqual(row['release_correction_m3s'][3], 100.)

    def test_disabled_penalty_accepts_legacy_while_enabled_requires_valid_audit(self):
        disabled = dict(settings(), penalty_enabled=False)
        deficit, cost = action_correction_penalty_values(disabled, {}, (5,))
        np.testing.assert_array_equal(deficit, np.zeros(5))
        np.testing.assert_array_equal(cost, np.zeros(5))
        with self.assertRaises(ValueError):
            action_correction_penalty_values(settings(), {}, (5,))
        row = audit()
        row['controls'][0] = .5
        with self.assertRaises(ValueError):
            action_correction_penalty_values(settings(), row, (5,))

    def test_scalar_and_counterfactual_batched_reward_parity_with_real_penalties(self):
        env, source = synthetic_dispatch(action=100)
        operation = OperationReward(env.data, dict(env.operation_reward.settings,
            action_correction=settings(correction_weight=2.)))
        transitions = []
        for executed in (20., 23., 40., 120.):
            tr = copy.deepcopy(source)
            tr.update(audit(executed=executed))
            operation.apply(tr)
            transitions.append(tr)
        keys = ('power', 'release', 'normalized_actions', 'realized_normalized_actions',
                'controls', 'inverse_action_valid', 'correction_applied', 'forced_dispatch',
                'numerical_adjustment', 'executed_outside_mapping_interval')
        batched = {k: np.stack([row[k] for row in transitions]) for k in keys}
        np.testing.assert_allclose(system_rewards_from_dispatch(operation, batched),
                                   [row['system_reward'] for row in transitions], atol=1e-15)


class CorrectionMetricTest(unittest.TestCase):
    def test_pooled_denominator_discharge_statistics_and_separate_forcing(self):
        row = audit(executed=[40., 20., 20.+1e-10, 120., 20.], controls=[1, 0, 1, 1, 1])
        first = ActionCorrectionMetrics()
        first.update(row)
        summary = first.summary()
        self.assertEqual(summary['valid_controllable_reservoir_days'], 3)
        self.assertEqual(summary['controllable_reservoir_days'], 4)
        self.assertEqual(summary['genuine_correction_events'], 2)
        self.assertEqual(summary['correction_violation_count'], 1)
        self.assertEqual(summary['correction_violation_rate'], 1/3)
        self.assertEqual(summary['forced_dispatch_events'], 1)
        self.assertEqual(summary['numerical_adjustment_events'], 1)
        self.assertEqual(summary['outside_mapping_interval_events'], 1)
        self.assertEqual(summary['discharge_correction_abs_sum_m3s'], 120.)
        self.assertAlmostEqual(summary['normalized_correction_abs_sum'], .2)
        second = ActionCorrectionMetrics()
        second.update(audit(executed=20.))
        combined = aggregate_action_correction_summaries([summary, second.summary()])
        self.assertEqual(combined['correction_violation_rate'], 1/8)
        self.assertEqual(combined['by_reservoir'][IDS[0]]['correction_violation_rate'], .5)
        json.dumps(combined, allow_nan=False)

    def test_failure_attempts_do_not_invent_five_reservoir_observations(self):
        metrics = ActionCorrectionMetrics(settings())
        metrics.record_failure(2)
        metrics.record_failure()
        summary = metrics.summary()
        self.assertEqual(summary['no_feasible_execution_events'], 2)
        self.assertEqual(summary['failed_unknown_reservoir_events'], 1)
        self.assertEqual(summary['by_reservoir'][IDS[2]]['no_feasible_execution_events'], 1)
        self.assertEqual(summary['valid_controllable_reservoir_days'], 0)
        self.assertIsNone(summary['correction_violation_rate'])
        with self.assertRaises(ValueError):
            metrics.record_failure(5)

    def test_tolerance_override_and_legacy_missing_data_are_explicit(self):
        metrics = ActionCorrectionMetrics()
        metrics.update(audit(executed=24., tau=.05))
        self.assertEqual(metrics.summary()['correction_violation_count'], 0)
        self.assertEqual(metrics.summary()['correction_tolerance_tau'], .05)
        with self.assertRaises(ValueError):
            metrics.update(audit(tau=.03))
        legacy = ActionCorrectionMetrics()
        legacy.update({})
        summary = legacy.summary()
        self.assertEqual(summary['unobserved_reservoir_days'], 5)
        self.assertEqual(summary['valid_controllable_reservoir_days'], 0)
        self.assertIsNone(summary['correction_violation_rate'])


if __name__ == '__main__':
    unittest.main()
