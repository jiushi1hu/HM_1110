"""Separate CSV proposal intervention, normalized Eq.16 and physical forcing."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harl.envs.cascade_reservoir.action_correction import action_correction_settings
from harl.envs.cascade_reservoir.correction_metrics import (
    ActionCorrectionMetrics, aggregate_action_correction_summaries, action_correction_metric_fields)
from harl.envs.cascade_reservoir.counterfactual_credit import system_rewards_from_dispatch
from harl.envs.cascade_reservoir.data_loader import IDS
from harl.envs.cascade_reservoir.reward import OperationReward, action_correction_penalty_values
from test_xld_dispatch_diagnostics import synthetic_dispatch


def settings(enabled=True):
    return action_correction_settings({'action_correction': dict(
        enabled=True, penalty_enabled=enabled, tau=.03, correction_weight=2.)})


def mixed_audit():
    """Five distinct contracts: outside, degenerate proposal, forced, ULP, valid."""
    original = np.array([.9, .8, .25, .2, .2])
    target = np.array([90., 50., 25., 20., 20.])
    executed = np.array([120., 80., 30., 20.+1e-10, 40.])
    width = np.array([100., 0., 100., 100., 100.])
    scaled = np.divide(np.abs(executed-target), width, out=np.zeros(5), where=width > 0)
    return dict(normalized_actions=original, target_release=target, executed_release=executed,
        release_correction_m3s=executed-target, physical_controls=np.array([1, 1, 0, 1, 1]),
        controls=np.array([1, 0, 0, 1, 1]), inverse_action_valid=np.array([0, 0, 0, 1, 1]),
        correction_applied=np.array([1, 1, 0, 0, 1]), forced_dispatch=np.array([0, 0, 1, 0, 0]),
        numerical_adjustment=np.array([0, 0, 0, 1, 0]), degenerate_proposal=np.array([0, 1, 0, 0, 0]),
        proposal_physical_disjoint=np.array([1, 1, 0, 0, 0]),
        executed_outside_mapping_interval=np.array([1, 1, 0, 0, 0]),
        realized_normalized_actions=np.array([0., 0., 0., executed[3]/100., .4]),
        normalized_action_correction_magnitude=np.array([0., 0., 0., abs(executed[3]/100.-.2), .2]),
        scaled_release_correction=scaled, scaled_release_correction_valid=width > 0,
        action_correction_tau=.03)


def identity_audit():
    row = mixed_audit()
    row.update(physical_controls=np.ones(5), controls=np.ones(5), inverse_action_valid=np.ones(5),
        correction_applied=np.zeros(5), forced_dispatch=np.zeros(5), numerical_adjustment=np.zeros(5),
        degenerate_proposal=np.zeros(5), proposal_physical_disjoint=np.zeros(5),
        executed_outside_mapping_interval=np.zeros(5), normalized_action_correction_magnitude=np.zeros(5),
        scaled_release_correction=np.zeros(5), scaled_release_correction_valid=np.ones(5),
        release_correction_m3s=np.zeros(5))
    row['executed_release'] = row['target_release'].copy()
    row['realized_normalized_actions'] = row['normalized_actions'].copy()
    return row


class CSVCorrectionMetricTest(unittest.TestCase):
    def test_physical_interventions_are_not_lost_to_proposal_or_inverse_filtering(self):
        metrics = ActionCorrectionMetrics(settings())
        metrics.update(mixed_audit())
        s = metrics.summary()
        self.assertEqual(s['observed_reservoir_days'], 5)
        self.assertEqual(s['physical_nonforced_reservoir_days'], 4)
        self.assertEqual(s['physical_intervention_count'], 3)
        self.assertEqual(s['genuine_correction_events'], 3)
        self.assertEqual(s['physical_intervention_rate'], .75)
        self.assertEqual(s['controllable_reservoir_days'], 3)
        self.assertEqual(s['valid_controllable_reservoir_days'], 2)
        self.assertEqual(s['correction_violation_count'], 1)
        self.assertEqual(s['correction_violation_rate'], .5)
        self.assertEqual(s['outside_mapping_interval_events'], 2)
        self.assertEqual(s['outside_correction_events'], 2)
        self.assertEqual(s['outside_correction_fraction'], 2/3)
        self.assertEqual(s['invalid_inverse_events'], 2)
        self.assertEqual(s['degenerate_proposal_events'], 1)
        self.assertEqual(s['proposal_physical_disjoint_events'], 2)
        self.assertEqual(s['forced_dispatch_events'], 1)
        self.assertEqual(s['forced_dispatch_frequency'], .2)
        self.assertEqual(s['numerical_adjustment_events'], 1)
        self.assertEqual(s['by_reservoir'][IDS[1]]['physical_intervention_rate'], 1.)
        self.assertIsNone(s['by_reservoir'][IDS[1]]['correction_violation_rate'])
        json.dumps(s, allow_nan=False)

    def test_magnitude_scope_retains_outside_and_scaled_forced_differences(self):
        metrics = ActionCorrectionMetrics()
        row = mixed_audit()
        metrics.update(row)
        s = metrics.summary()
        self.assertEqual(s['discharge_correction_abs_sum_m3s'], 80.)
        self.assertEqual(s['discharge_correction_abs_mean_m3s'], 80/3)
        self.assertEqual(s['discharge_correction_abs_max_m3s'], 30.)
        self.assertEqual(s['normalized_correction_event_count'], 1)
        self.assertEqual(s['normalized_correction_abs_mean'], .2)
        self.assertEqual(s['scaled_release_correction_valid_days'], 4)
        self.assertAlmostEqual(s['scaled_release_correction_sum'], .55, places=10)
        self.assertAlmostEqual(s['scaled_release_correction_mean'], .55/4, places=10)
        self.assertEqual(s['scaled_release_correction_max'], .3)
        self.assertGreater(s['all_release_difference_abs_sum_m3s'], 85.)
        self.assertAlmostEqual(s['by_reservoir'][IDS[2]]['scaled_release_correction_mean'], .05)
        self.assertEqual(s['by_reservoir'][IDS[2]]['physical_intervention_count'], 0)

    def test_pooling_uses_physical_and_normalized_event_counts_and_distinct_denominators(self):
        first, second = ActionCorrectionMetrics(), ActionCorrectionMetrics()
        first.update(mixed_audit())
        second.update(identity_audit())
        s = aggregate_action_correction_summaries([first.summary(), second.summary()])
        self.assertEqual(s['physical_intervention_rate'], 3/9)
        self.assertEqual(s['correction_violation_rate'], 1/7)
        self.assertEqual(s['forced_dispatch_frequency'], 1/10)
        self.assertEqual(s['discharge_correction_abs_mean_m3s'], 80/3)
        self.assertAlmostEqual(s['scaled_release_correction_mean'], .55/9, places=10)
        fields = action_correction_metric_fields(s)
        self.assertEqual(fields['physical_intervention_count'], 3)
        self.assertEqual(fields['physical_nonforced_reservoir_days'], 9)
        self.assertEqual(fields['outside_correction_events'], 2)

    def test_outside_scaled_correction_is_not_an_undocumented_penalty(self):
        deficit, cost = action_correction_penalty_values(settings(), mixed_audit(), (5,))
        np.testing.assert_allclose(deficit, [0., 0., 0., 0., .17])
        np.testing.assert_allclose(cost, [0., 0., 0., 0., .34])
        row = mixed_audit()
        row['scaled_release_correction'] *= 1000.
        np.testing.assert_array_equal(action_correction_penalty_values(settings(), row, (5,))[1], cost)
        np.testing.assert_array_equal(action_correction_penalty_values(settings(False), row, (5,))[1], np.zeros(5))

    def test_degenerate_proposal_and_physical_forcing_are_separate(self):
        row = mixed_audit()
        row.update(physical_controls=np.ones(5), controls=np.zeros(5),
            inverse_action_valid=np.zeros(5), correction_applied=np.ones(5),
            forced_dispatch=np.zeros(5), numerical_adjustment=np.zeros(5),
            degenerate_proposal=np.ones(5), executed_outside_mapping_interval=np.ones(5),
            scaled_release_correction_valid=np.zeros(5))
        metrics = ActionCorrectionMetrics()
        metrics.update(row)
        s = metrics.summary()
        self.assertEqual(s['physical_intervention_count'], 5)
        self.assertEqual(s['physical_intervention_rate'], 1.)
        self.assertEqual(s['degenerate_proposal_events'], 5)
        self.assertEqual(s['forced_dispatch_events'], 0)
        self.assertEqual(s['valid_controllable_reservoir_days'], 0)
        self.assertIsNone(s['correction_violation_rate'])
        self.assertIsNone(s['scaled_release_correction_mean'])
        np.testing.assert_array_equal(action_correction_penalty_values(settings(), row, (5,))[1], np.zeros(5))

    def test_numerical_outside_row_is_not_a_genuine_outside_correction(self):
        row = identity_audit()
        row['numerical_adjustment'][0] = 1
        row['executed_outside_mapping_interval'][0] = 1
        row['inverse_action_valid'][0] = 0
        metrics = ActionCorrectionMetrics()
        metrics.update(row)
        s = metrics.summary()
        self.assertEqual(s['outside_mapping_interval_events'], 1)
        self.assertEqual(s['outside_correction_events'], 0)
        self.assertEqual(s['physical_intervention_count'], 0)

    def test_new_fields_accept_previous_summary_without_inventing_scaled_samples(self):
        metrics = ActionCorrectionMetrics()
        metrics.update(identity_audit())
        old = metrics.summary()
        new_counts = ('observed_reservoir_days', 'physical_nonforced_reservoir_days',
            'normalized_correction_event_count', 'degenerate_proposal_events',
            'proposal_physical_disjoint_events', 'scaled_release_correction_valid_days',
            'outside_correction_events')
        for row in [old]+list(old['by_reservoir'].values()):
            for key in new_counts+('scaled_release_correction_sum', 'scaled_release_correction_max'):
                row.pop(key, None)
        merged = aggregate_action_correction_summaries([old])
        self.assertEqual(merged['physical_nonforced_reservoir_days'], 5)
        self.assertEqual(merged['observed_reservoir_days'], 5)
        self.assertEqual(merged['scaled_release_correction_valid_days'], 0)
        self.assertIsNone(merged['scaled_release_correction_mean'])

    def test_batched_counterfactual_cost_and_disabled_reward_remain_compatible(self):
        env, source = synthetic_dispatch(action=100)
        rows = []
        for audit in (mixed_audit(), identity_audit()):
            row = copy.deepcopy(source)
            row.update(audit)
            operation = OperationReward(env.data, dict(env.operation_reward.settings,
                action_correction=settings()))
            operation.apply(row)
            rows.append(row)
        keys = ('power', 'release', 'normalized_actions', 'realized_normalized_actions',
            'controls', 'inverse_action_valid', 'correction_applied', 'forced_dispatch',
            'numerical_adjustment', 'executed_outside_mapping_interval', 'degenerate_proposal')
        batch = {key:np.stack([row[key] for row in rows]) for key in keys}
        np.testing.assert_allclose(system_rewards_from_dispatch(operation, batch),
                                   [row['system_reward'] for row in rows], atol=1e-15)
        original = copy.deepcopy(source)
        extended = copy.deepcopy(source)
        extended.update(mixed_audit())
        disabled = OperationReward(env.data, dict(env.operation_reward.settings,
            action_correction=settings(False)))
        disabled.apply(original)
        disabled.apply(extended)
        for key in ('reward', 'local_rewards', 'mixed_rewards', 'system_reward'):
            np.testing.assert_array_equal(extended[key], original[key])


if __name__ == '__main__':
    unittest.main()
