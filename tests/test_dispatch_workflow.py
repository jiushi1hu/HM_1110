"""Frozen replay reconstruction and diagnostic accounting integration checks."""
import copy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.dispatch_diagnostics.analyze_run import DailyDispatchWriter, verify_replay
from analysis.dispatch_diagnostics.report import aggregate_rows
from harl.envs.cascade_reservoir.checkpoint import load_checkpoint, save_checkpoint
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.evaluation import evaluate_policy
from harl.envs.cascade_reservoir.periodic_evaluation import select_periodic_windows
from harl.envs.cascade_reservoir.xld_diagnostics import (
    CLASSIFICATION_DEFINITIONS, XLDDispatchDiagnosticError, aggregate_xld_summaries)
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner


class DispatchWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = yaml.safe_load((ROOT/'harl/configs/algos_cfgs/happo_cascade.yaml').read_text())
        cls.physical = yaml.safe_load((ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text())
        args['train'].update(episode_length=3, n_rollout_threads=1)
        args['model'].update(hidden_sizes=[8, 8], action_bins=7)
        args['experiment'].update(episodes=1, seeds=[1])
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new', cls.physical, args['experiment'])
        cls.runner = CascadeReservoirRunner(cls.data, cls.physical, args, 1,
            periodic_windows=select_periodic_windows(cls.data, episode_days=3))
        # This fixture tests inference/analysis; no optimizer updates are needed.
        cls.runner.update_count = cls.runner.completed_episodes = 1
        cls.runner.training_steps = 3
        cls.directory = tempfile.TemporaryDirectory()
        path = Path(cls.directory.name)/'policy.pt'
        save_checkpoint(cls.runner, path)
        cls.policy = load_checkpoint(path)

    @classmethod
    def tearDownClass(cls):
        cls.runner.close()
        cls.directory.cleanup()

    def test_baseline_matches_saved_policy_and_aggregates_without_double_counting(self):
        stream = io.StringIO()
        writer = DailyDispatchWriter(stream, self.policy, [1965], 'test_window')
        state = copy.deepcopy(self.policy.critic.state_dict())
        rng = torch.get_rng_state().clone()
        result = self.policy.replay([1965], writer=writer)
        reference = evaluate_policy(self.policy.actors, self.data, self.physical,
            seed=1, episode=1, training_steps=3, gamma=self.policy.gamma,
            source_years=[1965], episode_days=3, diagnostics=True)
        self.assertEqual(verify_replay(result, reference, periodic=False)['status'],
                         'matched_original_evaluation')
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in state.items():
            self.assertTrue(torch.equal(value, self.policy.critic.state_dict()[name]))
        self.assertEqual((writer.completed_days, writer.rows), (3, 15))
        np.testing.assert_array_equal(writer.env.storage, result['final_storage_m3'])
        rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
        report = aggregate_rows(rows, dict(gamma=self.policy.gamma))
        self.assertEqual(report['row_count'], 15)
        self.assertEqual(len(report['accounting']), 1)
        account = report['accounting'][0]
        self.assertEqual(account['executed_days'], 3)
        np.testing.assert_allclose(account['generation_MWh'], result['energy_MWh'], rtol=1e-12)
        self.assertAlmostEqual(account['discounted_legacy_team_reward'], result['episode_reward'], places=12)
        self.assertAlmostEqual(account['discounted_system_reward'], result['episode_system_reward'], places=12)
        for row in report['by_phase_term']:
            original = result['service_shortfalls'][row['term']]
            self.assertEqual(row['violation_count'], original['reservoir_day_count'])
            self.assertEqual(row['active_reservoir_days'], original['active_reservoir_days'])
        for row in report['mixed_reward_accounting']:
            i = ['WDD', 'BHT', 'XLD', 'XJB', 'THR'].index(row['reservoir_id'])
            self.assertAlmostEqual(row['discounted_mixed_return'], result['episode_mixed_rewards'][i], places=12)
        bad = dict(reference, energy_MWh=reference['energy_MWh']+1.)
        with self.assertRaises(AssertionError):
            verify_replay(result, bad, periodic=False)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            aggregate_rows(rows+rows[:1], dict(gamma=self.policy.gamma))

    def test_partial_or_wrong_order_replay_is_rejected(self):
        writer = DailyDispatchWriter(io.StringIO(), self.policy, [1965], 'test_window')
        with self.assertRaises(ValueError):
            writer.write(dict(reservoir='BHT', episode_step=1))
        with self.assertRaises(ValueError):
            writer.flush()


class XLDCompleteSummaryAggregationTest(unittest.TestCase):
    @staticmethod
    def summary(classes=None, **changes):
        counts = dict.fromkeys(CLASSIFICATION_DEFINITIONS, 0)
        counts.update(classes or dict(no_violation=2, same_day_policy_alternative=1,
            discrete_grid_gap=1, same_state_capacity_limit=1, same_state_water_limit=1,
            same_state_dispatch_limit=1, same_state_head_limit=1))
        days = sum(counts.values())
        active = days-counts['requirement_inactive']
        violations = active-counts['no_violation']
        result = dict(schema='xld_dispatch_v1', reservoir_id='XLD', phase='training_evaluation',
            guaranteed_output_mw=10., action_bins=100, head_model='storage_sensitivity',
            seed=1, evaluation_round=50, window_id='dry_1965', completed_steps=days,
            active_days=active, violation_count=violations,
            violation_rate=violations/active if active else None,
            classification_counts=counts)
        result.update(changes)
        return result

    def test_counts_are_pooled_and_nested_aggregation_is_equivalent(self):
        first = self.summary(dict(no_violation=8, same_day_policy_alternative=1,
                                  same_state_water_limit=1))
        second = self.summary(dict(no_violation=4, same_day_policy_alternative=4,
            discrete_grid_gap=2, same_state_capacity_limit=1, same_state_water_limit=3,
            same_state_dispatch_limit=2, same_state_head_limit=4),
            seed=2, window_id='wet_1970')
        before = copy.deepcopy([first, second])
        pooled = aggregate_xld_summaries([first, second])
        self.assertEqual(pooled['completed_steps'], 30)
        self.assertEqual(pooled['active_days'], 30)
        self.assertEqual(pooled['violation_count'], 18)
        self.assertEqual(pooled['violation_rate'], .6)
        self.assertNotEqual(pooled['violation_rate'],
                            (first['violation_rate']+second['violation_rate'])/2.)
        self.assertEqual(pooled['group_counts'], dict(actor_missed_feasible_discrete=5,
            discrete_grid_gap=2, same_state_continuous_unattainable=11))
        self.assertEqual(pooled['group_fractions_among_violations'],
            {name: value/18 for name, value in pooled['group_counts'].items()})
        self.assertEqual(sum(pooled['classification_counts'].values()), 30)
        self.assertEqual(sum(pooled['group_counts'].values()), 18)
        self.assertEqual(pooled['component_count'], 2)
        self.assertEqual(pooled['seeds'], [1, 2])
        nested = aggregate_xld_summaries([aggregate_xld_summaries([first]),
                                          aggregate_xld_summaries([second])])
        self.assertEqual(nested, pooled)
        self.assertEqual([first, second], before)
        json.dumps(pooled, allow_nan=False)
        self.assertIn('not proof', pooled['limitation'])
        self.assertIn('other four original sampled commands stay fixed', pooled['counterfactual_scope'])

    def test_no_violations_and_inactive_requirements_use_null_fractions(self):
        for summary in (self.summary(dict(no_violation=10)),
                        self.summary(dict(requirement_inactive=10), guaranteed_output_mw=0.)):
            with self.subTest(threshold=summary['guaranteed_output_mw']):
                result = aggregate_xld_summaries([summary])
                self.assertEqual(result['violation_count'], 0)
                self.assertTrue(all(value == 0 for value in result['group_counts'].values()))
                self.assertTrue(all(value is None
                                    for value in result['group_fractions_among_violations'].values()))
                self.assertEqual(result['violation_rate'],
                                 0. if result['active_days'] else None)
                json.dumps(result, allow_nan=False)

    def test_identity_mismatches_are_not_silently_pooled(self):
        first = self.summary()
        for changes in (dict(reservoir_id='WDD'), dict(guaranteed_output_mw=11.),
                        dict(action_bins=101), dict(head_model='fixed'),
                        dict(phase='evaluation')):
            with self.subTest(changes=changes), self.assertRaises(XLDDispatchDiagnosticError):
                aggregate_xld_summaries([first, self.summary(**changes)])
        for field in ('reservoir_id', 'guaranteed_output_mw', 'action_bins', 'head_model',
                      'phase', 'completed_steps', 'active_days', 'violation_count',
                      'classification_counts', 'violation_rate', 'window_id', 'seed',
                      'evaluation_round', 'schema'):
            bad = self.summary()
            del bad[field]
            with self.subTest(missing=field), self.assertRaises(XLDDispatchDiagnosticError):
                aggregate_xld_summaries([bad])

    def test_incomplete_or_unreconciled_counts_are_rejected(self):
        for changes in (dict(status='failed'), dict(completed=False), dict(complete=False),
                        dict(episode_horizon_days=9), dict(completed_steps=0),
                        dict(completed_steps=9), dict(active_days=7), dict(violation_count=7),
                        dict(violation_count=True), dict(violation_rate=.123),
                        dict(schema='unknown')):
            with self.subTest(changes=changes), self.assertRaises(XLDDispatchDiagnosticError):
                aggregate_xld_summaries([self.summary(**changes)])
        valid = self.summary(status='ok', completed=True, episode_horizon_days=8)
        self.assertEqual(aggregate_xld_summaries([valid])['completed_steps'], 8)
        for name, value in (('no_violation', -1), ('no_violation', True),
                            ('no_violation', 2.0), ('unrecognized_class', 0)):
            bad = self.summary()
            bad['classification_counts'][name] = value
            with self.subTest(classification=name, value=value), self.assertRaises(XLDDispatchDiagnosticError):
                aggregate_xld_summaries([bad])
        bad = self.summary()
        del bad['classification_counts']['same_state_head_limit']
        with self.assertRaises(XLDDispatchDiagnosticError):
            aggregate_xld_summaries([bad])
        for source in ([], None, [None], [self.summary(), None]):
            with self.subTest(source=source), self.assertRaises(XLDDispatchDiagnosticError):
                aggregate_xld_summaries(source)

    def test_nested_group_and_fraction_metadata_must_reconcile(self):
        original = aggregate_xld_summaries([self.summary()])
        for field, key, value in (
                ('group_counts', 'discrete_grid_gap', 2),
                ('group_counts', 'unknown_group', 0),
                ('group_fractions_among_violations', 'discrete_grid_gap', .9),
                ('group_fractions_among_violations', 'discrete_grid_gap', None),
                ('group_fractions_among_violations', 'discrete_grid_gap', float('nan'))):
            bad = copy.deepcopy(original)
            bad[field][key] = value
            with self.subTest(field=field, key=key, value=value), self.assertRaises(XLDDispatchDiagnosticError):
                aggregate_xld_summaries([bad])
        for field in ('group_counts', 'group_fractions_among_violations'):
            bad = copy.deepcopy(original)
            del bad[field]['discrete_grid_gap']
            with self.subTest(missing=field), self.assertRaises(XLDDispatchDiagnosticError):
                aggregate_xld_summaries([bad])
        for changes in (dict(component_count=2), dict(window_ids=[]), dict(seeds=[]),
                        dict(evaluation_rounds=[True])):
            with self.subTest(changes=changes), self.assertRaises(XLDDispatchDiagnosticError):
                aggregate_xld_summaries([dict(original, **changes)])



if __name__ == '__main__':
    unittest.main(verbosity=2)
