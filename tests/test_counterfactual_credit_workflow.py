"""Counterfactual credit through the real cascade collection/update lifecycle."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import csv
import gzip
import io
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples import train_cascade as cli
from examples import train_cascade_compare as compare
from harl.envs.cascade_reservoir.checkpoint import load_checkpoint
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.periodic_evaluation import select_periodic_windows
from harl.runners.cascade_mappo_runner import CascadeMAPPORunner
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
from harl.runners.on_policy_ha_runner import OnPolicyHARunner


def assert_nested_equal(case, first, second):
    if isinstance(first, torch.Tensor):
        case.assertTrue(torch.equal(first, second))
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        case.assertEqual(first.keys(), second.keys())
        for key in first:
            assert_nested_equal(case, first[key], second[key])
    elif isinstance(first, (tuple, list)):
        case.assertEqual(len(first), len(second))
        for left, right in zip(first, second):
            assert_nested_equal(case, left, right)
    else:
        case.assertEqual(first, second)


def training_state(runner):
    return copy.deepcopy(dict(
        actors=[actor.actor.state_dict() for actor in runner.actor],
        actor_optimizers=[actor.actor_optimizer.state_dict() for actor in runner.actor],
        critic=runner.critic.critic.state_dict(),
        critic_optimizer=runner.critic.critic_optimizer.state_dict(),
        normalization=None if runner.value_normalizer is None else runner.value_normalizer.state_dict(),
        python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
    ))


class CounterfactualActorDispatchTest(unittest.TestCase):
    def test_own_frozen_credit_is_not_centered_or_normalized_a_second_time(self):
        runner = OnPolicyHARunner()
        runner.num_agents, runner.fixed_order, runner.action_aggregation = 5, True, 'prod'
        runner.credit_assignment = 'counterfactual'
        runner.algo_args = dict(train=dict(episode_length=4, n_rollout_threads=2),
            algo=dict(happo_credit_assignment='counterfactual', normalize_shared_advantages=True,
                      mask_inactive_factors=True))
        runner.value_normalizer = None
        runner.critic_buffer = SimpleNamespace(returns=np.full((5, 2, 1), 100.),
                                              value_preds=np.full((5, 2, 1), -100.))
        credit = (np.arange(40, dtype=np.float32).reshape(4, 2, 5)-19.5)/7.
        runner.counterfactual_advantages = credit.copy()
        captured = []

        class Buffer:
            def __init__(self):
                self.obs = np.zeros((5, 2, 3), dtype=np.float32)
                self.actions = np.zeros((4, 2, 1), dtype=np.float32)
                self.active_masks = np.ones((5, 2, 1), dtype=np.float32)
                self.available_actions = None
            def update_factor(self, factor):
                self.factor = factor.copy()

        def train(index, buffer, advantage, state_type, **options):
            captured.append((index, advantage.copy(), state_type, options))
            return {}

        def evaluate(observations, *args):
            return torch.zeros((len(observations), 1)), None, None

        runner.actor_buffer = [Buffer() for _ in range(5)]
        runner.actor = [SimpleNamespace(train=lambda *args, index=i, **kwargs:train(index, *args, **kwargs),
                                       evaluate_actions=evaluate) for i in range(5)]
        runner.critic = SimpleNamespace(train=mock.Mock(return_value={'value_loss':0.}))
        before = runner.counterfactual_advantages.copy()
        runner.train()
        self.assertEqual(len(captured), 5)
        for index, advantage, state_type, options in captured:
            np.testing.assert_array_equal(advantage, credit[..., index:index+1])
            self.assertEqual(state_type, 'EP')
            self.assertTrue(options['advantages_normalized'])
            np.testing.assert_array_equal(runner.actor_buffer[index].factor, np.ones((4, 2, 1)))
        np.testing.assert_array_equal(before, runner.counterfactual_advantages)
        runner.critic.train.assert_called_once_with(runner.critic_buffer, None, update_normalizer_once=True)


class CounterfactualCreditWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args, cls.physical = cli.load_configuration()
        cls.args['train'].update(episode_length=3, n_rollout_threads=2)
        cls.args['model'].update(hidden_sizes=[8, 8], action_bins=5)
        cls.args['algo'].update(ppo_epoch=1, critic_epoch=1, counterfactual_batch_size=11)
        cls.args['experiment'].update(episodes=2, seeds=[1], diagnostics=False)
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new', cls.physical, cls.args['experiment'])
        cls.windows = select_periodic_windows(cls.data, episode_days=3)

    def runner(self, mode=None, **options):
        args = copy.deepcopy(self.args)
        args['algo'].pop('happo_credit_assignment', None)
        if mode is not None:
            args['algo']['happo_credit_assignment'] = mode
        runner = CascadeReservoirRunner(self.data, self.physical, args, 1,
                                        periodic_windows=self.windows, **options)
        self.addCleanup(runner.close)
        return runner

    def test_default_and_explicit_team_gae_preserve_exact_training_and_rng(self):
        with redirect_stdout(io.StringIO()):
            baseline = self.runner()
            baseline.run(2)
            expected = training_state(baseline)
            explicit = self.runner('team_gae')
            explicit.run(2)
            assert_nested_equal(self, expected, training_state(explicit))
        for left, right in zip(baseline.actor_buffer, explicit.actor_buffer):
            np.testing.assert_array_equal(left.actions, right.actions)
            np.testing.assert_array_equal(left.obs, right.obs)
        np.testing.assert_array_equal(baseline.critic_buffer.returns, explicit.critic_buffer.returns)

    def test_counterfactual_collection_preserves_actual_rollout_gae_and_rng(self):
        with redirect_stdout(io.StringIO()):
            baseline = self.runner('team_gae')
            baseline.collect_batch(2)
            before = training_state(baseline)
            traced = self.runner('counterfactual')
            with mock.patch.object(traced.value_normalizer, 'update',
                                   wraps=traced.value_normalizer.update) as normalizer_update:
                traced.collect_batch(2)
            normalizer_update.assert_not_called()
            assert_nested_equal(self, before, training_state(traced))
        for left, right in zip(baseline.actor_buffer, traced.actor_buffer):
            for name in ('obs','actions','action_log_probs','masks','active_masks'):
                np.testing.assert_array_equal(getattr(left,name), getattr(right,name))
        for name in ('rewards','returns','value_preds','masks','bad_masks','share_obs'):
            np.testing.assert_array_equal(getattr(baseline.critic_buffer,name),
                                          getattr(traced.critic_buffer,name))
        self.assertEqual(traced.critic_output_dim, 1)
        self.assertEqual(traced.actor_value_channels, (0,)*5)
        self.assertEqual(traced.counterfactual_advantages.shape, (3, 2, 5))
        self.assertTrue(np.isfinite(traced.counterfactual_advantages).all())
        self.assertTrue(any(not np.array_equal(traced.counterfactual_advantages[...,0],
                                               traced.counterfactual_advantages[...,i]) for i in range(1,5)))
        self.assertEqual(baseline.sampled_window_starts, traced.sampled_window_starts)

    def test_real_update_freezes_credit_before_actor_and_normalizer_changes(self):
        with redirect_stdout(io.StringIO()):
            runner = self.runner('counterfactual')
            runner.collect_batch(2)
            frozen = runner.counterfactual_advantages.copy()
            normalizer_before = copy.deepcopy(runner.value_normalizer.state_dict())
            seen, patches = [], []
            try:
                for i,actor in enumerate(runner.actor):
                    original = actor.train
                    def train(buffer, advantages, state_type, *, index=i, real_train=original, **options):
                        np.testing.assert_array_equal(advantages, frozen[...,index:index+1])
                        assert_nested_equal(self, normalizer_before, runner.value_normalizer.state_dict())
                        seen.append(index)
                        return real_train(buffer, advantages, state_type, **options)
                    patch = mock.patch.object(actor, 'train', side_effect=train)
                    patches.append(patch)
                    patch.start()
                with mock.patch.object(runner.value_normalizer, 'update',
                                       wraps=runner.value_normalizer.update) as update_normalizer:
                    runner.prep_training()
                    actors, critic = runner.train()
            finally:
                for patch in reversed(patches):
                    patch.stop()
            self.assertEqual(sorted(seen), list(range(5)))
            update_normalizer.assert_called_once()
            np.testing.assert_array_equal(update_normalizer.call_args.args[0],
                runner.critic_buffer.returns[:-1].reshape(-1,1))
            np.testing.assert_array_equal(runner.counterfactual_advantages, frozen)
            self.assertTrue(np.isfinite(critic['value_loss']))
            self.assertTrue(all(info['optimizer_steps']==1 for info in actors))

    def test_evaluation_does_not_recompute_training_credit(self):
        with redirect_stdout(io.StringIO()):
            runner = self.runner('counterfactual')
            runner.collect_batch(2)
            frozen = runner.counterfactual_advantages.copy()
            before = training_state(runner)
            with mock.patch.object(runner, '_compute_counterfactual_credit',
                    side_effect=AssertionError('Evaluation must not create training targets')):
                runner.periodic_evaluation()
            assert_nested_equal(self, before, training_state(runner))
            np.testing.assert_array_equal(frozen, runner.counterfactual_advantages)
            self.assertEqual(runner.periodic_evaluation_history[-1]['status'], 'ok')

    def test_checkpoint_replay_preserves_shared_value_and_credit_provenance(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            stream = io.StringIO()
            runner = self.runner('counterfactual', episode_log=stream, checkpoint_dir=Path(directory))
            runner.run(2)
            policy = load_checkpoint(Path(directory)/'update_000001.pt')
            self.assertEqual(policy.critic_reward_layout, 'shared_team')
            self.assertEqual(policy.critic_output_dim, 1)
            self.assertEqual(policy.actor_value_channels, (0,)*5)
            self.assertEqual(policy.metadata['settings']['algorithm']['algo']['happo_credit_assignment'],
                             'counterfactual')
            summary = policy.replay([1965])
            self.assertEqual(summary['episode_horizon_days'], 3)
            self.assertTrue(np.isfinite(summary['episode_system_reward']))
            entries = [json.loads(line) for line in stream.getvalue().splitlines()]
            episodes = [r for r in entries if r['record_type']=='training_episode']
            self.assertEqual(len(episodes), 2)
            for row in episodes:
                self.assertEqual(row['advantage_estimator'], 'one_step_counterfactual')
                self.assertEqual(row['actor_reward_sources'], ['system_reward']*5)
                self.assertEqual(row['critic_reward_layout'], 'shared_team')

    def test_compressed_trace_preserves_exact_frozen_credit_before_updates(self):
        from harl.envs.cascade_reservoir.counterfactual_logging import (
            CounterfactualCreditWriter, COUNTERFACTUAL_TRACE_FIELDS)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory)/'credit.csv.gz'
            with CounterfactualCreditWriter(path) as writer:
                runner = self.runner('counterfactual', counterfactual_log=writer)
                runner.run(2)
                self.assertEqual(writer.rows_written, 3*2*5)
            with gzip.open(path,'rt',newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3*2*5)
            self.assertEqual(set(rows[0]), set(COUNTERFACTUAL_TRACE_FIELDS))
            for row in rows:
                t, lane = int(row['episode_step'])-1, int(row['parallel_env_index'])
                i = ('WDD','BHT','XLD','XJB','THR').index(row['reservoir_id'])
                selected = int(row['sampled_action_index'])
                q = np.asarray(json.loads(row['frozen_q_values']))
                probability = np.asarray(json.loads(row['old_policy_probabilities']))
                self.assertEqual(q.shape, (5,))
                self.assertAlmostEqual(probability.sum(), 1., places=6)
                self.assertAlmostEqual(float(row['q_selected']), float(q[selected]), places=8)
                self.assertAlmostEqual(float(row['counterfactual_baseline']),
                                       float(np.dot(probability/probability.sum(),q)), places=7)
                self.assertAlmostEqual(float(row['counterfactual_raw_advantage']),
                    float(row['q_selected'])-float(row['counterfactual_baseline']), places=9)
                self.assertEqual(float(row['counterfactual_training_advantage']),
                                 float(runner.counterfactual_advantages[t,lane,i]))
                self.assertEqual(json.loads(row['joint_action_indices']),
                                 runner.counterfactual_context['actions'][t,lane].tolist())
            with self.assertRaises(FileExistsError):
                with CounterfactualCreditWriter(path):
                    pass

    def test_mappo_and_invalid_modes_are_rejected_at_direct_entry_points(self):
        for bad in ('invalid', None, True):
            args = copy.deepcopy(self.args)
            args['algo']['happo_credit_assignment'] = bad
            with self.subTest(mode=bad), self.assertRaises(ValueError):
                CascadeReservoirRunner(self.data,self.physical,args,1,periodic_windows=self.windows)
        args = copy.deepcopy(self.args)
        args['algo']['happo_credit_assignment'] = 'counterfactual'
        with self.assertRaisesRegex(ValueError, 'MAPPO|HAPPO|counterfactual'):
            CascadeMAPPORunner(self.data,self.physical,args,1,periodic_windows=self.windows)


class CounterfactualCLIWorkflowTest(unittest.TestCase):
    def test_cli_reaches_happo_configuration(self):
        def experiment(data, physical, config, options):
            self.assertEqual(config['algo']['happo_credit_assignment'],'counterfactual')
            self.assertEqual(config['algo']['counterfactual_batch_size'],17)
            return 'configured'
        with redirect_stdout(io.StringIO()):
            result = cli.main(['--happo-credit-assignment','counterfactual',
                '--counterfactual-batch-size','17','--no-plot'], experiment_runner=experiment)
        self.assertEqual(result,'configured')

    def test_mappo_only_counterfactual_rejected_before_loading_data(self):
        with redirect_stderr(io.StringIO()), mock.patch.object(cli,'CascadeData') as data:
            with self.assertRaises(SystemExit):
                compare.main(['--happo-credit-assignment','counterfactual',
                              '--algorithms','mappo','--no-plot'])
        data.assert_not_called()

    def test_counterfactual_happo_and_mappo_receive_their_own_credit_modes(self):
        with mock.patch.object(cli, 'CascadeData', return_value=mock.sentinel.data), mock.patch.object(compare, 'run_parallel_experiment', side_effect=lambda *args, **kwargs: {'algorithm': kwargs['algorithm_name']}) as run:
            with mock.patch.object(compare, 'validate_comparison', side_effect=lambda value: value), redirect_stdout(io.StringIO()):
                result = compare.main(['--algorithms', 'happo', 'mappo', '--happo-credit-assignment', 'counterfactual', '--no-plot'])
        self.assertEqual([entry['algorithm'] for entry in result['runs']], ['happo', 'mappo'])
        self.assertEqual([call.args[2]['algo']['happo_credit_assignment'] for call in run.call_args_list], ['counterfactual', 'team_gae'])
        self.assertTrue(all(call.args[3].episode_log is None for call in run.call_args_list))


if __name__ == '__main__':
    unittest.main(verbosity=2)
