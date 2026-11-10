"""Target/execution audits and raw PPO accounting through both cascade runners."""
import copy
import csv
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.train_cascade import load_configuration
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.episodes import episode_environment
from harl.envs.cascade_reservoir.periodic_evaluation import (
    select_periodic_windows, evaluate_periodic_policy)
from harl.envs.cascade_reservoir.evaluation import evaluate_policy
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade
from harl.envs.cascade_reservoir.reward import EpisodeRewardMetrics
from harl.envs.cascade_reservoir.training_metrics import episode_record, step_record, StepMetricsWriter
from harl.envs.cascade_reservoir.trajectory_diagnostics import TrajectoryDiagnosticsWriter
from harl.envs.cascade_reservoir.checkpoint import save_checkpoint, load_checkpoint, ReplayTraceWriter
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
from harl.runners.cascade_mappo_runner import CascadeMAPPORunner


class SynchronousCollector:
    """Use actual environment physics while isolating process startup in unit tests."""
    def __init__(self, data, physical, bins, days, count, gamma, trace, seed, forced=False):
        self.data, self.physical = copy.copy(data), copy.deepcopy(physical)
        if forced:
            self.data.external = np.zeros_like(data.external)
            self.data.initial_storage = lambda year: data.vmin.copy()
            self.physical['use_daily_level_limits'] = False
        self.bins, self.days, self.gamma, self.trace, self.seed = bins, days, gamma, trace, seed
        self.envs = []

    def reset(self, windows, ids):
        self.ids = ids
        self.envs = [episode_environment(self.data, self.physical, years, self.days, self.bins)
                     for years in windows]
        self.metrics = [EpisodeRewardMetrics(self.gamma, action_correction_config=self.physical) for _ in self.envs]
        self.energy = [np.zeros(5) for _ in self.envs]
        return [dict(obs=e.actor_observations(), state=e.state(), pid=1) for e in self.envs]

    def step(self, commands):
        result = []
        for lane, (env, action) in enumerate(zip(self.envs, commands)):
            env.step(action)
            tr = env.last_transition
            self.metrics[lane].update(tr)
            self.energy[lane] += tr['energy']
            done = env.index == env.stop
            row = dict(obs=env.actor_observations(), state=env.state(), transition=tr, done=done)
            if done:
                row.update(episode_record=episode_record(env, self.metrics[lane], self.energy[lane],
                    self.seed, self.ids[lane], self.gamma, self.trace, 0),
                    reward_metrics=self.metrics[lane].summary())
            result.append(row)
        return result

    def close(self):
        pass


class ActionCorrectionIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.physical = load_configuration()
        cls.config['train'].update(episode_length=3, n_rollout_threads=2)
        cls.config['model'].update(hidden_sizes=[8, 8], action_bins=5)
        cls.config['algo'].update(ppo_epoch=1, critic_epoch=1, counterfactual_batch_size=11)
        cls.config['experiment'].update(episodes=2, seeds=[1], diagnostics=False,
            trajectory_diagnostics=False, save_checkpoints=False, checkpoint_interval=0)
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new', cls.physical, cls.config['experiment'])
        cls.windows = select_periodic_windows(cls.data, episode_days=3)

    def test_environment_and_daily_exports_keep_raw_and_executed_decisions_separate(self):
        actions = np.array([0, 4, 2, 1, 3])
        baseline = None
        for enabled, penalty in ((False, False), (True, False), (True, True)):
            physical = copy.deepcopy(self.physical)
            physical['action_correction'] = dict(enabled=enabled, penalty_enabled=penalty,
                                                 tau=.03, correction_weight=1.)
            env = episode_environment(self.data, physical, [1965], 3, 5)
            env.step(actions)
            tr = env.last_transition
            np.testing.assert_array_equal(tr['sampled_actions'], actions)
            np.testing.assert_array_equal(tr['normalized_actions'], actions/4.)
            np.testing.assert_array_equal(tr['executed_release'], tr['release'])
            np.testing.assert_array_equal(tr['target_release'], tr['release'])
            self.assertFalse(tr['correction_applied'].any())
            self.assertEqual(float(tr['correction_deficit'].sum()), 0.)
            metrics = EpisodeRewardMetrics(.995)
            metrics.update(tr)
            summary = metrics.summary()['action_correction_summary']
            self.assertEqual(summary['genuine_correction_events'], 0)
            stream = io.StringIO()
            writer = StepMetricsWriter(stream)
            writer.write(step_record(tr, seed=1, episode=1, episode_step=1,
                                     phase='training', training_steps=1))
            row = next(csv.DictReader(io.StringIO(stream.getvalue())))
            self.assertEqual(float(row['genuine_action_correction_events']), 0.)
            comparable = {name:tr[name] for name in ('release', 'storage_end', 'power',
                           'reward', 'local_rewards', 'mixed_rewards', 'system_reward')}
            if baseline is None:
                baseline = copy.deepcopy(comparable)
            else:
                for name, value in comparable.items():
                    np.testing.assert_array_equal(value, baseline[name])

    def test_both_runners_train_evaluate_and_replay_with_auditable_correction_fields(self):
        for runner_class, mode in ((CascadeReservoirRunner, 'counterfactual'),
                                   (CascadeMAPPORunner, 'team_gae')):
            with self.subTest(algorithm=runner_class.algorithm_name):
                config, physical = copy.deepcopy(self.config), copy.deepcopy(self.physical)
                config['algo']['happo_credit_assignment'] = mode
                physical['action_correction'] = dict(enabled=True, penalty_enabled=True,
                                                     tau=.03, correction_weight=1.)
                with tempfile.TemporaryDirectory(dir=str(ROOT/'tests')) as directory:
                    directory = Path(directory)
                    trajectory = directory/'trajectory.csv'
                    stream = io.StringIO()
                    with TrajectoryDiagnosticsWriter(trajectory) as trace, mock.patch(
                            'harl.runners.cascade_reservoir_runner.ParallelCascadeEnvs', SynchronousCollector), redirect_stdout(io.StringIO()):
                        runner = runner_class(self.data, physical, config, 1,
                            periodic_windows=self.windows, episode_log=stream, trajectory_log=trace)
                        runner.run(2)
                    self.assertEqual(runner.update_count, 1)
                    self.assertEqual(runner.last_action_correction_summary['genuine_correction_events'], 0)
                    self.assertEqual(runner.last_reward_metrics['action_correction_active_reservoir_days'], 30)
                    self.assertEqual(runner.training_history[0]['action_correction_active_reservoir_days'], 30)
                    for evaluation in runner.periodic_evaluation_history:
                        self.assertEqual(evaluation['action_correction_violation_count'], 0)
                        self.assertEqual(len(evaluation['windows']), 3)
                    with trajectory.open(encoding='utf-8', newline='') as trace_stream:
                        rows = list(csv.DictReader(trace_stream))
                    self.assertEqual(len(rows), 30)
                    for row in rows:
                        self.assertEqual(int(row['ppo_action_index']), int(row['sampled_action_index']))
                        self.assertEqual(float(row['target_release_m3s']), float(row['executed_release_m3s']))
                        self.assertEqual(float(row['correction_deficit']), 0.)
                    checkpoint = directory/'policy.pt'
                    save_checkpoint(runner, checkpoint)
                    frozen = load_checkpoint(checkpoint)
                    replay_stream = io.StringIO()
                    replay = frozen.replay([1965], writer=ReplayTraceWriter(replay_stream))
                    self.assertEqual(replay['action_correction_violation_count'], 0)
                    replay_row = next(csv.DictReader(io.StringIO(replay_stream.getvalue())))
                    self.assertIn('correction_reason', replay_row)
                    self.assertEqual(float(replay_row['target_release_m3s']), float(replay_row['executed_release_m3s']))

    def test_execution_failure_keeps_diagnostics_and_counts_attempts_without_fake_days(self):
        physical = copy.deepcopy(self.physical)
        physical['action_correction'] = dict(enabled=True, penalty_enabled=False,
                                             tau=.11, correction_weight=1.)
        env = episode_environment(self.data, physical, [1965], 3, 5)
        error = InfeasibleCascade('No joint solution', dict(reservoir_index=2,
            failure_stage='execution_feasibility', test_marker='preserved'))
        before = env.storage.copy()
        with mock.patch.object(env.physics, 'transition', side_effect=error):
            with self.assertRaises(InfeasibleCascade) as caught:
                env.step(np.zeros(5, dtype=int))
        self.assertIs(caught.exception, error)
        self.assertEqual(error.diagnostics['test_marker'], 'preserved')
        np.testing.assert_array_equal(env.storage, before)
        class ConstantActor:
            args = dict(action_bins=5)
            def prep_rollout(self):
                pass
            def act(self, obs, deterministic=True):
                return np.array([0])
        with mock.patch('harl.envs.cascade_reservoir.evaluation.episode_environment', return_value=env), mock.patch.object(env, 'step', side_effect=error):
            with self.assertRaises(InfeasibleCascade):
                evaluate_policy([ConstantActor() for _ in range(5)], self.data, physical,
                    1, 1, 0, gamma=.995, source_years=[1965], episode_days=3)
        partial = error.evaluation_progress['partial_metrics']
        summary = partial['action_correction_summary']
        self.assertEqual(summary['no_feasible_execution_events'], 1)
        self.assertEqual(summary['correction_tolerance_tau'], .11)
        self.assertEqual(summary['valid_controllable_reservoir_days'], 0)
        self.assertIsNone(partial['action_correction_violation_rate'])
        with mock.patch('harl.envs.cascade_reservoir.periodic_evaluation.evaluate_policy', side_effect=error):
            round_result = evaluate_periodic_policy([], self.data, physical, seed=1,
                evaluation_round=0, training_steps=0, gamma=.995, windows=self.windows)
        self.assertIsNone(round_result['action_correction_violation_rate'])
        self.assertEqual(round_result['no_feasible_execution_events'], 3)
        for window in round_result['windows']:
            self.assertEqual(window['failure']['diagnostics']['test_marker'], 'preserved')

    def test_forced_dispatch_preserves_sampled_buffer_commands_and_log_probabilities(self):
        config, physical = copy.deepcopy(self.config), copy.deepcopy(self.physical)
        config['algo']['happo_credit_assignment'] = 'counterfactual'
        physical['use_daily_level_limits'] = False
        physical['action_correction'] = dict(enabled=True, penalty_enabled=True,
                                             tau=.03, correction_weight=1.)
        def factory(*args):
            return SynchronousCollector(*args, forced=True)
        with mock.patch('harl.runners.cascade_reservoir_runner.ParallelCascadeEnvs', factory), redirect_stdout(io.StringIO()):
            runner = CascadeReservoirRunner(self.data, physical, config, 1, periodic_windows=self.windows)
            self.addCleanup(runner.close)
            runner.collect_batch(2)
        self.assertTrue(any((buffer.actions != 0).any() for buffer in runner.actor_buffer))
        for i, buffer in enumerate(runner.actor_buffer):
            np.testing.assert_array_equal(buffer.active_masks[:-1], np.zeros_like(buffer.active_masks[:-1]))
            logs, _, _ = runner.actor[i].evaluate_actions(
                buffer.obs[:-1].reshape(-1, buffer.obs.shape[-1]),
                buffer.actions.reshape(-1, 1), return_entropy_per_sample=True)
            np.testing.assert_allclose(buffer.action_log_probs.reshape(-1, 1),
                                       logs.detach().cpu().numpy(), atol=1e-7)
            np.testing.assert_array_equal(buffer.actions[...,0], runner.counterfactual_context['actions'][...,i])
        summary = runner.last_action_correction_summary
        self.assertEqual(summary['forced_dispatch_events'], 30)
        self.assertEqual(summary['valid_controllable_reservoir_days'], 0)
        self.assertIsNone(summary['correction_violation_rate'])
        self.assertEqual(summary['genuine_correction_events'], 0)
        forced_data = copy.copy(self.data)
        forced_data.external = np.zeros_like(self.data.external)
        forced_data.initial_storage = lambda year: self.data.vmin.copy()
        forced_env = episode_environment(forced_data, physical, [1965], 3, 5)
        class ConstantActor:
            args = dict(action_bins=5)
            def prep_rollout(self):
                pass
            def act(self, obs, deterministic=True):
                return np.array([0])
        with mock.patch('harl.envs.cascade_reservoir.evaluation.episode_environment', return_value=forced_env):
            result = evaluate_policy([ConstantActor() for _ in range(5)], self.data, physical,
                1, 1, 0, gamma=.995, source_years=[1965], episode_days=3, diagnostics=True)
        self.assertEqual(result['mean_mapping_fraction'], [None]*5)
        self.assertEqual(result['mapping_fraction_valid_days'], [0]*5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
