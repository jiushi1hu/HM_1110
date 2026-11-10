"""Actual CSV execution, counterfactual credit, diagnostics and replay."""
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

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from examples.train_cascade import load_configuration
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.episodes import EpisodeData, episode_environment
from harl.envs.cascade_reservoir.cascade_reservoir_env import CascadeReservoirEnv
from harl.envs.cascade_reservoir.periodic_evaluation import select_periodic_windows
from harl.envs.cascade_reservoir.checkpoint import save_checkpoint, load_checkpoint
from harl.envs.cascade_reservoir.trajectory_diagnostics import TrajectoryDiagnosticsWriter
from harl.envs.cascade_reservoir.counterfactual_logging import CounterfactualCreditWriter
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
from harl.runners.cascade_mappo_runner import CascadeMAPPORunner
from test_action_correction_integration import SynchronousCollector

class CSVIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config,cls.physical=load_configuration()
        cls.physical['release_mapping_mode']='csv_proposal'
        cls.physical['action_correction'].update(enabled=True,penalty_enabled=True)
        cls.data=CascadeData(ROOT/'data/cascade_reservoir_new',cls.physical,cls.config['experiment'])

    def test_both_algorithms_collect_update_trace_and_replay_csv_context(self):
        for runner_class in (CascadeReservoirRunner,CascadeMAPPORunner):
            with self.subTest(algorithm=runner_class.algorithm_name):
                cfg=copy.deepcopy(self.config)
                cfg['train'].update(episode_length=3,n_rollout_threads=2)
                cfg['model'].update(hidden_sizes=[8,8],action_bins=5)
                cfg['algo'].update(ppo_epoch=1,critic_epoch=1,counterfactual_batch_size=11,
                    happo_credit_assignment='counterfactual' if runner_class is CascadeReservoirRunner else 'team_gae')
                cfg['experiment'].update(episodes=2,seeds=[1],diagnostics=False)
                windows=select_periodic_windows(self.data,episode_days=3)
                with tempfile.TemporaryDirectory(dir=str(ROOT/'tests')) as directory:
                    directory=Path(directory)
                    with TrajectoryDiagnosticsWriter(directory/'trace.csv') as trace, CounterfactualCreditWriter(directory/'credit.csv.gz') as credit, mock.patch('harl.runners.cascade_reservoir_runner.ParallelCascadeEnvs',SynchronousCollector),redirect_stdout(io.StringIO()):
                        runner=runner_class(self.data,copy.deepcopy(self.physical),cfg,1,periodic_windows=windows,
                            trajectory_log=trace,counterfactual_log=credit if runner_class is CascadeReservoirRunner else None)
                        runner.run(2)
                    with (directory/'trace.csv').open() as stream:
                        rows=list(csv.DictReader(stream))
                    self.assertEqual(len(rows),30)
                    for row in rows:
                        day=row['date'];i=self.data.release_bounds.date_index[__import__('datetime').date.fromisoformat(day)]
                        r=('WDD','BHT','XLD','XJB','THR').index(row['reservoir_id'])
                        self.assertEqual(float(row['proposal_low_m3s']),self.data.release_bounds.low[i,r])
                        self.assertEqual(int(row['ppo_action_index']),int(row['sampled_action_index']))
                        self.assertEqual(row['feasible_release_low_m3s'],row['physical_low_m3s'])
                        self.assertEqual(row['feasible_release_high_m3s'],row['physical_high_m3s'])
                        self.assertGreaterEqual(float(row['physical_high_m3s']),float(row['physical_low_m3s'])-1e-8)
                    if runner_class is CascadeReservoirRunner:
                        self.assertIn('proposal_low',runner.counterfactual_context)
                        self.assertTrue(np.isfinite(runner.counterfactual_credit['q_values']).all())
                    save_checkpoint(runner,directory/'policy.pt')
                    frozen=load_checkpoint(directory/'policy.pt')
                    replay=frozen.replay([1965])
                    self.assertIn('physical_intervention_rate',replay)
                    self.assertEqual(frozen.physical_config['release_mapping_mode'],'csv_proposal')

    def test_midyear_leap_day_lookup_uses_actual_action_date(self):
        view=EpisodeData(self.data,[1968],366)
        offset=58
        view.source_indices=view.source_indices[offset:offset+3]
        view.dates=view.dates[offset:offset+3]
        view.years=view.years[offset:offset+3]
        view.external=view.external[offset:offset+3]
        view.release_bounds=self.data.release_bounds.take(view.source_indices,data=view)
        env=CascadeReservoirEnv(view,self.physical,start_year=1968,episode_days=3,action_bins=5)
        for day in view.dates:
            env.step(np.array([0,1,2,3,4]))
            lo,hi=self.data.release_bounds.bounds_for_date(day)
            np.testing.assert_array_equal(env.last_transition['proposal_low'],lo)
            np.testing.assert_array_equal(env.last_transition['proposal_high'],hi)
        self.assertEqual(view.dates[1].isoformat(),'1968-02-29')

    def test_XLD_candidate_diagnostics_use_the_csv_mapping_and_correction(self):
        from harl.envs.cascade_reservoir.xld_diagnostics import XLDDispatchDiagnostics
        env=episode_environment(self.data,self.physical,[1965],10,100)
        diagnostic=XLDDispatchDiagnostics(env,seed=1,evaluation_round=0,
            window_id='csv_check',phase='training_evaluation')
        for day in range(10):
            env.step(np.array([(day*17+i*29)%100 for i in range(5)]))
            record=diagnostic.update(env.last_transition,day+1)
            self.assertTrue(np.isfinite(record['power_mw']))
        self.assertEqual(diagnostic.summary()['completed_steps'],10)

    def test_B_C_same_physics_penalty_only_changes_learning_rewards(self):
        environments=[]
        for penalty in (False,True):
            cfg=copy.deepcopy(self.physical)
            cfg['action_correction']['penalty_enabled']=penalty
            environments.append(episode_environment(self.data,cfg,[1965],10,100))
        for day in range(10):
            commands=np.array([(day*17+i*29)%100 for i in range(5)])
            for env in environments:
                env.step(commands)
            b,c=[env.last_transition for env in environments]
            for key in ('release','storage_end','power','reward'):
                np.testing.assert_array_equal(b[key],c[key])
            np.testing.assert_allclose(c['local_rewards'],b['local_rewards']+c['r_corr'])
            np.testing.assert_allclose(c['mixed_rewards'],b['mixed_rewards']+.5*c['r_corr'])
            self.assertAlmostEqual(c['system_reward'],b['system_reward']+.5*c['r_corr'].sum())

if __name__=='__main__':
    unittest.main()
