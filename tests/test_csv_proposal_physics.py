"""Conditional CSV release correction and original-mapping metadata contracts."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from test_xld_dispatch_diagnostics import synthetic_dispatch
from harl.envs.cascade_reservoir.reservoir_physics import ReservoirPhysics,InfeasibleCascade
from harl.envs.cascade_reservoir.batched_physics import batched_transition
from harl.envs.cascade_reservoir.reward import OperationReward,reward_settings

class CSVPhysicsTest(unittest.TestCase):
    def setUp(self):
        self.env,self.source=synthetic_dispatch()
        self.config=copy.deepcopy(self.env.physics.config)
        self.config.update(release_mapping_mode='csv_proposal',action_correction=dict(enabled=True,penalty_enabled=True,tau=.03,correction_weight=1.))
        self.physics=ReservoirPhysics(self.env.data,self.config)

    def execute(self,low,high,u=.5,storage=None,external=None,bounds=None):
        storage=self.source['storage_start'] if storage is None else storage
        external=self.source['external'] if external is None else external
        bounds=(self.source['storage_lower_m3'],self.source['storage_upper_m3']) if bounds is None else bounds
        args=(storage,external,np.broadcast_to(u,(5,)),*bounds)
        low,high=np.broadcast_to(low,(5,)),np.broadcast_to(high,(5,))
        row=self.physics.transition(*args,proposal_low=low,proposal_high=high)
        batch=batched_transition(self.physics,storage[None],external[None],args[2][None],*bounds,proposal_low=low,proposal_high=high)
        self.assertEqual(set(row),set(batch))
        for key,value in row.items():
            np.testing.assert_array_equal(batch[key][0],value,err_msg=key)
        np.testing.assert_array_equal(row['inflow'][1:],external[1:]+row['release'][:-1])
        np.testing.assert_array_equal(row['storage_end'],storage+(row['inflow']-row['release'])*86400.)
        tol=np.maximum(1.,1e-10*np.abs(storage))
        self.assertTrue((row['storage_end']>=bounds[0]-tol).all())
        self.assertTrue((row['storage_end']<=bounds[1]+tol).all())
        return row

    def test_feasible_target_identity_and_infeasible_target_repair(self):
        row=self.execute(0.,100.,.5)
        np.testing.assert_array_equal(row['target_release'],row['release'])
        self.assertFalse(row['correction_applied'].any())
        row=self.execute(0.,10000.,1.)
        self.assertEqual(row['target_release'][0],10000.)
        self.assertEqual(row['release'][0],5000.)
        self.assertTrue(row['correction_applied'][0])
        self.assertEqual(row['realized_normalized_actions'][0],.5)
        self.assertEqual(row['scaled_release_correction'][0],.5)
        self.assertEqual(row['inflow'][1],5000.)
        OperationReward(self.env.data,reward_settings(self.config)).apply(row)
        self.assertAlmostEqual(row['r_corr'][0],-.47)

    def test_execution_outside_csv_is_not_inverse_clipped_or_penalized(self):
        storage=self.source['storage_start'].copy();storage[0]=self.env.data.vmax[0]
        external=self.source['external'].copy();external[0]=1000.
        row=self.execute(0.,100.,.5,storage,external)
        self.assertEqual(row['target_release'][0],50.)
        self.assertEqual(row['release'][0],1000.)
        self.assertTrue(row['proposal_physical_disjoint'][0])
        self.assertTrue(row['executed_outside_mapping_interval'][0])
        self.assertFalse(row['inverse_action_valid'][0])
        self.assertEqual(row['scaled_release_correction'][0],9.5)
        OperationReward(self.env.data,reward_settings(self.config)).apply(row)
        self.assertEqual(row['r_corr'][0],0.)

    def test_physical_force_and_fixed_proposal_are_distinct(self):
        storage=self.source['storage_start'].copy()
        forced=self.execute(0.,10000.,1.,bounds=(storage,storage))
        self.assertTrue(forced['forced_dispatch'].all())
        self.assertFalse(forced['correction_applied'].any())
        self.assertTrue(forced['scaled_release_correction_valid'].all())
        fixed=self.execute(10000.,10000.,.5)
        self.assertFalse(fixed['forced_dispatch'][0])
        self.assertTrue(fixed['degenerate_proposal'][0])
        self.assertEqual(fixed['controls'][0],0.)
        self.assertEqual(fixed['physical_controls'][0],1.)
        self.assertTrue(fixed['correction_applied'][0])
        self.assertFalse(fixed['scaled_release_correction_valid'][0])

    def test_empty_joint_set_and_configuration_errors(self):
        storage=self.env.data.vmax.copy();external=np.zeros(5);external[-1]=2*self.env.data.release_cap[-1]
        with self.assertRaises(InfeasibleCascade):
            self.execute(0.,100.,storage=storage,external=external)
        with self.assertRaises(ValueError):
            ReservoirPhysics(self.env.data,dict(self.config,action_correction=dict(enabled=False)))
        with self.assertRaises(ValueError):
            self.physics.transition(self.source['storage_start'],self.source['external'],np.ones(5))

    def test_external_original_metadata_requires_coherent_ordered_bounds(self):
        args=(self.source['storage_start'],self.source['external'],np.full(5,9000.),self.source['storage_lower_m3'],self.source['storage_upper_m3'])
        for kwargs in (dict(normalized_actions=np.full(5,.9)),
            dict(normalized_actions=np.full(5,.9),mapping_low=np.zeros(5),mapping_high=np.full(5,5000.)),
            dict(normalized_actions=np.full(5,.9),mapping_low=np.full(5,100.),mapping_high=np.zeros(5))):
            with self.assertRaises(ValueError): self.physics.correct_releases(*args,**kwargs)
        row=self.physics.correct_releases(*args,normalized_actions=np.full(5,.9),mapping_low=np.zeros(5),mapping_high=np.full(5,10000.))
        self.assertAlmostEqual(row['correction_deficit'][0],.37)
        no_metadata=self.physics.correct_releases(*args)
        self.assertFalse(no_metadata['inverse_action_valid'].any())
        self.assertFalse(no_metadata['scaled_release_correction_valid'].any())

if __name__=='__main__': unittest.main()
