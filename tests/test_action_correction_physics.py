"""Physical correction contracts against the existing cascade constraints."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT/'tests'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from harl.envs.cascade_reservoir.action_correction import (
    action_correction_settings, action_correction_fields, correction_threshold_exceeded, project_release)
from harl.envs.cascade_reservoir.batched_physics import batched_transition
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade
from test_xld_dispatch_diagnostics import synthetic_dispatch


class ActionCorrectionPhysicsTest(unittest.TestCase):
    def assert_physical(self, physics, row):
        np.testing.assert_array_equal(row['inflow'][1:], row['external'][1:]+row['release'][:-1])
        expected = row['storage_start']+(row['inflow']-row['release'])*physics.dt
        np.testing.assert_array_equal(row['storage_end'], expected)
        tolerance = np.maximum(1., 1e-10*np.abs(row['storage_start']))
        self.assertTrue(np.all(row['storage_end'] >= row['storage_lower_m3']-tolerance))
        self.assertTrue(np.all(row['storage_end'] <= row['storage_upper_m3']+tolerance))
        self.assertTrue(np.all(row['release'] >= 0.))
        self.assertTrue(np.all(row['release'] <= physics.data.release_cap))
        self.assertLessEqual(abs(row['system_residual']), tolerance.sum())

    def test_feasible_targets_are_identity_and_inputs_are_unchanged(self):
        for head in ('fixed', 'storage_sensitivity'):
            env, original = synthetic_dispatch(head_model=head, external=100., action=37)
            target = original['release'].copy()
            storage, external = original['storage_start'].copy(), original['external'].copy()
            before = [v.copy() for v in (storage, external, target)]
            for v in (storage, external, target):
                v.setflags(write=False)
            actual = env.physics.correct_releases(storage, external, target,
                original['storage_lower_m3'], original['storage_upper_m3'])
            np.testing.assert_array_equal(actual['release'], target)
            self.assertFalse(actual['correction_applied'].any())
            self.assertFalse(actual['inverse_action_valid'].any())
            self.assertFalse(actual['original_action_available'].any())
            self.assert_physical(env.physics, actual)
            for value, unchanged in zip((storage, external, target), before):
                np.testing.assert_array_equal(value, unchanged)

    def test_genuine_repairs_recompute_downstream_flow_and_are_conditionally_nearest(self):
        env, original = synthetic_dispatch()
        target = np.full(5, 1e9)
        row = env.physics.correct_releases(original['storage_start'], original['external'], target,
            original['storage_lower_m3'], original['storage_upper_m3'])
        self.assertTrue(row['correction_applied'].all())
        self.assertEqual(row['release'][0], 5000.)
        self.assertEqual(row['inflow'][1], row['release'][0])
        np.testing.assert_array_equal(row['release'], np.minimum(row['high'], np.maximum(row['low'], target)))
        self.assertIn('next_day_storage_min', row['correction_reason'])
        self.assert_physical(env.physics, row)
        negative = env.physics.correct_releases(original['storage_start'], original['external'],
            -np.ones(5), original['storage_lower_m3'], original['storage_upper_m3'])
        self.assertTrue(negative['correction_applied'].all())
        self.assert_physical(env.physics, negative)

    def test_outside_original_mapping_interval_is_not_inverse_clipped(self):
        env, original = synthetic_dispatch()
        storage = original['storage_start'].copy()
        storage[0] = env.data.vmax[0]
        external = original['external'].copy()
        external[0] = 1000.
        row = env.physics.correct_releases(storage, external, np.full(5, 50.),
            original['storage_lower_m3'], original['storage_upper_m3'],
            normalized_actions=np.full(5, .5), mapping_low=np.zeros(5), mapping_high=np.full(5, 100.))
        self.assertEqual(row['release'][0], 1000.)
        self.assertTrue(row['executed_outside_mapping_interval'][0])
        self.assertFalse(row['inverse_action_valid'][0])
        self.assertEqual(row['correction_deficit'][0], 0.)
        self.assertEqual(row['release_correction_m3s'][0], 950.)
        np.testing.assert_array_equal(row['low'], np.zeros(5))
        np.testing.assert_array_equal(row['high'], np.full(5, 100.))
        self.assertEqual(row['execution_low'][0], 1000.)
        self.assert_physical(env.physics, row)

    def test_forced_and_tolerance_inverted_intervals_are_not_action_corrections(self):
        env, original = synthetic_dispatch()
        storage = original['storage_start'].copy()
        for enabled in (False, True):
            env.physics.correction_settings['enabled'] = enabled
            row = env.physics.transition(storage, original['external'], np.linspace(0., 1., 5),
                                         storage, storage)
            self.assertTrue(row['forced_dispatch'].all())
            self.assertFalse(row['controls'].any())
            self.assertFalse(row['inverse_action_valid'].any())
            self.assertFalse(row['correction_applied'].any())
            self.assertFalse(row['correction_deficit'].any())
            np.testing.assert_array_equal(row['normalized_actions'], np.linspace(0., 1., 5))
            np.testing.assert_array_equal(row['actions'], np.full(5, .5))
            storage_roundoff = storage.copy()
            storage_roundoff[0] = -env.physics.dt*5e-9
            tiny = env.physics.transition(storage_roundoff, np.zeros(5), np.full(5, .9),
                                         env.data.vmin, env.data.vmax)
            self.assertTrue(tiny['forced_dispatch'][0])
            self.assertEqual(tiny['release'][0], 0.)
            self.assertFalse(tiny['correction_applied'][0])
            self.assertFalse(tiny['numerical_adjustment'][0])

    def test_empty_joint_feasibility_is_explicit(self):
        env, original = synthetic_dispatch()
        storage = env.data.vmax.copy()
        external = np.zeros(5)
        external[-1] = 2.*env.data.release_cap[-1]
        for enabled in (False, True):
            env.physics.correction_settings['enabled'] = enabled
            for operation in (lambda: env.physics.transition(storage, external, np.full(5,.5)),
                              lambda: env.physics.correct_releases(storage, external, np.zeros(5))):
                with self.assertRaises(InfeasibleCascade) as caught:
                    operation()
                self.assertTrue(caught.exception.diagnostics['no_feasible_executed_solution'])
                self.assertEqual(caught.exception.failure_stage, 'action_execution')
                np.testing.assert_array_equal(storage, env.data.vmax)

    def test_linear_baseline_and_scalar_batched_projection_consistency(self):
        for head in ('fixed', 'storage_sensitivity'):
            env, original = synthetic_dispatch(head_model=head, external=100.)
            for action in (np.zeros(5), np.full(5,.25), np.full(5,.5),
                           np.full(5,.75), np.ones(5), np.array([.01,.3,.7,.99,1.])):
                env.physics.correction_settings['enabled'] = False
                old = env.physics.transition(original['storage_start'], original['external'], action,
                    original['storage_lower_m3'], original['storage_upper_m3'])
                expected = old['low']+action*(old['high']-old['low'])
                np.testing.assert_array_equal(old['target_release'], expected)
                active = old['controls'] > 0
                np.testing.assert_array_equal(old['release'][active], expected[active])
                for enabled in (False, True):
                    env.physics.correction_settings['enabled'] = enabled
                    new = env.physics.transition(original['storage_start'], original['external'], action,
                        original['storage_lower_m3'], original['storage_upper_m3'])
                    self.assertFalse(new['correction_applied'].any())
                    np.testing.assert_allclose(new['release'], old['release'], rtol=0., atol=1e-10)
                    actual = batched_transition(env.physics, original['storage_start'][None],
                        original['external'][None], action[None],
                        original['storage_lower_m3'], original['storage_upper_m3'])
                    self.assertEqual(set(actual), set(new))
                    for name in actual:
                        np.testing.assert_array_equal(actual[name][0], new[name])

    def test_extremely_narrow_controllable_interval_and_threshold_roundoff(self):
        env, original = synthetic_dispatch()
        storage = original['storage_start'].copy()
        storage[0] = env.physics.dt*1.01e-8
        env.physics.correction_settings['enabled'] = True
        for fraction in (0., .5, 1.):
            row = env.physics.transition(storage, np.zeros(5), np.full(5,fraction),
                                         env.data.vmin, env.data.vmax)
            self.assertEqual(row['controls'][0], 1.)
            self.assertTrue(row['inverse_action_valid'][0])
            self.assertAlmostEqual(row['realized_normalized_actions'][0], fraction)
            self.assertFalse(row['correction_applied'][0])
        self.assertFalse(bool(correction_threshold_exceeded(abs(.5-.47), .03)))
        self.assertTrue(bool(correction_threshold_exceeded(.031, .03)))
        settings = action_correction_settings({})
        fields = action_correction_fields(np.full(5,.2), np.full(5,2.),
            np.full(5,5.), np.zeros(5), np.full(5,10.), np.ones(5), settings)
        np.testing.assert_allclose(fields['correction_deficit'], .27)
        self.assertTrue(fields['correction_threshold_exceeded'].all())

    def test_ulp_boundary_adjustment_is_numerical_not_genuine(self):
        low, high = np.zeros(5), np.ones(5)
        target = np.nextafter(high, np.full(5, np.inf))
        executed = project_release(target, low, high, np.ones(5))
        np.testing.assert_array_equal(executed, high)
        fields = action_correction_fields(np.ones(5), target, executed, low, high,
                                          np.ones(5), action_correction_settings({}))
        self.assertTrue(fields['numerical_adjustment'].all())
        self.assertFalse(fields['correction_applied'].any())
        self.assertFalse(fields['correction_deficit'].any())
        self.assertFalse(fields['forced_dispatch'].any())
        self.assertTrue(fields['inverse_action_valid'].all())

    def test_configuration_validation_and_independent_switches(self):
        self.assertEqual(action_correction_settings({}),
            dict(enabled=False, tau=.03, penalty_enabled=False, correction_weight=1.))
        self.assertTrue(action_correction_settings(dict(action_correction=dict(
            enabled=False, penalty_enabled=True)))['penalty_enabled'])
        for raw in ({'enabled':1}, {'penalty_enabled':'yes'}, {'tau':-1.},
                    {'tau':float('nan')}, {'correction_weight':True}, {'unknown':1}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                action_correction_settings(dict(action_correction=raw))


if __name__ == '__main__':
    unittest.main(verbosity=2)
