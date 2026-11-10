"""MAPPO objective, independent updates and the common cascade lifecycle."""
import copy
import csv
from contextlib import redirect_stdout
import io
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
import yaml
from gym.spaces import Box, Discrete

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.actors.mappo import MAPPO
from harl.algorithms.actors.shared_mappo import (
    SharedMAPPOAgent, augment_agent_observations, joint_shared_actor_batch,
    make_shared_mappo_agents)
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.evaluation import evaluate_policy
from harl.envs.cascade_reservoir.periodic_evaluation import select_periodic_windows
from harl.runners.cascade_mappo_runner import CascadeMAPPORunner
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
from harl.runners.on_policy_ma_runner import OnPolicyMARunner


class MAPPOObjectiveTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.args = dict(hidden_sizes=[8, 8], activation_func='relu',
            use_feature_normalization=False, initialization_method='orthogonal_', gain=.01,
            lr=.003, opti_eps=1e-5, weight_decay=0., clip_param=.2, ppo_epoch=2,
            actor_num_mini_batch=1, entropy_coef=.01, use_max_grad_norm=True,
            max_grad_norm=.5, use_policy_active_masks=True, action_aggregation='prod',
            normalize_shared_advantages=True, gamma=.995, share_param=False)
        self.obs_space = Box(-np.inf, np.inf, (2,), dtype=np.float32)
        self.act_space = Discrete(3)

    def actor(self, cls=MAPPO, **changes):
        torch.manual_seed(21)
        return cls(dict(self.args, **changes), self.obs_space, self.act_space)

    def sample(self, actor):
        obs = np.array([[.2, -.3], [1., .5], [-.4, .8], [.9, -.7]], dtype=np.float32)
        actions = np.array([[0], [1], [2], [1]], dtype=np.float32)
        active = np.ones((4, 1), dtype=np.float32)
        with torch.no_grad():
            logp, _, _ = actor.evaluate_actions(obs, actions)
        old_logp = logp.numpy() + np.array([[.3], [-.3], [.2], [-.2]], dtype=np.float32)
        advantages = np.array([[-1.], [.2], [1.], [-.5]], dtype=np.float32)
        return (obs, actions, active.copy(), active, old_logp, advantages, None,
                np.ones_like(active))

    def buffer(self, actor):
        sample = self.sample(actor)
        buffer = OnPolicyActorBuffer(dict(episode_length=4, n_rollout_threads=1),
                                     self.obs_space, self.act_space)
        buffer.obs[:-1, 0] = sample[0]
        buffer.actions[:, 0] = sample[1]
        with torch.no_grad():
            logp, _, _ = actor.evaluate_actions(sample[0], sample[1])
        buffer.action_log_probs[:, 0] = logp.numpy()
        buffer.masks[-1] = 0.
        buffer.update_factor(np.ones_like(buffer.active_masks[:-1]))
        return buffer, sample[5].reshape(4, 1, 1)

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for x, y in zip(left, right):
                self.assert_state_equal(x, y)
        else:
            self.assertEqual(left, right)

    def assert_state_close(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-7)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_state_close(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for x, y in zip(left, right):
                self.assert_state_close(x, y)
        else:
            self.assertEqual(left, right)

    def test_ppo_moves_toward_advantages_and_clips_already_beneficial_changes(self):
        # One controllable decision isolates the policy response. Entropy is
        # disabled so only PPO's selected-action incentive can move weights.
        for advantage, ratio, clipped in ((1., 1., False), (-1., 1., False),
                                           (1., 1.5, True), (-1., .5, True)):
            with self.subTest(advantage=advantage, ratio=ratio):
                actor = self.actor(entropy_coef=0.)
                obs = np.array([[.2, -.3]], dtype=np.float32)
                action = np.array([[0]], dtype=np.float32)
                active = np.ones((1, 1), dtype=np.float32)
                with torch.no_grad():
                    before_logp, _, _ = actor.evaluate_actions(obs, action)
                before_model = copy.deepcopy(actor.actor.state_dict())
                old_logp = before_logp.numpy()-np.float32(np.log(ratio))
                sample = (obs, action, active.copy(), active, old_logp,
                          np.array([[advantage]], dtype=np.float32), None,
                          active.copy())
                actor.update(sample)
                with torch.no_grad():
                    after_logp, _, _ = actor.evaluate_actions(obs, action)
                if clipped:
                    self.assert_state_equal(before_model, actor.actor.state_dict())
                else:
                    self.assertGreater(advantage*float(after_logp-before_logp), 0.)

    def test_unit_factor_matches_existing_happo_single_actor_update(self):
        actor, happo = self.actor(), self.actor(HAPPO)
        self.assert_state_equal(actor.actor.state_dict(), happo.actor.state_dict())
        sample = self.sample(actor)
        left, right = actor.update(sample), happo.update(sample)
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, rtol=0., atol=0.)
        self.assert_state_equal(actor.actor.state_dict(), happo.actor.state_dict())

    def test_nonunit_or_missing_factors_are_rejected(self):
        actor = self.actor()
        for factor in (None, np.full((4, 1), .9), np.full((4, 1), np.nan),
                       np.ones((4, 2))):
            sample = list(self.sample(actor))
            sample[7] = factor
            with self.subTest(factor=factor), self.assertRaisesRegex(ValueError, 'unit buffer factors'):
                actor.update(tuple(sample))
        buffer, advantages = self.buffer(actor)
        buffer.factor[:] = 2.
        with self.assertRaisesRegex(ValueError, 'unit buffer factors'):
            actor.train(buffer, advantages, 'EP', diagnostics=True)

    def test_zero_control_does_not_advance_adam(self):
        for cls in (HAPPO, MAPPO):
            actor = self.actor(cls)
            actor.update(self.sample(actor))  # Populate Adam momentum first.
            model = copy.deepcopy(actor.actor.state_dict())
            optimizer = copy.deepcopy(actor.actor_optimizer.state_dict())
            sample = list(self.sample(actor))
            sample[3] = np.zeros_like(sample[3])
            with mock.patch.object(actor, 'evaluate_actions',
                                   side_effect=AssertionError('zero-mass batch was evaluated')):
                result = actor.update(tuple(sample))
            self.assertEqual(len(result), 4)
            self.assertEqual(float(result[0]), 0.)
            self.assert_state_equal(model, actor.actor.state_dict())
            self.assert_state_equal(optimizer, actor.actor_optimizer.state_dict())

    def test_forced_duplicate_keeps_full_row_mean_for_policy_and_entropy(self):
        for cls in (HAPPO, MAPPO):
            with self.subTest(algorithm=cls.__name__):
                single = self.actor(cls, use_max_grad_norm=False)
                masked = copy.deepcopy(single)
                sample = tuple(None if value is None else value[:1].copy()
                               for value in self.sample(single))
                doubled = [None if value is None else np.repeat(value, 2, axis=0)
                           for value in sample]
                doubled[3][1] = 0.
                # A forced placeholder's values cannot contribute to either
                # objective, but it remains a row in the minibatch mean.
                doubled[0][1] = [100., -100.]
                doubled[1][1] = 2.
                doubled[4][1] = 0.
                doubled[5][1] = 10000.
                one_result = single.update(sample)
                two_result = masked.update(tuple(doubled))
                for one, two in zip(one_result[:3], two_result[:3]):
                    torch.testing.assert_close(two, .5*one, rtol=1e-6, atol=1e-7)

    def test_active_masks_and_diagnostics_preserve_update_and_rng(self):
        plain = self.actor()
        measured = copy.deepcopy(plain)
        buffer, advantages = self.buffer(plain)
        buffer.masks[2] = 0.  # Episode boundaries do not reweight actor rows.
        buffer.active_masks[1] = 0.
        measured_buffer = copy.deepcopy(buffer)
        torch.manual_seed(72)
        plain_info = plain.train(buffer, advantages.copy(), 'EP')
        rng_after_plain = torch.get_rng_state().clone()
        torch.manual_seed(72)
        info = measured.train(measured_buffer, advantages.copy(), 'EP', diagnostics=True)
        self.assertTrue(torch.equal(rng_after_plain, torch.get_rng_state()))
        self.assert_state_equal(plain.actor.state_dict(), measured.actor.state_dict())
        self.assert_state_equal(plain.actor_optimizer.state_dict(), measured.actor_optimizer.state_dict())
        self.assertEqual(info['optimizer_steps'], 2.)
        self.assertEqual(info['ratio_samples'], 6.)
        self.assertEqual(info['ppo_epochs_completed'], 2.)
        self.assertEqual(plain_info['ratio_samples'], info['ratio_samples'])
        for suffix in ('before', 'after'):
            self.assertAlmostEqual(info['regularized_objective_'+suffix],
                info['clipped_surrogate_'+suffix]+measured.entropy_coef*info['weighted_entropy_'+suffix])

    def test_fixed_advantages_give_gamma_invariant_updates_without_row_age_preference(self):
        # Gamma belongs to target construction. Moving an unchanged decision
        # to a later rollout row must not reduce its PPO or entropy influence.
        identity = np.arange(4)
        moved = np.array([3, 0, 2, 1])
        python_state, numpy_state = random.getstate(), np.random.get_state()
        for cls in (HAPPO, MAPPO):
            initial = self.actor(cls, gamma=0.)
            original, advantages = self.buffer(initial)
            original.action_log_probs[:, 0] = self.sample(initial)[4]
            original.active_masks[1] = 0.
            original.actions[1] = 0.
            original.action_log_probs[1] = 0.
            original.masks[2] = 0.
            if cls is HAPPO:
                original.update_factor(np.array([.6, 1.2, 1.7, .9],
                                                dtype=np.float32).reshape(4, 1, 1))
            reference = None
            for gamma in (0., .995, 1.):
                for order in (identity, moved):
                    with self.subTest(algorithm=cls.__name__, gamma=gamma,
                                      row_order=order.tolist()):
                        actor = self.actor(cls, gamma=gamma)
                        buffer = copy.deepcopy(original)
                        for name in ('obs', 'masks', 'active_masks', 'available_actions'):
                            values = getattr(buffer, name)
                            values[:-1] = values[:-1][order].copy()
                        for name in ('actions', 'action_log_probs', 'factor'):
                            values = getattr(buffer, name)
                            values[:] = values[order].copy()
                        fixed_advantages = advantages[order].copy()
                        torch.manual_seed(72)
                        info = actor.train(buffer, fixed_advantages, 'EP', diagnostics=True)
                        observed = (copy.deepcopy(actor.actor.state_dict()),
                                    copy.deepcopy(actor.actor_optimizer.state_dict()),
                                    torch.get_rng_state().clone(), info)
                        if reference is None:
                            reference = observed
                        elif np.array_equal(order, identity):
                            self.assert_state_equal(observed[:3], reference[:3])
                            self.assertEqual(info, reference[3])
                        else:
                            # Permuting a reduction can change its final float32
                            # rounding; the update and Adam moments must agree.
                            self.assert_state_close(observed[:2], reference[:2])
                            self.assert_state_equal(observed[2], reference[2])
                            self.assertEqual(set(info), set(reference[3]))
                            for name in info:
                                # Reordering four float32 rows can change the
                                # reduction by a few rounding units. Gamma
                                # comparisons above remain exactly equal.
                                np.testing.assert_allclose(float(info[name]), float(reference[3][name]),
                                                           rtol=1e-5, atol=4*np.finfo(np.float32).eps)
                        self.assertEqual(info['optimizer_steps'], 2.)
                        self.assertEqual(info['ratio_samples'], 6.)
                        self.assertEqual(info['ppo_epochs_completed'], 2.)
                        self.assertGreater(info['weighted_entropy_before'], 0.)
        self.assert_state_equal(random.getstate(), python_state)
        self.assert_state_equal(np.random.get_state(), numpy_state)


class CascadeDiscountedReturnsTest(unittest.TestCase):
    def test_gamma_changes_six_channel_returns_and_gae_but_terminal_ignores_bootstrap(self):
        scales = np.arange(1, 7, dtype=np.float32)
        # Rewards are (2, 4)*scale and frozen values are (1, 3)*scale.
        # Lambda=.5 gives first-step GAE targets 2, 5.4825, 5.5 at the
        # three discounts; ordinary returns are 2, 5.98, 6 respectively.
        cases = ((0., 2., 2.), (.995, 5.4825, 5.98), (1., 5.5, 6.))
        for use_gae in (False, True):
            for gamma, gae_first, return_first in cases:
                with self.subTest(use_gae=use_gae, gamma=gamma):
                    buffer = OnPolicyCriticBufferEP(dict(episode_length=2,
                        n_rollout_threads=1, gamma=gamma, gae_lambda=.5,
                        use_gae=use_gae, use_proper_time_limits=True, critic_output_dim=6),
                        Box(-np.inf, np.inf, (2,), dtype=np.float32))
                    buffer.rewards[:, 0] = np.array([[2.], [4.]])*scales
                    buffer.value_preds[:-1, 0] = np.array([[1.], [3.]])*scales
                    buffer.masks[-1] = 0.
                    buffer.compute_returns(np.full((1, 6), 1e6, dtype=np.float32))
                    expected_first = gae_first if use_gae else return_first
                    np.testing.assert_allclose(buffer.returns[0, 0], expected_first*scales,
                                               rtol=1e-6, atol=1e-6)
                    np.testing.assert_array_equal(buffer.returns[1, 0], 4.*scales)

class MAPPORunnerDispatchTest(unittest.TestCase):
    def test_own_return_channels_without_prepost_likelihood_evaluation(self):
        calls, captured = [], {}
        runner = OnPolicyMARunner()
        runner.num_agents = 5
        runner.algo_args = dict(algo=dict(share_param=False, normalize_shared_advantages=True))
        runner.diagnostics_enabled = True
        raw = np.arange(24, dtype=np.float32).reshape(4, 1, 6)
        runner.critic_buffer = SimpleNamespace(returns=raw**2, value_preds=raw/7.)
        runner.value_normalizer = SimpleNamespace(denormalize=lambda values: 2.*values)

        class Buffer:
            active_masks = np.ones((4, 1, 1), dtype=np.float32)
            def update_factor(self, factor):
                self.factor = factor.copy()

        class Actor:
            def __init__(self, index):
                self.index = index
            def train(self, buffer, advantages, state_type, diagnostics=False):
                calls.append(self.index)
                captured[self.index] = advantages.copy()
                np.testing.assert_array_equal(buffer.factor, np.ones((3, 1, 1)))
                if state_type != 'EP' or not diagnostics:
                    raise AssertionError('Wrong training options')
                return dict(policy_loss=float(self.index))
            def evaluate_actions(self, *args, **kwargs):
                raise AssertionError('MAPPO runner must not evaluate cross-agent likelihood factors')

        class Critic:
            def train(self, buffer, normalizer, *, update_normalizer_once=False):
                if not update_normalizer_once or calls != [0, 1, 2, 3, 4]:
                    raise AssertionError('MAPPO must update ValueNorm once after all actor updates')
                calls.append('critic')
                return dict(value_loss=0.)

        runner.actor_buffer = [Buffer() for _ in range(5)]
        runner.actor = [Actor(i) for i in range(5)]
        runner.critic = Critic()
        infos, critic_info = runner.train()
        expected = raw[:-1]**2-2.*raw[:-1]/7.
        expected = (expected-expected.mean(axis=(0, 1), keepdims=True))/(
            expected.std(axis=(0, 1), keepdims=True)+1e-5)
        self.assertEqual(calls, [0, 1, 2, 3, 4, 'critic'])
        for i in range(5):
            np.testing.assert_array_equal(captured[i], expected[..., i:i+1])
            self.assertEqual(infos[i]['factor_min'], 1.)
            self.assertEqual(infos[i]['factor_max'], 1.)
        self.assertEqual(critic_info['value_loss'], 0.)


class SharedMAPPOObjectiveTest(unittest.TestCase):
    setUp = MAPPOObjectiveTest.setUp
    sample = MAPPOObjectiveTest.sample
    buffer = MAPPOObjectiveTest.buffer
    assert_state_equal = MAPPOObjectiveTest.assert_state_equal
    assert_state_close = MAPPOObjectiveTest.assert_state_close

    def views(self, **changes):
        torch.manual_seed(21)
        return make_shared_mappo_agents(dict(self.args, **changes),
            [self.obs_space]*5, [self.act_space]*5)

    def rollouts(self, views):
        buffers, advantages = [], []
        for i, view in enumerate(views):
            buffer, advantage = self.buffer(view)
            buffer.obs[:-1, 0] += np.float32(.1*i)
            with torch.no_grad():
                logp, _, _ = view.evaluate_actions(buffer.obs[:-1, 0], buffer.actions[:, 0])
            buffer.action_log_probs[:, 0] = logp.numpy()
            buffers.append(buffer)
            advantages.append(advantage.copy())
        return buffers, advantages

    def test_one_core_optimizer_and_identity_inputs_without_extra_initializers(self):
        views = self.views()
        after_factory = torch.get_rng_state().clone()
        torch.manual_seed(21)
        reference = MAPPO(dict(self.args, share_param=True),
            Box(np.r_[self.obs_space.low, np.zeros(5)].astype(np.float32),
                np.r_[self.obs_space.high, np.ones(5)].astype(np.float32), dtype=np.float32),
            self.act_space)
        self.assertTrue(torch.equal(after_factory, torch.get_rng_state()))
        self.assert_state_equal(views[0].actor.state_dict(), reference.actor.state_dict())
        self.assertEqual(len({id(view.actor) for view in views}), 1)
        self.assertEqual(len({id(view.actor_optimizer) for view in views}), 1)
        obs = np.array([[.2, -.3], [.4, .1]], dtype=np.float32)
        saved = obs.copy()
        for i, view in enumerate(views):
            self.assertIs(view.core, view.shared_core)
            self.assertEqual(view.obs_space.shape, (2,))
            augmented = augment_agent_observations(obs, i)
            np.testing.assert_array_equal(augmented[:, :2], obs)
            np.testing.assert_array_equal(augmented[:, 2:], np.eye(5, dtype=np.float32)[[i, i]])
            with torch.no_grad():
                expected = reference.get_actions(augmented, deterministic=True)
                actual = view.get_actions(obs, deterministic=True)
            for left, right in zip(expected, actual):
                torch.testing.assert_close(left, right, rtol=0., atol=0.)
        np.testing.assert_array_equal(obs, saved)
        views[2].actor_num_mini_batch = 2
        self.assertEqual(views[0].core.actor_num_mini_batch, 2)
        with self.assertRaisesRegex(ValueError, 'share_param_train'):
            views[0].train(None, None, 'EP')

    def test_identity_helper_supports_tensor_batches_and_frozen_views(self):
        views = self.views()
        obs = torch.ones((2, 3, 2), requires_grad=True)
        augmented = augment_agent_observations(obs, 3)
        self.assertEqual(tuple(augmented.shape), (2, 3, 7))
        augmented.sum().backward()
        torch.testing.assert_close(obs.grad, torch.ones_like(obs))
        with self.assertRaises(ValueError):
            augment_agent_observations(obs, 5)
        # Frozen replay needs neither an optimizer nor training actor methods.
        core = views[0].core
        frozen = SimpleNamespace(args=core.args, actor=core.actor, obs_space=core.obs_space)
        view = SharedMAPPOAgent(frozen, 2)
        raw = np.array([[.3, -.5]], dtype=np.float32)
        with torch.no_grad():
            action = view.act(raw, deterministic=True)
            expected = views[2].act(raw, deterministic=True)
            left = view.evaluate_actions(raw, action, return_entropy_per_sample=True)
            right = views[2].evaluate_actions(raw, action, return_entropy_per_sample=True)
        torch.testing.assert_close(action, expected, rtol=0., atol=0.)
        torch.testing.assert_close(left[0], right[0], rtol=0., atol=0.)
        torch.testing.assert_close(left[1], right[1], rtol=0., atol=0.)

    def test_joint_rows_preserve_own_targets_masks_and_source_buffers(self):
        views = self.views()
        buffers, advantages = self.rollouts(views)
        for i, buffer in enumerate(buffers):
            buffer.masks[1+i%3] = 0.
            buffer.active_masks[i%4] = 0.
            advantages[i] += np.float32(i)
        snapshots = [copy.deepcopy(buffer.__dict__) for buffer in buffers]
        joint, joined = joint_shared_actor_batch(views[0].core, buffers, advantages, 5)
        self.assertEqual(joint.obs.shape, (5, 5, 7))
        for i, buffer in enumerate(buffers):
            np.testing.assert_array_equal(joint.obs[:, i, :2], buffer.obs[:, 0])
            np.testing.assert_array_equal(joint.obs[:, i, 2:],
                np.broadcast_to(np.eye(5, dtype=np.float32)[i], (5, 5)))
            np.testing.assert_array_equal(joined[:, i], advantages[i][:, 0])
            np.testing.assert_array_equal(joint.action_log_probs[:, i], buffer.action_log_probs[:, 0])
            np.testing.assert_array_equal(joint.active_masks[:, i], buffer.active_masks[:, 0])
            np.testing.assert_array_equal(joint.masks[:, i], buffer.masks[:, 0])
            for name in ('obs', 'actions', 'action_log_probs', 'masks', 'active_masks', 'factor'):
                np.testing.assert_array_equal(getattr(buffer, name), snapshots[i][name])
        with self.assertRaisesRegex(ValueError, 'own advantage'):
            joint_shared_actor_batch(views[0].core, buffers, joined, 5)

    def test_shared_gradient_matches_one_joint_minibatch_and_one_adam_step(self):
        views = self.views(ppo_epoch=1)
        core = views[0].core
        buffers, advantages = self.rollouts(views)
        joint, joined = joint_shared_actor_batch(core, buffers, advantages, 5)
        expected = copy.deepcopy(core)
        torch.manual_seed(72)
        sample = next(joint.feed_forward_generator_actor(joined, 1))
        self.assertEqual(len(sample), 8)
        expected.update(sample)
        torch.manual_seed(72)
        with mock.patch.object(core, 'update', wraps=core.update) as update:
            info = core.share_param_train(buffers, advantages, 5, 'EP')
        self.assertEqual(update.call_count, 1)
        self.assertEqual(info['optimizer_steps'], 1.)
        self.assertEqual(info['ratio_samples'], 20.)
        self.assert_state_equal(expected.actor.state_dict(), core.actor.state_dict())
        self.assert_state_equal(expected.actor_optimizer.state_dict(), core.actor_optimizer.state_dict())
        self.assertTrue(all(float(state['step']) == 1.
                            for state in core.actor_optimizer.state.values()))

    def test_fallback_normalization_stays_per_reservoir_and_skips_double_normalization(self):
        views = self.views(normalize_shared_advantages=False)
        buffers, advantages = self.rollouts(views)
        for i in range(5):
            advantages[i] = advantages[i]*(i+1)+10.*i
        buffers[2].active_masks[:-1] = 0.
        buffers[3].active_masks[1] = 0.
        core = views[0].core
        joint, joined = joint_shared_actor_batch(core, buffers, advantages, 5)
        for i, buffer in enumerate(buffers):
            active = buffer.active_masks[:-1] > 0.
            if active.any():
                selected = advantages[i][active]
                expected = (advantages[i]-selected.mean())/(selected.std()+1e-5)
            else:
                expected = np.zeros_like(advantages[i])
            np.testing.assert_array_equal(joined[:, i:i+1], expected)
        with mock.patch.object(core, 'train', return_value={}) as train:
            core.share_param_train(buffers, advantages, 5, 'EP')
        self.assertTrue(train.call_args.kwargs['advantages_normalized'])
        np.testing.assert_array_equal(train.call_args.args[1], joined)

    def test_shared_diagnostics_preserve_updates_rng_and_accept_raw_view_buffers(self):
        views = self.views()
        core = views[0].core
        measured_core = copy.deepcopy(core)
        measured_views = [SharedMAPPOAgent(measured_core, i) for i in range(5)]
        buffers, advantages = self.rollouts(views)
        measured_buffers = copy.deepcopy(buffers)
        torch.manual_seed(72)
        core.share_param_train(buffers, advantages, 5, 'EP')
        rng = torch.get_rng_state().clone()
        torch.manual_seed(72)
        info = measured_core.share_param_train(measured_buffers, advantages, 5, 'EP', diagnostics=True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assert_state_equal(core.actor.state_dict(), measured_core.actor.state_dict())
        self.assert_state_equal(core.actor_optimizer.state_dict(), measured_core.actor_optimizer.state_dict())
        self.assertEqual(info['optimizer_steps'], 2.)
        self.assertEqual(info['ratio_samples'], 40.)
        self.assertEqual(info['ppo_epochs_completed'], 2.)
        for view, buffer, advantage in zip(measured_views, measured_buffers, advantages):
            distribution = view.rollout_distribution(buffer)
            obs = buffer.obs[:-1].reshape(-1, *buffer.obs.shape[2:])
            actions = buffer.actions.reshape(-1, buffer.actions.shape[-1])
            with torch.no_grad():
                logp, _, _ = view.evaluate_actions(obs, actions)
            chosen_actions = torch.as_tensor(actions, dtype=torch.int64).squeeze(-1)
            torch.testing.assert_close(distribution.log_prob(chosen_actions), logp.squeeze(-1))
            metrics = view.surrogate_metrics(buffer, advantage)
            self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_shared_fixed_channels_are_gamma_invariant_after_row_age_permutation(self):
        original_views = self.views(gamma=0.)
        original_buffers, original_advantages = self.rollouts(original_views)
        for i, buffer in enumerate(original_buffers):
            buffer.active_masks[i % 4] = 0.
            buffer.actions[i % 4] = 0.
            buffer.action_log_probs[i % 4] = 0.
        reference = None
        for gamma in (0., .995, 1.):
            for order in (np.arange(4), np.array([3, 0, 2, 1])):
                with self.subTest(gamma=gamma, row_order=order.tolist()):
                    core = self.views(gamma=gamma)[0].core
                    buffers = copy.deepcopy(original_buffers)
                    for buffer in buffers:
                        for name in ('obs', 'masks', 'active_masks', 'available_actions'):
                            values = getattr(buffer, name)
                            values[:-1] = values[:-1][order].copy()
                        for name in ('actions', 'action_log_probs', 'factor'):
                            values = getattr(buffer, name)
                            values[:] = values[order].copy()
                    advantages = [values[order].copy() for values in original_advantages]
                    torch.manual_seed(72)
                    info = core.share_param_train(buffers, advantages, 5, 'EP', diagnostics=True)
                    observed = (copy.deepcopy(core.actor.state_dict()),
                                copy.deepcopy(core.actor_optimizer.state_dict()),
                                torch.get_rng_state().clone(), info)
                    if reference is None:
                        reference = observed
                    elif np.array_equal(order, np.arange(4)):
                        self.assert_state_equal(observed[:3], reference[:3])
                        self.assertEqual(info, reference[3])
                    else:
                        self.assert_state_close(observed[:2], reference[:2])
                        self.assert_state_equal(observed[2], reference[2])
                        for name in info:
                            np.testing.assert_allclose(float(info[name]), float(reference[3][name]),
                                                       rtol=1e-5, atol=1e-7)
                    self.assertEqual(info['optimizer_steps'], 2.)
                    self.assertEqual(info['ratio_samples'], 30.)
                    self.assertEqual(info['ppo_epochs_completed'], 2.)



class CascadeMAPPOIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = yaml.safe_load((ROOT/'harl/configs/algos_cfgs/happo_cascade.yaml').read_text())
        cls.physical = yaml.safe_load((ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text())
        cls.args['train'].update(episode_length=4, n_rollout_threads=2)
        cls.args['model'].update(hidden_sizes=[8, 8], action_bins=7)
        cls.args['algo'].update(ppo_epoch=2, critic_epoch=2)
        cls.args['experiment'].update(episodes=2, seeds=[1], diagnostics=True)
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new', cls.physical, cls.args['experiment'])
        cls.windows = select_periodic_windows(cls.data, episode_days=4)

    def test_shared_actor_identity_inputs_and_repeatable_initialization(self):
        happo = CascadeReservoirRunner(self.data, self.physical, self.args, 1,
                                        periodic_windows=self.windows)
        self.addCleanup(happo.close)
        mappo = CascadeMAPPORunner(self.data, self.physical, self.args, 1,
                                   periodic_windows=self.windows)
        self.addCleanup(mappo.close)
        rng = torch.get_rng_state().clone()
        repeat = CascadeMAPPORunner(self.data, self.physical, self.args, 1,
                                    periodic_windows=self.windows)
        self.addCleanup(repeat.close)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(len({id(a.actor) for a in happo.actor}), 5)
        self.assertEqual(len({id(a.actor) for a in mappo.actor}), 1)
        self.assertEqual(len({id(a.actor_optimizer) for a in mappo.actor}), 1)
        for ma in mappo.actor:
            self.assertIsInstance(ma, SharedMAPPOAgent)
            self.assertIs(ma.shared_core, mappo.actor[0].shared_core)
            self.assertEqual(ma.obs_space.shape, happo.actor[0].obs_space.shape)
            self.assertEqual(ma.core.obs_space.shape[0], ma.obs_space.shape[0]+5)
        for name, value in mappo.actor[0].actor.state_dict().items():
            self.assertTrue(torch.equal(value, repeat.actor[0].actor.state_dict()[name]))
        for name, value in mappo.critic.critic.state_dict().items():
            self.assertTrue(torch.equal(value, repeat.critic.critic.state_dict()[name]))
        self.assertEqual(mappo.critic_output_dim, 6)

    def test_tiny_real_update_diagnostics_checkpoint_and_replay(self):
        from harl.envs.cascade_reservoir.checkpoint import load_checkpoint
        from harl.envs.cascade_reservoir.trajectory_diagnostics import TrajectoryDiagnosticsWriter

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            directory = Path(directory)
            trace = directory/'trajectory.csv'
            with TrajectoryDiagnosticsWriter(trace) as writer:
                runner = CascadeMAPPORunner(self.data, self.physical, self.args, 1,
                    periodic_windows=self.windows, trajectory_log=writer,
                    checkpoint_dir=directory/'checkpoints')
                self.addCleanup(runner.close)
                initial_actor = copy.deepcopy(runner.actor[0].actor.state_dict())
                initial_critic = copy.deepcopy(runner.critic.critic.state_dict())
                with mock.patch('harl.runners.on_policy_ha_runner.OnPolicyHARunner.train',
                                side_effect=AssertionError('HAPPO dispatch used for MAPPO')):
                    runner.run(2)
            self.assertEqual(runner.update_count, 1)
            self.assertEqual(len(runner.diagnostics_history), 1)
            self.assertTrue(any(not torch.equal(value, runner.actor[0].actor.state_dict()[name])
                                for name, value in initial_actor.items()))
            self.assertTrue(any(not torch.equal(value, runner.critic.critic.state_dict()[name])
                                for name, value in initial_critic.items()))
            for i, buffer in enumerate(runner.actor_buffer):
                np.testing.assert_array_equal(buffer.factor, np.ones((4, 2, 1)))
                self.assertEqual(runner.training_history[0]['actor'+str(i)+'_factor_max'], 1.)
                self.assertEqual(runner.training_history[0]['actor'+str(i)+'_shared_actor'], 1.)
                self.assertEqual(runner.training_history[0]['actor'+str(i)+'_optimizer_steps'], 2.)
            with trace.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4*2*5)
            self.assertTrue(all(float(row['happo_factor']) == 1. for row in rows))
            checkpoint = directory/'checkpoints/update_000001.pt'
            rng = torch.get_rng_state().clone()
            policy = load_checkpoint(checkpoint)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(len({id(a.actor) for a in policy.actors}), 1)
            replay = policy.replay([1965])
            reference = evaluate_policy(runner.actor, self.data, self.physical,
                seed=1, episode=2, training_steps=8, gamma=self.args['algo']['gamma'],
                source_years=[1965], episode_days=4, diagnostics=True)
            for field in ('energy_MWh', 'episode_reward', 'episode_mixed_rewards',
                          'episode_system_reward', 'spill_volume_m3'):
                np.testing.assert_allclose(replay[field], reference[field], rtol=0., atol=1e-10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
