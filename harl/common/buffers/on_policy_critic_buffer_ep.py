"""On-policy buffer for critic that uses Environment-Provided (EP) state."""
import torch
import numpy as np
from harl.utils.envs_tools import get_shape_from_obs_space


def validate_critic_minibatches(batch_size, count):
    """Require complete, equal-sized critic minibatches before shuffling."""
    if (isinstance(count, (bool, np.bool_)) or
            not isinstance(count, (int, np.integer)) or count < 1):
        raise ValueError('critic_num_mini_batch must be a positive integer')
    if batch_size < count or batch_size % count:
        raise ValueError('critic_num_mini_batch must divide the rollout size without dropping rows')
    return int(count)


class OnPolicyCriticBufferEP:
    """On-policy buffer for critic that uses Environment-Provided (EP) state."""

    def __init__(self, args, share_obs_space):
        """Initialize on-policy critic buffer.
        Args:
            args: (dict) arguments
            share_obs_space: (gym.Space or list) share observation space
        """
        self.episode_length = args["episode_length"]
        self.n_rollout_threads = args["n_rollout_threads"]
        self.gamma = args["gamma"]
        self.gae_lambda = args["gae_lambda"]
        self.use_gae = args["use_gae"]
        self.use_proper_time_limits = args["use_proper_time_limits"]
        output_dim = args.get("critic_output_dim", 1)
        if (isinstance(output_dim, (bool, np.bool_)) or
                not isinstance(output_dim, (int, np.integer)) or output_dim < 1):
            raise ValueError("critic_output_dim must be a positive integer")
        self.critic_output_dim = int(output_dim)

        share_obs_shape = get_shape_from_obs_space(share_obs_space)
        # Buffer for share observations
        self.share_obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, *share_obs_shape),
            dtype=np.float32,
        )

        # Buffer for value predictions made by this critic
        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.critic_output_dim), dtype=np.float32
        )

        # Buffer for returns calculated at each timestep
        self.returns = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.critic_output_dim), dtype=np.float32
        )

        # Buffer for rewards received by agents at each timestep
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, self.critic_output_dim), dtype=np.float32
        )

        # Buffer for masks indicating whether an episode is done at each timestep
        self.masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32
        )

        # Buffer for bad masks indicating truncation and termination. If 0, trunction; if 1 and masks is 0, termination; else, not done yet.
        self.bad_masks = np.ones_like(self.masks)

        self.step = 0

    def insert(
        self, share_obs, value_preds, rewards, masks, bad_masks
    ):
        """Insert data into buffer."""
        expected = (self.n_rollout_threads, self.critic_output_dim)
        if np.shape(value_preds) != expected or np.shape(rewards) != expected:
            raise ValueError("Critic values and rewards must match every rollout lane and reward channel")
        self.share_obs[self.step + 1] = share_obs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        self.bad_masks[self.step + 1] = bad_masks.copy()

        self.step = (self.step + 1) % self.episode_length

    def compute_returns(self, next_value, value_normalizer=None):
        """Compute independent reward-channel returns with shared episode boundaries.
        Args:
            next_value: (np.ndarray) value predictions for the step after the last episode step.
            value_normalizer: (ValueNorm) If not None, ValueNorm value normalizer instance.
        """
        if np.shape(next_value) != (self.n_rollout_threads, self.critic_output_dim):
            raise ValueError("Critic bootstrap must match every rollout lane and reward channel")
        if (
            self.use_proper_time_limits
        ):  # consider the difference between truncation and termination
            if self.use_gae:  # use GAE
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if value_normalizer is not None:  # use ValueNorm
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(self.value_preds[step + 1])
                            * self.masks[step + 1]
                            - value_normalizer.denormalize(self.value_preds[step])
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        gae = self.bad_masks[step + 1] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:  # do not use ValueNorm
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * self.value_preds[step + 1]
                            * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        gae = self.bad_masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:  # do not use GAE
                # Return storage is in reward units, including its bootstrap.
                self.returns[-1] = (next_value if value_normalizer is None else
                                    value_normalizer.denormalize(next_value))
                for step in reversed(range(self.rewards.shape[0])):
                    if value_normalizer is not None:  # use ValueNorm
                        self.returns[step] = (
                            self.returns[step + 1] * self.gamma * self.masks[step + 1]
                            + self.rewards[step]
                        ) * self.bad_masks[step + 1] + (
                            1 - self.bad_masks[step + 1]
                        ) * value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:  # do not use ValueNorm
                        self.returns[step] = (
                            self.returns[step + 1] * self.gamma * self.masks[step + 1]
                            + self.rewards[step]
                        ) * self.bad_masks[step + 1] + (
                            1 - self.bad_masks[step + 1]
                        ) * self.value_preds[
                            step
                        ]
        else:  # do not consider the difference between truncation and termination, i.e. all done episodes are terminated
            if self.use_gae:  # use GAE
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if value_normalizer is not None:  # use ValueNorm
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(self.value_preds[step + 1])
                            * self.masks[step + 1]
                            - value_normalizer.denormalize(self.value_preds[step])
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        self.returns[step] = gae + value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:  # do not use ValueNorm
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * self.value_preds[step + 1]
                            * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        self.returns[step] = gae + self.value_preds[step]
            else:  # do not use GAE
                self.returns[-1] = (next_value if value_normalizer is None else
                                    value_normalizer.denormalize(next_value))
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = (
                        self.returns[step + 1] * self.gamma * self.masks[step + 1]
                        + self.rewards[step]
                    )

    def feed_forward_generator_critic(
        self, critic_num_mini_batch
    ):
        """Training data generator for critic that uses MLP network.
        Args:
            critic_num_mini_batch: (int) Number of mini batches for critic.
        """

        # get episode_length, n_rollout_threads, mini_batch_size
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        critic_num_mini_batch = validate_critic_minibatches(batch_size, critic_num_mini_batch)
        mini_batch_size = batch_size // critic_num_mini_batch

        # shuffle indices
        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(critic_num_mini_batch)
        ]

        # Combine the first two dimensions (episode_length and n_rollout_threads) to form batch.
        # Take share_obs shape as an example:
        # (episode_length + 1, n_rollout_threads, *share_obs_shape) --> (episode_length, n_rollout_threads, *share_obs_shape)
        # --> (episode_length * n_rollout_threads, *share_obs_shape)
        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        value_preds = self.value_preds[:-1].reshape(-1, self.critic_output_dim)
        returns = self.returns[:-1].reshape(-1, self.critic_output_dim)
        masks = self.masks[:-1].reshape(-1, 1)

        for indices in sampler:
            # share_obs shape:
            # (episode_length * n_rollout_threads, *share_obs_shape) --> (mini_batch_size, *share_obs_shape)
            share_obs_batch = share_obs[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]

            yield share_obs_batch, value_preds_batch, return_batch, masks_batch
