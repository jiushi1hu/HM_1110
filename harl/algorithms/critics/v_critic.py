"""V Critic."""
import torch
import torch.nn as nn
from harl.utils.models_tools import (
    get_grad_norm,
    huber_loss,
    mse_loss,
    update_linear_schedule,
)
from harl.utils.envs_tools import check
from harl.models.value_function_models.v_net import VNet


def validate_value_normalization(use_clipped_value_loss, use_valuenorm):
    """Reject clipping across moving value scales until it is implemented safely."""
    if use_clipped_value_loss and use_valuenorm:
        raise ValueError(
            'Clipped value loss with ValueNorm is unsupported: use unclipped value loss '
            'or disable ValueNorm; clipping requires a consistent value scale')


class VCritic:
    """V Critic.
    Critic that learns a V-function.
    """

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.clip_param = args["clip_param"]
        self.critic_epoch = args["critic_epoch"]
        self.critic_num_mini_batch = args["critic_num_mini_batch"]
        self.value_loss_coef = args["value_loss_coef"]
        self.max_grad_norm = args["max_grad_norm"]
        self.huber_delta = args["huber_delta"]

        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.use_clipped_value_loss = args["use_clipped_value_loss"]
        self.use_huber_loss = args["use_huber_loss"]

        self.critic_lr = args["critic_lr"]
        self.opti_eps = args["opti_eps"]
        self.weight_decay = args["weight_decay"]

        self.share_obs_space = cent_obs_space

        self.critic = VNet(args, self.share_obs_space, self.device)

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode, episodes):
        """Decay the actor and critic learning rates.
        Args:
            episode: (int) current training episode.
            episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_values(self, cent_obs):
        """Get value function predictions.
        Args:
            cent_obs: (np.ndarray) centralized input to the critic.
        Returns:
            values: (torch.Tensor) value function predictions.
        """
        return self.critic(cent_obs)

    def _validate_value_normalizer(self, value_normalizer):
        validate_value_normalization(self.use_clipped_value_loss, value_normalizer is not None)

    def cal_value_loss(
        self, values, value_preds_batch, return_batch, value_normalizer=None,
        *, update_normalizer=True
    ):
        """Calculate value function loss.
        Args:
            values: (torch.Tensor) value function predictions.
            value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
            return_batch: (torch.Tensor) reward to go returns.
            value_normalizer: (ValueNorm) normalize the rewards, denormalize critic outputs.
            update_normalizer: update running statistics from this minibatch;
                false when train() has already updated once from the rollout.
        Returns:
            value_loss: (torch.Tensor) value function loss.
        """
        self._validate_value_normalizer(value_normalizer)
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )
        if value_normalizer is not None:
            if update_normalizer:
                value_normalizer.update(return_batch)
            error_clipped = (
                value_normalizer.normalize(return_batch) - value_pred_clipped
            )
            error_original = value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self.use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self.use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        value_loss = value_loss.mean()

        return value_loss

    def update(self, sample, value_normalizer=None, *, update_normalizer=True):
        """Update critic network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
            value_normalizer: (ValueNorm) normalize the rewards, denormalize critic outputs.
            update_normalizer: whether this minibatch updates running statistics.
        Returns:
            value_loss: (torch.Tensor) value function loss.
            critic_grad_norm: (torch.Tensor) gradient norm from critic update.
        """
        self._validate_value_normalizer(value_normalizer)
        (
            share_obs_batch,
            value_preds_batch,
            return_batch,
            _masks_batch,
        ) = sample

        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)

        values = self.get_values(share_obs_batch)

        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch, value_normalizer=value_normalizer,
            update_normalizer=update_normalizer
        )

        self.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self.use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm
            )
        else:
            critic_grad_norm = get_grad_norm(self.critic.parameters())

        self.critic_optimizer.step()

        return value_loss, critic_grad_norm

    def train(self, critic_buffer, value_normalizer=None, *, update_normalizer_once=False):
        """Perform a training update using minibatch GD.
        Args:
            critic_buffer: (OnPolicyCriticBufferEP or OnPolicyCriticBufferFP) buffer containing training data related to critic.
            value_normalizer: (ValueNorm) normalize the rewards, denormalize critic outputs.
            update_normalizer_once: the HAPPO and MAPPO runners update from the
                complete rollout once before all critic epochs. The default
                retains the per-minibatch mode for direct low-level callers.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        self._validate_value_normalizer(value_normalizer)
        if update_normalizer_once and value_normalizer is not None:
            # Returns are in reward units and GAE has already used the rollout's
            # original statistics. Exclude the terminal/bootstrap slot; flatten
            # time and environments, retaining the separate value channels.
            returns = critic_buffer.returns[:-1]
            value_normalizer.update(returns.reshape(-1, returns.shape[-1]))
        train_info = {}

        train_info["value_loss"] = 0
        train_info["critic_grad_norm"] = 0

        for _ in range(self.critic_epoch):
            data_generator = critic_buffer.feed_forward_generator_critic(
                self.critic_num_mini_batch
            )

            for sample in data_generator:
                value_loss, critic_grad_norm = self.update(
                    sample, value_normalizer=value_normalizer,
                    update_normalizer=not update_normalizer_once
                )

                train_info["value_loss"] += value_loss.item()
                train_info["critic_grad_norm"] += critic_grad_norm

        num_updates = self.critic_epoch * self.critic_num_mini_batch

        for k, _ in train_info.items():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        """Prepare for training."""
        self.critic.train()

    def prep_rollout(self):
        """Prepare for rollout."""
        self.critic.eval()
