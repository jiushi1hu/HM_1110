"""Categorical action head for the reservoir's scalar discrete controls."""
import torch.nn as nn
from harl.models.base.distributions import Categorical


class ACTLayer(nn.Module):
    """Map actor features to a masked categorical distribution."""

    def __init__(self, action_space, inputs_dim, initialization_method, gain):
        super(ACTLayer, self).__init__()
        if action_space.__class__.__name__ != "Discrete":
            raise ValueError("Cascade controls require Discrete actions")
        self.action_out = Categorical(
            inputs_dim, action_space.n, initialization_method, gain
        )

    def forward(self, x, available_actions=None, deterministic=False):
        distribution = self.action_out(x, available_actions)
        actions = distribution.mode() if deterministic else distribution.sample()
        return actions, distribution.log_probs(actions)

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None,
                         return_entropy_per_sample=False):
        """Return likelihoods and row entropies or the masked mean entropy."""
        distribution = self.action_out(x, available_actions)
        action_log_probs = distribution.log_probs(action)
        if return_entropy_per_sample:
            return action_log_probs, distribution.entropy(), distribution
        if active_masks is not None:
            entropy = (
                distribution.entropy() * active_masks.squeeze(-1)
            ).sum() / active_masks.sum().clamp_min(1.0)
        else:
            entropy = distribution.entropy().mean()
        return action_log_probs, entropy, distribution
