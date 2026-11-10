"""Base class for on-policy algorithms."""

import torch
from harl.models.policy_models.stochastic_policy import StochasticPolicy
from harl.utils.models_tools import update_linear_schedule


class OnPolicyBase:
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize Base class.
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        # save arguments
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.action_aggregation = args["action_aggregation"]

        self.lr = args["lr"]
        self.opti_eps = args["opti_eps"]
        self.weight_decay = args["weight_decay"]
        # save observation and action spaces
        self.obs_space = obs_space
        self.act_space = act_space
        # create actor network
        self.actor = StochasticPolicy(args, self.obs_space, self.act_space, self.device)
        # create actor optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode, episodes):
        """Decay the learning rates.
        Args:
            episode: (int) current training episode.
            episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)

    def get_actions(
        self, obs, available_actions=None, deterministic=False
    ):
        """Compute actions for the given inputs.
        Args:
            obs: (np.ndarray) local agent inputs to the actor.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, action_log_probs = self.actor(
            obs, available_actions, deterministic
        )
        return actions, action_log_probs

    def evaluate_actions(
        self,
        obs,
        action,
        available_actions=None,
        active_masks=None,
        return_entropy_per_sample=False,
    ):
        """Get action logprobs, entropy, and distributions for actor update.
        Args:
            obs: (np.ndarray / torch.Tensor) local agent inputs to the actor.
            action: (np.ndarray / torch.Tensor) actions whose log probabilities and entropy to compute.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                    (if None, all actions available)
            active_masks: (np.ndarray / torch.Tensor) denotes whether an agent is active or dead.
            return_entropy_per_sample: return an unreduced entropy vector so
                the caller can apply active masks before the full-row mean.
        """

        (
            action_log_probs,
            dist_entropy,
            action_distribution,
        ) = self.actor.evaluate_actions(
            obs, action, available_actions, active_masks,
            return_entropy_per_sample=return_entropy_per_sample,
        )
        return action_log_probs, dist_entropy, action_distribution

    def act(
        self, obs, available_actions=None, deterministic=False
    ):
        """Compute actions using the given inputs.
        Args:
            obs: (np.ndarray) local agent inputs to the actor.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                    (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _ = self.actor(
            obs, available_actions, deterministic
        )
        return actions

    def prep_training(self):
        """Prepare for training."""
        self.actor.train()

    def prep_rollout(self):
        """Prepare for rollout."""
        self.actor.eval()
