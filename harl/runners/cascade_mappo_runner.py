"""MAPPO using the unchanged cascade environment and collection lifecycle."""
import copy

from harl.algorithms.actors.mappo import MAPPO
from harl.algorithms.actors.shared_mappo import make_shared_mappo_agents
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
from harl.runners.on_policy_ma_runner import OnPolicyMARunner


class CascadeMAPPORunner(CascadeReservoirRunner):
    """One ID-conditioned actor shared by five reservoirs and one value model."""

    actor_class = MAPPO
    algorithm_name = 'mappo'
    share_actor_parameters = True
    critic_reward_layout = 'mixed_by_reservoir_then_legacy_team'
    actor_value_channels = (0, 1, 2, 3, 4)

    def __init__(self, data, physical_config, algo_args, seed, *args, **kwargs):
        requested_args = copy.deepcopy(algo_args)
        effective_args = copy.deepcopy(algo_args)
        effective_args['algo']['share_param'] = True
        super().__init__(data, physical_config, effective_args, seed, *args, **kwargs)
        # The common comparison settings remain caller-owned. Record the
        # architecture's effective sharing flag separately from that request.
        self.requested_args = requested_args

    def _create_actors(self, network_args, env):
        """Construct the single policy before the centralized critic is built."""
        return make_shared_mappo_agents(network_args, env.observation_space,
                                       env.action_space, self.device)

    def train(self):
        """Dispatch the MAPPO update while reusing the cascade workflow."""
        return OnPolicyMARunner.train(self)
