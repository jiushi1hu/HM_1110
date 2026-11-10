from numbers import Integral

import torch
import torch.nn as nn
from harl.models.base.mlp import MLPBase
from harl.utils.envs_tools import check, get_shape_from_obs_space
from harl.utils.models_tools import init, get_init_method


class VNet(nn.Module):
    """One centralized feature network with one value output per reward target."""

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        """Initialize VNet model.
        Args:
            args: (dict) arguments containing relevant model information.
            cent_obs_space: (gym.Space) centralized observation space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(VNet, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.initialization_method = args["initialization_method"]
        output_dim = args.get("critic_output_dim", 1)
        if isinstance(output_dim, bool) or not isinstance(output_dim, Integral) or output_dim < 1:
            raise ValueError("critic_output_dim must be a positive integer")
        self.critic_output_dim = int(output_dim)
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = get_init_method(self.initialization_method)

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        self.base = MLPBase(args, cent_obs_shape)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_sizes[-1], self.critic_output_dim))

        self.to(device)

    def forward(self, cent_obs):
        """Compute values from the given observations.
        Args:
            cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        Returns:
            values: (torch.Tensor) value function predictions.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        values = self.v_out(critic_features)

        return values
