"""Tensor and space helpers used by HAPPO; no game environment factories."""
import numpy as np
import torch


def check(value):
    """Check if value is a numpy array, if so, convert it to a torch tensor."""
    output = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    return output


def get_shape_from_obs_space(obs_space):
    """Get shape from observation space.
    Args:
        obs_space: one-dimensional gym.spaces.Box observation space
    Returns:
        obs_shape: (tuple) observation shape
    """
    if obs_space.__class__.__name__ != "Box" or len(obs_space.shape) != 1:
        raise ValueError("Cascade networks require one-dimensional Box observations")
    return obs_space.shape


def get_shape_from_act_space(act_space):
    """Get shape from action space.
    Args:
        act_space: gym.spaces.Discrete action space
    Returns:
        int: scalar discrete action width
    """
    if act_space.__class__.__name__ != "Discrete":
        raise ValueError("Cascade controls require Discrete actions")
    return 1

