"""Centralized value critic used by HAPPO."""
from harl.algorithms.critics.v_critic import VCritic

CRITIC_REGISTRY = {"happo": VCritic}
