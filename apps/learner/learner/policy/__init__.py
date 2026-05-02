from __future__ import annotations

import torch

from learner.config.learner import LearnerConfig

from .export import export_policy_to_onnx
from .actor_critic import ActorCritic
from .distribution import TanhGaussianAction
from .distribution import TanhGaussianConfig
from .normalization import ObservationNormalizer

PolicyModule = ActorCritic


def build_policy(
    config: LearnerConfig, action_dim: int, device: torch.device | None = None
) -> ActorCritic:
    return ActorCritic.from_config(config, action_dim, device=device)


__all__ = [
    "export_policy_to_onnx",
    "ActorCritic",
    "ObservationNormalizer",
    "PolicyModule",
    "TanhGaussianAction",
    "TanhGaussianConfig",
    "build_policy",
]
