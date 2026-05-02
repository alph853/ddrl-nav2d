from __future__ import annotations

from dataclasses import dataclass, field
import yaml
from pathlib import Path


@dataclass
class LearnerConfig:
    trainer_mode: str = "rl"
    gamma: float = 0.99
    clip_rho: float = 1.0
    clip_c: float = 1.0
    entropy_coef: float = 0.001
    value_loss_coef: float = 0.5
    bc_coef: float = 1.0
    bc_decay_updates: int = 0
    bc_floor_coef: float = 0.01
    action_init_std: float = 0.03
    reset_action_std_on_resume: bool = False
    lr: float = 5e-4
    grad_clip: float = 40.0
    reward_clip_abs: float = 200.0
    rollouts_per_update: int = 1024
    num_epochs_per_update: int = 1
    checkpoint_interval: int = 20
    policy_push_warmup_updates: int = 0
    device: str = "auto"
    azimuth_bins: int = 32
    z_bands: list[tuple[float, float]] | list[list[float]] | None = field(
        default_factory=lambda: [(0.0, 3.0)]
    )
    range_max_m: float = 35.0
    state_dim: int = 5  # [vx, vy, yaw_rate, distance, heading_error]
    action_dim: int = 2
    policy_model: str = "mlp"
    mlp_hidden_dim: int = 256
    lstm_hidden_dim: int = 256
    normalize_observation: bool = True
    observation_clip: float = 10.0

    def __post_init__(self) -> None:
        if self.z_bands is None:
            self.z_bands = [(0.0, 0.4), (0.4, 1.6), (1.6, 3.0)]

    @property
    def observation_dim(self) -> int:
        return int(self.azimuth_bins) * len(self.z_bands) + int(self.state_dim)

    @staticmethod
    def from_policy_arch_yaml(path: Path) -> "LearnerConfig":
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"Policy architecture config at {path} must be a mapping.")
        policy = data.get("policy", {})
        arch = data.get("architecture")
        if arch is None and isinstance(policy, dict):
            arch = policy.get("architecture")
        if not isinstance(arch, dict):
            raise ValueError(
                f"Policy architecture config at {path} must contain 'architecture' "
                "or 'policy.architecture'."
            )

        return LearnerConfig(
            azimuth_bins=arch.get("azimuth_bins", LearnerConfig.azimuth_bins),
            z_bands=arch.get("z_bands"),
            range_max_m=arch.get("range_max_m", LearnerConfig.range_max_m),
            state_dim=arch.get("state_dim", LearnerConfig.state_dim),
            action_dim=arch.get("action_dim", LearnerConfig.action_dim),
            action_init_std=arch.get("action_init_std", LearnerConfig.action_init_std),
            policy_model=arch.get("policy_model", LearnerConfig.policy_model),
            mlp_hidden_dim=arch.get("mlp_hidden_dim", LearnerConfig.mlp_hidden_dim),
            lstm_hidden_dim=arch.get("lstm_hidden_dim", LearnerConfig.lstm_hidden_dim),
            normalize_observation=arch.get(
                "normalize_observation", LearnerConfig.normalize_observation
            ),
            observation_clip=arch.get(
                "observation_clip", LearnerConfig.observation_clip
            ),
        )

    @staticmethod
    def from_rl_yaml(path: Path) -> "LearnerConfig":
        return LearnerConfig.from_policy_arch_yaml(path)
