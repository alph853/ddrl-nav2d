from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Normal


@dataclass(frozen=True)
class TanhGaussianConfig:
    log_std_min: float = -5.0
    log_std_max: float = -2.0
    eps: float = 1e-6
    latent_clip: float = 20.0


class TanhGaussianAction:
    """Diagonal Gaussian followed by tanh, producing normalized actions in [-1, 1]."""

    def __init__(self, config: TanhGaussianConfig | None = None) -> None:
        self.config = config or TanhGaussianConfig()

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            log_std,
            min=float(self.config.log_std_min),
            max=float(self.config.log_std_max),
        )

    def mean_to_action(self, mean: torch.Tensor) -> torch.Tensor:
        return torch.tanh(mean)

    def sample(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_std = self.clamp_log_std(log_std)
        dist = Normal(mean, log_std.exp())
        latent = mean if deterministic else dist.rsample()
        action = torch.tanh(latent)
        log_prob = self.log_prob_from_latent(mean, log_std, latent, action=action)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def log_prob(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        actions: torch.Tensor,
        raw_actions: torch.Tensor | None = None,
        raw_action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_std = self.clamp_log_std(log_std)
        action = torch.clamp(
            actions,
            min=-1.0 + float(self.config.eps),
            max=1.0 - float(self.config.eps),
        )
        latent = self._atanh(action)

        if raw_actions is not None:
            raw_latent = torch.clamp(
                raw_actions, -float(self.config.latent_clip), float(self.config.latent_clip)
            )
            raw_action = torch.tanh(raw_latent)
            if raw_action_mask is None:
                raw_mask = torch.ones_like(actions[..., :1], dtype=torch.bool)
            else:
                raw_mask = (raw_action_mask > 0.5).unsqueeze(-1)
            latent = torch.where(raw_mask, raw_latent, latent)
            action = torch.where(raw_mask, raw_action, action)

        log_prob = self.log_prob_from_latent(mean, log_std, latent, action=action)
        entropy = Normal(mean, log_std.exp()).entropy().sum(dim=-1)
        return log_prob, entropy

    def log_prob_from_latent(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        latent: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        log_std = self.clamp_log_std(log_std)
        latent = torch.clamp(
            latent, -float(self.config.latent_clip), float(self.config.latent_clip)
        )
        if action is None:
            action = torch.tanh(latent)
        dist = Normal(mean, log_std.exp())
        log_det = torch.log(1.0 - action * action + float(self.config.eps))
        return (dist.log_prob(latent) - log_det).sum(dim=-1)

    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))
