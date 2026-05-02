from __future__ import annotations

import math

import torch
import torch.nn as nn

from learner.config.learner import LearnerConfig

from .distribution import TanhGaussianAction
from .distribution import TanhGaussianConfig
from .normalization import ObservationNormalizer


class ActorCritic(nn.Module):
    """Shared actor-critic policy with optional LSTM recurrence."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        actor_hidden_dim: int,
        critic_hidden_dim: int | None = None,
        recurrent_hidden_dim: int | None = None,
        recurrent_model: str | None = None,
        init_std: float = 0.03,
        normalize_observation: bool = True,
        observation_clip: float = 10.0,
        action_distribution: TanhGaussianConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.actor_hidden_dim = max(32, int(actor_hidden_dim))
        self.critic_hidden_dim = max(32, int(critic_hidden_dim or actor_hidden_dim))
        self.recurrent_model = (recurrent_model or "").lower() or None
        self.recurrent_hidden_dim = max(
            32, int(recurrent_hidden_dim or actor_hidden_dim)
        )
        self.observation_normalizer = ObservationNormalizer(
            self.observation_dim,
            device=device,
            clip=float(observation_clip),
            enabled=bool(normalize_observation),
        )
        self.action_distribution = TanhGaussianAction(action_distribution)

        trunk_dim = self.observation_dim
        if self.recurrent_model is not None:
            if self.recurrent_model != "lstm":
                raise ValueError(f"Unsupported recurrent_model: {recurrent_model}")
            self.recurrent = nn.LSTM(
                input_size=self.observation_dim,
                hidden_size=self.recurrent_hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            trunk_dim = self.recurrent_hidden_dim
        else:
            self.recurrent = None

        self.actor = nn.Sequential(
            nn.Linear(trunk_dim, self.actor_hidden_dim),
            nn.ELU(),
        )
        self.critic = nn.Sequential(
            nn.Linear(trunk_dim, self.critic_hidden_dim),
            nn.ELU(),
        )
        self.actor_mean = nn.Linear(self.actor_hidden_dim, self.action_dim)
        self.critic_value = nn.Linear(self.critic_hidden_dim, 1)
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), math.log(max(float(init_std), 1e-6)))
        )

        self._init_weights()

    @classmethod
    def from_config(
        cls, config: LearnerConfig, action_dim: int, device: torch.device | None = None
    ) -> "ActorCritic":
        model_name = str(config.policy_model).lower()
        recurrent_model = "lstm" if model_name == "lstm" else None
        hidden_dim = (
            int(config.lstm_hidden_dim)
            if recurrent_model == "lstm"
            else int(config.mlp_hidden_dim)
        )
        return cls(
            observation_dim=config.observation_dim,
            action_dim=action_dim,
            actor_hidden_dim=hidden_dim,
            critic_hidden_dim=hidden_dim,
            recurrent_hidden_dim=int(config.lstm_hidden_dim),
            recurrent_model=recurrent_model,
            init_std=float(config.action_init_std),
            normalize_observation=bool(config.normalize_observation),
            observation_clip=float(config.observation_clip),
            device=device,
        )

    @torch.no_grad()
    def update_observation_stats(
        self, observation: torch.Tensor, mask: torch.Tensor | None = None
    ) -> None:
        self.observation_normalizer.update(observation, mask)

    def initial_recurrent_state(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.recurrent is None:
            return None
        shape = (1, int(batch_size), self.recurrent_hidden_dim)
        return (
            torch.zeros(shape, device=device, dtype=torch.float32),
            torch.zeros(shape, device=device, dtype=torch.float32),
        )

    def forward(
        self,
        observation: torch.Tensor,
        recurrent_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        batch, _time, _dim = observation.shape
        normalized_observation = self.observation_normalizer(observation)
        trunk = normalized_observation
        next_state = None
        if self.recurrent is not None:
            if recurrent_state is None:
                recurrent_state = self.initial_recurrent_state(
                    batch, observation.device
                )
            trunk, next_state = self.recurrent(normalized_observation, recurrent_state)

        actor_features = self.actor(trunk)
        critic_features = self.critic(trunk)
        mean = self.actor_mean(actor_features)
        value = self.critic_value(critic_features).squeeze(-1)
        log_std = self.log_std.view(1, 1, -1).expand_as(mean)
        return mean, log_std, value, next_state

    def act(
        self,
        observation: torch.Tensor,
        recurrent_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        deterministic: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        mean, log_std, value, next_state = self.forward(observation, recurrent_state)
        action, log_prob, _entropy = self.action_distribution.sample(
            mean, log_std, deterministic=deterministic
        )
        return action, log_prob, value, next_state

    def action_log_prob(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        actions: torch.Tensor,
        raw_actions: torch.Tensor | None = None,
        raw_action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.action_distribution.log_prob(
            mean,
            log_std,
            actions,
            raw_actions=raw_actions,
            raw_action_mask=raw_action_mask,
        )

    def mean_to_action(self, mean: torch.Tensor) -> torch.Tensor:
        return self.action_distribution.mean_to_action(mean)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, a=0.0, mode="fan_in", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.recurrent is None:
            return
        for name, param in self.recurrent.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)
