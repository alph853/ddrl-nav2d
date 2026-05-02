from __future__ import annotations

import torch
import torch.nn as nn


class ObservationNormalizer(nn.Module):
    """Running mean/variance normalizer whose statistics are part of the policy state."""

    def __init__(
        self,
        observation_dim: int,
        device: torch.device | None = None,
        eps: float = 1e-5,
        clip: float = 10.0,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.eps = float(eps)
        self.clip = float(clip)
        self.register_buffer(
            "mean", torch.zeros(int(observation_dim), dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "var", torch.ones(int(observation_dim), dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "count", torch.tensor(float(eps), dtype=torch.float32, device=device)
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return observation
        mean_buffer = self.mean_buffer
        var_buffer = self.var_buffer
        mean = mean_buffer.view(*([1] * (observation.ndim - 1)), -1)
        var = var_buffer.view(*([1] * (observation.ndim - 1)), -1)
        normalized = (observation - mean) / torch.sqrt(var + self.eps)
        return torch.clamp(normalized, -self.clip, self.clip)

    @torch.no_grad()
    def update(
        self, observation: torch.Tensor, mask: torch.Tensor | None = None
    ) -> None:
        if not self.enabled or observation.numel() == 0:
            return
        flat = observation.reshape(-1, observation.shape[-1])
        if mask is not None:
            valid = mask.reshape(-1) > 0.0
            flat = flat[valid]
        if flat.numel() == 0:
            return
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False)
        batch_count = torch.tensor(
            float(flat.shape[0]), device=flat.device, dtype=flat.dtype
        )
        self._update_from_moments(batch_mean, batch_var, batch_count)

    @torch.no_grad()
    def _update_from_moments(
        self,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: torch.Tensor,
    ) -> None:
        mean_buffer = self.mean_buffer
        var_buffer = self.var_buffer
        count_buffer = self.count_buffer

        batch_mean = batch_mean.to(device=mean_buffer.device, dtype=mean_buffer.dtype)
        batch_var = batch_var.to(device=var_buffer.device, dtype=var_buffer.dtype)
        batch_count = batch_count.to(
            device=count_buffer.device, dtype=count_buffer.dtype
        )

        delta = batch_mean - mean_buffer
        total_count = count_buffer + batch_count
        new_mean = mean_buffer + delta * batch_count / total_count

        current_m2 = var_buffer * count_buffer
        batch_m2 = batch_var * batch_count
        delta_m2 = delta.pow(2) * count_buffer * batch_count / total_count
        new_var = (current_m2 + batch_m2 + delta_m2) / total_count

        mean_buffer.copy_(new_mean)
        var_buffer.copy_(torch.clamp(new_var, min=self.eps))
        count_buffer.copy_(total_count)

    @property
    def mean_buffer(self) -> torch.Tensor:
        return self._buffer("mean")

    @property
    def var_buffer(self) -> torch.Tensor:
        return self._buffer("var")

    @property
    def count_buffer(self) -> torch.Tensor:
        return self._buffer("count")

    def _buffer(self, name: str) -> torch.Tensor:
        value = self.get_buffer(name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"normalizer buffer {name!r} is not a tensor")
        return value
