from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VTraceConfig:
    clip_rho: float = 1.0
    clip_c: float = 1.0


@dataclass(frozen=True)
class VTraceReturns:
    values: torch.Tensor
    policy_advantages: torch.Tensor
    rhos: torch.Tensor


def vtrace_from_logits(
    rewards: torch.Tensor,
    discounts: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    behaviour_log_probs: torch.Tensor,
    target_log_probs: torch.Tensor,
    mask: torch.Tensor,
    config: VTraceConfig,
) -> VTraceReturns:
    """Batched IMPALA V-trace for padded [B, T] rollout tensors."""
    valid = mask > 0.0
    log_rhos = torch.where(
        valid,
        target_log_probs - behaviour_log_probs,
        torch.zeros_like(target_log_probs),
    )
    rhos = torch.where(valid, torch.exp(log_rhos), torch.zeros_like(log_rhos))
    rho_clipped = torch.clamp(rhos, max=float(config.clip_rho))
    c = torch.clamp(rhos, max=float(config.clip_c))

    next_values = torch.cat([values[:, 1:], bootstrap_value.unsqueeze(1)], dim=1)
    deltas = rho_clipped * (rewards + discounts * next_values - values)

    vs = torch.zeros_like(values)
    acc = bootstrap_value
    for t in reversed(range(values.size(1))):
        new_acc = values[:, t] + deltas[:, t] + discounts[:, t] * c[:, t] * (
            acc - next_values[:, t]
        )
        acc = torch.where(valid[:, t], new_acc, acc)
        vs[:, t] = acc

    vs_next = torch.cat([vs[:, 1:], bootstrap_value.unsqueeze(1)], dim=1)
    pg_adv = rho_clipped * (rewards + discounts * vs_next - values)
    return VTraceReturns(values=vs * mask, policy_advantages=pg_adv * mask, rhos=rhos)
