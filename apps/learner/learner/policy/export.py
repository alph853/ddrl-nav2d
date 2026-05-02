from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from learner.config.learner import LearnerConfig

from .actor_critic import ActorCritic


class _PolicyExportWrapper(nn.Module):
    def __init__(self, policy: ActorCritic) -> None:
        super().__init__()
        self.policy = policy

    def forward(  # type: ignore[override]
        self,
        observation: torch.Tensor,
        hidden_in: torch.Tensor | None = None,
        cell_in: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        recurrent_state = None
        if hidden_in is not None and cell_in is not None:
            recurrent_state = (hidden_in, cell_in)
        mean, log_std, _value, recurrent_out = self.policy(observation, recurrent_state)
        if recurrent_out is None:
            return mean, log_std
        hidden_out, cell_out = recurrent_out
        return mean, log_std, hidden_out, cell_out


def export_policy_to_onnx(
    policy: ActorCritic, config: LearnerConfig, opset_version: int = 17
) -> bytes:
    policy = policy.eval()
    device = next(policy.parameters()).device

    batch, time = 1, 1
    observation = torch.zeros(
        (batch, time, config.observation_dim), device=device, dtype=torch.float32
    )
    wrapper = _PolicyExportWrapper(policy)

    recurrent_state = policy.initial_recurrent_state(batch, device)
    if recurrent_state is None:
        with tempfile.NamedTemporaryFile(suffix=".onnx") as handle:
            output_path = Path(handle.name)
            torch.onnx.export(
                wrapper,
                (observation,),
                output_path,
                opset_version=opset_version,
                input_names=["observation"],
                output_names=["mean", "log_std"],
                dynamic_axes={
                    "observation": {0: "batch", 1: "time"},
                    "mean": {0: "batch", 1: "time"},
                    "log_std": {0: "batch", 1: "time"},
                },
                do_constant_folding=True,
                dynamo=False,
            )
            return output_path.read_bytes()

    hidden_in, cell_in = recurrent_state
    with tempfile.NamedTemporaryFile(suffix=".onnx") as handle:
        output_path = Path(handle.name)
        torch.onnx.export(
            wrapper,
            (observation, hidden_in, cell_in),
            output_path,
            opset_version=opset_version,
            input_names=["observation", "hidden_in", "cell_in"],
            output_names=["mean", "log_std", "hidden_out", "cell_out"],
            dynamic_axes={
                "observation": {0: "batch", 1: "time"},
                "mean": {0: "batch", 1: "time"},
                "log_std": {0: "batch", 1: "time"},
                "hidden_in": {1: "batch"},
                "cell_in": {1: "batch"},
                "hidden_out": {1: "batch"},
                "cell_out": {1: "batch"},
            },
            do_constant_folding=True,
            dynamo=False,
        )
        return output_path.read_bytes()
