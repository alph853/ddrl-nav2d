#!/usr/bin/env python3
"""Export a saved learner checkpoint to the ONNX format consumed by C++ workers."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
LEARNER_ROOT = ROOT / "apps" / "learner"
if str(LEARNER_ROOT) not in sys.path:
    sys.path.insert(0, str(LEARNER_ROOT))

from learner.config.learner import LearnerConfig
from learner.policy import build_policy
from learner.policy.export import export_policy_to_onnx


def _load_config(raw: Any) -> LearnerConfig:
    if not isinstance(raw, dict):
        return LearnerConfig()
    allowed = {field.name for field in fields(LearnerConfig)}
    filtered = {key: value for key, value in raw.items() if key in allowed}
    return LearnerConfig(**filtered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert learner_checkpoints/policy_vXXXXX.pt to worker-ready ONNX."
    )
    parser.add_argument(
        "checkpoint", type=Path, help="Path to a learner .pt checkpoint."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output .onnx path. Defaults to the checkpoint path with .onnx suffix.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used while loading/exporting the model. Default: cpu.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = args.checkpoint
    output_path = args.output or checkpoint_path.with_suffix(".onnx")

    payload = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Checkpoint {checkpoint_path} did not contain a mapping payload."
        )

    config = _load_config(payload.get("config"))
    action_dim = int(payload.get("action_dim") or config.action_dim)
    model_state = payload.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} is missing model_state.")

    model = build_policy(config, action_dim).to(args.device)
    model.load_state_dict(model_state)
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(export_policy_to_onnx(model, config))
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
