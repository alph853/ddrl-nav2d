from __future__ import annotations

from dataclasses import asdict
from dataclasses import fields
from pathlib import Path
from typing import Dict

import torch

from learner.config.learner import LearnerConfig


RUNTIME_OWNED_CONFIG_KEYS = {
    "trainer_mode",
    "rollouts_per_update",
    "num_epochs_per_update",
    "checkpoint_interval",
    "policy_push_warmup_updates",
    "device",
    "entropy_coef",
    "bc_coef",
    "bc_decay_updates",
    "bc_floor_coef",
    "action_init_std",
    "reset_action_std_on_resume",
}


def load_checkpoint_payload(checkpoint_path: Path, device: torch.device) -> dict:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Checkpoint {checkpoint_path} did not contain a mapping payload."
        )
    return payload


def apply_checkpoint_config(config: LearnerConfig, payload_config: object) -> None:
    if not isinstance(payload_config, dict):
        return
    allowed = {field.name for field in fields(LearnerConfig)}
    for key, value in payload_config.items():
        if key in allowed and key not in RUNTIME_OWNED_CONFIG_KEYS:
            setattr(config, key, value)


def checkpoint_restore_optimizer(payload: dict) -> bool:
    checkpoint_stage = str(
        payload.get("checkpoint_stage") or payload.get("stage") or ""
    )
    stats = payload.get("stats")
    if not checkpoint_stage and isinstance(stats, dict):
        checkpoint_stage = str(stats.get("stage") or "")
    restore_optimizer = payload.get("restore_optimizer_on_resume", True)
    if not isinstance(restore_optimizer, bool):
        restore_optimizer = True
    if checkpoint_stage == "bc_pretrain":
        restore_optimizer = False
    return restore_optimizer


def save_checkpoint(
    checkpoint_dir: Path,
    version: int,
    config: LearnerConfig,
    model_state: Dict[str, torch.Tensor],
    optimizer_state: Dict[str, torch.Tensor],
    action_dim: int,
    stats: Dict[str, float],
    num_updates: int,
    bc_decay_offset_steps: int = 0,
) -> Path:
    import json
    import time

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"policy_v{version:05d}.pt"

    payload = {
        "version": version,
        "timestamp": time.time(),
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "config": asdict(config),
        "stats": stats,
        "action_dim": action_dim,
        "num_updates": num_updates,
        "bc_decay_offset_steps": max(0, int(bc_decay_offset_steps)),
    }
    torch.save(payload, ckpt_path)

    meta = {
        "version": version,
        "timestamp": time.time(),
        "stats": stats,
        "num_updates": num_updates,
    }
    ckpt_path.with_suffix(".json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return ckpt_path
