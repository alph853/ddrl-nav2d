#!/usr/bin/env python3
"""Offline behavior-cloning pretrain from recorded rollout batches."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from dataclasses import fields
from pathlib import Path
import sys
import time
from typing import Any
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[3]
LEARNER_ROOT = ROOT / "apps" / "learner"
if str(LEARNER_ROOT) not in sys.path:
    sys.path.insert(0, str(LEARNER_ROOT))

from proto import rollouts_pb2
from learner.config.learner import LearnerConfig
from learner.config.rollouts import rollouts_from_proto
from learner.policy import PolicyModule
from learner.policy import build_policy


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self, sequences: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> None:
        self._sequences = sequences

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._sequences[index]


def _load_config(raw: Any) -> LearnerConfig:
    if not isinstance(raw, dict):
        return LearnerConfig()
    allowed = {field.name for field in fields(LearnerConfig)}
    filtered = {key: value for key, value in raw.items() if key in allowed}
    return LearnerConfig(**filtered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain the compact policy with offline BC from recorded expert rollouts."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("learner_checkpoints/bc_pretrain"),
        help="Directory for per-epoch checkpoints and model_best.pt.",
    )
    parser.add_argument(
        "--policy-arch-config",
        type=Path,
        default=Path("config/policy_arch.yaml"),
        help="Shared policy architecture configuration.",
    )
    parser.add_argument(
        "--rl-config",
        type=Path,
        default=None,
        help="Deprecated alias for --policy-arch-config.",
    )
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss", type=str, choices=("nll", "mse"), default="mse")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=64,
        help="Fixed LSTM training window length in steps.",
    )
    parser.add_argument(
        "--sequence-stride",
        type=int,
        default=16,
        help="Sliding stride between successive LSTM windows.",
    )
    parser.add_argument("--weight-corrective-steps", action="store_true")
    parser.add_argument("--reverse-step-weight", type=float, default=2.5)
    parser.add_argument("--high-steer-weight", type=float, default=1.5)
    parser.add_argument("--high-heading-weight", type=float, default=1.5)
    parser.add_argument("--high-steer-threshold", type=float, default=0.6)
    parser.add_argument("--high-heading-threshold-rad", type=float, default=0.75)
    return parser.parse_args()


def format_observation(step, observation_dim: int) -> np.ndarray:
    arr = np.zeros(observation_dim, dtype=np.float32)
    if step.observation:
        data = np.asarray(step.observation, dtype=np.float32)
        arr[: min(observation_dim, data.shape[0])] = data[:observation_dim]
    return arr


def iter_rollout_paths(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("rollout_batch_*.pb"))


def _sequence_windows(
    seq_obs: list[np.ndarray],
    seq_actions: list[np.ndarray],
    sequence_length: int,
    sequence_stride: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    total = len(seq_obs)
    if total == 0:
        return []

    if total <= sequence_length:
        return [
            (
                torch.from_numpy(np.stack(seq_obs, axis=0)),
                torch.from_numpy(np.stack(seq_actions, axis=0)),
            )
        ]

    windows: list[tuple[torch.Tensor, torch.Tensor]] = []
    last_start = total - sequence_length
    starts = list(range(0, last_start + 1, sequence_stride))
    if starts[-1] != last_start:
        starts.append(last_start)

    for start in starts:
        end = start + sequence_length
        windows.append(
            (
                torch.from_numpy(np.stack(seq_obs[start:end], axis=0)),
                torch.from_numpy(np.stack(seq_actions[start:end], axis=0)),
            )
        )
    return windows


def load_dataset(
    input_dir: Path, config: LearnerConfig, sequence_length: int, sequence_stride: int
) -> tuple[Dataset[tuple[torch.Tensor, ...]], int]:
    rollout_paths = iter_rollout_paths(input_dir)
    if not rollout_paths:
        raise FileNotFoundError(f"No rollout_batch_*.pb files found under {input_dir}")

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    sequences: list[tuple[torch.Tensor, torch.Tensor]] = []
    expert_steps = 0

    for path in rollout_paths:
        batch = rollouts_pb2.RolloutBatch()
        if not batch.ParseFromString(path.read_bytes()):
            raise ValueError(f"Failed to parse rollout batch protobuf: {path}")
        for rollout in rollouts_from_proto(batch):
            rollout_is_expert = float(rollout.metrics.get("is_expert", 0.0)) >= 0.5
            seq_obs: list[np.ndarray] = []
            seq_actions: list[np.ndarray] = []
            for step in rollout.steps:
                target_action = target_action_from_step(
                    step, rollout_is_expert, config.action_dim
                )
                if target_action is None:
                    continue
                obs_np = format_observation(step, config.observation_dim)
                action_np = target_action
                observations.append(obs_np)
                actions.append(action_np)
                seq_obs.append(obs_np)
                seq_actions.append(action_np)
                expert_steps += 1
            if seq_obs:
                sequences.extend(
                    _sequence_windows(
                        seq_obs,
                        seq_actions,
                        sequence_length=sequence_length,
                        sequence_stride=sequence_stride,
                    )
                )

    if expert_steps == 0:
        raise ValueError(f"No expert-tagged steps found in {input_dir}")
    if str(config.policy_model).lower() == "lstm":
        return SequenceDataset(sequences), expert_steps
    return (
        TensorDataset(
            torch.from_numpy(np.stack(observations, axis=0)),
            torch.from_numpy(np.stack(actions, axis=0)),
        ),
        expert_steps,
    )


def bc_step_weights(
    obs: torch.Tensor,
    action: torch.Tensor,
    config: LearnerConfig,
    args: argparse.Namespace,
) -> torch.Tensor:
    weights = torch.ones(obs.shape[:-1], device=obs.device, dtype=obs.dtype)
    if not args.weight_corrective_steps:
        return weights

    state_offset = config.observation_dim - int(config.state_dim)
    heading_feature_idx = state_offset + 4

    reverse_mask = action[..., 0] < 0.0
    steer_frac = action[..., 1].abs()
    high_steer_mask = steer_frac >= float(args.high_steer_threshold)
    heading_error = obs[..., heading_feature_idx].abs() * float(np.pi)
    high_heading_mask = heading_error >= float(args.high_heading_threshold_rad)

    weights = torch.where(
        reverse_mask, weights * float(args.reverse_step_weight), weights
    )
    weights = torch.where(
        high_steer_mask, weights * float(args.high_steer_weight), weights
    )
    weights = torch.where(
        high_heading_mask, weights * float(args.high_heading_weight), weights
    )
    return weights


def bc_loss_from_outputs(
    model: PolicyModule,
    mean: torch.Tensor,
    log_std: torch.Tensor,
    action: torch.Tensor,
    config: LearnerConfig,
    args: argparse.Namespace,
    obs: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_action = model.mean_to_action(mean)
    if args.loss == "mse":
        per_step = (pred_action - action).pow(2).mean(dim=-1)
    else:
        log_prob, _entropy = model.action_log_prob(mean, log_std, action)
        per_step = -log_prob
    weights = (
        torch.ones_like(per_step, dtype=action.dtype, device=action.device)
        if obs is None
        else bc_step_weights(obs, action, config, args)
    )
    if mask is not None:
        weights = weights * mask
    loss = (per_step * weights).sum() / weights.sum().clamp(min=1.0)
    return loss, pred_action


def clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def clone_optimizer_state(state: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(state)


def target_action_from_step(
    step: Any, rollout_is_expert: bool, action_dim: int
) -> np.ndarray | None:
    if not rollout_is_expert:
        return None
    action = np.asarray(step.action, dtype=np.float32)
    if action.size < action_dim:
        return None
    return action[:action_dim].astype(np.float32, copy=False)


def split_dataset(
    dataset: Dataset[tuple[torch.Tensor, ...]],
    val_ratio: float,
    split_seed: int,
) -> tuple[Dataset[tuple[torch.Tensor, ...]], Dataset[tuple[torch.Tensor, ...]] | None]:
    ratio = float(val_ratio)
    if ratio <= 0.0 or len(dataset) < 2:
        return dataset, None
    ratio = min(ratio, 0.49)
    num_val = int(round(len(dataset) * ratio))
    num_val = max(1, min(num_val, len(dataset) - 1))
    generator = torch.Generator().manual_seed(int(split_seed))
    permutation = torch.randperm(len(dataset), generator=generator).tolist()
    return Subset(dataset, permutation[num_val:]), Subset(
        dataset, permutation[:num_val]
    )


def make_loader(
    dataset: Dataset[tuple[torch.Tensor, ...]], batch_size: int, shuffle: bool
) -> DataLoader[tuple[torch.Tensor, ...]]:
    is_sequence_dataset = isinstance(dataset, SequenceDataset) or (
        isinstance(dataset, Subset) and isinstance(dataset.dataset, SequenceDataset)
    )
    if not is_sequence_dataset:
        return DataLoader(
            dataset, batch_size=max(1, batch_size), shuffle=shuffle, drop_last=False
        )

    def collate_fn(
        batch: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([item[0].shape[0] for item in batch], dtype=torch.long)
        max_len = int(lengths.max().item())
        obs = torch.zeros(
            (len(batch), max_len, batch[0][0].shape[-1]), dtype=torch.float32
        )
        action = torch.zeros(
            (len(batch), max_len, batch[0][1].shape[-1]), dtype=torch.float32
        )
        for i, (obs_i, act_i) in enumerate(batch):
            seq_len = obs_i.shape[0]
            obs[i, :seq_len] = obs_i
            action[i, :seq_len] = act_i
        return obs, action, lengths

    return DataLoader(
        dataset,
        batch_size=max(1, batch_size),
        shuffle=shuffle,
        drop_last=False,
        collate_fn=collate_fn,
    )


def _unpack_batch(
    batch: tuple[torch.Tensor, ...], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(batch) == 3:
        obs, action, lengths = batch
        obs = obs.to(device)
        action = action.to(device)
        lengths = lengths.to(device)
        max_len = int(lengths.max().item())
        mask = (torch.arange(max_len, device=device)[None, :] < lengths[:, None]).to(
            torch.float32
        )
        return obs, action, mask

    obs, action = batch
    obs = obs.to(device).unsqueeze(1)
    action = action.to(device).unsqueeze(1)
    mask = torch.ones((obs.shape[0], 1), device=device, dtype=torch.float32)
    return obs, action, mask


@torch.no_grad()
def evaluate_loader(
    model: PolicyModule,
    loader: Iterable[tuple[torch.Tensor, ...]],
    device: torch.device,
    config: LearnerConfig,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    total_speed_abs_err = 0.0
    total_steer_abs_err = 0.0
    total_pred_abs_speed = 0.0
    total_pred_abs_steer = 0.0
    total_target_abs_speed = 0.0
    total_target_abs_steer = 0.0
    total_pred_speed_sat = 0.0
    total_pred_steer_sat = 0.0
    total_target_speed_sat = 0.0
    total_target_steer_sat = 0.0
    action_sat_thresh = 0.95

    for batch in loader:
        obs, action, mask = _unpack_batch(batch, device)
        recurrent_state = model.initial_recurrent_state(obs.shape[0], device)
        mean, log_std, _value, _state_out = model(obs, recurrent_state)
        loss, pred_action = bc_loss_from_outputs(
            model, mean, log_std, action, config, args, obs, mask
        )

        step_count = int(mask.sum().item())
        total_loss += float(loss.item()) * step_count
        total_steps += step_count
        total_speed_abs_err += float(
            (((pred_action[..., 0] - action[..., 0]).abs()) * mask).sum().item()
        )
        total_steer_abs_err += float(
            (((pred_action[..., 1] - action[..., 1]).abs()) * mask).sum().item()
        )
        total_pred_abs_speed += float((pred_action[..., 0].abs() * mask).sum().item())
        total_pred_abs_steer += float((pred_action[..., 1].abs() * mask).sum().item())
        total_target_abs_speed += float((action[..., 0].abs() * mask).sum().item())
        total_target_abs_steer += float((action[..., 1].abs() * mask).sum().item())
        total_pred_speed_sat += float(
            ((pred_action[..., 0].abs() >= action_sat_thresh).to(mask.dtype) * mask)
            .sum()
            .item()
        )
        total_pred_steer_sat += float(
            ((pred_action[..., 1].abs() >= action_sat_thresh).to(mask.dtype) * mask)
            .sum()
            .item()
        )
        total_target_speed_sat += float(
            ((action[..., 0].abs() >= action_sat_thresh).to(mask.dtype) * mask)
            .sum()
            .item()
        )
        total_target_steer_sat += float(
            ((action[..., 1].abs() >= action_sat_thresh).to(mask.dtype) * mask)
            .sum()
            .item()
        )

    total_steps = max(total_steps, 1)
    return {
        "loss": total_loss / total_steps,
        "speed_mae": total_speed_abs_err / total_steps,
        "steer_mae": total_steer_abs_err / total_steps,
        "pred_abs_speed": total_pred_abs_speed / total_steps,
        "pred_abs_steer": total_pred_abs_steer / total_steps,
        "target_abs_speed": total_target_abs_speed / total_steps,
        "target_abs_steer": total_target_abs_steer / total_steps,
        "pred_speed_sat_frac": total_pred_speed_sat / total_steps,
        "pred_steer_sat_frac": total_pred_steer_sat / total_steps,
        "target_speed_sat_frac": total_target_speed_sat / total_steps,
        "target_steer_sat_frac": total_target_steer_sat / total_steps,
    }


def bc_trainable_parameters(model: PolicyModule) -> list[torch.nn.Parameter]:
    if getattr(model, "recurrent", None) is not None:
        return (
            list(model.recurrent.parameters())
            + list(model.actor.parameters())
            + list(model.actor_mean.parameters())
        )
    return list(model.actor.parameters()) + list(model.actor_mean.parameters())


def write_checkpoint(
    output_path: Path,
    config: LearnerConfig,
    model: PolicyModule,
    best_model_state: dict[str, torch.Tensor],
    best_optimizer_state: dict[str, Any],
    num_steps: int,
    start_version: int,
    global_step: int,
    best_epoch: int,
    best_val_loss: float,
    elapsed: float,
    train_dataset_len: int,
    val_dataset_len: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_model_state, strict=True)
    payload = {
        "checkpoint_stage": "bc_pretrain",
        "restore_optimizer_on_resume": False,
        "version": max(1, start_version),
        "timestamp": time.time(),
        "model_state": best_model_state,
        "optimizer_state": best_optimizer_state,
        "config": asdict(config),
        "stats": {
            "stage": "bc_pretrain",
            "samples": int(num_steps),
            "elapsed_sec": float(elapsed),
            "train_samples": int(train_dataset_len),
            "val_samples": int(val_dataset_len),
            "best_epoch": int(best_epoch),
            "best_loss": float(best_val_loss),
        },
        "action_dim": config.action_dim,
        "num_updates": int(global_step),
    }
    torch.save(payload, output_path)


def checkpoint_path(output_dir: Path, stem: str) -> Path:
    return output_dir / f"{stem}.pt"


def main() -> int:
    args = parse_args()
    if not 0.0 <= float(args.val_ratio) < 0.5:
        raise ValueError("--val-ratio must be in [0, 0.5).")
    if int(args.sequence_length) <= 0:
        raise ValueError("--sequence-length must be > 0.")
    if int(args.sequence_stride) <= 0:
        raise ValueError("--sequence-stride must be > 0.")

    policy_arch_config = args.rl_config or args.policy_arch_config
    config = LearnerConfig.from_policy_arch_yaml(policy_arch_config)
    config.lr = args.lr

    device = (args.device or "auto").lower()
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, num_steps = load_dataset(
        args.input_dir,
        config,
        sequence_length=int(args.sequence_length),
        sequence_stride=int(args.sequence_stride),
    )
    train_dataset, val_dataset = split_dataset(dataset, args.val_ratio, args.split_seed)
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True)
    train_eval_loader = make_loader(train_dataset, args.batch_size, shuffle=False)
    val_loader = (
        make_loader(val_dataset, args.batch_size, shuffle=False)
        if val_dataset is not None
        else None
    )

    model = build_policy(config, config.action_dim).to(device)
    optimizer = torch.optim.Adam(bc_trainable_parameters(model), lr=args.lr, eps=1e-5)
    start_version = 1

    if args.resume_checkpoint is not None:
        payload = torch.load(
            args.resume_checkpoint, map_location=device, weights_only=False
        )
        if not isinstance(payload, dict):
            raise ValueError(
                f"Checkpoint {args.resume_checkpoint} did not contain a mapping payload."
            )
        model_state = payload.get("model_state")
        if not isinstance(model_state, dict):
            raise ValueError(
                f"Checkpoint {args.resume_checkpoint} is missing model_state."
            )
        model.load_state_dict(model_state, strict=True)
        start_version = max(1, int(payload.get("version", 1)))

    global_step = 0
    t0 = time.perf_counter()
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = clone_state_dict(model.state_dict())
    best_optimizer_state = clone_optimizer_state(optimizer.state_dict())
    interrupted = False

    print(
        "dataset:"
        f" total={len(dataset)} train={len(train_dataset)}"
        f" val={len(val_dataset) if val_dataset is not None else 0}"
        f" expert_steps={num_steps}"
        f" policy_model={config.policy_model}"
    )

    try:
        for epoch in range(max(1, args.epochs)):
            model.train()
            for batch in train_loader:
                obs, action, mask = _unpack_batch(batch, device)

                optimizer.zero_grad(set_to_none=True)
                recurrent_state = model.initial_recurrent_state(obs.shape[0], device)
                mean, log_std, _value, _state_out = model(obs, recurrent_state)
                loss, _pred_action = bc_loss_from_outputs(
                    model, mean, log_std, action, config, args, obs, mask
                )
                loss.backward()
                if config.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

                global_step += 1
                if args.log_interval > 0 and global_step % args.log_interval == 0:
                    print(
                        f"step={global_step} epoch={epoch + 1}/{args.epochs} bc_loss={loss.item():.6f}"
                    )

            train_metrics = evaluate_loader(
                model, train_eval_loader, device, config, args
            )
            val_metrics = (
                evaluate_loader(model, val_loader, device, config, args)
                if val_loader
                else None
            )
            score_loss = (
                val_metrics["loss"]
                if val_metrics is not None
                else train_metrics["loss"]
            )
            if score_loss < best_val_loss:
                best_val_loss = score_loss
                best_epoch = epoch + 1
                best_model_state = clone_state_dict(model.state_dict())
                best_optimizer_state = clone_optimizer_state(optimizer.state_dict())

            epoch_line = (
                f"epoch={epoch + 1}/{args.epochs} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"train_speed_mae={train_metrics['speed_mae']:.4f} "
                f"train_steer_mae={train_metrics['steer_mae']:.4f} "
                f"train_pred_abs_steer={train_metrics['pred_abs_steer']:.4f} "
                f"train_target_abs_steer={train_metrics['target_abs_steer']:.4f} "
                f"train_pred_steer_sat={train_metrics['pred_steer_sat_frac']:.3f}"
            )
            if val_metrics is not None:
                epoch_line += (
                    f" val_loss={val_metrics['loss']:.6f} "
                    f"val_speed_mae={val_metrics['speed_mae']:.4f} "
                    f"val_steer_mae={val_metrics['steer_mae']:.4f} "
                    f"val_pred_abs_steer={val_metrics['pred_abs_steer']:.4f} "
                    f"val_target_abs_steer={val_metrics['target_abs_steer']:.4f} "
                    f"val_pred_steer_sat={val_metrics['pred_steer_sat_frac']:.3f}"
                )
            print(epoch_line)

            write_checkpoint(
                output_path=checkpoint_path(args.output_dir, f"model_{epoch + 1}"),
                config=config,
                model=model,
                best_model_state=clone_state_dict(model.state_dict()),
                best_optimizer_state=clone_optimizer_state(optimizer.state_dict()),
                num_steps=num_steps,
                start_version=start_version,
                global_step=global_step,
                best_epoch=epoch + 1,
                best_val_loss=score_loss,
                elapsed=time.perf_counter() - t0,
                train_dataset_len=len(train_dataset),
                val_dataset_len=len(val_dataset) if val_dataset is not None else 0,
            )
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Saving best checkpoint so far...")
    finally:
        elapsed = time.perf_counter() - t0
        write_checkpoint(
            output_path=checkpoint_path(args.output_dir, "model_best"),
            config=config,
            model=model,
            best_model_state=best_model_state,
            best_optimizer_state=best_optimizer_state,
            num_steps=num_steps,
            start_version=start_version,
            global_step=global_step,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            elapsed=elapsed,
            train_dataset_len=len(train_dataset),
            val_dataset_len=len(val_dataset) if val_dataset is not None else 0,
        )

    print(
        f"wrote {checkpoint_path(args.output_dir, 'model_best')} with {num_steps} expert steps "
        f"(best_epoch={best_epoch} best_loss={best_val_loss:.6f})"
    )
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
