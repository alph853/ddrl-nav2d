from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Sequence

from learner.config.learner import LearnerConfig

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - depends on optional tensorboard package.
    SummaryWriter = None  # type: ignore[assignment]


class LearnerMetrics:
    UPDATE_LOSS_TAGS = {
        "loss_total": "loss/total",
        "loss_actor": "loss/actor",
        "loss_value": "loss/value",
        "loss_entropy": "loss/entropy",
        "loss_bc": "loss/bc_anchor",
    }
    UPDATE_POLICY_TAGS = {
        "bc_coef": "policy/bc_coef",
        "bc_decay_steps": "policy/bc_decay_steps",
        "bc_speed_mae": "policy/bc_speed_mae",
        "bc_steer_mae": "policy/bc_steer_mae",
        "mean_rho": "policy/mean_rho",
        "max_rho": "policy/max_rho",
        "grad_norm": "policy/grad_norm",
        "behaviour_log_prob_mean": "policy/behaviour_log_prob_mean",
        "policy_mean_speed": "policy/mean_speed",
        "policy_mean_steer": "policy/mean_steer",
        "policy_sigma_speed": "policy/sigma_speed",
        "policy_sigma_steer": "policy/sigma_steer",
    }
    ROLLOUT_TAGS = {
        "episode_return": "rollout/episode_return",
        "steps": "rollout/steps",
        "episode_terminal": "rollout/episode_terminal",
        "mean_reward": "rollout/mean_reward",
        "mean_speed": "rollout/mean_speed",
        "mean_abs_steer": "rollout/mean_abs_steer",
    }
    CHECKPOINT_TAGS = {
        "updates_since_checkpoint": "checkpoint/optimizer_steps",
        "rollouts_processed": "checkpoint/rollouts_processed",
        "rollout_steps_processed": "checkpoint/rollout_steps_processed",
        "checkpoint_elapsed_sec": "checkpoint/elapsed_sec",
        "rollouts_per_sec": "checkpoint/rollouts_per_sec",
        "expert_rollouts": "checkpoint/expert_rollouts",
        "neural_rollouts": "checkpoint/neural_rollouts",
        "policy_version": "checkpoint/policy_version",
        "batch_mean_return": "checkpoint/batch_mean_return",
    }
    MEAN_FIELDS = {
        "loss",
        "actor_loss",
        "value_loss",
        "entropy",
        "bc_loss",
        "steps",
        "mean_rho",
        "max_rho",
        "bc_coef",
        "behaviour_log_prob_mean",
        "behaviour_log_prob_min",
        "behaviour_log_prob_max",
        "action_speed_mean",
        "action_steer_mean",
        "raw_speed_mean",
        "raw_steer_mean",
        "reward_mean",
        "reward_min",
        "reward_max",
        "grad_norm",
        "max_abs_grad",
        "grad_params",
        "bc_speed_mae",
        "bc_steer_mae",
        "policy_mean_speed",
        "policy_mean_steer",
        "policy_sigma_speed",
        "policy_sigma_steer",
        "rollout_epoch",
        "rollout_epochs_per_update",
    }
    SUM_FIELDS = {"expert_rollouts", "neural_rollouts", "rl_steps", "bc_steps"}

    def __init__(
        self,
        config: LearnerConfig,
        logger,
        *,
        tensorboard_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._logger = logger
        self._rollouts_ingested = 0
        self._batches_ingested = 0
        self._rollouts_recorded = 0
        self._last_rate_time = time.monotonic()
        self._pending_update_count = 0
        self._pending_return_sum = 0.0
        self._pending_return_count = 0
        self._pending_stats_sum: Dict[str, float] = {}
        self._pending_rollouts_processed = 0
        self._pending_rollout_steps_processed = 0
        self._last_checkpoint_time = self._last_rate_time
        self._tb_writer = self._open_tensorboard_writer(tensorboard_dir)

    def note_batch(self, rollouts: int) -> None:
        self._batches_ingested += 1
        self._rollouts_ingested += int(rollouts)

    def record_rollout(
        self,
        rollout,
        num_updates: int,
        buffer_size: int,
        rollouts_dropped: int,
    ) -> None:
        steps = rollout.steps
        rewards = [float(step.reward) for step in steps]
        speeds = [float(step.action[0]) if step.action else 0.0 for step in steps]
        steers = [
            float(step.action[1]) if len(step.action) > 1 else 0.0 for step in steps
        ]
        self._rollouts_recorded += 1
        row = {
            "timestamp": time.time(),
            "event": "rollout",
            "num_updates": num_updates,
            "sim_id": rollout.sim_id,
            "rollout_id": rollout.rollout_id,
            "steps": len(steps),
            "episode_return": float(rollout.episode_return),
            "episode_terminal": int(bool(rollout.episode_terminal)),
            "last_step_terminal": int(bool(steps and steps[-1].terminal)),
            "sum_reward": float(sum(rewards)) if rewards else 0.0,
            "mean_reward": self._safe_mean(rewards),
            "min_reward": float(min(rewards)) if rewards else 0.0,
            "max_reward": float(max(rewards)) if rewards else 0.0,
            "mean_speed": self._safe_mean(speeds),
            "mean_abs_steer": self._safe_mean(abs(s) for s in steers),
            "buffer_size": buffer_size,
            "rollouts_dropped": rollouts_dropped,
        }
        self._write_tensorboard(row, step=self._rollouts_recorded, prefix="rollout")
        self._pending_rollouts_processed += 1
        self._pending_rollout_steps_processed += len(steps)

    def record_update(
        self,
        rollouts: Sequence,
        stats: Dict[str, float],
        num_updates: int,
        buffer_size: int,
        rollouts_dropped: int,
    ) -> None:
        returns = [float(r.episode_return) for r in rollouts]
        rollout_steps = sum(len(getattr(r, "steps", [])) for r in rollouts)
        self._pending_update_count += 1
        self._pending_return_sum += float(sum(returns))
        self._pending_return_count += len(returns)
        for field in self.MEAN_FIELDS | self.SUM_FIELDS:
            self._pending_stats_sum[field] = self._pending_stats_sum.get(
                field, 0.0
            ) + float(stats.get(field, 0.0))

        row = self._stats_row(
            event="update",
            stats=stats,
            num_updates=num_updates,
            batch_mean_return=self._safe_mean(returns),
            buffer_size=buffer_size,
            rollouts_dropped=rollouts_dropped,
        )
        row["rollouts_processed"] = len(rollouts)
        row["rollout_steps_processed"] = rollout_steps
        row["updates_since_checkpoint"] = 1
        self._write_tensorboard(row, step=num_updates, prefix="update")

    def record_checkpoint(
        self,
        rollouts: Sequence,
        stats: Dict[str, float],
        num_updates: int,
        policy_version: int,
        buffer_size: int,
        rollouts_dropped: int,
    ) -> None:
        if self._pending_update_count <= 0:
            self.record_update(
                rollouts,
                stats,
                num_updates=num_updates,
                buffer_size=buffer_size,
                rollouts_dropped=rollouts_dropped,
            )

        now = time.monotonic()
        window = max(now - self._last_rate_time, 1e-6)
        checkpoint_elapsed_sec = max(now - self._last_checkpoint_time, 0.0)
        rollouts_per_sec = self._rollouts_ingested / window
        batches_per_sec = self._batches_ingested / window
        self._rollouts_ingested = 0
        self._batches_ingested = 0
        self._last_rate_time = now
        self._last_checkpoint_time = now

        checkpoint_stats = self._checkpoint_stats(stats)
        row = self._stats_row(
            event="checkpoint",
            stats=checkpoint_stats,
            num_updates=num_updates,
            batch_mean_return=self._pending_return_sum
            / max(self._pending_return_count, 1),
            buffer_size=buffer_size,
            rollouts_dropped=rollouts_dropped,
        )
        row.update(
            {
                "policy_version": policy_version,
                "updates_since_checkpoint": self._pending_update_count,
                "rollouts_processed": self._pending_rollouts_processed,
                "rollout_steps_processed": self._pending_rollout_steps_processed,
                "checkpoint_elapsed_sec": checkpoint_elapsed_sec,
                "rollouts_per_sec": rollouts_per_sec,
                "batches_per_sec": batches_per_sec,
            }
        )
        self._write_tensorboard(row, step=num_updates, prefix="checkpoint")
        self._log_checkpoint(
            row,
            num_updates,
            rollouts_per_sec,
            batches_per_sec,
            buffer_size,
            rollouts_dropped,
        )
        self._pending_update_count = 0
        self._pending_return_sum = 0.0
        self._pending_return_count = 0
        self._pending_stats_sum.clear()
        self._pending_rollouts_processed = 0
        self._pending_rollout_steps_processed = 0

    def _stats_row(
        self,
        *,
        event: str,
        stats: Dict[str, float],
        num_updates: int,
        batch_mean_return: float,
        buffer_size: int,
        rollouts_dropped: int,
    ) -> Dict[str, float | int | str]:
        return {
            "timestamp": time.time(),
            "event": event,
            "num_updates": num_updates,
            "batch_mean_return": batch_mean_return,
            "loss_total": stats.get("loss", 0.0),
            "loss_actor": stats.get("actor_loss", 0.0),
            "loss_value": stats.get("value_loss", 0.0),
            "loss_entropy": stats.get("entropy", 0.0),
            "loss_bc": stats.get("bc_loss", 0.0),
            "steps_in_update": stats.get("steps", 0.0),
            "mean_rho": stats.get("mean_rho", 0.0),
            "max_rho": stats.get("max_rho", 0.0),
            "buffer_size": buffer_size,
            "rollouts_dropped": rollouts_dropped,
            "bc_coef": stats.get("bc_coef", 0.0),
            "bc_decay_steps": stats.get("bc_decay_steps", 0.0),
            "expert_rollouts": stats.get("expert_rollouts", 0.0),
            "neural_rollouts": stats.get("neural_rollouts", 0.0),
            "rl_steps": stats.get("rl_steps", 0.0),
            "bc_steps": stats.get("bc_steps", 0.0),
            "behaviour_log_prob_mean": stats.get("behaviour_log_prob_mean", 0.0),
            "behaviour_log_prob_min": stats.get("behaviour_log_prob_min", 0.0),
            "behaviour_log_prob_max": stats.get("behaviour_log_prob_max", 0.0),
            "action_speed_mean": stats.get("action_speed_mean", 0.0),
            "action_steer_mean": stats.get("action_steer_mean", 0.0),
            "raw_speed_mean": stats.get("raw_speed_mean", 0.0),
            "raw_steer_mean": stats.get("raw_steer_mean", 0.0),
            "reward_mean": stats.get("reward_mean", 0.0),
            "reward_min": stats.get("reward_min", 0.0),
            "reward_max": stats.get("reward_max", 0.0),
            "grad_norm": stats.get("grad_norm", 0.0),
            "max_abs_grad": stats.get("max_abs_grad", 0.0),
            "grad_params": stats.get("grad_params", 0.0),
            "bc_speed_mae": stats.get("bc_speed_mae", 0.0),
            "bc_steer_mae": stats.get("bc_steer_mae", 0.0),
            "policy_mean_speed": stats.get("policy_mean_speed", 0.0),
            "policy_mean_steer": stats.get("policy_mean_steer", 0.0),
            "policy_sigma_speed": stats.get("policy_sigma_speed", 0.0),
            "policy_sigma_steer": stats.get("policy_sigma_steer", 0.0),
            "rollout_epoch": stats.get("rollout_epoch", 0.0),
            "rollout_epochs_per_update": stats.get(
                "rollout_epochs_per_update", 0.0
            ),
        }

    def _log_checkpoint(
        self,
        row: Dict[str, float | int | str],
        num_updates: int,
        rollouts_per_sec: float,
        batches_per_sec: float,
        buffer_size: int,
        rollouts_dropped: int,
    ) -> None:
        if self._logger is None:
            return
        self._logger.info(
            "checkpoint_policy_v{} update={} mode={} updates_since_checkpoint={} "
            "rollouts(expert={}, neural={}) steps(rl={}, bc={}) "
            "loss(total={:.4f} actor={:.4f} value={:.4f} entropy={:.4f} bc={:.4f} bc_coef={:.4f}) "
            "bc_mae(speed={:.4f} steer={:.4f}) "
            "rho(mean={:.3f} max={:.3f}) "
            "behaviour_log_prob(mean={:.3f} min={:.3f} max={:.3f}) "
            "actions(mean_speed={:.3f} mean_steer={:.3f} raw_speed={:.3f} raw_steer={:.3f}) "
            "rewards(mean={:.4f} min={:.4f} max={:.4f}) "
            "grad(norm={:.4f} max_abs={:.4f}) "
            "processed(rollouts={} rollout_steps={} optimizer_steps={} elapsed_sec={:.2f}) "
            "returns(batch_mean={:.3f}) ingest(rollouts/s={:.1f} batches/s={:.1f}) buffer={} dropped={}",
            int(float(row.get("policy_version", 0) or 0)),
            num_updates,
            self._config.trainer_mode,
            int(float(row.get("updates_since_checkpoint", 0) or 0)),
            int(float(row.get("expert_rollouts", 0) or 0)),
            int(float(row.get("neural_rollouts", 0) or 0)),
            int(float(row.get("rl_steps", 0) or 0)),
            int(float(row.get("bc_steps", 0) or 0)),
            float(row.get("loss_total", 0.0) or 0.0),
            float(row.get("loss_actor", 0.0) or 0.0),
            float(row.get("loss_value", 0.0) or 0.0),
            float(row.get("loss_entropy", 0.0) or 0.0),
            float(row.get("loss_bc", 0.0) or 0.0),
            float(row.get("bc_coef", 0.0) or 0.0),
            float(row.get("bc_speed_mae", 0.0) or 0.0),
            float(row.get("bc_steer_mae", 0.0) or 0.0),
            float(row.get("mean_rho", 0.0) or 0.0),
            float(row.get("max_rho", 0.0) or 0.0),
            float(row.get("behaviour_log_prob_mean", 0.0) or 0.0),
            float(row.get("behaviour_log_prob_min", 0.0) or 0.0),
            float(row.get("behaviour_log_prob_max", 0.0) or 0.0),
            float(row.get("action_speed_mean", 0.0) or 0.0),
            float(row.get("action_steer_mean", 0.0) or 0.0),
            float(row.get("raw_speed_mean", 0.0) or 0.0),
            float(row.get("raw_steer_mean", 0.0) or 0.0),
            float(row.get("reward_mean", 0.0) or 0.0),
            float(row.get("reward_min", 0.0) or 0.0),
            float(row.get("reward_max", 0.0) or 0.0),
            float(row.get("grad_norm", 0.0) or 0.0),
            float(row.get("max_abs_grad", 0.0) or 0.0),
            int(float(row.get("rollouts_processed", 0) or 0)),
            int(float(row.get("rollout_steps_processed", 0) or 0)),
            int(float(row.get("updates_since_checkpoint", 0) or 0)),
            float(row.get("checkpoint_elapsed_sec", 0.0) or 0.0),
            float(row.get("batch_mean_return", 0.0) or 0.0),
            rollouts_per_sec,
            batches_per_sec,
            buffer_size,
            rollouts_dropped,
        )

    def _checkpoint_stats(self, latest_stats: Dict[str, float]) -> Dict[str, float]:
        count = max(self._pending_update_count, 1)
        out: Dict[str, float] = {}
        for field in self.MEAN_FIELDS:
            out[field] = self._pending_stats_sum.get(
                field, float(latest_stats.get(field, 0.0))
            ) / count
        for field in self.SUM_FIELDS:
            out[field] = self._pending_stats_sum.get(
                field, float(latest_stats.get(field, 0.0))
            )
        return out

    def close(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()
            self._tb_writer = None

    def _open_tensorboard_writer(self, tensorboard_dir: Path | None) -> Any | None:
        if tensorboard_dir is None:
            return None
        if SummaryWriter is None:
            if self._logger is not None:
                self._logger.warning(
                    "tensorboard logging requested at {}, but tensorboard is not installed",
                    tensorboard_dir,
                )
            return None
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tensorboard_dir))
        if self._logger is not None:
            self._logger.info("writing TensorBoard metrics to {}", tensorboard_dir)
        return writer

    def _write_tensorboard(
        self, row: Dict[str, object], *, step: int, prefix: str
    ) -> None:
        if self._tb_writer is None:
            return
        if prefix == "update":
            tags = self.UPDATE_LOSS_TAGS | self.UPDATE_POLICY_TAGS
        elif prefix == "rollout":
            tags = self.ROLLOUT_TAGS
        elif prefix == "checkpoint":
            tags = self.CHECKPOINT_TAGS
        else:
            return

        for field, tag in tags.items():
            value = row.get(field)
            scalar = self._to_float(value)
            if scalar is None:
                continue
            self._tb_writer.add_scalar(tag, scalar, step)
        if prefix == "update":
            self._tb_writer.flush()

    @staticmethod
    def _safe_mean(values: Iterable[float]) -> float:
        vals = list(values)
        return float(sum(vals) / len(vals)) if vals else 0.0

    @staticmethod
    def _to_float(value: object) -> float | None:
        if value in ("", None):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
