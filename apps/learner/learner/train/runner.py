from __future__ import annotations

from pathlib import Path
import random
from typing import Dict, List, Optional

import torch
from loguru import logger

from learner.config.learner import LearnerConfig
from learner.config.rollouts import RolloutSample
from learner.io.policy_pusher import PolicyPusher
from learner.io.sources_grpc import RolloutSource
from learner.train.checkpoint import apply_checkpoint_config
from learner.train.checkpoint import checkpoint_restore_optimizer
from learner.train.checkpoint import load_checkpoint_payload
from learner.train.checkpoint import save_checkpoint
from learner.train.metrics import LearnerMetrics
from learner.train.trainer import PolicyTrainer


class Runner:
    """Coordinates rollout ingestion, policy updates, checkpointing, and streaming."""

    def __init__(
        self,
        rollout_source: RolloutSource,
        checkpoint_dir: Path,
        config: LearnerConfig,
        policy_pusher: Optional[PolicyPusher] = None,
        resume_checkpoint: Optional[Path] = None,
        tensorboard_dir: Optional[Path] = None,
    ) -> None:
        self._rollout_source = rollout_source
        self._checkpoint_dir = checkpoint_dir
        self._config = config
        self._device = self._resolve_device(config.device)
        self._policy_pusher = policy_pusher

        resume_payload = None
        if resume_checkpoint is not None:
            resume_payload = load_checkpoint_payload(resume_checkpoint, self._device)
            apply_checkpoint_config(self._config, resume_payload.get("config"))

        self._trainer = PolicyTrainer(self._config, self._device)
        self._batch_size = max(1, self._config.rollouts_per_update)
        self._buffer: List[RolloutSample] = []
        self._policy_version = 0
        self._num_updates = 0
        self._logger = logger.bind(logger_name="ddrl.learner")
        self._metrics = LearnerMetrics(
            self._config,
            self._logger,
            tensorboard_dir=tensorboard_dir,
        )
        self._rollouts_dropped = 0

        if resume_checkpoint is not None:
            self._restore_checkpoint_payload(resume_checkpoint, resume_payload)
        self._publish_initial_policy()

    async def run(self) -> None:
        async for batch in self._rollout_source.batches():
            self._metrics.note_batch(len(batch))
            for rollout in batch:
                self._handle_rollout(rollout)

    def close(self) -> None:
        self._metrics.close()

    def _publish_initial_policy(self) -> None:
        if self._policy_pusher is None:
            return
        if self._policy_version == 0:
            self._policy_version = 1
        push = self._trainer.build_policy_push(self._policy_version, {})
        self._policy_pusher.submit(push)
        if self._logger:
            self._logger.info(
                "published initial policy v{} (action_dim={})",
                self._policy_version,
                self._trainer.action_dim,
            )

    def _restore_checkpoint_payload(
        self, checkpoint_path: Path, payload: dict | None
    ) -> None:
        if not isinstance(payload, dict):
            raise ValueError(
                f"Checkpoint {checkpoint_path} did not contain a mapping payload."
            )
        model_state = payload.get("model_state")
        optimizer_state = payload.get("optimizer_state")
        if not isinstance(model_state, dict):
            raise ValueError(f"Checkpoint {checkpoint_path} is missing model_state.")
        restore_optimizer = checkpoint_restore_optimizer(payload)
        if restore_optimizer and not isinstance(optimizer_state, dict):
            raise ValueError(
                f"Checkpoint {checkpoint_path} is missing optimizer_state."
            )
        if not isinstance(optimizer_state, dict):
            optimizer_state = {}

        action_dim = int(payload.get("action_dim") or self._config.action_dim)
        if action_dim <= 0:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has invalid action_dim={action_dim}."
            )
        if action_dim != self._trainer.action_dim:
            raise ValueError(
                f"Checkpoint {checkpoint_path} action_dim={action_dim} does not match "
                f"learner action_dim={self._trainer.action_dim}."
            )

        self._trainer.restore(
            model_state=model_state,
            optimizer_state=optimizer_state,
            opt_steps=int(payload.get("num_updates", 0)),
            restore_optimizer=restore_optimizer,
        )
        self._policy_version = max(1, int(payload.get("version", 1)))
        self._num_updates = max(0, int(payload.get("num_updates", 0)))
        if self._logger:
            self._logger.info(
                "resumed from {} (policy_version={}, num_updates={})",
                checkpoint_path,
                self._policy_version,
                self._num_updates,
            )
            self._logger.info(
                "effective runtime config after resume: trainer_mode={} rollouts_per_update={} num_epochs_per_update={} checkpoint_interval={} device={}",
                self._config.trainer_mode,
                self._config.rollouts_per_update,
                self._config.num_epochs_per_update,
                self._config.checkpoint_interval,
                self._config.device,
            )
            self._logger.info(
                "bc anchor config after resume: bc_coef={} bc_floor_coef={} bc_decay_updates={} bc_decay_steps={} entropy_coef={} reset_action_std={} action_init_std={}",
                self._config.bc_coef,
                self._config.bc_floor_coef,
                self._config.bc_decay_updates,
                self._trainer.bc_decay_steps,
                self._config.entropy_coef,
                self._config.reset_action_std_on_resume,
                self._config.action_init_std,
            )

    def _handle_rollout(self, rollout: RolloutSample) -> None:
        if not rollout.steps:
            return
        rollout_is_expert = (
            float(getattr(rollout, "metrics", {}).get("is_expert", 0.0)) >= 0.5
        )
        if self._config.trainer_mode == "online_bc" and not rollout_is_expert:
            self._rollouts_dropped += 1
            if self._logger:
                self._logger.debug(
                    "dropping non-expert rollout in online_bc mode (sim_id={} rollout_id={} policy_version={})",
                    rollout.sim_id,
                    rollout.rollout_id,
                    rollout.policy_version,
                )
            return
        if self._config.trainer_mode == "rl" and rollout_is_expert:
            self._rollouts_dropped += 1
            if self._logger:
                self._logger.debug(
                    "dropping expert rollout in rl mode (sim_id={} rollout_id={} policy_version={})",
                    rollout.sim_id,
                    rollout.rollout_id,
                    rollout.policy_version,
                )
            return

        action_dim = len(rollout.steps[0].action)
        if action_dim == 0:
            return
        if action_dim != self._trainer.action_dim:
            if self._logger:
                self._logger.warning(
                    "rollout action_dim {} does not match learner action_dim {}; dropping rollout",
                    action_dim,
                    self._trainer.action_dim,
                )
            return
        self._buffer.append(rollout)
        self._metrics.record_rollout(
            rollout,
            num_updates=self._num_updates,
            buffer_size=len(self._buffer),
            rollouts_dropped=self._rollouts_dropped,
        )
        if len(self._buffer) > self._batch_size:
            # Drop oldest to keep the freshest rollouts.
            self._buffer.pop(0)
            self._rollouts_dropped += 1

        if len(self._buffer) >= self._batch_size:
            rollouts = self._buffer[: self._batch_size]
            del self._buffer[: self._batch_size]
            self._optimize_rollout_batch(rollouts)

    def _optimize_rollout_batch(self, rollouts: List[RolloutSample]) -> None:
        epochs = max(1, int(self._config.num_epochs_per_update))
        for epoch in range(epochs):
            shuffled = list(rollouts)
            random.shuffle(shuffled)
            if self._config.trainer_mode == "online_bc":
                stats = self._trainer.optimize_bc(shuffled)
            else:
                stats = self._trainer.optimize(shuffled)
            stats["rollout_epoch"] = float(epoch + 1)
            stats["rollout_epochs_per_update"] = float(epochs)

            self._num_updates += 1
            self._metrics.record_update(
                shuffled,
                stats,
                num_updates=self._num_updates,
                buffer_size=len(self._buffer),
                rollouts_dropped=self._rollouts_dropped,
            )
            warmup = max(0, int(self._config.policy_push_warmup_updates))
            interval = max(1, int(self._config.checkpoint_interval))
            if (
                self._num_updates >= warmup
                and (self._num_updates - warmup) % interval == 0
            ):
                policy_version = self._emit_policy_checkpoint(stats)
                self._metrics.record_checkpoint(
                    shuffled,
                    stats,
                    num_updates=self._num_updates,
                    policy_version=policy_version,
                    buffer_size=len(self._buffer),
                    rollouts_dropped=self._rollouts_dropped,
                )

    def _emit_policy_checkpoint(self, stats: Dict[str, float]) -> int:
        self._policy_version += 1
        save_checkpoint(
            checkpoint_dir=self._checkpoint_dir,
            version=self._policy_version,
            config=self._config,
            model_state=self._trainer.model_state,
            optimizer_state=self._trainer.optimizer_state,
            action_dim=self._trainer.action_dim,
            stats=stats,
            num_updates=self._num_updates,
        )

        if self._policy_pusher is not None:
            push = self._trainer.build_policy_push(self._policy_version, stats)
            self._policy_pusher.submit(push)
        return self._policy_version

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "cpu":
            return torch.device("cpu")
        if requested in ("cuda", "gpu"):
            if torch.cuda.is_available():
                return torch.device("cuda")
            logger.bind(logger_name="ddrl.learner").warning(
                "CUDA requested but unavailable, falling back to CPU"
            )
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
