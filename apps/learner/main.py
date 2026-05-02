#!/usr/bin/env python3
"""
IMPALA-style learner entrypoint.

Workers stream rollouts over gRPC using `ddrl.comm.RolloutBatch`, while the
Python learner consumes those batches, optimises a V-trace actor-critic, and
pushes checkpoints back to workers via gRPC.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from loguru import logger
from learner.io.coordinator_client import LearnerCoordinatorClient
from learner.io.policy_pusher import PolicyPusher
from learner.io.rpc_server import RpcServer
from learner.io.sources_grpc import GrpcRolloutSource
from learner.train.runner import Runner
from learner.config.learner import LearnerConfig
from learner.config.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IMPALA learner with V-trace updates.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("learner_checkpoints"),
        help="Directory to store policy checkpoints.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Learner checkpoint (.pt) to restore and continue training from.",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument(
        "--clip-rho", type=float, default=1.0, help="Importance weight clip for rho."
    )
    parser.add_argument(
        "--clip-c", type=float, default=1.0, help="Importance weight clip for c."
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.001,
        help="Entropy regularization factor.",
    )
    parser.add_argument(
        "--value-loss-coef", type=float, default=0.5, help="Value loss scaling."
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="Optimizer learning rate."
    )
    parser.add_argument(
        "--grad-clip", type=float, default=40.0, help="Gradient clipping threshold."
    )
    parser.add_argument(
        "--trainer-mode",
        type=str,
        default="rl",
        choices=("rl", "online_bc", "rl_bc_anchor"),
        help="Learner update mode.",
    )
    parser.add_argument(
        "--bc-coef",
        type=float,
        default=1.0,
        help="BC loss coefficient for rl_bc_anchor mode.",
    )
    parser.add_argument(
        "--bc-decay-updates",
        type=int,
        default=0,
        help="Linearly decay --bc-coef to --bc-floor-coef over this many learner updates; <=0 disables decay.",
    )
    parser.add_argument(
        "--bc-floor-coef",
        type=float,
        default=0.01,
        help="Lower bound for decayed BC coefficient in rl_bc_anchor mode.",
    )
    parser.add_argument(
        "--reset-action-std-on-resume",
        action="store_true",
        help="Reset policy log_std to --policy-arch-config action_init_std when loading a checkpoint.",
    )
    parser.add_argument(
        "--rollouts-per-update",
        type=int,
        default=64,
        help="Rollouts per learner update.",
    )
    parser.add_argument(
        "--num-epochs-per-update",
        type=int,
        default=3,
        help="Optimizer passes over each rollout update batch; the batch is shuffled before each pass.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=20,
        help="Updates per checkpoint emission.",
    )
    parser.add_argument(
        "--policy-push-warmup-updates",
        type=int,
        default=0,
        help="Number of learner updates to wait before pushing new policies.",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Training device: auto/cpu/cuda."
    )
    parser.add_argument(
        "--grpc-rollout-queue-size",
        type=int,
        default=int(os.environ.get("DDRL_GRPC_ROLLOUT_QUEUE_SIZE", "1024")),
        help="Max number of rollout batches buffered from the gRPC stream.",
    )
    parser.add_argument(
        "--grpc-bind",
        type=str,
        default="127.0.0.1:5600",
        help="Bind address for learner gRPC services (host:port).",
    )
    parser.add_argument(
        "--grpc-advertise",
        type=str,
        default=None,
        help="Advertised endpoint for workers (host:port). Defaults to --grpc-bind.",
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
    parser.add_argument(
        "--learner-version",
        type=str,
        default=os.environ.get("DDRL_LEARNER_VERSION", "python-learner"),
        help="Version string advertised to the coordinator.",
    )
    parser.add_argument(
        "--coordinator-target",
        type=str,
        default=os.environ.get("DDRL_COORDINATOR_TARGET"),
        help="Coordinator gRPC endpoint to register this learner (host:port).",
    )
    parser.add_argument(
        "--coordinator-retry-delay",
        type=float,
        default=2.0,
        help="Seconds to wait between coordinator registration retries.",
    )
    parser.add_argument(
        "--coordinator-max-retries",
        type=int,
        default=10,
        help="Maximum coordinator registration attempts before exit (<=0 means retry forever).",
    )
    parser.add_argument(
        "--coordinator-rpc-timeout",
        type=float,
        default=5.0,
        help="Timeout (seconds) per coordinator RegisterLearner/LearnerHeartbeat RPC.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.environ.get("LOG_LEVEL", "INFO"),
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Log level for console/file sinks.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to write logs to disk (includes timestamps).",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=None,
        help="Directory for TensorBoard event metrics. Defaults to <checkpoint-dir>/tensorboard.",
    )
    parser.add_argument(
        "--disable-tensorboard",
        action="store_true",
        help="Disable TensorBoard metric event writing.",
    )
    return parser.parse_args()


def _build_config_from_args(args: argparse.Namespace) -> LearnerConfig:
    policy_arch_config = args.rl_config or args.policy_arch_config
    base = LearnerConfig.from_policy_arch_yaml(policy_arch_config)
    base.gamma = args.gamma
    base.clip_rho = args.clip_rho
    base.clip_c = args.clip_c
    base.entropy_coef = args.entropy_coef
    base.value_loss_coef = args.value_loss_coef
    base.bc_coef = args.bc_coef
    base.bc_decay_updates = args.bc_decay_updates
    base.bc_floor_coef = args.bc_floor_coef
    base.reset_action_std_on_resume = args.reset_action_std_on_resume
    base.lr = args.lr
    base.grad_clip = args.grad_clip
    base.trainer_mode = args.trainer_mode
    base.rollouts_per_update = args.rollouts_per_update
    base.num_epochs_per_update = args.num_epochs_per_update
    base.checkpoint_interval = args.checkpoint_interval
    base.policy_push_warmup_updates = args.policy_push_warmup_updates
    base.device = args.device.lower()
    return base


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level, logger_name="ddrl.learner", log_file=args.log_file)
    app_logger = logger.bind(logger_name="ddrl.learner")

    config = _build_config_from_args(args)
    rpc_server = RpcServer(args.grpc_bind)
    policy_pusher = PolicyPusher(rpc_server)
    coordinator_client: LearnerCoordinatorClient | None = None
    rollout_source = GrpcRolloutSource(
        rpc_server, max_queue_size=args.grpc_rollout_queue_size
    )
    rpc_server.start()

    if args.coordinator_target:
        grpc_advertise = args.grpc_advertise or args.grpc_bind
        coordinator_client = LearnerCoordinatorClient(
            args.coordinator_target,
            retry_delay=args.coordinator_retry_delay,
            max_retries=args.coordinator_max_retries,
            rpc_timeout=args.coordinator_rpc_timeout,
        )
        
        try:
            coordinator_client.start(
                grpc_advertise,
                args.learner_version,
                [config.policy_model],
                metrics_supplier=rollout_source.health_metrics,
            )
        except RuntimeError as exc:
            app_logger.error("Unable to register learner with coordinator: {}", exc)
            raise SystemExit(1)

    runner = Runner(
        rollout_source,
        args.checkpoint_dir,
        config,
        policy_pusher,
        resume_checkpoint=args.resume_checkpoint,
        tensorboard_dir=None
        if args.disable_tensorboard
        else (args.tensorboard_dir or args.checkpoint_dir / "tensorboard"),
    )
    try:
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        app_logger.info("received keyboard interrupt, shutting down")
    finally:
        runner.close()
        if coordinator_client is not None:
            coordinator_client.close()
        rpc_server.shutdown()


if __name__ == "__main__":
    main()
