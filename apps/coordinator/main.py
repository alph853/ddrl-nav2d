#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import signal
from pathlib import Path

from loguru import logger
from src.config import CoordinatorConfig
from src.health import ensure_kafka_cluster_active
from src.logging_config import setup_logging
from src.server import CoordinatorServer

LOGGER = logger.bind(logger_name="ddrl.coordinator")


def parse_args() -> argparse.Namespace:
    env = os.environ
    parser = argparse.ArgumentParser(description="DDRL coordinator gRPC service.")
    parser.add_argument("--host", type=str, default=env.get("COORDINATOR_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(env.get("COORDINATOR_PORT", "5700")))
    parser.add_argument(
        "--kafka-bootstrap",
        type=str,
        default=env.get("KAFKA_BOOTSTRAP_SERVERS_SING", "127.0.0.1:19095"),
        help="Comma-separated Kafka bootstrap servers for optional health checks.",
    )
    parser.add_argument(
        "--worker-heartbeat-interval",
        type=float,
        default=float(env.get("WORKER_HEARTBEAT_INTERVAL", "2.0")),
        help="Seconds between worker heartbeat messages suggested by the coordinator.",
    )
    parser.add_argument(
        "--worker-heartbeat-timeout",
        type=float,
        default=float(env.get("WORKER_HEARTBEAT_TIMEOUT", "10.0")),
        help="Seconds before a worker heartbeat is considered stale.",
    )
    parser.add_argument(
        "--learner-heartbeat-interval",
        type=float,
        default=float(env.get("LEARNER_HEARTBEAT_INTERVAL", "2.0")),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=env.get("LOG_LEVEL", "INFO"),
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to write logs to disk (includes timestamps).",
    )
    parser.add_argument(
        "--enable-kafka-healthcheck",
        action="store_true",
        default=env.get("DDRL_ENABLE_KAFKA_HEALTHCHECK", "").lower() in {"1", "true", "yes"},
        help="Enable Kafka reachability checks before start and in the monitor loop.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> CoordinatorConfig:
    return CoordinatorConfig(
        host=args.host,
        port=args.port,
        kafka_bootstrap=args.kafka_bootstrap,
        worker_heartbeat_interval=args.worker_heartbeat_interval,
        worker_heartbeat_timeout=args.worker_heartbeat_timeout,
        learner_heartbeat_interval=args.learner_heartbeat_interval,
        enable_kafka_healthcheck=args.enable_kafka_healthcheck,
    )


async def _serve_forever(config: CoordinatorConfig) -> None:
    server = CoordinatorServer(config)
    await server.start()
    LOGGER.info("coordinator listening on {}", config.listen_address)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _trigger_stop() -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _trigger_stop)
        except NotImplementedError:
            # Windows (or embedded interpreter) may not support signal handlers.
            pass

    try:
        await stop_event.wait()
    finally:
        LOGGER.info("shutting down coordinator ...")
        await server.stop()
        LOGGER.info("coordinator stopped")


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level, logger_name="ddrl.coordinator", log_file=args.log_file)
    if args.enable_kafka_healthcheck:
        try:
            ensure_kafka_cluster_active(args.kafka_bootstrap)
        except RuntimeError as exc:
            LOGGER.error("Unable to start coordinator: {}", exc)
            raise SystemExit(1)
    else:
        LOGGER.info("Kafka healthcheck disabled (set DDRL_ENABLE_KAFKA_HEALTHCHECK=1 to enable)")

    config = build_config(args)
    try:
        asyncio.run(_serve_forever(config))
    except KeyboardInterrupt:
        LOGGER.info("received keyboard interrupt, exiting")


if __name__ == "__main__":
    main()
