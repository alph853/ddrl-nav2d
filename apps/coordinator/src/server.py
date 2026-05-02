from __future__ import annotations

import asyncio
import contextlib
from typing import Optional

import grpc
from loguru import logger

from proto import coordinator_pb2_grpc

from .config import CoordinatorConfig
from .health import ensure_kafka_cluster_active
from .service import CoordinatorService
from .state import CoordinatorState


LOGGER = logger.bind(logger_name="ddrl.coordinator")


class CoordinatorServer:
    """Wrapper around the grpc.aio server lifecycle."""

    def __init__(self, config: CoordinatorConfig) -> None:
        self._config = config
        self._state = CoordinatorState()
        self._server = grpc.aio.server(options=(("grpc.so_reuseport", 0),))
        self._monitor_task: Optional[asyncio.Task] = None
        self._bound_port = 0
        coordinator_pb2_grpc.add_CoordinatorControlServicer_to_server(
            CoordinatorService(config, self._state),
            self._server,
        )
        self._bound_port = self._server.add_insecure_port(self._config.listen_address)
        if self._bound_port == 0:
            raise RuntimeError(
                f"failed to bind coordinator gRPC server to {self._config.listen_address}"
            )

    async def start(self) -> None:
        await self._server.start()
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop(self, grace: float = 2.0) -> None:
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task
            self._monitor_task = None
        await self._server.stop(grace)

    async def wait_for_termination(self) -> None:
        await self._server.wait_for_termination()

    async def _monitor_loop(self) -> None:
        worker_timeout = self._config.worker_heartbeat_timeout
        learner_timeout = max(self._config.learner_heartbeat_interval * 2.0, worker_timeout)
        poll_interval = max(min(worker_timeout / 2.0, 5.0), 1.0)
        kafka_check_interval = max(worker_timeout, 10.0)
        last_kafka_check = 0.0
        enable_kafka_check = self._config.enable_kafka_healthcheck

        LOGGER.info(
            "coordinator monitor loop started (worker_timeout={:.1f}s, learner_timeout={:.1f}s)",
            worker_timeout,
            learner_timeout,
        )

        try:
            loop = asyncio.get_running_loop()
            while True:
                self._state.cleanup_stale_workers(worker_timeout)
                self._state.cleanup_stale_learners(learner_timeout)

                now = loop.time()
                if enable_kafka_check and now - last_kafka_check >= kafka_check_interval:
                    try:
                        ensure_kafka_cluster_active(self._config.kafka_bootstrap, timeout=1.0)
                    except RuntimeError as exc:
                        LOGGER.warning("Kafka health check failed: {}", exc)
                    finally:
                        last_kafka_check = now

                await asyncio.sleep(poll_interval)
        except asyncio.CancelledError:
            LOGGER.info("coordinator monitor loop cancelled")
            raise
        except Exception:
            LOGGER.exception("coordinator monitor loop crashed")
            raise
