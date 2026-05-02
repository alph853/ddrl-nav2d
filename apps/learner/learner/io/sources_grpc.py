from __future__ import annotations

import asyncio
import queue
import time
from typing import AsyncIterator, List, Protocol

from loguru import logger

from proto import rollouts_pb2, rollouts_pb2_grpc
from learner.io.rpc_server import RpcServer
from learner.config.rollouts import RolloutSample, rollouts_from_proto

_INFO_THROTTLE_STATE: dict[str, float] = {}


def INFO_THROTTLE(logger, period_seconds: float, key: str, message: str, *args) -> None:
    now = time.monotonic()
    last = _INFO_THROTTLE_STATE.get(key, 0.0)
    if now - last >= period_seconds:
        _INFO_THROTTLE_STATE[key] = now
        logger.info(message, *args)


class RolloutSource(Protocol):
    def batches(self) -> AsyncIterator[List[RolloutSample]]: ...


class _RolloutStreamServicer(rollouts_pb2_grpc.RolloutStreamServicer):
    def __init__(self, source: "GrpcRolloutSource") -> None:
        self._source = source
        self._logger = logger.bind(logger_name="ddrl.learner")

    async def StreamRollouts(self, request_iterator, context):
        async for request in request_iterator:
            batch = request.batch
            if not batch.rollouts:
                continue
            rollouts = rollouts_from_proto(batch)
            if rollouts:
                steps = sum(len(rollout.steps) for rollout in rollouts)
                payload_bytes = request.ByteSize()
                self._source.enqueue(rollouts, batch_bytes=payload_bytes, steps=steps)
        if self._logger:
            self._logger.info("Rollout stream closed")
        if False:  # pragma: no cover - keep generator type
            yield rollouts_pb2.RolloutStreamResponse()


class GrpcRolloutSource:
    """Async wrapper that receives rollout batches over gRPC streaming."""

    def __init__(self, rpc_server: RpcServer, max_queue_size: int = 256) -> None:
        self._queue: "queue.Queue[List[RolloutSample]]" = queue.Queue(
            maxsize=max_queue_size
        )
        self._rpc_server = rpc_server
        self._logger = logger.bind(logger_name="ddrl.learner")
        self._dropped = 0
        self._received = 0
        self._bytes_received = 0
        self._steps_received = 0
        self._batches_received = 0
        self._stats_window_start = time.monotonic()
        self._rpc_server.register_servicer(self._register_servicer)

    def enqueue(
        self, rollouts: List[RolloutSample], batch_bytes: int = 0, steps: int = 0
    ) -> None:
        if not rollouts:
            return
        try:
            self._queue.put_nowait(rollouts)
            self._received += len(rollouts)
            self._bytes_received += max(0, int(batch_bytes))
            self._steps_received += max(0, int(steps))
            self._batches_received += 1
        except queue.Full:
            self._dropped += len(rollouts)
            if self._logger:
                self._logger.warning(
                    "Rollout queue full; dropping {} rollouts", len(rollouts)
                )
        self._log_stats()

    async def batches(self) -> AsyncIterator[List[RolloutSample]]:
        while True:
            batch = await asyncio.to_thread(self._queue.get)
            yield batch

    def health_metrics(self) -> dict:
        return {
            "grpc_rollouts_received": float(self._received),
            "grpc_rollouts_dropped": float(self._dropped),
            "grpc_queue_depth": float(self._queue.qsize()),
        }

    def _log_stats(self) -> None:
        now = time.monotonic()
        elapsed = now - self._stats_window_start
        if elapsed < 5.0:
            return
        if self._logger:
            mb_per_sec = (self._bytes_received / (1024.0 * 1024.0)) / max(elapsed, 1e-6)
            rollouts_per_sec = self._received / max(elapsed, 1e-6)
            steps_per_sec = self._steps_received / max(elapsed, 1e-6)
            avg_batch_bytes = (
                self._bytes_received / self._batches_received
                if self._batches_received > 0
                else 0.0
            )
            avg_step_bytes = (
                self._bytes_received / self._steps_received
                if self._steps_received > 0
                else 0.0
            )
            INFO_THROTTLE(
                self._logger,
                5.0,
                "learner.grpc.rollouts",
                "Rollout ingest stats: batches={} rollouts={} steps={} serialized_bytes={} "
                "throughput={:.2f}MB/s rollouts/s={:.1f} steps/s={:.1f} "
                "avg_batch_bytes={:.0f} avg_step_bytes={:.1f} queue_depth={}",
                self._batches_received,
                self._received,
                self._steps_received,
                self._bytes_received,
                mb_per_sec,
                rollouts_per_sec,
                steps_per_sec,
                avg_batch_bytes,
                avg_step_bytes,
                self._queue.qsize(),
            )
        self._stats_window_start = now
        self._bytes_received = 0
        self._steps_received = 0
        self._batches_received = 0
        self._received = 0

    def _register_servicer(self, server) -> None:
        rollouts_pb2_grpc.add_RolloutStreamServicer_to_server(
            _RolloutStreamServicer(self), server
        )
