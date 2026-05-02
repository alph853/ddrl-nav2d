from __future__ import annotations

import asyncio
import time
import uuid
from typing import Dict, Optional

import grpc
from loguru import logger

from proto import policy_pb2, policy_pb2_grpc
from .rpc_server import RpcServer


class _Subscriber:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[policy_pb2.PolicyPush] = asyncio.Queue(
            maxsize=32
        )
        self.last_acked_version: int = 0
        self.last_acked_at_unix: float = 0.0
        self.last_ack_worker_unix_ms: int = 0


class _PolicyStreamServicer(policy_pb2_grpc.PolicyServiceServicer):
    def __init__(self, state: PolicyPusher) -> None:
        self._state = state
        self._logger = logger.bind(logger_name="ddrl.learner")

    async def StreamPolicyUpdates(self, request_iterator, context) -> None:  # type: ignore[override]
        try:
            first = await request_iterator.__anext__()
        except StopAsyncIteration:
            return

        if not first.HasField("subscribe"):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "expected subscribe payload"
            )

        subscribe = first.subscribe
        worker_id = subscribe.worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        subscriber = _Subscriber()
        async with self._state.state_lock:
            self._state.subscribers[worker_id] = subscriber
            latest = self._state.latest
            subscriber.last_acked_version = subscribe.last_version
            subscriber.last_acked_at_unix = time.time()

        self._logger.info("worker {} subscribed to policy stream", worker_id)

        ack_task = asyncio.create_task(self._consume_acks(worker_id, request_iterator))

        try:
            if latest is not None and latest.version > subscribe.last_version:
                yield latest

            while not context.done():
                try:
                    update = await asyncio.wait_for(subscriber.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                yield update
        finally:
            ack_task.cancel()
            async with self._state.state_lock:
                self._state.subscribers.pop(worker_id, None)
            self._logger.info("worker {} disconnected from policy stream", worker_id)

    async def _consume_acks(self, worker_id: str, request_iterator) -> None:
        try:
            async for request in request_iterator:
                if request.HasField("ack"):
                    ack = request.ack
                    ts = ack.received_unix_ms / 1000.0
                    now = time.time()
                    latest_version = 0
                    async with self._state.state_lock:
                        sub = self._state.subscribers.get(worker_id)
                        if sub is not None:
                            sub.last_acked_version = ack.version
                            sub.last_ack_worker_unix_ms = ack.received_unix_ms
                            sub.last_acked_at_unix = now
                            if self._state.latest is not None:
                                latest_version = self._state.latest.version
                    self._logger.info(
                        "worker {} acked v{} at {}",
                        worker_id,
                        ack.version,
                        time.strftime("%H:%M:%S", time.localtime(ts)),
                    )
                    if latest_version and ack.version + 5 < latest_version:
                        self._logger.warning(
                            "worker {} is {} versions behind (latest v{})",
                            worker_id,
                            latest_version - ack.version,
                            latest_version,
                        )
        except asyncio.CancelledError:  # pragma: no cover - best effort logging
            pass


class PolicyPusher:
    """Manages policy broadcast state and registers its servicer on a shared RpcServer."""

    def __init__(self, server: RpcServer) -> None:
        self.subscribers: Dict[str, _Subscriber] = {}
        self.state_lock = asyncio.Lock()
        self.latest: Optional[policy_pb2.PolicyPush] = None
        self._logger = logger.bind(logger_name="ddrl.learner")

        self._rpc_server = server
        self._rpc_server.register_servicer(self._register_servicer)

    def submit(self, push: policy_pb2.PolicyPush) -> None:
        msg = policy_pb2.PolicyPush()
        msg.CopyFrom(push)

        async def _broadcast(payload: policy_pb2.PolicyPush) -> None:
            async with self.state_lock:
                self.latest = payload
                subscribers = list(self.subscribers.items())
            for worker_id, sub in subscribers:
                frame = policy_pb2.PolicyPush()
                frame.CopyFrom(payload)
                try:
                    sub.queue.put_nowait(frame)
                except asyncio.QueueFull:
                    try:
                        sub.queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    sub.queue.put_nowait(frame)
                except RuntimeError as exc:  # pragma: no cover - queue closed during shutdown
                    self._logger.warning(
                        "failed to enqueue policy for worker {}: {}", worker_id, exc
                    )

        self._rpc_server.run_coroutine(_broadcast(msg))

    async def subscriber_stats(self) -> Dict[str, Dict[str, float]]:
        """Return a snapshot of subscriber ACK metadata for monitoring/debugging."""
        async with self.state_lock:
            latest_version = self.latest.version if self.latest is not None else 0
            now = time.time()
            stats: Dict[str, Dict[str, float]] = {}
            for worker_id, sub in self.subscribers.items():
                lag_versions = max(0, latest_version - sub.last_acked_version)
                lag_seconds: Optional[float]
                if sub.last_acked_at_unix:
                    lag_seconds = max(0.0, now - sub.last_acked_at_unix)
                else:
                    lag_seconds = None
                stats[worker_id] = {
                    "last_acked_version": sub.last_acked_version,
                    "last_acked_at_unix": sub.last_acked_at_unix,
                    "last_ack_worker_unix_ms": sub.last_ack_worker_unix_ms,
                    "lag_versions": lag_versions,
                    "lag_seconds": lag_seconds if lag_seconds is not None else -1.0,
                }
        return stats

    # ------------------------------------------------------------------ internals
    def _register_servicer(self, server: grpc.aio.Server) -> None:
        policy_pb2_grpc.add_PolicyServiceServicer_to_server(
            _PolicyStreamServicer(self), server
        )
