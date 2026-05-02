from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional

import grpc
from google.protobuf import empty_pb2
from loguru import logger

from proto import coordinator_pb2, coordinator_pb2_grpc

from .config import CoordinatorConfig
from .state import CoordinatorState

LOGGER = logger.bind(logger_name="ddrl.coordinator")


class CoordinatorService(coordinator_pb2_grpc.CoordinatorControlServicer):
    def __init__(self, config: CoordinatorConfig, state: CoordinatorState) -> None:
        self._config = config
        self._state = state

    # worker RPCs
    async def RegisterWorker(
        self,
        request: coordinator_pb2.WorkerHello,
        context: grpc.aio.ServicerContext,
    ) -> coordinator_pb2.WorkerSession:
        session = self._state.register_worker(
            request,
            heartbeat_interval_ms=int(self._config.worker_heartbeat_interval * 1000),
        )
        LOGGER.info(
            "worker registered id={} hostname={} version={} learner_target={}",
            session.worker_id,
            request.hostname or "unknown",
            request.version or "unknown",
            session.learner_target or "<empty>",
        )
        return session

    async def WorkerHeartbeatStream(
        self,
        request_iterator: AsyncIterator[coordinator_pb2.WorkerHeartbeat],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[coordinator_pb2.ControlDirective]:
        worker_id: Optional[str] = None
        forward_task: Optional[asyncio.Task] = None
        send_queue: asyncio.Queue = asyncio.Queue()
        heartbeat_task = asyncio.create_task(self._next_heartbeat(request_iterator))
        send_task: Optional[asyncio.Task] = None

        try:
            while True:
                wait_set = {heartbeat_task}
                if send_task is None:
                    send_task = asyncio.create_task(send_queue.get())
                wait_set.add(send_task)

                done, _ = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)

                if send_task in done:
                    directive = send_task.result()
                    send_task = None
                    yield directive

                if heartbeat_task in done:
                    heartbeat = heartbeat_task.result()
                    if heartbeat is None:
                        break
                    worker_id = heartbeat.worker_id
                    try:
                        record = self._state.record_worker_heartbeat(heartbeat)
                    except KeyError as exc:
                        await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
                    except PermissionError:
                        await context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid session token")

                    if forward_task is None:
                        forward_task = asyncio.create_task(
                            self._forward_directives(record.directive_queue, send_queue)
                        )
                    heartbeat_task = asyncio.create_task(self._next_heartbeat(request_iterator))
        except asyncio.CancelledError:
            LOGGER.info("heartbeat stream cancelled for worker={}", worker_id)
            raise
        finally:
            heartbeat_task.cancel()
            if send_task is not None:
                send_task.cancel()
            if forward_task is not None:
                forward_task.cancel()
            if worker_id:
                self._state.mark_worker_disconnected(worker_id)

    async def _next_heartbeat(
        self,
        request_iterator: AsyncIterator[coordinator_pb2.WorkerHeartbeat],
    ) -> Optional[coordinator_pb2.WorkerHeartbeat]:
        try:
            return await request_iterator.__anext__()
        except StopAsyncIteration:
            return None

    async def _forward_directives(
        self,
        source_queue: asyncio.Queue,
        sink_queue: asyncio.Queue,
    ) -> None:
        try:
            while True:
                directive = await source_queue.get()
                await sink_queue.put(directive)
        except asyncio.CancelledError:
            return

    # learner RPCs
    async def RegisterLearner(
        self,
        request: coordinator_pb2.LearnerHello,
        context: grpc.aio.ServicerContext,
    ) -> coordinator_pb2.LearnerSession:
        record = self._state.register_learner(request, int(self._config.learner_heartbeat_interval * 1000))
        LOGGER.info("learner registered id={} endpoint={}", record.learner_id, record.endpoint)
        return coordinator_pb2.LearnerSession(
            learner_id=record.learner_id,
            session_token=record.session_token,
            heartbeat_interval_ms=int(self._config.learner_heartbeat_interval * 1000),
        )

    async def LearnerHeartbeat(
        self,
        request: coordinator_pb2.LearnerHeartbeatRequest,
        context: grpc.aio.ServicerContext,
    ) -> empty_pb2.Empty:
        record = self._state.record_learner_heartbeat(request)
        LOGGER.debug(
            "learner heartbeat id={} endpoint={} version={}",
            record.learner_id,
            record.endpoint,
            record.version,
        )
        return empty_pb2.Empty()
