from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, Coroutine, List, Optional, TypeVar
from concurrent.futures import Future

import grpc
from loguru import logger

T = TypeVar("T")


class RpcServer:
    """Runs a grpc.aio.Server on a background loop and accepts multiple servicers."""

    _MAX_GRPC_MESSAGE_BYTES = 128 * 1024 * 1024

    def __init__(self, bind_target: str) -> None:
        self._bind_target = bind_target
        self._loop = asyncio.new_event_loop()
        self._server: Optional[grpc.aio.Server] = None
        self._thread: Optional[threading.Thread] = None
        self._register_callbacks: List[Callable[[grpc.aio.Server], None]] = []
        self._started = False
        self._logger = logger.bind(logger_name="ddrl.learner")

    def register_servicer(self, register_fn: Callable[[grpc.aio.Server], None]) -> None:
        """Queue a registration callback to run before the server starts."""
        if self._started:
            raise RuntimeError("cannot register new servicers after server start")
        self._register_callbacks.append(register_fn)

    def start(self) -> None:
        if self._started:
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._started = True

    def run_coroutine(self, coro: Coroutine[Any, Any, T]) -> Future[T]:
        """Schedule a coroutine on the server loop."""
        if not self._started:
            raise RuntimeError("server not started")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def shutdown(self) -> None:
        if not self._started or self._server is None or self._thread is None:
            return
        fut = asyncio.run_coroutine_threadsafe(self._server.stop(grace=1), self._loop)
        fut.result(timeout=2)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)
        self._server = None
        self._thread = None
        self._started = False

    # ------------------------------------------------------------------ internals
    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._start_server())
        self._loop.run_forever()

    async def _start_server(self) -> None:
        self._server = grpc.aio.server(
            options=(
                ("grpc.max_receive_message_length", self._MAX_GRPC_MESSAGE_BYTES),
                ("grpc.max_send_message_length", self._MAX_GRPC_MESSAGE_BYTES),
            )
        )
        for callback in self._register_callbacks:
            callback(self._server)
        self._server.add_insecure_port(self._bind_target)
        await self._server.start()
        self._logger.info("policy stream listening on {}", self._bind_target)
