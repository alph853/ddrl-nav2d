from __future__ import annotations

import socket
import threading
import time
from typing import Callable, Optional, Sequence

import grpc
from loguru import logger

from proto import coordinator_pb2, coordinator_pb2_grpc

LOGGER = logger.bind(logger_name="ddrl.learner")


class LearnerCoordinatorClient:
    """Register the learner and maintain coordinator heartbeats in the background."""

    def __init__(
        self,
        target: str,
        retry_delay: float = 2.0,
        max_retries: int = 10,
        rpc_timeout: float = 5.0,
        fail_fast: bool = False,
    ) -> None:
        if not target:
            raise ValueError("coordinator target is required")
        self._target = target
        self._retry_delay = max(0.5, retry_delay)
        self._max_retries = max_retries
        self._rpc_timeout = max(1.0, rpc_timeout)
        self._fail_fast = fail_fast
        self._channel = None
        self._stub = None
        self._stop_event = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._session: Optional[coordinator_pb2.LearnerSession] = None
        self._endpoint: str = ""
        self._version: str = ""
        self._models: list[str] = []
        self._metrics_supplier: Optional[Callable[[], dict]] = None
        self._last_heartbeat_ok_monotonic: float = 0.0
        self._last_heartbeat_log_monotonic: float = 0.0
        self._reconnect_channel()

    def start(
        self,
        endpoint: str,
        version: str,
        models: Sequence[str] | None = None,
        metrics_supplier: Callable[[], dict] | None = None,
    ) -> coordinator_pb2.LearnerSession:
        """Register the learner endpoint and launch the heartbeat loop."""
        session = self._register(endpoint, version, models)
        self._stop_event.clear()
        self._endpoint = endpoint
        self._version = version or "unknown"
        self._models = list(models or [])
        self._metrics_supplier = metrics_supplier
        heartbeat_interval = max(0.5, (session.heartbeat_interval_ms or 0) / 1000.0)
        thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(heartbeat_interval,),
            name="learner-heartbeat",
            daemon=True,
        )
        thread.start()
        self._heartbeat_thread = thread
        return session

    def close(self) -> None:
        """Signal the heartbeat loop to stop and close the channel."""
        self._stop_event.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
            self._heartbeat_thread = None
        if self._channel is not None:
            self._channel.close()
            self._channel = None
        self._stub = None

    def _reconnect_channel(self) -> None:
        if self._channel is not None:
            self._channel.close()
        self._channel = grpc.insecure_channel(self._target)
        self._stub = coordinator_pb2_grpc.CoordinatorControlStub(self._channel)

    def _register(
        self,
        endpoint: str,
        version: str,
        models: Sequence[str] | None,
    ) -> coordinator_pb2.LearnerSession:
        if not endpoint:
            raise ValueError(
                "learner endpoint is required for coordinator registration"
            )
        hello = coordinator_pb2.LearnerHello(
            endpoint=endpoint, version=version or "unknown"
        )
        if models:
            hello.models.extend(models)
        hello.hostname = socket.gethostname()

        attempt = 0
        while True:
            attempt += 1
            try:
                if self._channel is None or self._stub is None:
                    self._reconnect_channel()
                assert self._stub is not None and self._channel is not None
                grpc.channel_ready_future(self._channel).result(
                    timeout=self._rpc_timeout
                )
                session = self._stub.RegisterLearner(hello, timeout=self._rpc_timeout)
                self._session = session
                LOGGER.info(
                    "Registered learner with coordinator id={} heartbeat_interval={}ms",
                    session.learner_id or "unknown",
                    session.heartbeat_interval_ms,
                )
                return session
            except grpc.FutureTimeoutError as exc:
                LOGGER.warning(
                    "Coordinator channel not ready for RegisterLearner (attempt {} target={} timeout={}s)",
                    attempt,
                    self._target,
                    self._rpc_timeout,
                )
                if self._fail_fast:
                    raise RuntimeError(
                        f"Coordinator channel was not ready within {self._rpc_timeout}s for target {self._target}"
                    ) from exc
                if self._max_retries > 0 and attempt >= self._max_retries:
                    raise RuntimeError(
                        f"Coordinator channel was not ready after {self._max_retries} attempts for target {self._target}"
                    ) from exc
                self._reconnect_channel()
                time.sleep(self._retry_delay)
            except grpc.RpcError as exc:
                LOGGER.warning(
                    "Coordinator RegisterLearner failed (attempt {} target={} code={}): {}",
                    attempt,
                    self._target,
                    exc.code() if hasattr(exc, "code") else None,
                    exc.details() if hasattr(exc, "details") else exc,
                )
                if self._fail_fast:
                    raise RuntimeError(
                        f"Coordinator registration failed: {exc.details() if hasattr(exc, 'details') else exc}"
                    ) from exc
                if self._max_retries > 0 and attempt >= self._max_retries:
                    raise RuntimeError(
                        f"Coordinator registration exceeded {self._max_retries} attempts (last error: "
                        f"{exc.details() if hasattr(exc, 'details') else exc})"
                    ) from exc
                self._reconnect_channel()
                time.sleep(self._retry_delay)

    def _heartbeat_loop(self, interval_seconds: float) -> None:
        """Send periodic heartbeats; retries indefinitely unless fail-fast."""
        next_delay = 0.0
        while not self._stop_event.wait(next_delay):
            try:
                if self._session is None:
                    next_delay = self._retry_delay
                    continue
                if self._stub is None:
                    self._reconnect_channel()
                assert self._stub is not None and self._channel is not None

                heartbeat = coordinator_pb2.LearnerHeartbeat(
                    learner_id=self._session.learner_id,
                    session_token=self._session.session_token,
                    unix_ms=int(time.time() * 1000),
                    endpoint=self._endpoint,
                    version=self._version,
                )
                if self._models:
                    heartbeat.models.extend(self._models)
                metrics = self._collect_metrics()
                if metrics:
                    heartbeat.metrics.update(metrics)
                request = coordinator_pb2.LearnerHeartbeatRequest(heartbeat=heartbeat)
                self._stub.LearnerHeartbeat(request, timeout=self._rpc_timeout)
                self._last_heartbeat_ok_monotonic = time.monotonic()
                if (
                    self._last_heartbeat_log_monotonic == 0.0
                    or self._last_heartbeat_ok_monotonic
                    - self._last_heartbeat_log_monotonic
                    >= 15.0
                ):
                    self._last_heartbeat_log_monotonic = (
                        self._last_heartbeat_ok_monotonic
                    )
                next_delay = interval_seconds
            except grpc.RpcError as exc:
                code = exc.code() if hasattr(exc, "code") else None
                LOGGER.warning(
                    "Coordinator LearnerHeartbeat failed (target={} code={}): {}",
                    self._target,
                    code,
                    exc.details() if hasattr(exc, "details") else exc,
                )
                if code in (grpc.StatusCode.NOT_FOUND, grpc.StatusCode.UNAUTHENTICATED):
                    LOGGER.warning(
                        "Coordinator rejected learner session (code={}); re-registering learner {}",
                        code,
                        self._endpoint or "<unknown>",
                    )
                    try:
                        self._session = self._register(
                            self._endpoint, self._version, self._models
                        )
                        next_delay = 0.0
                        continue
                    except RuntimeError as reg_exc:
                        LOGGER.warning("Learner re-registration failed: {}", reg_exc)
                self._reconnect_channel()
                if self._fail_fast:
                    self._stop_event.set()
                    LOGGER.error(
                        "Learner heartbeat failed with fail-fast enabled: {}",
                        exc.details() if hasattr(exc, "details") else exc,
                    )
                    return
                next_delay = self._retry_delay
            except Exception as exc:  # pragma: no cover - defensive thread hardening
                LOGGER.exception("Learner heartbeat loop crashed unexpectedly: {}", exc)
                self._reconnect_channel()
                if self._fail_fast:
                    self._stop_event.set()
                    return
                next_delay = self._retry_delay

    def _collect_metrics(self) -> dict:
        if self._metrics_supplier is None:
            return {}
        try:
            metrics = self._metrics_supplier()
            return dict(metrics) if isinstance(metrics, dict) else {}
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("metrics supplier failed: {}", exc)
            return {}
