from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loguru import logger
from proto import coordinator_pb2


def now_ms() -> int:
    return int(time.time() * 1000)


LOGGER = logger.bind(logger_name="ddrl.coordinator")


@dataclass
class WorkerRecord:
    worker_id: str
    session_token: str
    hostname: str
    version: str
    labels: Dict[str, str]
    capabilities: Dict[str, str]
    last_heartbeat_ms: int
    status: coordinator_pb2.WorkerStatus = field(default_factory=coordinator_pb2.WorkerStatus)
    metrics: Dict[str, float] = field(default_factory=dict)
    directive_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=8))


@dataclass
class LearnerRecord:
    learner_id: str
    session_token: str
    hostname: str
    endpoint: str
    version: str
    models: List[str]
    last_heartbeat_ms: int
    metrics: Dict[str, float] = field(default_factory=dict)


class CoordinatorState:
    """In-memory registry for workers, learners, and control directives."""

    def __init__(self) -> None:
        self._worker_seq = 0
        self._learner_seq = 0
        self._workers: Dict[str, WorkerRecord] = {}
        self._learners: Dict[str, LearnerRecord] = {}
        self._active_learner_id: Optional[str] = None

    # workers
    def _find_worker_match(self, hello: coordinator_pb2.WorkerHello) -> Optional[WorkerRecord]:
        preferred_identity = hello.labels.get("identity") or hello.labels.get("worker_identity")
        if preferred_identity:
            for record in self._workers.values():
                labels = record.labels or {}
                if labels.get("identity") == preferred_identity or labels.get("worker_identity") == preferred_identity:
                    return record

        hostname = hello.hostname
        if hostname:
            for record in self._workers.values():
                if record.hostname == hostname:
                    return record
        return None

    def register_worker(
        self,
        hello: coordinator_pb2.WorkerHello,
        heartbeat_interval_ms: int,
    ) -> coordinator_pb2.WorkerSession:
        session_token = secrets.token_hex(16)
        hostname = hello.hostname or "unknown"
        existing = self._find_worker_match(hello)

        if existing is None:
            self._worker_seq += 1
            worker_id = f"worker-{self._worker_seq:04d}"
            record = WorkerRecord(
                worker_id=worker_id,
                session_token=session_token,
                hostname=hostname,
                version=hello.version or "unknown",
                labels=dict(hello.labels),
                capabilities={cap.name: cap.value for cap in hello.capabilities if cap.name},
                last_heartbeat_ms=now_ms(),
            )
            self._workers[worker_id] = record
        else:
            worker_id = existing.worker_id
            existing.session_token = session_token
            existing.hostname = hostname
            existing.version = hello.version or "unknown"
            existing.labels = dict(hello.labels)
            existing.capabilities = {cap.name: cap.value for cap in hello.capabilities if cap.name}
            existing.last_heartbeat_ms = now_ms()
            existing.status = coordinator_pb2.WorkerStatus()
            existing.metrics = {}
            record = existing
            LOGGER.info("reusing worker_id={} for reconnecting host={}", worker_id, hostname)

        return coordinator_pb2.WorkerSession(
            worker_id=worker_id,
            session_token=session_token,
            learner_target=self.current_learner_endpoint(),
            heartbeat_interval_ms=heartbeat_interval_ms,
        )

    def validate_worker_session(self, worker_id: str, session_token: str) -> WorkerRecord:
        record = self._workers.get(worker_id)
        if record is None:
            raise KeyError(f"worker {worker_id} is not registered")
        if record.session_token != session_token:
            raise PermissionError("invalid session token")
        return record

    def record_worker_heartbeat(self, heartbeat: coordinator_pb2.WorkerHeartbeat) -> WorkerRecord:
        record = self.validate_worker_session(heartbeat.worker_id, heartbeat.session_token)
        record.last_heartbeat_ms = heartbeat.unix_ms or now_ms()
        if heartbeat.HasField("status"):
            record.status.CopyFrom(heartbeat.status)
        record.metrics = dict(heartbeat.metrics)
        return record

    def mark_worker_disconnected(self, worker_id: str, detail: Optional[str] = None) -> None:
        record = self._workers.get(worker_id)
        if record is None:
            return
        was_offline = record.status.phase == "offline"
        record.last_heartbeat_ms = now_ms()
        record.status.phase = "offline"
        record.status.detail = detail or "heartbeat stream closed"
        if not was_offline:
            online = sum(
                1 for rec in self._workers.values() if rec.status.phase != "offline"
            )
            LOGGER.info(
                "worker {} marked offline (reason={}); online_workers={}/{}",
                worker_id,
                record.status.detail,
                online,
                len(self._workers),
            )

    def push_directive(
        self,
        worker_id: str,
        directive: coordinator_pb2.ControlDirective,
    ) -> None:
        record = self._workers.get(worker_id)
        if record is None:
            return
        queue = record.directive_queue
        while queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        queue.put_nowait(directive)

    def push_directive_to_all(self, directive: coordinator_pb2.ControlDirective) -> None:
        for worker_id in list(self._workers.keys()):
            self.push_directive(worker_id, directive)

    def _push_switch_learner_directive(self, learner_target: str) -> None:
        directive = coordinator_pb2.ControlDirective(
            type=coordinator_pb2.CONTROL_SWITCH_LEARNER,
            message=f"Switch to learner at {learner_target or 'none'}",
            learner_target=learner_target,
        )
        self.push_directive_to_all(directive)

    def cleanup_stale_workers(self, timeout_seconds: float) -> None:
        if timeout_seconds <= 0:
            return
        threshold_ms = now_ms() - int(timeout_seconds * 1000)
        stale_workers = [wid for wid, rec in self._workers.items() if rec.last_heartbeat_ms < threshold_ms and rec.status.phase != "offline"]
        if not stale_workers:
            return
        for worker_id in stale_workers:
            LOGGER.warning("worker {} considered stale after {:.1f}s without heartbeat", worker_id, timeout_seconds)
            self.mark_worker_disconnected(worker_id, detail="heartbeat timed out")

    # learner
    def _find_learner_match(self, hello: coordinator_pb2.LearnerHello) -> Optional[LearnerRecord]:
        for record in self._learners.values():
            if record.endpoint == hello.endpoint and record.endpoint:
                return record
            if hello.hostname and record.hostname == hello.hostname:
                return record
        return None

    def _update_active_learner(self, learner_id: Optional[str] = None) -> Optional[LearnerRecord]:
        if learner_id is not None and learner_id in self._learners:
            self._active_learner_id = learner_id
        elif self._active_learner_id not in self._learners:
            self._active_learner_id = None

        current = self._learners.get(self._active_learner_id) if self._active_learner_id is not None else None
        if current is not None and current.endpoint:
            return current

        if self._learners:
            candidates = [record for record in self._learners.values() if record.endpoint]
            if candidates:
                newest = max(candidates, key=lambda record: record.last_heartbeat_ms)
                self._active_learner_id = newest.learner_id
            elif self._active_learner_id is None:
                newest = max(self._learners.values(), key=lambda record: record.last_heartbeat_ms)
                self._active_learner_id = newest.learner_id

        if self._active_learner_id is None:
            return None
        return self._learners.get(self._active_learner_id)

    def current_learner_endpoint(self) -> str:
        active = self._update_active_learner()
        return active.endpoint if active else ""

    def register_learner(
        self,
        hello: coordinator_pb2.LearnerHello,
        heartbeat_interval_ms: int,
    ) -> LearnerRecord:
        session_token = secrets.token_hex(16)
        previous_endpoint = self.current_learner_endpoint()
        existing = self._find_learner_match(hello)

        if existing is None:
            self._learner_seq += 1
            learner_id = f"learner-{self._learner_seq:04d}"
            record = LearnerRecord(
                learner_id=learner_id,
                session_token=session_token,
                hostname=hello.hostname or "unknown",
                endpoint=hello.endpoint,
                version=hello.version or "unknown",
                models=list(hello.models),
                last_heartbeat_ms=now_ms(),
            )
            self._learners[learner_id] = record
        else:
            existing.session_token = session_token
            existing.hostname = hello.hostname or existing.hostname
            existing.endpoint = hello.endpoint
            existing.version = hello.version or "unknown"
            existing.models = list(hello.models)
            existing.last_heartbeat_ms = now_ms()
            existing.metrics = {}
            record = existing
            LOGGER.info("reusing learner_id={} for endpoint={}", record.learner_id, record.endpoint)

        active_record = self._update_active_learner(record.learner_id)

        if previous_endpoint != self.current_learner_endpoint():
            self._push_switch_learner_directive(self.current_learner_endpoint())
        elif active_record is not None and previous_endpoint != active_record.endpoint:
            self._push_switch_learner_directive(active_record.endpoint)

        return record

    def validate_learner_session(self, learner_id: str, session_token: str) -> LearnerRecord:
        record = self._learners.get(learner_id)
        if record is None:
            raise KeyError(f"learner {learner_id} is not registered")
        if record.session_token != session_token:
            raise PermissionError("invalid session token")
        return record

    def record_learner_heartbeat(
        self, request: coordinator_pb2.LearnerHeartbeatRequest
    ) -> LearnerRecord:
        if not request.HasField("heartbeat"):
            raise ValueError("learner heartbeat payload is missing")
        heartbeat = request.heartbeat
        previous_endpoint = self.current_learner_endpoint()
        record = self.validate_learner_session(heartbeat.learner_id, heartbeat.session_token)

        record.last_heartbeat_ms = heartbeat.unix_ms or now_ms()
        if heartbeat.endpoint:
            record.endpoint = heartbeat.endpoint
        if heartbeat.version:
            record.version = heartbeat.version
        if heartbeat.models:
            record.models = list(heartbeat.models)
        record.metrics = dict(heartbeat.metrics)

        self._update_active_learner(record.learner_id)

        if previous_endpoint != self.current_learner_endpoint():
            self._push_switch_learner_directive(self.current_learner_endpoint())

        return record

    def cleanup_stale_learners(self, timeout_seconds: float) -> None:
        if timeout_seconds <= 0 or not self._learners:
            return

        threshold_ms = now_ms() - int(timeout_seconds * 1000)
        stale_learner_ids = [lid for lid, rec in self._learners.items() if rec.last_heartbeat_ms < threshold_ms]
        if not stale_learner_ids:
            return

        previous_endpoint = self.current_learner_endpoint()
        for learner_id in stale_learner_ids:
            record = self._learners.pop(learner_id, None)
            if record is not None:
                LOGGER.warning(
                    "learner {} considered stale after {:.1f}s without heartbeat",
                    learner_id,
                    timeout_seconds,
                )

        self._update_active_learner()

        if previous_endpoint != self.current_learner_endpoint():
            self._push_switch_learner_directive(self.current_learner_endpoint())
