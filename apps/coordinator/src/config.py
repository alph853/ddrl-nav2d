from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoordinatorConfig:
    host: str
    port: int
    kafka_bootstrap: str
    worker_heartbeat_interval: float
    worker_heartbeat_timeout: float
    learner_heartbeat_interval: float
    enable_kafka_healthcheck: bool = False

    @property
    def listen_address(self) -> str:
        return f"{self.host}:{self.port}"
