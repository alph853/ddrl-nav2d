from __future__ import annotations

import socket
from typing import Sequence

from loguru import logger

LOGGER = logger.bind(logger_name="ddrl.coordinator")
CONNECTED = False

def ensure_kafka_cluster_active(bootstrap: str, timeout: float = 3.0) -> None:
    """Raise RuntimeError if none of the bootstrap brokers accepts a TCP connection."""
    endpoints: Sequence[str] = [entry.strip() for entry in bootstrap.split(",") if entry.strip()]
    if not endpoints:
        raise RuntimeError("no Kafka bootstrap servers were provided")

    failures = []
    for endpoint in endpoints:
        try:
            host, port_str = endpoint.rsplit(":", 1)
            port = int(port_str)
        except ValueError:
            failures.append(f"{endpoint} (invalid host:port)")
            continue

        try:
            with socket.create_connection((host, port), timeout=timeout):
                global CONNECTED
                if not CONNECTED:
                    LOGGER.info("connected to Kafka bootstrap broker {}", endpoint)
                CONNECTED = True
                return
        except OSError as exc:  # pragma: no cover - best effort logging
            failures.append(f"{endpoint} ({exc})")

    failure_text = "; ".join(failures)
    raise RuntimeError(f"Kafka cluster is not reachable via bootstrap '{bootstrap}': {failure_text}")
