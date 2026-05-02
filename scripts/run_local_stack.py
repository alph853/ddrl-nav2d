#!/usr/bin/env python3
"""Fire-and-forget launcher for the native coordinator, worker, and learner.

You can optionally choose which components to run via ``--service``.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# --- tweak these paths/addresses for your local dev stack ---------------------
ENV_FILE = ".env"
WORKER_BIN = "build/dev/apps/worker/ddrl_worker"
LEARNER_SCRIPT = "apps/learner/main.py"
COORDINATOR_SCRIPT = "apps/coordinator/main.py"
PYTHON = sys.executable
COORDINATOR_HOST = "127.0.0.1"
COORDINATOR_PORT = 10800
LEARNER_BIND = "0.0.0.0:6600"
LEARNER_ADVERTISE = "127.0.0.1:6600"
POLICY_ARCH_CONFIG = "config/policy_arch.yaml"
CHECKPOINT_DIR = "learner_checkpoints"
COORDINATOR_DELAY = 1.0
LEARNER_DELAY = 2.0
# -----------------------------------------------------------------------------#

ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _resolve(project_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (project_root / path)


def _load_env(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not path.exists():
        return env
    base_scope = dict(os.environ)
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :]
            if "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            scope = base_scope.copy()
            scope.update(env)
            env[key.strip()] = ENV_PATTERN.sub(
                lambda m: scope.get(m.group(1), ""),
                raw_value.strip().strip('"').strip("'"),
            )
    return env


def _spawn(
    name: str, cmd: list[str], cwd: Path, env: Dict[str, str]
) -> subprocess.Popen:
    print(f"[launch] starting {name}: {' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=str(cwd), env=env, start_new_session=True)


def _terminate(proc: Optional[subprocess.Popen], name: str) -> None:
    if not proc or proc.poll() is not None:
        return
    print(f"[launch] stopping {name} ...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print(f"[launch] force killing {name}")
        proc.kill()


def _wait_for_tcp_ready(host: str, port: int, timeout_s: float, label: str) -> None:
    deadline = time.monotonic() + max(timeout_s, 0.1)
    last_error: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                print(f"[launch] {label} is accepting TCP connections at {host}:{port}")
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.2)
    raise RuntimeError(
        f"{label} did not become ready at {host}:{port} within {timeout_s:.1f}s"
        + (f" (last error: {last_error})" if last_error is not None else "")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch local DDRL native stack (coordinator, worker, learner) "
            "or a subset via --service."
        )
    )
    parser.add_argument(
        "--service",
        "-s",
        choices=("all", "coordinator", "worker", "learner"),
        default="all",
        help=(
            "Which component(s) to launch: 'all' (default), 'coordinator', "
            "'worker', or 'learner'."
        ),
    )
    parser.add_argument(
        "--worker-identity",
        "-wi",
        type=str,
        default="",
        help="Explicit worker identity (set per instance when running multiple workers on the same host).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env.update(_load_env(_resolve(project_root, ENV_FILE)))

    coordinator_host = env.get("DDRL_COORDINATOR_BIND_HOST", COORDINATOR_HOST)
    coordinator_port = int(env.get("DDRL_COORDINATOR_BIND_PORT", COORDINATOR_PORT))
    coordinator_target = env.get(
        "DDRL_COORDINATOR_TARGET", f"{coordinator_host}:{coordinator_port}"
    )

    env["COORDINATOR_HOST"] = coordinator_host
    env["COORDINATOR_PORT"] = str(coordinator_port)
    env["DDRL_COORDINATOR_TARGET"] = coordinator_target
    learner_bind = env.get("DDRL_LEARNER_GRPC_BIND", LEARNER_BIND)
    learner_advertise = env.get("DDRL_LEARNER_GRPC_ADVERTISE", LEARNER_ADVERTISE)
    env["DDRL_WORKER_GRPC_TARGET"] = learner_bind

    worker_bin = _resolve(project_root, WORKER_BIN)
    learner_script = _resolve(project_root, LEARNER_SCRIPT)
    coordinator_script = _resolve(project_root, COORDINATOR_SCRIPT)
    policy_arch_config = _resolve(project_root, POLICY_ARCH_CONFIG)
    checkpoint_dir = _resolve(project_root, CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    worker_cmd = [
        str(worker_bin),
        "--coordinator-target",
        coordinator_target,
    ]

    if args.worker_identity:
        worker_cmd += ["--worker-identity", args.worker_identity]

    coordinator_cmd = [
        PYTHON,
        str(coordinator_script),
        "--host",
        coordinator_host,
        "--port",
        str(coordinator_port),
    ]

    learner_device = env.get("DDRL_LEARNER_DEVICE", "auto")
    learner_cmd = [
        PYTHON,
        str(learner_script),
        "--grpc-bind",
        learner_bind,
        "--grpc-advertise",
        learner_advertise,
        "--policy-arch-config",
        str(policy_arch_config),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--coordinator-target",
        coordinator_target,
        "--device",
        learner_device,
    ]

    print("========================================")
    print("DDRL Native Stack Launcher")
    print("========================================")
    print(f"Project root:     {project_root}")
    print(f"Worker binary:    {worker_bin}")
    if args.service in ("all", "coordinator"):
        print(f"Coordinator:      {coordinator_script} ({coordinator_host}:{coordinator_port})")
    print(f"Coordinator tgt:  {coordinator_target}")
    print(f"Learner script:   {learner_script}")
    print(f"Learner bind ->   {learner_bind}")
    print(f"Learner advert -> {learner_advertise}")
    print(f"Service mode:     {args.service}")
    print("========================================")

    coordinator_proc = worker_proc = learner_proc = None
    try:
        # Launch components based on selected service mode.
        if args.service in ("all", "coordinator"):
            coordinator_proc = _spawn("coordinator", coordinator_cmd, project_root, env)
            _wait_for_tcp_ready(coordinator_host, coordinator_port, max(COORDINATOR_DELAY, 1.0), "coordinator")

        if args.service in ("all", "learner"):
            if args.service == "all":
                time.sleep(max(LEARNER_DELAY, 0.0))
            learner_proc = _spawn("learner", learner_cmd, project_root, env)

        if args.service in ("all", "worker"):
            worker_proc = _spawn("worker", worker_cmd, project_root, env)

        # Poll whichever processes were started; exit when the "main" one ends.
        # For single-service modes, this collapses to just that process.
        while True:
            worker_rc = worker_proc.poll() if worker_proc else None
            learner_rc = learner_proc.poll() if learner_proc else None

            if worker_proc and worker_rc is not None:
                print(f"[launch] worker exited with code {worker_rc}")
                if learner_proc and learner_proc.poll() is None:
                    _terminate(learner_proc, "learner")
                if coordinator_proc and coordinator_proc.poll() is None:
                    _terminate(coordinator_proc, "coordinator")
                return worker_rc

            if learner_proc and learner_rc is not None:
                print(f"[launch] learner exited with code {learner_rc}")
                if worker_proc and worker_proc.poll() is None:
                    _terminate(worker_proc, "worker")
                if coordinator_proc and coordinator_proc.poll() is None:
                    _terminate(coordinator_proc, "coordinator")
                return learner_rc

            # If only coordinator was launched, just wait on it.
            if not worker_proc and not learner_proc and coordinator_proc:
                rc = coordinator_proc.poll()
                if rc is not None:
                    print(f"[launch] coordinator exited with code {rc}")
                    return rc

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[launch] Ctrl-C received, shutting down...")
        return 130
    finally:
        _terminate(learner_proc, "learner")
        _terminate(worker_proc, "worker")
        _terminate(coordinator_proc, "coordinator")


if __name__ == "__main__":
    raise SystemExit(main())
