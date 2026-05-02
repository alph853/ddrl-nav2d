#!/usr/bin/env python3
"""Wrapper for multi-machine demos (advertise host IP for remote workers)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch local coordinator/learner/worker with learner advertising a "
            "reachable host IP for remote workers."
        )
    )
    parser.add_argument(
        "--host-ip",
        default="192.168.1.191",
        help="Host IP address that remote workers can reach (used for advertising).",
    )
    parser.add_argument(
        "--coordinator-port",
        type=int,
        default=10700,
        help="Coordinator gRPC port to bind and advertise.",
    )
    parser.add_argument(
        "--learner-port",
        type=int,
        default=6600,
        help="Learner gRPC port to bind and advertise.",
    )
    parser.add_argument(
        "--service",
        "-s",
        choices=("all", "coordinator", "worker", "learner"),
        default="all",
        help="Which component(s) to launch (passed to run_local_stack.py).",
    )
    parser.add_argument(
        "--worker-identity",
        "-wi",
        type=str,
        default="",
        help="Explicit worker identity (passed to run_local_stack.py).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    env = os.environ.copy()
    env["DDRL_COORDINATOR_BIND_HOST"] = "0.0.0.0"
    env["DDRL_COORDINATOR_BIND_PORT"] = str(args.coordinator_port)
    env["DDRL_COORDINATOR_TARGET"] = f"{args.host_ip}:{args.coordinator_port}"
    env["DDRL_LEARNER_GRPC_BIND"] = f"0.0.0.0:{args.learner_port}"
    env["DDRL_LEARNER_GRPC_ADVERTISE"] = f"{args.host_ip}:{args.learner_port}"

    script_path = Path(__file__).resolve().parent / "run_local_stack.py"
    cmd = [sys.executable, str(script_path), "--service", args.service]
    if args.worker_identity:
        cmd += ["--worker-identity", args.worker_identity]

    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
