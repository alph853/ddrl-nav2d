#!/usr/bin/env python3
"""Generate gRPC bindings for DDRL services."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parent.parent
PROTO_DIR = ROOT_DIR / "apps" / "comm" / "proto"
CPP_OUT = ROOT_DIR / "apps" / "worker" / "proto"
PY_OUT = ROOT_DIR / "apps" / "learner" / "proto"
COOR_OUT = ROOT_DIR / "apps" / "coordinator" / "proto"
PY_PROTO_FILES: tuple[str, ...] = (
    "rollouts.proto",
    "policy.proto",
    "coordinator.proto",
)
CPP_PROTO_FILES: tuple[str, ...] = (
    "policy.proto",
    "rollouts.proto",
    "coordinator.proto",
)
SERVICE_CHOICES = ("worker", "learner", "coordinator", "all")


def run_command(command: Iterable[str]) -> None:
    """Run a subprocess command, surfacing any failure immediately."""
    subprocess.run(list(command), check=True)


def find_grpc_cpp_plugin() -> str:
    plugin = shutil.which("grpc_cpp_plugin")
    if plugin:
        return plugin
    raise RuntimeError(
        "Unable to locate grpc_cpp_plugin on PATH. "
        "Ensure gRPC C++ plugins are installed and available."
    )


def generate_worker(plugin_path: str) -> None:
    CPP_OUT.mkdir(parents=True, exist_ok=True)
    for proto in CPP_PROTO_FILES:
        run_command(
            (
                "protoc",
                "-I",
                str(PROTO_DIR),
                "--cpp_out",
                str(CPP_OUT),
                "--grpc_out",
                str(CPP_OUT),
                f"--plugin=protoc-gen-grpc={plugin_path}",
                str(PROTO_DIR / proto),
            )
        )


def generate_python_bindings(output_dir: Path, proto_files: Iterable[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for proto in proto_files:
        run_command(
            (
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                f"--proto_path={PROTO_DIR}",
                f"--python_out={output_dir}",
                f"--pyi_out={output_dir}",
                f"--grpc_python_out={output_dir}",
                str(PROTO_DIR / proto),
            )
        )


def generate_learner() -> None:
    generate_python_bindings(PY_OUT, PY_PROTO_FILES)
    fix_python_relative_imports(PY_OUT)


def generate_coordinator() -> None:
    generate_python_bindings(COOR_OUT, ("coordinator.proto",))
    fix_python_relative_imports(COOR_OUT)


def fix_python_relative_imports(target_dir: Path) -> None:
    if not target_dir.exists():
        return

    local_modules = {path.stem for path in target_dir.glob("*_pb2*.py")}

    for file_path in target_dir.glob("*_pb2*.py"):
        lines = file_path.read_text().splitlines()
        changed = False
        rewritten: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import ") and " as " in stripped:
                module_part, alias = stripped[len("import ") :].split(" as ", maxsplit=1)
                module = module_part.strip()
                alias = alias.strip()
                if module in local_modules and not module.startswith("."):
                    line = f"from . import {module} as {alias}"
                    changed = True
            rewritten.append(line)
        if changed:
            file_path.write_text("\n".join(rewritten) + "\n")

    pycache_dir = target_dir / "__pycache__"
    if pycache_dir.exists():
        shutil.rmtree(pycache_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate gRPC bindings for worker, learner, and coordinator services.",
    )
    parser.add_argument(
        "--service",
        "-s",
        choices=SERVICE_CHOICES,
        default="all",
        help="Select which service's bindings to regenerate (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    grpc_cpp_plugin = None
    if args.service in {"all", "worker"}:
        grpc_cpp_plugin = find_grpc_cpp_plugin()
        generate_worker(grpc_cpp_plugin)

    if args.service in {"all", "learner"}:
        generate_learner()

    if args.service in {"all", "coordinator"}:
        generate_coordinator()


if __name__ == "__main__":
    main()
