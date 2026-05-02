#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ONNXRUNTIME_VERSION="${ONNXRUNTIME_VERSION:-1.23.2}"
ONNXRUNTIME_ARCHIVE="onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz"
ONNXRUNTIME_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_ARCHIVE}"

if ! command -v apt-get >/dev/null 2>&1; then
  echo "This installer currently targets Ubuntu/Debian systems with apt-get." >&2
  exit 1
fi

SUDO=()
if [[ "${EUID}" -ne 0 ]]; then
  SUDO=(sudo)
fi

"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  cmake \
  curl \
  git \
  libcli11-dev \
  libgl1-mesa-dev \
  libglu1-mesa-dev \
  libgrpc++-dev \
  libprotobuf-dev \
  libspdlog-dev \
  libvtk9-dev \
  libx11-dev \
  libxcursor-dev \
  libxi-dev \
  libxmu-dev \
  libxrandr-dev \
  libyaml-cpp-dev \
  ninja-build \
  pkg-config \
  protobuf-compiler \
  protobuf-compiler-grpc \
  qtbase5-dev \
  qttools5-dev \
  qttools5-dev-tools \
  libqt5opengl5-dev
"${SUDO[@]}" rm -rf /var/lib/apt/lists/*

mkdir -p "${PROJECT_ROOT}/third_party"
curl -L "${ONNXRUNTIME_URL}" -o "/tmp/${ONNXRUNTIME_ARCHIVE}"
rm -rf "${PROJECT_ROOT}/third_party/onnxruntime"
tar -xzf "/tmp/${ONNXRUNTIME_ARCHIVE}" -C "${PROJECT_ROOT}/third_party"
mv \
  "${PROJECT_ROOT}/third_party/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}" \
  "${PROJECT_ROOT}/third_party/onnxruntime"

echo "Installed native build dependencies and ONNX Runtime ${ONNXRUNTIME_VERSION}."
