#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-${SOURCE_ROOT}/build-aarch64}"
RKNN_SDK_ROOT="${RKNN_SDK_ROOT:-${SOURCE_ROOT}/deps/rknn_runtime_1.6.0}"
OPENCV_ROOT="${EDGEVISION_OPENCV_ROOT:?Set EDGEVISION_OPENCV_ROOT to the verified ARM64 OpenCV root}"

"${SOURCE_ROOT}/scripts/stage_rknn_deps.sh"
cmake -S "${SOURCE_ROOT}" -B "${BUILD_ROOT}" \
    -DCMAKE_TOOLCHAIN_FILE="${SOURCE_ROOT}/cmake/toolchain-aarch64.cmake" \
    -DEDGEVISION_OPENCV_ROOT="${OPENCV_ROOT}" \
    -DRKNN_INCLUDE_DIR="${RKNN_SDK_ROOT}/include" \
    -DRKNN_LIBRARY="${RKNN_SDK_ROOT}/lib/aarch64/librknnrt.so" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_ROOT}" -- -j2
file "${BUILD_ROOT}/edge_vision"
