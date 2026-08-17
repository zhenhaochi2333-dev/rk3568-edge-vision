#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-${SOURCE_ROOT}/build-native}"
RKNN_INCLUDE="${SOURCE_ROOT}/deps/rknn_runtime_1.6.0/include"
RKNN_LIBRARY="${SOURCE_ROOT}/deps/rknn_runtime_1.6.0/lib/aarch64/librknnrt.so"

if [[ ! -f "${RKNN_INCLUDE}/rknn_api.h" ]]; then
    echo 'Project-local rknn_api.h is missing; stage the verified RKNN 1.6.0 SDK first.' >&2
    exit 1
fi
if [[ ! -f "${RKNN_LIBRARY}" ]]; then
    if [[ -f /lib/librknnrt.so ]] && file /lib/librknnrt.so | grep -q 'ARM aarch64'; then
        mkdir -p "$(dirname "${RKNN_LIBRARY}")"
        cp /lib/librknnrt.so "${RKNN_LIBRARY}"
    else
        echo 'No verified ARM64 RKNN runtime available for native build.' >&2
        exit 1
    fi
fi

# Runtime deployment is project-local. The executable RPATH is $ORIGIN/lib;
# never replace the board's /lib or /usr/lib installation.
mkdir -p "${SOURCE_ROOT}/lib"
cp "${RKNN_LIBRARY}" "${SOURCE_ROOT}/lib/librknnrt.so"

cmake -S "${SOURCE_ROOT}" -B "${BUILD_ROOT}" \
    -DRKNN_INCLUDE_DIR="${RKNN_INCLUDE}" \
    -DRKNN_LIBRARY="${RKNN_LIBRARY}" \
    -DEDGEVISION_WITH_VIDEO=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_ROOT}" -- -j2
file "${BUILD_ROOT}/edge_vision"
