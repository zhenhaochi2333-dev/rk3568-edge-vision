#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK_ROOT="${RKNN_SDK_ROOT:-${HOME}/rk3568_sdk/rknn_runtime_1.6.0}"
DEST_ROOT="${SOURCE_ROOT}/deps/rknn_runtime_1.6.0"

if [[ ! -f "${SDK_ROOT}/include/rknn_api.h" ]]; then
    echo "Missing verified RKNN header: ${SDK_ROOT}/include/rknn_api.h" >&2
    exit 1
fi

mkdir -p "${DEST_ROOT}/include" "${DEST_ROOT}/lib/aarch64"
if [[ "$(readlink -f "${SDK_ROOT}/include/rknn_api.h")" != "$(readlink -f "${DEST_ROOT}/include/rknn_api.h")" ]]; then
    cp "${SDK_ROOT}/include/rknn_api.h" "${DEST_ROOT}/include/rknn_api.h"
fi

if [[ -f "${SDK_ROOT}/lib/aarch64/librknnrt.so" ]]; then
    if [[ "$(readlink -f "${SDK_ROOT}/lib/aarch64/librknnrt.so")" != "$(readlink -f "${DEST_ROOT}/lib/aarch64/librknnrt.so")" ]]; then
        cp "${SDK_ROOT}/lib/aarch64/librknnrt.so" "${DEST_ROOT}/lib/aarch64/librknnrt.so"
    fi
elif [[ -f /lib/librknnrt.so ]]; then
    file /lib/librknnrt.so | grep -q 'ARM aarch64' || {
        echo 'Existing /lib/librknnrt.so is not an ARM64 ELF' >&2
        exit 1
    }
    cp /lib/librknnrt.so "${DEST_ROOT}/lib/aarch64/librknnrt.so"
else
    echo 'No matching ARM64 librknnrt.so found' >&2
    exit 1
fi

file "${DEST_ROOT}/lib/aarch64/librknnrt.so"
echo "Staged RKNN 1.6.0 dependencies under ${DEST_ROOT}"
