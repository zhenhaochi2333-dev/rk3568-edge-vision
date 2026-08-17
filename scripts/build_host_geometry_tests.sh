#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT="${1:-/tmp/edgevision_host_geometry_tests}"
OPENCV_FLAGS="$(pkg-config --cflags --libs opencv)"

g++ -std=c++17 -Wall -Wextra -Wpedantic -I"${SOURCE_ROOT}/include" \
    "${SOURCE_ROOT}/src/display_composer.cpp" \
    "${SOURCE_ROOT}/src/cli_parser.cpp" \
    "${SOURCE_ROOT}/src/image_processor.cpp" \
    "${SOURCE_ROOT}/src/iou_tracker.cpp" \
    "${SOURCE_ROOT}/src/region_monitor.cpp" \
    "${SOURCE_ROOT}/tests/test_display_composer.cpp" \
    "${SOURCE_ROOT}/tests/test_geometry.cpp" \
    "${SOURCE_ROOT}/tests/test_host_geometry_main.cpp" \
    "${SOURCE_ROOT}/tests/test_iou_tracker.cpp" \
    "${SOURCE_ROOT}/tests/test_region_monitor.cpp" \
    ${OPENCV_FLAGS} \
    -o "${OUTPUT}"

"${OUTPUT}"

CLI_OUTPUT="${OUTPUT}_cli"
g++ -std=c++17 -Wall -Wextra -Wpedantic -I"${SOURCE_ROOT}/include" \
    "${SOURCE_ROOT}/src/cli_parser.cpp" \
    "${SOURCE_ROOT}/tests/test_host_cli.cpp" \
    -o "${CLI_OUTPUT}"
"${CLI_OUTPUT}"
