cmake_minimum_required(VERSION 3.10)

project(edgevision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(EDGEVISION_BUILD_TESTS "Build lightweight Phase 0-5 tests" ON)
option(EDGEVISION_WITH_VIDEO "Build VideoIO and video pipeline support" ON)
option(EDGEVISION_WITH_RTSP "Build the GStreamer RTSP detection stream" OFF)
option(EDGEVISION_BUILD_BENCHMARKS "Build standalone RKNN timing tools" OFF)
option(EDGEVISION_BUILD_PC_BRIDGE "Build the Windows C++ two-window PC bridge only" OFF)
option(EDGEVISION_BUILD_PC_EVENT_LOGGER "Build the Windows lightweight TCP event logger only" OFF)

if(EDGEVISION_BUILD_PC_BRIDGE OR EDGEVISION_BUILD_PC_EVENT_LOGGER)
    if(NOT WIN32)
        message(FATAL_ERROR "PC tools are Windows-only targets")
    endif()
    if(EDGEVISION_BUILD_PC_BRIDGE)
        add_executable(edgevision_pc_bridge tools/pc_two_window_bridge.cpp)
    endif()
    if(EDGEVISION_BUILD_PC_EVENT_LOGGER)
        add_executable(edgevision_event_logger tools/pc_event_logger.cpp)
        target_link_libraries(edgevision_event_logger PRIVATE ws2_32)
    endif()
    if(MSVC)
        if(TARGET edgevision_pc_bridge)
            target_compile_options(edgevision_pc_bridge PRIVATE /W4)
        endif()
        if(TARGET edgevision_event_logger)
            target_compile_options(edgevision_event_logger PRIVATE /W4)
        endif()
    else()
        if(TARGET edgevision_pc_bridge)
            target_compile_options(edgevision_pc_bridge PRIVATE -Wall -Wextra -Wpedantic)
        endif()
        if(TARGET edgevision_event_logger)
            target_compile_options(edgevision_event_logger PRIVATE -Wall -Wextra -Wpedantic)
        endif()
    endif()
    return()
endif()
set(RKNN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/rknn_runtime_1.6.0/include" CACHE PATH
    "Directory containing rknn_api.h")
set(RKNN_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/deps/rknn_runtime_1.6.0/lib/aarch64/librknnrt.so" CACHE FILEPATH
    "Matching ARM64 librknnrt.so")

if(NOT EXISTS "${RKNN_INCLUDE_DIR}/rknn_api.h")
    message(FATAL_ERROR "Missing ${RKNN_INCLUDE_DIR}/rknn_api.h; run scripts/stage_rknn_deps.sh")
endif()
if(NOT EXISTS "${RKNN_LIBRARY}")
    message(FATAL_ERROR "Missing ${RKNN_LIBRARY}; stage the verified ARM64 RKNN runtime")
endif()

set(EDGEVISION_OPENCV_COMPONENTS core imgproc imgcodecs)
if(EDGEVISION_WITH_VIDEO)
    list(APPEND EDGEVISION_OPENCV_COMPONENTS videoio highgui)
endif()
find_package(OpenCV REQUIRED COMPONENTS ${EDGEVISION_OPENCV_COMPONENTS})
find_package(Threads REQUIRED)

set(EDGEVISION_SOURCES
    src/application.cpp
    src/cli_parser.cpp
    src/display_composer.cpp
    src/image_processor.cpp
    src/iou_tracker.cpp
    src/label_loader.cpp
    src/logger.cpp
    src/network_camera_source.cpp
    src/perf_monitor.cpp
    src/region_monitor.cpp
    src/rknn_model.cpp
    src/semantic_stabilizer.cpp
    src/tcp_server.cpp
    src/yolo11_detector.cpp
)
if(EDGEVISION_WITH_VIDEO)
    list(APPEND EDGEVISION_SOURCES
        src/camera_source.cpp
        src/video_io.cpp
    )
endif()
if(EDGEVISION_WITH_RTSP)
    if(NOT EDGEVISION_WITH_VIDEO)
        message(FATAL_ERROR "EDGEVISION_WITH_RTSP requires EDGEVISION_WITH_VIDEO")
    endif()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GST_APP REQUIRED IMPORTED_TARGET gstreamer-app-1.0)
    pkg_check_modules(GST_RTSP REQUIRED IMPORTED_TARGET gstreamer-rtsp-server-1.0)
    list(APPEND EDGEVISION_SOURCES src/rtsp_streamer.cpp)
endif()

add_library(edgevision_core STATIC ${EDGEVISION_SOURCES})
target_include_directories(edgevision_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${RKNN_INCLUDE_DIR}"
        "${OpenCV_INCLUDE_DIRS}"
)
target_link_libraries(edgevision_core PUBLIC ${OpenCV_LIBS} "${RKNN_LIBRARY}" Threads::Threads)
target_compile_options(edgevision_core PRIVATE -Wall -Wextra -Wpedantic)
if(EDGEVISION_WITH_VIDEO)
    target_compile_definitions(edgevision_core PUBLIC EDGEVISION_WITH_VIDEO=1)
endif()
if(EDGEVISION_WITH_RTSP)
    target_compile_definitions(edgevision_core PUBLIC EDGEVISION_WITH_RTSP=1)
    target_link_libraries(edgevision_core PUBLIC PkgConfig::GST_APP PkgConfig::GST_RTSP)
    if(UNIX)
        find_package(X11 REQUIRED)
        target_compile_definitions(edgevision_core PRIVATE EDGEVISION_WITH_X11_THREADS=1)
        target_link_libraries(edgevision_core PUBLIC X11::X11)
    endif()
endif()

add_executable(edge_vision src/main.cpp)
target_link_libraries(edge_vision PRIVATE edgevision_core)
set_target_properties(edge_vision PROPERTIES
    INSTALL_RPATH "\$ORIGIN/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

if(EDGEVISION_BUILD_BENCHMARKS)
    add_executable(edgevision_rknn_benchmark
        tools/rknn_run_benchmark.cpp
        src/rknn_model.cpp
        src/logger.cpp
    )
    target_include_directories(edgevision_rknn_benchmark
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${RKNN_INCLUDE_DIR}"
            "${OpenCV_INCLUDE_DIRS}"
    )
    target_link_libraries(edgevision_rknn_benchmark PRIVATE "${RKNN_LIBRARY}")
    target_compile_options(edgevision_rknn_benchmark PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(EDGEVISION_BUILD_TESTS)
    enable_testing()
    add_executable(edgevision_tests
        tests/test_cli.cpp
        tests/test_display_composer.cpp
        tests/test_geometry.cpp
        tests/test_iou_tracker.cpp
        tests/test_yolo11_postprocess.cpp
        tests/test_region_monitor.cpp
        tests/test_semantic_stabilizer.cpp
        tests/test_perf_monitor.cpp
    )
    if(EDGEVISION_WITH_VIDEO)
        target_sources(edgevision_tests PRIVATE
            tests/test_camera_source.cpp
            tests/test_network_camera_source.cpp
            tests/test_video_io.cpp
        )
    endif()
    if(EDGEVISION_WITH_RTSP)
        target_sources(edgevision_tests PRIVATE tests/test_rtsp_streamer.cpp)
    endif()
    if(UNIX)
        target_sources(edgevision_tests PRIVATE tests/test_tcp_server.cpp)
    endif()
    target_link_libraries(edgevision_tests PRIVATE edgevision_core)
    target_compile_definitions(edgevision_tests PRIVATE EDGEVISION_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME edgevision_tests COMMAND edgevision_tests)
endif()

install(TARGETS edge_vision RUNTIME DESTINATION .)
