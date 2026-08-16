cmake_minimum_required(VERSION 3.10)

project(edgevision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(EDGEVISION_BUILD_TESTS "Build lightweight Phase 0-5 tests" ON)
option(EDGEVISION_WITH_VIDEO "Build VideoIO and video pipeline support" OFF)
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
    list(APPEND EDGEVISION_OPENCV_COMPONENTS videoio)
endif()
find_package(OpenCV REQUIRED COMPONENTS ${EDGEVISION_OPENCV_COMPONENTS})

set(EDGEVISION_SOURCES
    src/application.cpp
    src/cli_parser.cpp
    src/image_processor.cpp
    src/label_loader.cpp
    src/logger.cpp
    src/perf_monitor.cpp
    src/rknn_model.cpp
    src/yolov5_detector.cpp
    src/visualizer.cpp
)
if(EDGEVISION_WITH_VIDEO)
    list(APPEND EDGEVISION_SOURCES src/video_io.cpp)
endif()

add_library(edgevision_core STATIC ${EDGEVISION_SOURCES})
target_include_directories(edgevision_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${RKNN_INCLUDE_DIR}"
        "${OpenCV_INCLUDE_DIRS}"
)
target_link_libraries(edgevision_core PUBLIC ${OpenCV_LIBS} "${RKNN_LIBRARY}")
target_compile_options(edgevision_core PRIVATE -Wall -Wextra -Wpedantic)
if(EDGEVISION_WITH_VIDEO)
    target_compile_definitions(edgevision_core PUBLIC EDGEVISION_WITH_VIDEO=1)
endif()

add_executable(edge_vision src/main.cpp)
target_link_libraries(edge_vision PRIVATE edgevision_core)
set_target_properties(edge_vision PROPERTIES
    INSTALL_RPATH "\$ORIGIN/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

if(EDGEVISION_BUILD_TESTS)
    enable_testing()
    add_executable(edgevision_tests
        tests/test_cli.cpp
        tests/test_geometry.cpp
        tests/test_postprocess.cpp
        tests/test_visualizer.cpp
        tests/test_perf_monitor.cpp
    )
    target_link_libraries(edgevision_tests PRIVATE edgevision_core)
    target_compile_definitions(edgevision_tests PRIVATE EDGEVISION_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME edgevision_tests COMMAND edgevision_tests)
endif()

install(TARGETS edge_vision RUNTIME DESTINATION .)
