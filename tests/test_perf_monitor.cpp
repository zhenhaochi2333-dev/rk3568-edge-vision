#include "edgevision/perf_monitor.hpp"

#include <cassert>
#include <cmath>

void run_perf_monitor_tests()
{
    edgevision::PerfMonitor monitor(2U);
    edgevision::FrameMetrics first;
    first.preprocess_ms = 2.0;
    first.inference_ms = 10.0;
    first.postprocess_ms = 3.0;
    first.visualization_ms = 4.0;
    first.end_to_end_ms = 19.0;
    first.object_count = 2.0;
    edgevision::FrameMetrics second = first;
    second.inference_ms = 20.0;
    second.end_to_end_ms = 29.0;
    second.object_count = 4.0;
    edgevision::FrameMetrics third = second;
    third.inference_ms = 30.0;
    third.end_to_end_ms = 39.0;
    third.object_count = 6.0;

    monitor.add_frame(first);
    monitor.add_frame(second);
    monitor.add_frame(third);
    const edgevision::FrameMetrics moving = monitor.moving_average();
    const edgevision::FrameMetrics final = monitor.final_average();
    assert(monitor.frame_count() == 3U);
    assert(std::fabs(moving.inference_ms - 25.0) < 1e-9);
    assert(std::fabs(final.inference_ms - 20.0) < 1e-9);
    assert(std::fabs(final.end_to_end_ms - 29.0) < 1e-9);
    assert(std::fabs(final.object_count - 4.0) < 1e-9);
    monitor.set_throughput_fps(12.5);
    assert(std::fabs(monitor.throughput_fps() - 12.5) < 1e-9);
}
