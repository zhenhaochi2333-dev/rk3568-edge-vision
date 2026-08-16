#pragma once

#include "edgevision/core_types.hpp"

#include <cstddef>
#include <deque>

namespace edgevision {

class PerfMonitor {
public:
    explicit PerfMonitor(std::size_t moving_window = 20U);

    void add_frame(const FrameMetrics& metrics);
    void set_throughput_fps(double fps);

    FrameMetrics moving_average() const;
    FrameMetrics final_average() const;
    double throughput_fps() const { return throughput_fps_; }
    std::size_t frame_count() const { return frame_count_; }

private:
    static FrameMetrics divide(const FrameMetrics& metrics, double divisor);
    static void add(FrameMetrics& destination, const FrameMetrics& source);
    static void subtract(FrameMetrics& destination, const FrameMetrics& source);

    std::size_t moving_window_;
    std::deque<FrameMetrics> recent_;
    FrameMetrics recent_sum_;
    FrameMetrics total_sum_;
    std::size_t frame_count_ = 0U;
    double throughput_fps_ = 0.0;
};

}  // namespace edgevision
