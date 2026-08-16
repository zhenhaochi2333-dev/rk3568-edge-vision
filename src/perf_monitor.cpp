#include "edgevision/perf_monitor.hpp"

#include <stdexcept>

namespace edgevision {

PerfMonitor::PerfMonitor(std::size_t moving_window)
    : moving_window_(moving_window)
{
    if (moving_window_ == 0U) {
        throw std::runtime_error("PerfMonitor moving window must be greater than zero");
    }
}

void PerfMonitor::add(FrameMetrics& destination, const FrameMetrics& source)
{
    destination.preprocess_ms += source.preprocess_ms;
    destination.inference_ms += source.inference_ms;
    destination.postprocess_ms += source.postprocess_ms;
    destination.visualization_ms += source.visualization_ms;
    destination.end_to_end_ms += source.end_to_end_ms;
    destination.fps += source.fps;
    destination.object_count += source.object_count;
}

void PerfMonitor::subtract(FrameMetrics& destination, const FrameMetrics& source)
{
    destination.preprocess_ms -= source.preprocess_ms;
    destination.inference_ms -= source.inference_ms;
    destination.postprocess_ms -= source.postprocess_ms;
    destination.visualization_ms -= source.visualization_ms;
    destination.end_to_end_ms -= source.end_to_end_ms;
    destination.fps -= source.fps;
    destination.object_count -= source.object_count;
}

FrameMetrics PerfMonitor::divide(const FrameMetrics& metrics, double divisor)
{
    if (divisor <= 0.0) {
        return FrameMetrics{};
    }
    FrameMetrics result = metrics;
    result.preprocess_ms /= divisor;
    result.inference_ms /= divisor;
    result.postprocess_ms /= divisor;
    result.visualization_ms /= divisor;
    result.end_to_end_ms /= divisor;
    result.fps /= divisor;
    result.object_count /= divisor;
    return result;
}

void PerfMonitor::add_frame(const FrameMetrics& metrics)
{
    recent_.push_back(metrics);
    add(recent_sum_, metrics);
    add(total_sum_, metrics);
    ++frame_count_;

    if (recent_.size() > moving_window_) {
        subtract(recent_sum_, recent_.front());
        recent_.pop_front();
    }
}

void PerfMonitor::set_throughput_fps(double fps)
{
    throughput_fps_ = fps;
}

FrameMetrics PerfMonitor::moving_average() const
{
    return divide(recent_sum_, static_cast<double>(recent_.size()));
}

FrameMetrics PerfMonitor::final_average() const
{
    return divide(total_sum_, static_cast<double>(frame_count_));
}

}  // namespace edgevision
