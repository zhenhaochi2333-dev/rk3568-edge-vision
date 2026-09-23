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

// 所有 FrameMetrics 字段采用相同的求和策略，保持加、减和均值字段集合一致。
// 这里累加的是逐帧样本值；FPS 字段也因此是帧 FPS 的算术平均，不等于整体吞吐率。
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

// 零样本时返回默认全零结构，避免对空任务做除零；非空时逐字段除以样本数。
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

// 每帧同时进入“全程累计和”与“近期窗口和”。窗口淘汰只影响移动平均，
// 所以长任务的 final_average 不会因 recent_ 容量固定而遗失早期样本。
void PerfMonitor::add_frame(const FrameMetrics& metrics)
{
    recent_.push_back(metrics);
    add(recent_sum_, metrics);
    add(total_sum_, metrics);
    ++frame_count_;

    // 先加入再淘汰：窗口大小为 2 时，第三帧到达后保留第二、三帧，
    // 这与调用方查询时“最近 N 个已完成帧”的直觉一致。
    if (recent_.size() > moving_window_) {
        subtract(recent_sum_, recent_.front());
        recent_.pop_front();
    }
}

// 由外部按任务起止时间计算完整吞吐率，避免把短时间逐帧瞬时 FPS 的平均误当作
// 全程“处理帧数 / 总时长”。本类不推导该时间边界，也不检查 fps 合法范围。
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
