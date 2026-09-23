#pragma once

#include "edgevision/core_types.hpp"

#include <cstddef>
#include <deque>

namespace edgevision {

// 汇总逐帧 FrameMetrics：recent_ 维护最近 N 帧，total_sum_ 累积整个任务。
// 这是调用线程内的简单统计器，没有锁；并发 add/query 必须由上层同步。
class PerfMonitor {
public:
    // moving_window 是最近帧平均的样本数而非秒数；必须大于零。
    explicit PerfMonitor(std::size_t moving_window = 20U);

    // 输入一帧已经算好的指标。超出窗口时只从 recent_sum_ 移除最旧样本，
    // total_sum_ 和 frame_count_ 仍保留它，因此最终平均覆盖整个处理期间。
    void add_frame(const FrameMetrics& metrics);
    // 设置由调用方按完整处理区间测出的吞吐 FPS；它是单独指标，不会从逐帧 fps 求平均。
    void set_throughput_fps(double fps);

    // 最近 min(已收帧数, moving_window) 帧的逐字段算术平均；空样本返回全零指标。
    FrameMetrics moving_average() const;
    // 从第一次 add_frame 至今所有帧的逐字段平均，通常用于任务结束摘要。
    FrameMetrics final_average() const;
    double throughput_fps() const { return throughput_fps_; }
    std::size_t frame_count() const { return frame_count_; }

private:
    static FrameMetrics divide(const FrameMetrics& metrics, double divisor);
    static void add(FrameMetrics& destination, const FrameMetrics& source);
    static void subtract(FrameMetrics& destination, const FrameMetrics& source);

    std::size_t moving_window_;
    // 双端队列保留窗口样本以便 O(1) 移除最旧帧；两个和分别服务滑动与累计平均。
    std::deque<FrameMetrics> recent_;
    FrameMetrics recent_sum_;
    FrameMetrics total_sum_;
    std::size_t frame_count_ = 0U;
    double throughput_fps_ = 0.0;
};

}  // namespace edgevision
