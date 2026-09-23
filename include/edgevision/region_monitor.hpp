#pragma once

#include "edgevision/core_types.hpp"

#include <chrono>
#include <cstddef>
#include <deque>
#include <map>
#include <vector>

namespace edgevision {

enum class RegionEventType {
    // 框中心从 ROI 外进入，或首次观察时已在 ROI 内（除非 suppress_enter）。
    Enter,
    // 已进入对象的框中心被再次观察到 ROI 外；长时间缺失也会产生延迟 Exit。
    Exit,
    // 在连续累计的实际在场时间达到 dwell_seconds 后，每次进入周期最多一次。
    Dwell,
};

struct RegionEvent {
    RegionEventType type = RegionEventType::Enter;
    // 兼容既有 TCP/API 字段。经稳定器输入时这里同样填写 logical_id，因此不能当作 raw track_id。
    int track_id = -1;
    // 事件发生时最近一次观测的稳定类别、源帧时间和融合置信度；缺失超时 EXIT 使用缓存值。
    int class_id = -1;
    std::chrono::steady_clock::time_point source_timestamp{};
    float confidence = 0.0F;
    // 稳定身份字段；旧式只带 raw track_id 的输入会把选出的身份同时写入此字段。
    int logical_id = -1;
};

struct RegionSnapshot {
    std::size_t occupancy = 0U;
    std::vector<RegionEvent> new_events;
    std::vector<RegionEvent> recent_events;
};

class RegionMonitor {
public:
    // ROI 使用归一化原图坐标闭区间；dwell_seconds 是单次进入周期内累计的可见在场秒数。
    // max_recent_events 限制快照中的历史队列长度（必须非零）；max_lost_seconds 是对象缺席
    // 后仍保留 ROI 生命周期的最长时间（必须为正），超时会发出延迟 EXIT 并删除状态。
    explicit RegionMonitor(NormalizedRoi roi, double dwell_seconds = 3.0,
                           std::size_t max_recent_events = 16U,
                           double max_lost_seconds = 2.0);

    // 建议输入稳定器输出：若 logical_id 有效但生命周期非 Active 则忽略；否则使用
    // logical_id，缺省时回退为 track_id。占用数只计本轮实际观测且中心在 ROI 内的身份。
    // 时间使用 source_timestamp 的差值而非帧数；漏检期间不增加 dwell，短暂漏检恢复后接续累计。
    // ROI 判断基于原始帧坐标中的检测框中心，边界点也视为在 ROI 内。
    RegionSnapshot update(const std::vector<Detection>& tracked_detections,
                          std::chrono::steady_clock::time_point source_timestamp,
                          int frame_width, int frame_height);

    // 清除每个身份的进出状态和近期事件环形队列，不修改 ROI 或计时参数。
    void reset();
    const NormalizedRoi& roi() const { return roi_; }

    static bool contains(const NormalizedRoi& roi, float normalized_x, float normalized_y);

private:
    struct TrackState {
        // 最近看到的事件元数据；若对象缺失直至超时，这些值用于构造延迟 EXIT。
        int class_id = -1;
        float confidence = 0.0F;
        // 当前 ROI 生命周期是否已经进入，以及该周期的 DWELL 是否已发过。
        bool inside = false;
        bool dwell_emitted = false;
        // entered_at 记录首次进入时间（当前实现用于状态说明）；last_observed_at 是驻留累计锚点。
        std::chrono::steady_clock::time_point entered_at{};
        std::chrono::steady_clock::time_point last_observed_at{};
        // 仅将连续两次实际观察都在 ROI 内的间隔加进来；缺席间隔被暂停而不是算在驻留内。
        double dwell_accumulated_seconds = 0.0;
        // 自最近一次观察后是否有 update 未见此身份；重新观察时仅恢复标志，不合成缺席时间。
        bool missing = false;
    };

    void append_event(const RegionEvent& event, RegionSnapshot& snapshot);

    NormalizedRoi roi_;
    double dwell_seconds_;
    std::size_t max_recent_events_;
    double max_lost_seconds_;
    std::map<int, TrackState> states_;
    std::deque<RegionEvent> recent_events_;
};

}  // namespace edgevision
