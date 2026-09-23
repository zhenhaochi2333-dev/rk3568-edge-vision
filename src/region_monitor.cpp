/* 用源图坐标中的框中心判断归一化 ROI，按 logical_id 维护 ENTER/DWELL/EXIT。 */
#include "edgevision/region_monitor.hpp"

#include <algorithm>
#include <stdexcept>

namespace edgevision {

namespace {

double elapsed_seconds(std::chrono::steady_clock::time_point now,
                       std::chrono::steady_clock::time_point before)
{
    // 源帧时间异常倒退时不产生负驻留；正常运行依赖调用方按时间顺序传入帧。
    return std::max(0.0, std::chrono::duration<double>(now - before).count());
}

}  // namespace

RegionMonitor::RegionMonitor(NormalizedRoi roi, double dwell_seconds,
                             std::size_t max_recent_events, double max_lost_seconds)
    : roi_(roi),
      dwell_seconds_(dwell_seconds),
      max_recent_events_(max_recent_events),
      max_lost_seconds_(max_lost_seconds)
{
    // 归一化矩形必须有正面积且完整落在图像范围内；边恰好到 1.0 是有效配置。
    if (roi_.x < 0.0F || roi_.y < 0.0F || roi_.width <= 0.0F || roi_.height <= 0.0F ||
        roi_.x + roi_.width > 1.0F || roi_.y + roi_.height > 1.0F) {
        throw std::runtime_error("ROI must be inside normalized [0,1] coordinates");
    }
    // dwell=0 合法；进入分支在本帧直接 continue，因此首次创建状态时不会发 DWELL，后续一次在内观测才触发。
    // 队列容量为零则无有效快照历史。
    if (dwell_seconds_ < 0.0 || max_recent_events_ == 0U || max_lost_seconds_ <= 0.0) {
        throw std::runtime_error("invalid RegionMonitor configuration");
    }
}

bool RegionMonitor::contains(const NormalizedRoi& roi, float normalized_x, float normalized_y)
{
    // 四条边都包含：恰落在 ROI 边缘的中心属于区域内。
    return normalized_x >= roi.x && normalized_x <= roi.x + roi.width &&
           normalized_y >= roi.y && normalized_y <= roi.y + roi.height;
}

void RegionMonitor::reset()
{
    // 重置监控会清掉所有已进入身份，后续再次观察在区内会被视为首次 ENTER。
    states_.clear();
    recent_events_.clear();
}

void RegionMonitor::append_event(const RegionEvent& event, RegionSnapshot& snapshot)
{
    // 同一事件同时进入本轮增量列表和有界历史队列；历史满时丢弃最早事件。
    snapshot.new_events.push_back(event);
    recent_events_.push_back(event);
    while (recent_events_.size() > max_recent_events_) {
        recent_events_.pop_front();
    }
}

// 输入来自稳定器的活动对象；返回本次新事件、当前占用数和有界的近期事件列表。
RegionSnapshot RegionMonitor::update(
    const std::vector<Detection>& stabilized_detections,
    std::chrono::steady_clock::time_point source_timestamp,
    int frame_width, int frame_height)
{
    if (frame_width <= 0 || frame_height <= 0) {
        throw std::runtime_error("RegionMonitor requires positive frame dimensions");
    }

    // observed 记录本轮出现过的逻辑身份（即便它在 ROI 外），用来区别“离开 ROI”和“完全漏检”。
    RegionSnapshot snapshot;
    std::map<int, bool> observed;
    for (const Detection& detection : stabilized_detections) {
        if (detection.logical_id >= 0 &&
            detection.lifecycle_state != LogicalObjectState::Active) {
            continue;
        }
        // 稳定 ID 优先；旧调用方没有 logical_id 时才回退到 raw track_id。
        const int logical_id = detection.logical_id >= 0 ? detection.logical_id : detection.track_id;
        if (logical_id < 0) {
            continue;
        }
        observed[logical_id] = true;

        // ROI 参数在 [0,1] 坐标系；检测框必须先除以原始帧宽高，不能用 640 模型坐标。
        const float center_x = (detection.box.x + detection.box.width * 0.5F) /
                               static_cast<float>(frame_width);
        const float center_y = (detection.box.y + detection.box.height * 0.5F) /
                               static_cast<float>(frame_height);
        const bool inside = contains(roi_, center_x, center_y);
        // occupancy 是瞬时观测值，不包括缺席中的保留状态，也不等同于状态表大小。
        if (inside) {
            ++snapshot.occupancy;
        }

        auto state_it = states_.find(logical_id);
        if (state_it == states_.end()) {
            TrackState state;
            state.class_id = detection.class_id;
            state.confidence = detection.confidence;
            state.inside = inside;
            state.last_observed_at = source_timestamp;
            state.missing = false;
            if (inside) {
                // 首次被监控看见且中心已在 ROI 内，相当于从区外进入；基线静音仅阻止此 ENTER。
                state.entered_at = source_timestamp;
                if (!detection.suppress_enter) {
                    RegionEvent event{RegionEventType::Enter, logical_id, detection.class_id,
                                      source_timestamp, detection.confidence};
                    event.logical_id = logical_id;
                    append_event(event, snapshot);
                }
            }
            states_.emplace(logical_id, state);
            continue;
        }

        TrackState& state = state_it->second;
        state.class_id = detection.class_id;
        state.confidence = detection.confidence;
        // 临时漏检暂停驻留累计；重新观察到对象时继续同一次区域生命周期。
        if (state.inside && state.missing) {
            // 短暂缺席只暂停驻留累计，不结束本次 ROI 生命周期；恢复帧从当前时刻重新作为累计锚点。
            state.missing = false;
            state.last_observed_at = source_timestamp;
        } else if (state.inside && inside) {
            state.dwell_accumulated_seconds +=
                elapsed_seconds(source_timestamp, state.last_observed_at);
            state.last_observed_at = source_timestamp;
        }

        // 下面根据上次生命周期状态与本帧中心位置转移；先前的缺席恢复逻辑已处理驻留间隔。
        if (!state.inside && inside) {
            state.inside = true;
            state.dwell_emitted = false;
            state.entered_at = source_timestamp;
            state.last_observed_at = source_timestamp;
            state.dwell_accumulated_seconds = 0.0;
            state.missing = false;
            RegionEvent event{RegionEventType::Enter, logical_id, detection.class_id,
                              source_timestamp, detection.confidence};
            event.logical_id = logical_id;
            append_event(event, snapshot);
        } else if (state.inside && !inside) {
            // 明确观察到中心在外，立即 EXIT；这与对象完全消失后的超时 EXIT 是两条不同路径。
            state.inside = false;
            state.dwell_emitted = false;
            state.entered_at = std::chrono::steady_clock::time_point{};
            state.last_observed_at = source_timestamp;
            state.dwell_accumulated_seconds = 0.0;
            state.missing = false;
            RegionEvent event{RegionEventType::Exit, logical_id, detection.class_id,
                              source_timestamp, detection.confidence};
            event.logical_id = logical_id;
            append_event(event, snapshot);
        } else if (state.inside && !state.dwell_emitted &&
                   state.dwell_accumulated_seconds >= dwell_seconds_) {
            // 只有本轮仍确认在内才检查阈值；缺席时不会借经过的墙钟时间触发 DWELL。
            // 一个区域停留周期只发一次 DWELL，离开再进入才重新允许触发。
            state.dwell_emitted = true;
            RegionEvent event{RegionEventType::Dwell, logical_id, detection.class_id,
                              source_timestamp, detection.confidence};
            event.logical_id = logical_id;
            append_event(event, snapshot);
        }
    }

    // 完全未观测的对象不计入占用数，但仍保留原 ROI 状态直到缺失窗口届满；届满发一次延迟 EXIT，
    // 随即删除状态，避免同一旧身份之后重新出现时继承已经结束的区域停留周期。
    for (auto state_it = states_.begin(); state_it != states_.end();) {
        TrackState& state = state_it->second;
        if (observed.find(state_it->first) != observed.end() || !state.inside) {
            ++state_it;
            continue;
        }
        state.missing = true;
        // 到达超时边界（elapsed >= max_lost_seconds）时发一次延迟 EXIT 并删除；未进入状态无事件。
        if (elapsed_seconds(source_timestamp, state.last_observed_at) < max_lost_seconds_) {
            ++state_it;
            continue;
        }

        RegionEvent event{RegionEventType::Exit, state_it->first, state.class_id,
                          source_timestamp, state.confidence};
        event.logical_id = state_it->first;
        append_event(event, snapshot);
        state_it = states_.erase(state_it);
    }

    snapshot.recent_events.assign(recent_events_.begin(), recent_events_.end());
    return snapshot;
}

}  // namespace edgevision
