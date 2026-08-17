#include "edgevision/region_monitor.hpp"

#include <algorithm>
#include <stdexcept>

namespace edgevision {

RegionMonitor::RegionMonitor(NormalizedRoi roi, double dwell_seconds,
                             std::size_t max_recent_events)
    : roi_(roi), dwell_seconds_(dwell_seconds), max_recent_events_(max_recent_events)
{
    if (roi_.x < 0.0F || roi_.y < 0.0F || roi_.width <= 0.0F || roi_.height <= 0.0F ||
        roi_.x + roi_.width > 1.0F || roi_.y + roi_.height > 1.0F) {
        throw std::runtime_error("ROI must be inside normalized [0,1] coordinates");
    }
    if (dwell_seconds_ < 0.0 || max_recent_events_ == 0U) {
        throw std::runtime_error("invalid RegionMonitor configuration");
    }
}

bool RegionMonitor::contains(const NormalizedRoi& roi, float normalized_x, float normalized_y)
{
    return normalized_x >= roi.x && normalized_x <= roi.x + roi.width &&
           normalized_y >= roi.y && normalized_y <= roi.y + roi.height;
}

void RegionMonitor::reset()
{
    states_.clear();
    recent_events_.clear();
}

void RegionMonitor::append_event(const RegionEvent& event, RegionSnapshot& snapshot)
{
    snapshot.new_events.push_back(event);
    recent_events_.push_back(event);
    while (recent_events_.size() > max_recent_events_) {
        recent_events_.pop_front();
    }
}

RegionSnapshot RegionMonitor::update(
    const std::vector<Detection>& tracked_detections,
    std::chrono::steady_clock::time_point source_timestamp,
    int frame_width, int frame_height)
{
    if (frame_width <= 0 || frame_height <= 0) {
        throw std::runtime_error("RegionMonitor requires positive frame dimensions");
    }

    RegionSnapshot snapshot;
    for (const Detection& detection : tracked_detections) {
        if (detection.track_id < 0) {
            continue;
        }
        const float center_x = (detection.box.x + detection.box.width * 0.5F) /
                               static_cast<float>(frame_width);
        const float center_y = (detection.box.y + detection.box.height * 0.5F) /
                               static_cast<float>(frame_height);
        const bool inside = contains(roi_, center_x, center_y);
        if (inside) {
            ++snapshot.occupancy;
        }

        auto state_it = states_.find(detection.track_id);
        if (state_it == states_.end()) {
            TrackState state;
            state.class_id = detection.class_id;
            state.inside = inside;
            state.entered_at = inside ? source_timestamp : std::chrono::steady_clock::time_point{};
            state_it = states_.emplace(detection.track_id, state).first;
            if (inside) {
                append_event(RegionEvent{RegionEventType::Enter, detection.track_id,
                                         detection.class_id, source_timestamp,
                                         detection.confidence}, snapshot);
            }
        } else {
            TrackState& state = state_it->second;
            state.class_id = detection.class_id;
            if (!state.inside && inside) {
                state.inside = true;
                state.dwell_emitted = false;
                state.entered_at = source_timestamp;
                append_event(RegionEvent{RegionEventType::Enter, detection.track_id,
                                         detection.class_id, source_timestamp,
                                         detection.confidence}, snapshot);
            } else if (state.inside && !inside) {
                state.inside = false;
                state.dwell_emitted = false;
                state.entered_at = std::chrono::steady_clock::time_point{};
                append_event(RegionEvent{RegionEventType::Exit, detection.track_id,
                                         detection.class_id, source_timestamp,
                                         detection.confidence}, snapshot);
            } else if (state.inside && !state.dwell_emitted &&
                       std::chrono::duration<double>(source_timestamp - state.entered_at).count() >=
                           dwell_seconds_) {
                state.dwell_emitted = true;
                append_event(RegionEvent{RegionEventType::Dwell, detection.track_id,
                                         detection.class_id, source_timestamp,
                                         detection.confidence}, snapshot);
            }
        }
    }

    snapshot.recent_events.assign(recent_events_.begin(), recent_events_.end());
    return snapshot;
}

}  // namespace edgevision
