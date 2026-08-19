#include "edgevision/region_monitor.hpp"

#include <algorithm>
#include <stdexcept>

namespace edgevision {

namespace {

double elapsed_seconds(std::chrono::steady_clock::time_point now,
                       std::chrono::steady_clock::time_point before)
{
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
    if (roi_.x < 0.0F || roi_.y < 0.0F || roi_.width <= 0.0F || roi_.height <= 0.0F ||
        roi_.x + roi_.width > 1.0F || roi_.y + roi_.height > 1.0F) {
        throw std::runtime_error("ROI must be inside normalized [0,1] coordinates");
    }
    if (dwell_seconds_ < 0.0 || max_recent_events_ == 0U || max_lost_seconds_ <= 0.0) {
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
    const std::vector<Detection>& stabilized_detections,
    std::chrono::steady_clock::time_point source_timestamp,
    int frame_width, int frame_height)
{
    if (frame_width <= 0 || frame_height <= 0) {
        throw std::runtime_error("RegionMonitor requires positive frame dimensions");
    }

    RegionSnapshot snapshot;
    std::map<int, bool> observed;
    for (const Detection& detection : stabilized_detections) {
        if (detection.logical_id >= 0 &&
            detection.lifecycle_state != LogicalObjectState::Active) {
            continue;
        }
        const int logical_id = detection.logical_id >= 0 ? detection.logical_id : detection.track_id;
        if (logical_id < 0) {
            continue;
        }
        observed[logical_id] = true;

        const float center_x = (detection.box.x + detection.box.width * 0.5F) /
                               static_cast<float>(frame_width);
        const float center_y = (detection.box.y + detection.box.height * 0.5F) /
                               static_cast<float>(frame_height);
        const bool inside = contains(roi_, center_x, center_y);
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
        if (state.inside && state.missing) {
            // A short absence pauses dwell instead of ending the lifecycle.
            state.missing = false;
            state.last_observed_at = source_timestamp;
        } else if (state.inside && inside) {
            state.dwell_accumulated_seconds +=
                elapsed_seconds(source_timestamp, state.last_observed_at);
            state.last_observed_at = source_timestamp;
        }

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
            state.dwell_emitted = true;
            RegionEvent event{RegionEventType::Dwell, logical_id, detection.class_id,
                              source_timestamp, detection.confidence};
            event.logical_id = logical_id;
            append_event(event, snapshot);
        }
    }

    // Missing objects are not counted in occupancy. They remain in the ROI
    // lifecycle until the real-time loss window expires, then produce one
    // delayed EXIT and are forgotten so a future logical id can enter cleanly.
    for (auto state_it = states_.begin(); state_it != states_.end();) {
        TrackState& state = state_it->second;
        if (observed.find(state_it->first) != observed.end() || !state.inside) {
            ++state_it;
            continue;
        }
        state.missing = true;
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
