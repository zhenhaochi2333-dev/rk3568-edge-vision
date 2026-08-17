#pragma once

#include "edgevision/core_types.hpp"

#include <chrono>
#include <cstddef>
#include <deque>
#include <map>
#include <vector>

namespace edgevision {

enum class RegionEventType {
    Enter,
    Exit,
    Dwell,
};

struct RegionEvent {
    RegionEventType type = RegionEventType::Enter;
    int track_id = -1;
    int class_id = -1;
    std::chrono::steady_clock::time_point source_timestamp{};
    float confidence = 0.0F;
};

struct RegionSnapshot {
    std::size_t occupancy = 0U;
    std::vector<RegionEvent> new_events;
    std::vector<RegionEvent> recent_events;
};

class RegionMonitor {
public:
    explicit RegionMonitor(NormalizedRoi roi, double dwell_seconds = 3.0,
                           std::size_t max_recent_events = 16U);

    RegionSnapshot update(const std::vector<Detection>& tracked_detections,
                          std::chrono::steady_clock::time_point source_timestamp,
                          int frame_width, int frame_height);

    void reset();
    const NormalizedRoi& roi() const { return roi_; }

    static bool contains(const NormalizedRoi& roi, float normalized_x, float normalized_y);

private:
    struct TrackState {
        int class_id = -1;
        bool inside = false;
        bool dwell_emitted = false;
        std::chrono::steady_clock::time_point entered_at{};
    };

    void append_event(const RegionEvent& event, RegionSnapshot& snapshot);

    NormalizedRoi roi_;
    double dwell_seconds_;
    std::size_t max_recent_events_;
    std::map<int, TrackState> states_;
    std::deque<RegionEvent> recent_events_;
};

}  // namespace edgevision
