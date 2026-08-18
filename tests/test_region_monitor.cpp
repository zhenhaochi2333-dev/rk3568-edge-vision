#include "edgevision/region_monitor.hpp"

#include <opencv2/core.hpp>

#include <cassert>
#include <chrono>
#include <vector>

namespace {

edgevision::Detection tracked(int track_id, int class_id, float center_x, float center_y)
{
    edgevision::Detection detection;
    detection.track_id = track_id;
    detection.class_id = class_id;
    detection.confidence = 0.8F;
    detection.box = cv::Rect2f(center_x - 5.0F, center_y - 5.0F, 10.0F, 10.0F);
    return detection;
}

}  // namespace

void run_region_monitor_tests()
{
    const auto base = std::chrono::steady_clock::time_point{};
    edgevision::RegionMonitor full_frame(edgevision::NormalizedRoi{});
    const auto full_frame_detection = full_frame.update(
        {tracked(20, 0, 95.0F, 95.0F)}, base, 100, 100);
    assert(full_frame_detection.occupancy == 1U);
    assert(full_frame_detection.new_events.size() == 1U);
    assert(full_frame_detection.new_events.front().type == edgevision::RegionEventType::Enter);

    const edgevision::NormalizedRoi roi{0.25F, 0.20F, 0.50F, 0.60F};
    edgevision::RegionMonitor monitor(roi, 100.0);

    const auto outside = monitor.update({tracked(1, 0, 10.0F, 10.0F)}, base, 100, 100);
    assert(outside.occupancy == 0U);
    assert(outside.new_events.empty());

    const auto entered = monitor.update({tracked(1, 0, 50.0F, 50.0F)}, base + std::chrono::seconds(1),
                                        100, 100);
    assert(entered.occupancy == 1U);
    assert(entered.new_events.size() == 1U);
    assert(entered.new_events.front().type == edgevision::RegionEventType::Enter);

    const auto stayed = monitor.update({tracked(1, 0, 50.0F, 50.0F)}, base + std::chrono::seconds(2),
                                       100, 100);
    assert(stayed.occupancy == 1U);
    assert(stayed.new_events.empty());

    const auto missing = monitor.update({}, base + std::chrono::seconds(3), 100, 100);
    assert(missing.occupancy == 0U);
    assert(missing.new_events.empty());

    const auto reobserved_inside = monitor.update(
        {tracked(1, 0, 50.0F, 50.0F)}, base + std::chrono::seconds(4), 100, 100);
    assert(reobserved_inside.new_events.empty());

    const auto exited = monitor.update({tracked(1, 0, 90.0F, 90.0F)},
                                       base + std::chrono::seconds(5), 100, 100);
    assert(exited.occupancy == 0U);
    assert(exited.new_events.size() == 1U);
    assert(exited.new_events.front().type == edgevision::RegionEventType::Exit);

    edgevision::RegionMonitor independent(roi);
    const auto two_tracks = independent.update(
        {tracked(7, 1, 50.0F, 50.0F), tracked(8, 2, 90.0F, 90.0F)}, base, 100, 100);
    assert(two_tracks.occupancy == 1U);
    assert(two_tracks.new_events.size() == 1U);
    assert(two_tracks.new_events.front().track_id == 7);

    edgevision::Detection logical_only = tracked(-1, 1, 50.0F, 50.0F);
    logical_only.logical_id = 42;
    logical_only.lifecycle_state = edgevision::LogicalObjectState::Confirmed;
    edgevision::RegionMonitor logical_monitor(roi);
    const auto logical_event = logical_monitor.update(
        {logical_only}, base, 100, 100);
    assert(logical_event.new_events.size() == 1U);
    assert(logical_event.new_events.front().logical_id == 42);
    assert(logical_event.new_events.front().track_id == 42);

    edgevision::RegionMonitor dwell_monitor(roi);
    dwell_monitor.update({tracked(3, 0, 50.0F, 50.0F)}, base, 100, 100);
    const auto before_dwell = dwell_monitor.update(
        {tracked(3, 0, 50.0F, 50.0F)}, base + std::chrono::milliseconds(2999), 100, 100);
    assert(before_dwell.new_events.empty());
    const auto dwell = dwell_monitor.update(
        {tracked(3, 0, 50.0F, 50.0F)}, base + std::chrono::milliseconds(3000), 100, 100);
    assert(dwell.new_events.size() == 1U);
    assert(dwell.new_events.front().type == edgevision::RegionEventType::Dwell);
    const auto after_dwell = dwell_monitor.update(
        {tracked(3, 0, 50.0F, 50.0F)}, base + std::chrono::seconds(4), 100, 100);
    assert(after_dwell.new_events.empty());
}
