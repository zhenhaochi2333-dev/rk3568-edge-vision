#include "edgevision/iou_tracker.hpp"

#include <opencv2/core.hpp>

#include <cassert>
#include <vector>

namespace {

edgevision::Detection detection(int class_id, float x, float y, float width, float height)
{
    return edgevision::Detection{class_id, 0.8F, cv::Rect2f(x, y, width, height)};
}

}  // namespace

void run_iou_tracker_tests()
{
    edgevision::IouTracker tracker;
    const std::vector<edgevision::Detection> first{
        detection(0, 10.0F, 10.0F, 40.0F, 40.0F),
        detection(1, 100.0F, 10.0F, 40.0F, 40.0F)};
    const std::vector<edgevision::Detection> first_result = tracker.update(first);
    assert(first_result.size() == 2U);
    assert(first_result[0].track_id == 1);
    assert(first_result[1].track_id == 2);

    const std::vector<edgevision::Detection> second{
        detection(0, 12.0F, 11.0F, 40.0F, 40.0F),
        detection(1, 102.0F, 11.0F, 40.0F, 40.0F)};
    const std::vector<edgevision::Detection> second_result = tracker.update(second);
    assert(second_result.size() == 2U);
    assert(second_result[0].track_id == 1);
    assert(second_result[1].track_id == 2);

    // A small object can move far enough under camera shake to fall below
    // the old 0.30 IoU gate. The default tracker keeps the same raw identity.
    edgevision::IouTracker jitter_tracker;
    const auto jitter_first = jitter_tracker.update({detection(0, 100, 100, 40, 40)});
    const auto jitter_second = jitter_tracker.update({detection(0, 124, 100, 40, 40)});
    assert(jitter_first.front().track_id == jitter_second.front().track_id);

    // Different classes cannot steal an existing track, and matches are one-to-one.
    const std::vector<edgevision::Detection> class_changed{
        detection(2, 12.0F, 11.0F, 40.0F, 40.0F),
        detection(0, 13.0F, 12.0F, 40.0F, 40.0F)};
    const std::vector<edgevision::Detection> changed_result = tracker.update(class_changed);
    assert(changed_result.size() == 2U);
    assert(changed_result[0].track_id != changed_result[1].track_id);

    edgevision::IouTracker short_tracker(edgevision::IouTrackerConfig{0.3F, 1U});
    const std::vector<edgevision::Detection> tracked = short_tracker.update({detection(0, 0, 0, 10, 10)});
    assert(tracked.front().track_id == 1);
    assert(short_tracker.update({}).empty());
    assert(short_tracker.update({}).empty());
    const std::vector<edgevision::Detection> reappeared =
        short_tracker.update({detection(0, 0, 0, 10, 10)});
    assert(reappeared.front().track_id == 2);
}
