#include "edgevision/semantic_stabilizer.hpp"

#include <opencv2/core.hpp>

#include <cassert>
#include <chrono>
#include <vector>

namespace {

edgevision::Detection detection(int raw_id, int class_id, float x, float y,
                                float confidence = 0.8F)
{
    edgevision::Detection result;
    result.track_id = raw_id;
    result.class_id = class_id;
    result.confidence = confidence;
    result.box = cv::Rect2f(x, y, 40.0F, 40.0F);
    return result;
}

}  // namespace

void run_semantic_stabilizer_tests()
{
    edgevision::SemanticStabilizer stabilizer;
    const auto base = std::chrono::steady_clock::time_point{};

    // Cold-start mute: an object becomes visible to the business layer only
    // after three observations.
    assert(stabilizer.update({detection(10, 0, 20.0F, 20.0F)}, base).empty());
    assert(stabilizer.update({detection(10, 0, 21.0F, 20.0F)},
                             base + std::chrono::milliseconds(100)).empty());
    const auto confirmed = stabilizer.update({detection(10, 0, 22.0F, 20.0F)},
                                             base + std::chrono::milliseconds(200));
    assert(confirmed.size() == 1U);
    assert(confirmed.front().logical_id == 1);
    assert(confirmed.front().lifecycle_state == edgevision::LogicalObjectState::Confirmed);
    assert(confirmed.front().presence_score >= 0.99F);

    // The raw tracker id changes and the detector class flickers once, but
    // geometric reassociation and temporal class fusion keep the logical
    // object and class stable.
    const auto flicker = stabilizer.update({detection(11, 1, 23.0F, 20.0F)},
                                           base + std::chrono::milliseconds(300));
    assert(flicker.size() == 1U);
    assert(flicker.front().logical_id == 1);
    assert(flicker.front().class_id == 0);

    const auto stable_again = stabilizer.update({detection(11, 0, 24.0F, 20.0F)},
                                                base + std::chrono::milliseconds(400));
    assert(stable_again.size() == 1U);
    assert(stable_again.front().logical_id == 1);

    // A short absence does not create EXIT/ENTER churn in the logical layer;
    // reappearance with another raw id keeps the same identity.
    const auto missing = stabilizer.update({}, base + std::chrono::milliseconds(500));
    assert(missing.empty());
    const auto reappeared = stabilizer.update({detection(12, 0, 25.0F, 20.0F)},
                                              base + std::chrono::milliseconds(600));
    assert(reappeared.size() == 1U);
    assert(reappeared.front().logical_id == 1);
    assert(reappeared.front().lifecycle_state == edgevision::LogicalObjectState::Confirmed);

    // A far-away detection is a new logical object, not a reassociation.
    const auto second = stabilizer.update({detection(13, 2, 500.0F, 400.0F)},
                                          base + std::chrono::milliseconds(700));
    assert(second.empty());
    assert(stabilizer.update({detection(13, 2, 500.0F, 400.0F)},
                             base + std::chrono::milliseconds(800)).empty());
    const auto second_confirmed = stabilizer.update({detection(13, 2, 500.0F, 400.0F)},
                                                    base + std::chrono::milliseconds(900));
    assert(second_confirmed.size() == 1U);
    assert(second_confirmed.front().logical_id == 2);
    assert(second_confirmed.front().class_id == 2);
}
