#include "edgevision/region_monitor.hpp"
#include "edgevision/semantic_stabilizer.hpp"

#include <opencv2/core.hpp>

#include <cassert>
#include <chrono>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

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

edgevision::SemanticStabilizer make_stabilizer(double bootstrap_seconds = 0.0)
{
    edgevision::SemanticStabilizerConfig config;
    config.bootstrap_mute_seconds = bootstrap_seconds;
    return edgevision::SemanticStabilizer(config);
}

std::vector<edgevision::Detection> activate(edgevision::SemanticStabilizer& stabilizer,
                                            Clock::time_point base, int raw_id,
                                            int class_id = 0, float x = 100.0F,
                                            float confidence = 0.8F)
{
    assert(stabilizer.update({detection(raw_id, class_id, x, 100.0F, confidence)}, base).empty());
    assert(stabilizer.update({detection(raw_id, class_id, x + 1.0F, 100.0F, confidence)},
                             base + std::chrono::milliseconds(400))
               .empty());
    return stabilizer.update({detection(raw_id, class_id, x + 2.0F, 100.0F, confidence)},
                             base + std::chrono::milliseconds(800));
}

}  // namespace

void run_semantic_stabilizer_tests()
{
    const Clock::time_point base{};

    // A/B: the same logical object survives a short miss and a raw id change.
    {
        auto stabilizer = make_stabilizer();
        const auto active = activate(stabilizer, base, 10);
        assert(active.size() == 1U);
        const int logical_id = active.front().logical_id;
        assert(active.front().lifecycle_state == edgevision::LogicalObjectState::Active);
        assert(stabilizer.update({}, base + std::chrono::seconds(1)).empty());
        const auto recovered = stabilizer.update(
            {detection(11, 0, 104.0F, 100.0F)}, base + std::chrono::milliseconds(1200));
        assert(recovered.size() == 1U);
        assert(recovered.front().logical_id == logical_id);
    }

    // C: one low-confidence class flicker does not change the stable class.
    {
        auto stabilizer = make_stabilizer();
        const auto active = activate(stabilizer, base, 20, 0);
        assert(active.front().class_id == 0);
        const auto flicker = stabilizer.update(
            {detection(20, 1, 103.0F, 100.0F, 0.05F)}, base + std::chrono::seconds(1));
        assert(flicker.size() == 1U && flicker.front().class_id == 0);
        const auto stable_again = stabilizer.update(
            {detection(20, 0, 104.0F, 100.0F)}, base + std::chrono::milliseconds(1200));
        assert(stable_again.size() == 1U && stable_again.front().class_id == 0);
    }

    // D: sustained, materially stronger evidence eventually switches class.
    {
        auto stabilizer = make_stabilizer();
        assert(activate(stabilizer, base, 30, 0).size() == 1U);
        bool switched = false;
        for (int index = 0; index < 16; ++index) {
            const auto output = stabilizer.update(
                {detection(30, 1, 102.0F, 100.0F, 1.0F)},
                base + std::chrono::milliseconds(1000 + index * 200));
            if (!output.empty() && output.front().class_id == 1) {
                switched = true;
                break;
            }
        }
        assert(switched);
    }

    // E: a far-away object is never merged with the nearby logical object.
    {
        auto stabilizer = make_stabilizer();
        const auto first = activate(stabilizer, base, 40, 0, 100.0F);
        assert(first.size() == 1U);
        const int first_id = first.front().logical_id;
        assert(stabilizer.update({detection(41, 2, 1000.0F, 500.0F)},
                                 base + std::chrono::seconds(1))
                   .empty());
        assert(stabilizer.update({detection(41, 2, 1000.0F, 500.0F)},
                                 base + std::chrono::milliseconds(1400))
                   .empty());
        const auto second = stabilizer.update({detection(41, 2, 1000.0F, 500.0F)},
                                              base + std::chrono::milliseconds(1800));
        assert(second.size() == 1U && second.front().logical_id != first_id);
    }

    // F: after the reassociation window a nearby detection starts a new id.
    {
        edgevision::SemanticStabilizerConfig config;
        config.reassociation_window_seconds = 1.2;
        config.max_lost_time_seconds = 2.0;
        auto stabilizer = edgevision::SemanticStabilizer(config);
        const auto first = activate(stabilizer, base, 50, 0, 100.0F);
        assert(first.size() == 1U);
        assert(stabilizer.update({}, base + std::chrono::seconds(1)).empty());
        assert(stabilizer.update({detection(51, 0, 104.0F, 100.0F)},
                                 base + std::chrono::milliseconds(2400))
                   .empty());
        assert(stabilizer.update({detection(51, 0, 104.0F, 100.0F)},
                                 base + std::chrono::milliseconds(2800))
                   .empty());
        const auto second = stabilizer.update({detection(51, 0, 104.0F, 100.0F)},
                                              base + std::chrono::milliseconds(3200));
        assert(second.size() == 1U && second.front().logical_id != first.front().logical_id);
    }

    // F2: a far-away class flip cannot inherit the old object's label.
    {
        auto stabilizer = make_stabilizer();
        const auto first = activate(stabilizer, base, 55, 41, 100.0F);
        assert(first.size() == 1U && first.front().class_id == 41);
        const auto flipped = stabilizer.update(
            {detection(56, 72, 700.0F, 500.0F, 0.9F)},
            base + std::chrono::milliseconds(1200));
        assert(flipped.empty());
    }

    // G: a brief flash never reaches the ENTER hysteresis.
    {
        auto stabilizer = make_stabilizer();
        edgevision::RegionMonitor monitor(edgevision::NormalizedRoi{}, 3.0);
        assert(monitor.update(stabilizer.update(
                   {detection(60, 0, 100.0F, 100.0F)}, base), base, 1280, 720)
                   .new_events.empty());
        assert(monitor.update(stabilizer.update(
                   {detection(60, 0, 100.0F, 100.0F)}, base + std::chrono::milliseconds(200)),
                              base + std::chrono::milliseconds(200), 1280, 720)
                   .new_events.empty());
        assert(monitor.update(stabilizer.update({}, base + std::chrono::milliseconds(400)),
                              base + std::chrono::milliseconds(400), 1280, 720)
                   .new_events.empty());
    }

    // G2: duplicate same-class boxes from camera shake do not create a second
    // logical cup/object, even when the raw tracker ids change every frame.
    {
        auto stabilizer = make_stabilizer();
        const auto first = activate(stabilizer, base, 90, 41, 300.0F, 0.9F);
        assert(first.size() == 1U);
        const int logical_id = first.front().logical_id;
        for (int index = 0; index < 8; ++index) {
            const auto timestamp = base + std::chrono::milliseconds(1000 + index * 200);
            const auto output = stabilizer.update(
                {detection(100 + index * 2, 41, 300.0F + index, 100.0F, 0.9F),
                 detection(101 + index * 2, 41, 324.0F + index, 100.0F, 0.7F)},
                timestamp);
            assert(output.size() == 1U);
            assert(output.front().logical_id == logical_id);
        }
    }

    // G3: two nearly identical boxes with different raw classes still render
    // as one physical object; the stronger box wins.
    {
        auto stabilizer = make_stabilizer();
        assert(stabilizer.update(
                   {detection(200, 62, 500.0F, 200.0F, 0.80F),
                    detection(201, 68, 501.0F, 201.0F, 0.70F)},
                   base)
                   .empty());
        assert(stabilizer.update(
                   {detection(202, 62, 500.0F, 200.0F, 0.80F),
                    detection(203, 68, 501.0F, 201.0F, 0.70F)},
                   base + std::chrono::milliseconds(400))
                   .empty());
        const auto output = stabilizer.update(
            {detection(204, 62, 500.0F, 200.0F, 0.80F),
             detection(205, 68, 501.0F, 201.0F, 0.70F)},
            base + std::chrono::milliseconds(800));
        assert(output.size() == 1U);
        assert(output.front().class_id == 62);
    }

    // H/I: a short absence does not EXIT, and DWELL is emitted once.
    {
        auto stabilizer = make_stabilizer();
        edgevision::RegionMonitor monitor(edgevision::NormalizedRoi{}, 1.0);
        const auto first = activate(stabilizer, base, 70, 0, 100.0F);
        const auto entered = monitor.update(first, base + std::chrono::milliseconds(800),
                                            1280, 720);
        assert(entered.new_events.size() == 1U);
        assert(entered.new_events.front().type == edgevision::RegionEventType::Enter);
        const auto missing = monitor.update(stabilizer.update({}, base + std::chrono::seconds(1)),
                                            base + std::chrono::seconds(1), 1280, 720);
        assert(missing.new_events.empty());
        const auto recovered = stabilizer.update(
            {detection(71, 0, 104.0F, 100.0F)}, base + std::chrono::milliseconds(1200));
        assert(recovered.size() == 1U);
        assert(monitor.update(recovered, base + std::chrono::milliseconds(1200), 1280, 720)
                   .new_events.empty());
        const auto dwell_output = stabilizer.update(
            {detection(71, 0, 105.0F, 100.0F)}, base + std::chrono::milliseconds(2200));
        const auto dwell = monitor.update(dwell_output, base + std::chrono::milliseconds(2200),
                                          1280, 720);
        assert(dwell.new_events.size() == 1U);
        assert(dwell.new_events.front().type == edgevision::RegionEventType::Dwell);
        const auto after_dwell_output = stabilizer.update(
            {detection(71, 0, 106.0F, 100.0F)}, base + std::chrono::milliseconds(3200));
        const auto after_dwell = monitor.update(after_dwell_output,
                                                base + std::chrono::milliseconds(3200), 1280, 720);
        assert(after_dwell.new_events.empty());
    }

    // J: bootstrap mute establishes an already-present object without a
    // backfilled ENTER; an object born later still emits ENTER normally.
    {
        auto stabilizer = make_stabilizer(3.0);
        edgevision::RegionMonitor monitor(edgevision::NormalizedRoi{}, 3.0);
        std::vector<edgevision::Detection> baseline;
        for (int index = 0; index <= 8; ++index) {
            const auto timestamp = base + std::chrono::milliseconds(index * 400);
            baseline = stabilizer.update({detection(80, 0, 100.0F, 100.0F)}, timestamp);
            assert(monitor.update(baseline, timestamp, 1280, 720).new_events.empty());
        }
        assert(!baseline.empty() && baseline.front().suppress_enter);

        const auto new_object_time = base + std::chrono::milliseconds(3400);
        const auto first = stabilizer.update(
            {detection(80, 0, 100.0F, 100.0F), detection(81, 1, 800.0F, 500.0F)},
            new_object_time);
        monitor.update(first, new_object_time, 1280, 720);
        const auto second = stabilizer.update(
            {detection(80, 0, 100.0F, 100.0F), detection(81, 1, 800.0F, 500.0F)},
            base + std::chrono::milliseconds(3800));
        monitor.update(second, base + std::chrono::milliseconds(3800), 1280, 720);
        const auto third = stabilizer.update(
            {detection(80, 0, 100.0F, 100.0F), detection(81, 1, 800.0F, 500.0F)},
            base + std::chrono::milliseconds(4200));
        const auto events = monitor.update(third, base + std::chrono::milliseconds(4200), 1280, 720);
        assert(events.new_events.size() == 1U);
        assert(events.new_events.front().type == edgevision::RegionEventType::Enter);
        assert(events.new_events.front().logical_id != baseline.front().logical_id);
    }
}
