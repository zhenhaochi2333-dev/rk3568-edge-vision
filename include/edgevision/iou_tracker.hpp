#pragma once

#include "edgevision/core_types.hpp"

#include <cstddef>
#include <vector>

namespace edgevision {

struct IouTrackerConfig {
    float iou_threshold = 0.30F;
    std::size_t max_missed = 4U;
};

class IouTracker {
public:
    explicit IouTracker(IouTrackerConfig config = {});

    std::vector<Detection> update(const std::vector<Detection>& detections);
    void reset();

private:
    struct Track {
        int id = -1;
        Detection detection;
        std::size_t missed = 0U;
    };

    static float intersection_over_union(const cv::Rect2f& first,
                                         const cv::Rect2f& second);

    IouTrackerConfig config_;
    std::vector<Track> tracks_;
    int next_id_ = 1;
};

}  // namespace edgevision
