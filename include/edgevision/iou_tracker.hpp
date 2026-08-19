#pragma once

#include "edgevision/core_types.hpp"

#include <cstddef>
#include <vector>

namespace edgevision {

struct IouTrackerConfig {
    // Camera shake can move a small object enough to drop IoU below 0.30
    // even though it is still the same track. The semantic layer remains the
    // identity authority, so keep the raw track alive a little longer.
    float iou_threshold = 0.20F;
    std::size_t max_missed = 6U;
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
