#pragma once

#include "edgevision/core_types.hpp"

#include <chrono>
#include <cstddef>
#include <map>
#include <vector>

namespace edgevision {

struct SemanticStabilizerConfig {
    float reassociation_iou_threshold = 0.20F;
    float reassociation_center_distance_ratio = 1.50F;
    std::size_t max_missed_frames = 8U;
    std::size_t confirm_frames = 3U;
    float missing_presence_decay = 0.80F;
    float class_evidence_decay = 0.85F;
    std::size_t class_switch_confirmations = 3U;
};

class SemanticStabilizer {
public:
    explicit SemanticStabilizer(SemanticStabilizerConfig config = {});

    // The returned vector contains only confirmed objects observed in this
    // update. Missing objects remain in the internal lifecycle so a short
    // detector/tracker gap does not create a new logical identity.
    std::vector<Detection> update(
        const std::vector<Detection>& tracked_detections,
        std::chrono::steady_clock::time_point source_timestamp);

    void reset();

    static float intersection_over_union(const cv::Rect2f& first,
                                         const cv::Rect2f& second);

private:
    struct LogicalObject {
        int logical_id = -1;
        int raw_track_id = -1;
        cv::Rect2f last_box;
        int stable_class_id = -1;
        int pending_class_id = -1;
        std::size_t pending_class_frames = 0U;
        std::size_t observations = 0U;
        std::size_t missed_frames = 0U;
        float presence_score = 0.0F;
        float fused_confidence = 0.0F;
        LogicalObjectState state = LogicalObjectState::Cold;
        bool ever_confirmed = false;
        std::map<int, float> class_evidence;
        std::chrono::steady_clock::time_point last_seen{};
    };

    static float area(const cv::Rect2f& box);
    static float center_distance(const cv::Rect2f& first,
                                 const cv::Rect2f& second);

    bool can_reassociate(const LogicalObject& object, const Detection& detection) const;
    void update_class_fusion(LogicalObject& object, const Detection& detection);

    SemanticStabilizerConfig config_;
    std::vector<LogicalObject> objects_;
    int next_logical_id_ = 1;
};

}  // namespace edgevision
