#pragma once

#include "edgevision/core_types.hpp"

#include <chrono>
#include <cstddef>
#include <map>
#include <optional>
#include <vector>

namespace edgevision {

struct SemanticStabilizerConfig {
    // The board inference loop is slower than the 15 FPS camera stream. A
    // fixed scene can therefore have a 2 s detector gap without the object
    // having left the frame.
    double reassociation_window_seconds = 2.5;
    double max_lost_time_seconds = 4.0;
    float reassociation_center_distance_ratio = 0.20F;
    float reassociation_iou_threshold = 0.05F;
    // At the validated 15 FPS network input this reaches the enter gate in
    // roughly 0.4 s for a strong detection, while the stability hold keeps
    // one-frame flashes out of the event stream.
    float presence_alpha = 3.0F;
    float presence_beta = 0.5F;
    float enter_threshold = 0.45F;
    float exit_threshold = 0.20F;
    double enter_stability_seconds = 0.20;
    double bootstrap_mute_seconds = 3.0;
    double exited_retention_seconds = 5.0;
    std::size_t max_live_objects = 50U;
    float class_switch_evidence_ratio = 1.30F;
    double class_switch_hold_seconds = 0.70;
    float class_evidence_decay_per_second = 0.995F;
};

class SemanticStabilizer {
public:
    explicit SemanticStabilizer(SemanticStabilizerConfig config = {});

    // The returned vector contains only ACTIVE objects observed in this
    // update. CANDIDATE and LOST_PENDING objects remain internal so a short
    // detector/tracker gap does not create a new logical identity.
    std::vector<Detection> update(
        const std::vector<Detection>& tracked_detections,
        std::chrono::steady_clock::time_point source_timestamp,
        int frame_width = 1280,
        int frame_height = 720);

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
        std::optional<std::chrono::steady_clock::time_point> class_switch_started_at;
        std::optional<std::chrono::steady_clock::time_point> enter_threshold_reached_at;
        std::chrono::steady_clock::time_point first_seen{};
        std::chrono::steady_clock::time_point last_update{};
        std::chrono::steady_clock::time_point last_seen{};
        std::chrono::steady_clock::time_point exited_at{};
        float presence_score = 0.0F;
        float fused_confidence = 0.0F;
        LogicalObjectState state = LogicalObjectState::Candidate;
        bool bootstrap_baseline = false;
        bool initialized = false;
        bool has_seen = false;
        bool active_before_loss = false;
        std::map<int, float> class_evidence;
    };

    static float area(const cv::Rect2f& box);
    static float center_distance(const cv::Rect2f& first,
                                 const cv::Rect2f& second);

    bool can_reassociate(const LogicalObject& object,
                         const Detection& detection,
                         std::chrono::steady_clock::time_point source_timestamp,
                         float image_diagonal) const;
    void update_class_fusion(LogicalObject& object,
                             const Detection& detection,
                             std::chrono::steady_clock::time_point source_timestamp,
                             double dt_seconds);
    void evict_for_capacity();

    SemanticStabilizerConfig config_;
    std::vector<LogicalObject> objects_;
    int next_logical_id_ = 1;
    std::optional<std::chrono::steady_clock::time_point> bootstrap_started_at_;
};

}  // namespace edgevision
