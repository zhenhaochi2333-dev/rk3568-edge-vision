#include "edgevision/semantic_stabilizer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace edgevision {

namespace {

float clamp_unit(float value)
{
    return std::max(0.0F, std::min(1.0F, value));
}

double elapsed_seconds(std::chrono::steady_clock::time_point now,
                       std::chrono::steady_clock::time_point before)
{
    return std::max(0.0, std::chrono::duration<double>(now - before).count());
}

}  // namespace

SemanticStabilizer::SemanticStabilizer(SemanticStabilizerConfig config)
    : config_(config)
{
    if (config_.reassociation_window_seconds <= 0.0 ||
        config_.max_lost_time_seconds < config_.reassociation_window_seconds ||
        config_.reassociation_center_distance_ratio <= 0.0F ||
        config_.reassociation_iou_threshold < 0.0F ||
        config_.reassociation_iou_threshold > 1.0F || config_.presence_alpha <= 0.0F ||
        config_.presence_beta <= 0.0F || config_.enter_threshold < 0.0F ||
        config_.enter_threshold > 1.0F || config_.exit_threshold < 0.0F ||
        config_.exit_threshold > config_.enter_threshold ||
        config_.enter_stability_seconds < 0.0 || config_.bootstrap_mute_seconds < 0.0 ||
        config_.exited_retention_seconds <= 0.0 || config_.max_live_objects == 0U ||
        config_.class_switch_evidence_ratio <= 1.0F ||
        config_.class_switch_hold_seconds < 0.0 ||
        config_.class_evidence_decay_per_second <= 0.0F ||
        config_.class_evidence_decay_per_second > 1.0F) {
        throw std::runtime_error("invalid SemanticStabilizer configuration");
    }
}

void SemanticStabilizer::reset()
{
    objects_.clear();
    next_logical_id_ = 1;
    bootstrap_started_at_.reset();
}

float SemanticStabilizer::area(const cv::Rect2f& box)
{
    return std::max(0.0F, box.width) * std::max(0.0F, box.height);
}

float SemanticStabilizer::intersection_over_union(const cv::Rect2f& first,
                                                  const cv::Rect2f& second)
{
    const float left = std::max(first.x, second.x);
    const float top = std::max(first.y, second.y);
    const float right = std::min(first.x + first.width, second.x + second.width);
    const float bottom = std::min(first.y + first.height, second.y + second.height);
    const float intersection = area(cv::Rect2f(left, top, right - left, bottom - top));
    const float union_area = area(first) + area(second) - intersection;
    return union_area > 0.0F ? intersection / union_area : 0.0F;
}

float SemanticStabilizer::center_distance(const cv::Rect2f& first,
                                          const cv::Rect2f& second)
{
    const float first_x = first.x + first.width * 0.5F;
    const float first_y = first.y + first.height * 0.5F;
    const float second_x = second.x + second.width * 0.5F;
    const float second_y = second.y + second.height * 0.5F;
    return std::sqrt((first_x - second_x) * (first_x - second_x) +
                     (first_y - second_y) * (first_y - second_y));
}

bool SemanticStabilizer::can_reassociate(
    const LogicalObject& object, const Detection& detection,
    std::chrono::steady_clock::time_point source_timestamp, float image_diagonal) const
{
    if (object.state == LogicalObjectState::Exited || !object.initialized ||
        !object.has_seen) {
        return false;
    }

    const double gap = elapsed_seconds(source_timestamp, object.last_seen);
    if (gap > config_.reassociation_window_seconds) {
        return false;
    }

    const float overlap = intersection_over_union(object.last_box, detection.box);
    const float normalized_distance = center_distance(object.last_box, detection.box) /
                                      std::max(1.0F, image_diagonal);
    if (object.stable_class_id >= 0 && detection.class_id != object.stable_class_id) {
        // A class flip may keep the identity only when the new box is almost
        // the same physical box. Without this gate a distant low-confidence
        // class (for example a large furniture box) can inherit the cup's
        // stable label and create a second cup identity beside it.
        return overlap >= 0.15F || normalized_distance <= 0.05F;
    }
    // Center distance is the primary cue. IoU is only auxiliary evidence and
    // does not impose a class-dependent gate.
    return normalized_distance <= config_.reassociation_center_distance_ratio ||
           overlap >= config_.reassociation_iou_threshold;
}

void SemanticStabilizer::update_class_fusion(
    LogicalObject& object, const Detection& detection,
    std::chrono::steady_clock::time_point source_timestamp, double dt_seconds)
{
    const float decay = std::pow(config_.class_evidence_decay_per_second,
                                 static_cast<float>(std::max(0.0, dt_seconds)));
    for (auto& evidence : object.class_evidence) {
        evidence.second *= decay;
    }

    object.class_evidence[detection.class_id] +=
        std::max(0.01F, clamp_unit(detection.confidence));

    // Keep the evidence sparse. The stable class is retained even when it is
    // not in the three strongest current observations.
    while (object.class_evidence.size() > 3U) {
        auto weakest = object.class_evidence.end();
        for (auto it = object.class_evidence.begin(); it != object.class_evidence.end(); ++it) {
            if (it->first == object.stable_class_id) {
                continue;
            }
            if (weakest == object.class_evidence.end() || it->second < weakest->second) {
                weakest = it;
            }
        }
        if (weakest == object.class_evidence.end()) {
            break;
        }
        object.class_evidence.erase(weakest);
    }

    if (object.stable_class_id < 0) {
        object.stable_class_id = detection.class_id;
        object.pending_class_id = -1;
        object.class_switch_started_at.reset();
    } else {
        int strongest_class = object.stable_class_id;
        float strongest_evidence = object.class_evidence[strongest_class];
        for (const auto& evidence : object.class_evidence) {
            if (evidence.second > strongest_evidence) {
                strongest_class = evidence.first;
                strongest_evidence = evidence.second;
            }
        }

        const float stable_evidence = object.class_evidence[object.stable_class_id];
        if (strongest_class != object.stable_class_id &&
            strongest_evidence >= stable_evidence * config_.class_switch_evidence_ratio) {
            if (object.pending_class_id != strongest_class) {
                object.pending_class_id = strongest_class;
                object.class_switch_started_at = source_timestamp;
            } else if (object.class_switch_started_at.has_value() &&
                       elapsed_seconds(source_timestamp,
                                      *object.class_switch_started_at) >=
                           config_.class_switch_hold_seconds) {
                object.stable_class_id = strongest_class;
                object.pending_class_id = -1;
                object.class_switch_started_at.reset();
            }
        } else {
            object.pending_class_id = -1;
            object.class_switch_started_at.reset();
        }
    }

    const float current_confidence = clamp_unit(detection.confidence);
    if (!object.has_seen) {
        object.fused_confidence = current_confidence;
    } else {
        object.fused_confidence = object.fused_confidence * 0.75F + current_confidence * 0.25F;
    }
}

void SemanticStabilizer::evict_for_capacity()
{
    while (objects_.size() > config_.max_live_objects) {
        auto victim = objects_.end();
        int victim_priority = std::numeric_limits<int>::max();
        for (auto it = objects_.begin(); it != objects_.end(); ++it) {
            int priority = 3;
            if (it->state == LogicalObjectState::Exited) {
                priority = 0;
            } else if (it->state == LogicalObjectState::LostPending) {
                priority = 1;
            } else if (it->state == LogicalObjectState::Candidate) {
                priority = 2;
            }
            if (victim == objects_.end() || priority < victim_priority ||
                (priority == victim_priority && it->last_seen < victim->last_seen)) {
                victim = it;
                victim_priority = priority;
            }
        }
        if (victim == objects_.end()) {
            break;
        }
        objects_.erase(victim);
    }
}

std::vector<Detection> SemanticStabilizer::update(
    const std::vector<Detection>& tracked_detections,
    std::chrono::steady_clock::time_point source_timestamp,
    int frame_width, int frame_height)
{
    if (frame_width <= 0 || frame_height <= 0) {
        throw std::runtime_error("SemanticStabilizer requires positive frame dimensions");
    }
    if (!bootstrap_started_at_.has_value()) {
        bootstrap_started_at_ = source_timestamp;
    }

    objects_.erase(std::remove_if(objects_.begin(), objects_.end(),
                                  [&](const LogicalObject& object) {
                                      return object.state == LogicalObjectState::Exited &&
                                             object.initialized &&
                                             elapsed_seconds(source_timestamp, object.exited_at) >=
                                                 config_.exited_retention_seconds;
                                  }),
                   objects_.end());
    evict_for_capacity();

    const float image_diagonal = std::sqrt(static_cast<float>(frame_width * frame_width) +
                                           static_cast<float>(frame_height * frame_height));
    // A detector/NMS pair can briefly return two boxes for one small object
    // when the camera shakes. If both are allowed to create candidates, the
    // second raw track eventually becomes a second logical identity. Collapse
    // only same-class boxes that overlap or touch; separate adjacent objects
    // with a gap remain independent.
    std::vector<Detection> detections;
    detections.reserve(tracked_detections.size());
    for (const Detection& candidate : tracked_detections) {
        int duplicate_index = -1;
        for (std::size_t kept_index = 0U; kept_index < detections.size(); ++kept_index) {
            const Detection& kept = detections[kept_index];
            const float overlap = intersection_over_union(candidate.box, kept.box);
            const float candidate_area = area(candidate.box);
            const float kept_area = area(kept.box);
            const float size_similarity =
                (candidate_area > 0.0F && kept_area > 0.0F)
                    ? std::min(candidate_area, kept_area) /
                          std::max(candidate_area, kept_area)
                    : 0.0F;
            const float smallest_box_scale =
                std::sqrt(std::max(0.0F, std::min(candidate_area, kept_area)));
            const bool same_class_shaking_object =
                overlap >= 0.15F ||
                (overlap > 0.0F && smallest_box_scale > 0.0F &&
                 center_distance(candidate.box, kept.box) <= 0.75F * smallest_box_scale);
            const bool same_box_different_class =
                candidate.class_id != kept.class_id && overlap >= 0.80F &&
                size_similarity >= 0.75F;
            if ((candidate.class_id == kept.class_id && same_class_shaking_object) ||
                same_box_different_class) {
                duplicate_index = static_cast<int>(kept_index);
                break;
            }
        }
        if (duplicate_index < 0) {
            detections.push_back(candidate);
        } else if (candidate.confidence >
                   detections[static_cast<std::size_t>(duplicate_index)].confidence) {
            detections[static_cast<std::size_t>(duplicate_index)] = candidate;
        }
    }

    std::vector<int> assignments(detections.size(), -1);
    std::vector<bool> object_matched(objects_.size(), false);

    // The raw tracker id is the strongest available cue while it remains
    // within the short real-time reassociation window.
    for (std::size_t detection_index = 0U; detection_index < detections.size();
         ++detection_index) {
        const Detection& detection = detections[detection_index];
        if (detection.track_id < 0) {
            continue;
        }
        for (std::size_t object_index = 0U; object_index < objects_.size(); ++object_index) {
            LogicalObject& object = objects_[object_index];
            if (object_matched[object_index] || object.state == LogicalObjectState::Exited ||
                object.raw_track_id != detection.track_id || !object.initialized ||
                !object.has_seen ||
                elapsed_seconds(source_timestamp, object.last_seen) >
                    config_.reassociation_window_seconds) {
                continue;
            }
            assignments[detection_index] = static_cast<int>(object_index);
            object_matched[object_index] = true;
            break;
        }
    }

    // A new raw id can still be the same physical object after a short gap.
    // Greedy matching is sufficient for the small object counts in this
    // embedded path; the score deliberately does not use class as a gate.
    for (std::size_t detection_index = 0U; detection_index < detections.size();
         ++detection_index) {
        if (assignments[detection_index] >= 0) {
            continue;
        }
        const Detection& detection = detections[detection_index];
        int best_object = -1;
        float best_score = -std::numeric_limits<float>::infinity();
        for (std::size_t object_index = 0U; object_index < objects_.size(); ++object_index) {
            const LogicalObject& object = objects_[object_index];
            if (object_matched[object_index] ||
                !can_reassociate(object, detection, source_timestamp, image_diagonal)) {
                continue;
            }

            const float overlap = intersection_over_union(object.last_box, detection.box);
            const float normalized_distance = center_distance(object.last_box, detection.box) /
                                              std::max(1.0F, image_diagonal);
            const float old_area = area(object.last_box);
            const float new_area = area(detection.box);
            const float size_similarity =
                (old_area > 0.0F && new_area > 0.0F)
                    ? std::min(old_area, new_area) / std::max(old_area, new_area)
                    : 0.0F;
            const float distance_score =
                std::exp(-normalized_distance /
                         std::max(0.001F, config_.reassociation_center_distance_ratio));
            const float time_score = clamp_unit(
                1.0F - static_cast<float>(elapsed_seconds(source_timestamp, object.last_seen) /
                                           config_.reassociation_window_seconds));
            const float score = 0.45F * distance_score + 0.20F * size_similarity +
                                0.20F * overlap + 0.15F * time_score;
            if (score > best_score) {
                best_score = score;
                best_object = static_cast<int>(object_index);
            }
        }
        if (best_object >= 0) {
            assignments[detection_index] = best_object;
            object_matched[static_cast<std::size_t>(best_object)] = true;
        }
    }

    // Advance real-time presence for every live object, including objects
    // missing from this detector update.
    std::vector<double> object_dt(objects_.size(), 0.0);
    for (std::size_t object_index = 0U; object_index < objects_.size(); ++object_index) {
        LogicalObject& object = objects_[object_index];
        const double dt = object.initialized
                              ? elapsed_seconds(source_timestamp, object.last_update)
                              : 0.0;
        object_dt[object_index] = dt;
        if (!object_matched[object_index]) {
            object.presence_score = clamp_unit(
                object.presence_score - config_.presence_beta * static_cast<float>(dt));
            if (object.state != LogicalObjectState::Exited) {
                if (object.state == LogicalObjectState::Active) {
                    object.active_before_loss = true;
                }
                object.state = LogicalObjectState::LostPending;
            }
            if (object.has_seen &&
                elapsed_seconds(source_timestamp, object.last_seen) >=
                    config_.max_lost_time_seconds &&
                object.presence_score <= config_.exit_threshold) {
                object.state = LogicalObjectState::Exited;
                object.exited_at = source_timestamp;
            }
            object.last_update = source_timestamp;
        }
    }

    std::vector<Detection> stabilized;
    stabilized.reserve(detections.size());
    for (std::size_t detection_index = 0U; detection_index < detections.size();
         ++detection_index) {
        const Detection& detection = detections[detection_index];
        int object_index = assignments[detection_index];
        if (object_index < 0) {
            if (objects_.size() >= config_.max_live_objects) {
                continue;
            }
            objects_.push_back(LogicalObject{});
            object_index = static_cast<int>(objects_.size() - 1U);
            LogicalObject& object = objects_.back();
            object.logical_id = next_logical_id_++;
            object.first_seen = source_timestamp;
            object.last_seen = source_timestamp;
            object.last_update = source_timestamp;
            object.initialized = true;
            object.has_seen = false;
            object.raw_track_id = detection.track_id;
            object.last_box = detection.box;
            object_matched.push_back(true);
        }

        LogicalObject& object = objects_[static_cast<std::size_t>(object_index)];
        const double dt = object.has_seen
                              ? object_dt[static_cast<std::size_t>(object_index)]
                              : 0.0;
        const bool was_active = object.state == LogicalObjectState::Active ||
                                object.active_before_loss;
        object.presence_score = clamp_unit(
            object.presence_score + config_.presence_alpha * clamp_unit(detection.confidence) *
                                      static_cast<float>(dt));
        object.raw_track_id = detection.track_id;
        object.last_box = detection.box;
        object.last_seen = source_timestamp;
        object.last_update = source_timestamp;
        object.initialized = true;
        update_class_fusion(object, detection, source_timestamp, dt);
        object.has_seen = true;
        object.active_before_loss = false;

        if (was_active) {
            // LOST_PENDING -> ACTIVE recovery never creates a second ENTER.
            object.state = LogicalObjectState::Active;
        } else if (object.state != LogicalObjectState::Exited) {
            object.state = LogicalObjectState::Candidate;
            if (object.presence_score >= config_.enter_threshold) {
                if (!object.enter_threshold_reached_at.has_value()) {
                    object.enter_threshold_reached_at = source_timestamp;
                }
                if (elapsed_seconds(source_timestamp,
                                    *object.enter_threshold_reached_at) >=
                    config_.enter_stability_seconds) {
                    object.state = LogicalObjectState::Active;
                    const double age = elapsed_seconds(source_timestamp, object.first_seen);
                    object.bootstrap_baseline =
                        age < config_.bootstrap_mute_seconds &&
                        elapsed_seconds(source_timestamp, *bootstrap_started_at_) <
                            config_.bootstrap_mute_seconds;
                }
            } else {
                object.enter_threshold_reached_at.reset();
            }
        }

        if (object.state != LogicalObjectState::Active) {
            continue;
        }
        Detection output = detection;
        output.class_id = object.stable_class_id;
        output.confidence = object.fused_confidence;
        output.logical_id = object.logical_id;
        output.presence_score = object.presence_score;
        output.lifecycle_state = LogicalObjectState::Active;
        output.suppress_enter = object.bootstrap_baseline;
        stabilized.push_back(output);
    }

    return stabilized;
}

}  // namespace edgevision
