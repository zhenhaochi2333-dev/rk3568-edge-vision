#include "edgevision/semantic_stabilizer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace edgevision {

namespace {

float clamp_confidence(float confidence)
{
    return std::max(0.0F, std::min(1.0F, confidence));
}

}  // namespace

SemanticStabilizer::SemanticStabilizer(SemanticStabilizerConfig config)
    : config_(config)
{
    if (!(config_.reassociation_iou_threshold >= 0.0F &&
          config_.reassociation_iou_threshold <= 1.0F) ||
        config_.reassociation_center_distance_ratio <= 0.0F ||
        config_.confirm_frames == 0U || config_.missing_presence_decay < 0.0F ||
        config_.missing_presence_decay > 1.0F || config_.class_evidence_decay < 0.0F ||
        config_.class_evidence_decay > 1.0F || config_.class_switch_confirmations == 0U) {
        throw std::runtime_error("invalid SemanticStabilizer configuration");
    }
}

void SemanticStabilizer::reset()
{
    objects_.clear();
    next_logical_id_ = 1;
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

bool SemanticStabilizer::can_reassociate(const LogicalObject& object,
                                          const Detection& detection) const
{
    if (object.state == LogicalObjectState::Removed ||
        object.missed_frames > config_.max_missed_frames) {
        return false;
    }
    if (detection.track_id >= 0 && detection.track_id == object.raw_track_id) {
        return true;
    }

    const float overlap = intersection_over_union(object.last_box, detection.box);
    const float reference_size = std::max(1.0F, std::sqrt(area(object.last_box)));
    const float normalized_distance = center_distance(object.last_box, detection.box) /
                                      reference_size;
    return overlap >= config_.reassociation_iou_threshold ||
           normalized_distance <= config_.reassociation_center_distance_ratio;
}

void SemanticStabilizer::update_class_fusion(LogicalObject& object,
                                             const Detection& detection)
{
    for (auto& evidence : object.class_evidence) {
        evidence.second *= config_.class_evidence_decay;
    }
    object.class_evidence[detection.class_id] += std::max(0.05F, clamp_confidence(detection.confidence));

    if (object.stable_class_id < 0) {
        object.stable_class_id = detection.class_id;
        object.pending_class_id = -1;
        object.pending_class_frames = 0U;
    } else if (detection.class_id == object.stable_class_id) {
        object.pending_class_id = -1;
        object.pending_class_frames = 0U;
    } else if (object.pending_class_id == detection.class_id) {
        ++object.pending_class_frames;
        if (object.pending_class_frames >= config_.class_switch_confirmations) {
            object.stable_class_id = detection.class_id;
            object.pending_class_id = -1;
            object.pending_class_frames = 0U;
        }
    } else {
        object.pending_class_id = detection.class_id;
        object.pending_class_frames = 1U;
    }

    const float current_confidence = clamp_confidence(detection.confidence);
    if (object.observations == 0U) {
        object.fused_confidence = current_confidence;
    } else {
        object.fused_confidence = object.fused_confidence * 0.75F + current_confidence * 0.25F;
    }
}

std::vector<Detection> SemanticStabilizer::update(
    const std::vector<Detection>& tracked_detections,
    std::chrono::steady_clock::time_point source_timestamp)
{
    std::vector<int> assignments(tracked_detections.size(), -1);
    std::vector<bool> object_matched(objects_.size(), false);

    // Prefer the raw tracker identity while it is still available. This is
    // both cheaper and more precise than geometric matching.
    for (std::size_t detection_index = 0U; detection_index < tracked_detections.size();
         ++detection_index) {
        const Detection& detection = tracked_detections[detection_index];
        if (detection.track_id < 0) {
            continue;
        }
        for (std::size_t object_index = 0U; object_index < objects_.size(); ++object_index) {
            if (object_matched[object_index] ||
                objects_[object_index].raw_track_id != detection.track_id ||
                !can_reassociate(objects_[object_index], detection)) {
                continue;
            }
            assignments[detection_index] = static_cast<int>(object_index);
            object_matched[object_index] = true;
            break;
        }
    }

    // A new raw track id can still be the same physical object after a short
    // detector gap. Match class-agnostically so one-frame class flicker does
    // not prevent reassociation.
    for (std::size_t detection_index = 0U; detection_index < tracked_detections.size();
         ++detection_index) {
        if (assignments[detection_index] >= 0) {
            continue;
        }
        int best_object = -1;
        float best_score = -std::numeric_limits<float>::infinity();
        const Detection& detection = tracked_detections[detection_index];
        for (std::size_t object_index = 0U; object_index < objects_.size(); ++object_index) {
            if (object_matched[object_index] || !can_reassociate(objects_[object_index], detection)) {
                continue;
            }
            const float overlap = intersection_over_union(objects_[object_index].last_box,
                                                         detection.box);
            const float reference_size = std::max(1.0F, std::sqrt(area(objects_[object_index].last_box)));
            const float normalized_distance = center_distance(objects_[object_index].last_box,
                                                              detection.box) /
                                              reference_size;
            const float score = overlap +
                                std::max(0.0F, 1.0F - normalized_distance /
                                                   config_.reassociation_center_distance_ratio) *
                                    0.10F;
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

    for (std::size_t object_index = 0U; object_index < objects_.size(); ++object_index) {
        LogicalObject& object = objects_[object_index];
        if (object_matched[object_index]) {
            continue;
        }
        ++object.missed_frames;
        object.presence_score *= config_.missing_presence_decay;
        object.state = object.missed_frames > config_.max_missed_frames
                           ? LogicalObjectState::Removed
                           : (object.ever_confirmed ? LogicalObjectState::Missing
                                                    : LogicalObjectState::Cold);
    }

    std::vector<Detection> stabilized;
    stabilized.reserve(tracked_detections.size());
    for (std::size_t detection_index = 0U; detection_index < tracked_detections.size();
         ++detection_index) {
        const Detection& detection = tracked_detections[detection_index];
        int object_index = assignments[detection_index];
        if (object_index < 0) {
            objects_.push_back(LogicalObject{});
            object_index = static_cast<int>(objects_.size() - 1U);
            LogicalObject& object = objects_.back();
            object.logical_id = next_logical_id_++;
            object.state = LogicalObjectState::Cold;
        }

        LogicalObject& object = objects_[static_cast<std::size_t>(object_index)];
        object.raw_track_id = detection.track_id;
        object.last_box = detection.box;
        object.last_seen = source_timestamp;
        object.missed_frames = 0U;
        object.presence_score = std::min(
            1.0F, object.presence_score + 1.0F / static_cast<float>(config_.confirm_frames));
        update_class_fusion(object, detection);
        ++object.observations;
        if (object.ever_confirmed) {
            object.state = LogicalObjectState::Confirmed;
        } else if (object.observations >= config_.confirm_frames) {
            object.ever_confirmed = true;
            object.state = LogicalObjectState::Confirmed;
        } else {
            object.state = LogicalObjectState::Cold;
        }

        if (object.state != LogicalObjectState::Confirmed) {
            continue;  // cold-start mute
        }
        Detection output = detection;
        output.class_id = object.stable_class_id;
        output.confidence = object.fused_confidence;
        output.logical_id = object.logical_id;
        output.presence_score = object.presence_score;
        output.lifecycle_state = object.state;
        stabilized.push_back(output);
    }

    objects_.erase(std::remove_if(objects_.begin(), objects_.end(),
                                  [](const LogicalObject& object) {
                                      return object.state == LogicalObjectState::Removed;
                                  }),
                   objects_.end());
    return stabilized;
}

}  // namespace edgevision
