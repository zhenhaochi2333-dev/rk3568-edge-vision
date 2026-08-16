#include "edgevision/iou_tracker.hpp"

#include <algorithm>
#include <stdexcept>

namespace edgevision {

namespace {

float area(const cv::Rect2f& box)
{
    return std::max(0.0F, box.width) * std::max(0.0F, box.height);
}

}  // namespace

IouTracker::IouTracker(IouTrackerConfig config)
    : config_(config)
{
    if (!(config_.iou_threshold >= 0.0F && config_.iou_threshold <= 1.0F)) {
        throw std::runtime_error("IoU tracker threshold must be within [0, 1]");
    }
}

void IouTracker::reset()
{
    tracks_.clear();
    next_id_ = 1;
}

float IouTracker::intersection_over_union(const cv::Rect2f& first,
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

std::vector<Detection> IouTracker::update(const std::vector<Detection>& detections)
{
    std::vector<bool> matched(detections.size(), false);

    for (Track& track : tracks_) {
        int best_index = -1;
        float best_iou = config_.iou_threshold;
        for (std::size_t index = 0U; index < detections.size(); ++index) {
            if (matched[index] || detections[index].class_id != track.detection.class_id) {
                continue;
            }
            const float current_iou = intersection_over_union(track.detection.box,
                                                              detections[index].box);
            if (current_iou >= best_iou) {
                best_iou = current_iou;
                best_index = static_cast<int>(index);
            }
        }

        if (best_index >= 0) {
            track.detection = detections[static_cast<std::size_t>(best_index)];
            track.detection.track_id = track.id;
            track.missed = 0U;
            matched[static_cast<std::size_t>(best_index)] = true;
        } else {
            ++track.missed;
        }
    }

    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [this](const Track& track) {
                                     return track.missed > config_.max_missed;
                                 }),
                   tracks_.end());

    for (std::size_t index = 0U; index < detections.size(); ++index) {
        if (matched[index]) {
            continue;
        }
        Detection detection = detections[index];
        detection.track_id = next_id_++;
        tracks_.push_back(Track{detection.track_id, detection, 0U});
    }

    std::vector<Detection> output;
    output.reserve(detections.size());
    for (const Track& track : tracks_) {
        if (track.missed == 0U) {
            output.push_back(track.detection);
        }
    }
    return output;
}

}  // namespace edgevision
