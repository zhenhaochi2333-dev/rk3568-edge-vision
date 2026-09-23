/* 轻量逐帧 IoU 关联产生短期 track_id；业务上的连续存在身份由后续稳定器维护。 */
#include "edgevision/iou_tracker.hpp"

#include <algorithm>
#include <stdexcept>

namespace edgevision {

namespace {

float area(const cv::Rect2f& box)
{
    // 将负宽/高按零处理，避免无效矩形产生负面积并污染 IoU 分母。
    return std::max(0.0F, box.width) * std::max(0.0F, box.height);
}

}  // namespace

IouTracker::IouTracker(IouTrackerConfig config)
    : config_(config)
{
    if (!(config_.iou_threshold >= 0.0F && config_.iou_threshold <= 1.0F)) {
        // !(a && b) 也会拒绝 NaN，因为 NaN 与两端比较都为 false。
        throw std::runtime_error("IoU tracker threshold must be within [0, 1]");
    }
}

void IouTracker::reset()
{
    // 同时重置编号和状态；旧 track_id 可能在 reset 后被新对象再次使用，不能跨会话比较。
    tracks_.clear();
    next_id_ = 1;
}

float IouTracker::intersection_over_union(const cv::Rect2f& first,
                                          const cv::Rect2f& second)
{
    // 先求两个半开矩形的交集，再用并集归一化；没有正面积的并集定义为无重叠。
    const float left = std::max(first.x, second.x);
    const float top = std::max(first.y, second.y);
    const float right = std::min(first.x + first.width, second.x + second.width);
    const float bottom = std::min(first.y + first.height, second.y + second.height);
    const float intersection = area(cv::Rect2f(left, top, right - left, bottom - top));
    const float union_area = area(first) + area(second) - intersection;
    return union_area > 0.0F ? intersection / union_area : 0.0F;
}

// 每条旧轨迹选同类别且 IoU 达阈值的未匹配框。遍历按 tracks_ 顺序贪心占用检测，
// 所以多个轨迹争同一检测时先处理者优先；这是轻量关联，不是全局最佳的二分图分配。
std::vector<Detection> IouTracker::update(const std::vector<Detection>& detections)
{
    std::vector<bool> matched(detections.size(), false);

    for (Track& track : tracks_) {
        int best_index = -1;
        // 初值直接取门槛，故恰好等于门槛也可匹配；best_index 仍用 -1 区分无候选。
        float best_iou = config_.iou_threshold;
        for (std::size_t index = 0U; index < detections.size(); ++index) {
            if (matched[index] || detections[index].class_id != track.detection.class_id) {
                continue;
            }
            const float current_iou = intersection_over_union(track.detection.box,
                                                              detections[index].box);
            // >= 意味着相同 IoU 时后出现的 detection 会成为当前最佳项。
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
            // 漏检只累计，不把旧框复制到输出，避免下游把预测框误当成本帧观测。
            ++track.missed;
        }
    }

    // 连续漏检超过上限才删轨迹：max_missed=N 时可保留 N 次空更新，第 N+1 次删除。
    // 删除发生在分配本帧新检测之前；若该物体此时才回来，会拿到新的 raw track_id。
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
        // 仅复制检测后写入 tracker 元数据，类别、置信度和框仍取检测器本轮结果。
        detection.track_id = next_id_++;
        tracks_.push_back(Track{detection.track_id, detection, 0U});
    }

    std::vector<Detection> output;
    output.reserve(detections.size());
    for (const Track& track : tracks_) {
        // 内部保留的 missed 轨迹只为未来重连；输出严格是当前帧观察到的对象集合。
        if (track.missed == 0U) {
            output.push_back(track.detection);
        }
    }
    return output;
}

}  // namespace edgevision
