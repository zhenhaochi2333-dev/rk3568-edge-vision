/* raw track_id -> 连续存在的 logical_id：以源帧时间推进短时重关联、
 * 置信度证据、类别融合与 Candidate/Active/LostPending/Exited 生命周期。
 */
#include "edgevision/semantic_stabilizer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace edgevision {

namespace {

float clamp_unit(float value)
{
    // presence 和置信度运算统一夹在概率尺度内，避免累积超出 [0,1]。
    return std::max(0.0F, std::min(1.0F, value));
}

double elapsed_seconds(std::chrono::steady_clock::time_point now,
                       std::chrono::steady_clock::time_point before)
{
    // 所有生命周期计时采用输入图像的单调源时间；倒序时间按零间隔处理，避免负 dt 反向加分。
    return std::max(0.0, std::chrono::duration<double>(now - before).count());
}

}  // namespace

SemanticStabilizer::SemanticStabilizer(SemanticStabilizerConfig config)
    : config_(config)
{
    // 构造时一次性校验配置，使后续除法、阈值比较和幂运算都具有定义良好的边界。
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
    // 新会话丢弃所有历史证据；下一个对象重新从 logical_id=1 编号。
    objects_.clear();
    next_logical_id_ = 1;
    bootstrap_started_at_.reset();
}

float SemanticStabilizer::area(const cv::Rect2f& box)
{
    // 无效框的负边长按零计面积，后续尺寸相似度/IoU 因而退化为零证据。
    return std::max(0.0F, box.width) * std::max(0.0F, box.height);
}

float SemanticStabilizer::intersection_over_union(const cv::Rect2f& first,
                                                  const cv::Rect2f& second)
{
    // 与 IoU tracker 同一几何定义；退化到零并集时返回零而不是 NaN。
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
    // 以框中心衡量位移，尺寸变化不会直接改变距离；随后除以图像对角线获得尺寸无关比例。
    const float first_x = first.x + first.width * 0.5F;
    const float first_y = first.y + first.height * 0.5F;
    const float second_x = second.x + second.width * 0.5F;
    const float second_y = second.y + second.height * 0.5F;
    return std::sqrt((first_x - second_x) * (first_x - second_x) +
                     (first_y - second_y) * (first_y - second_y));
}

// 短时丢失后用框位置/重叠判断是否仍为同一物体，避免 raw track_id 变化就重发事件。
bool SemanticStabilizer::can_reassociate(
    const LogicalObject& object, const Detection& detection,
    std::chrono::steady_clock::time_point source_timestamp, float image_diagonal) const
{
    if (object.state == LogicalObjectState::Exited || !object.initialized ||
        !object.has_seen) {
        return false;
    }

    const double gap = elapsed_seconds(source_timestamp, object.last_seen);
    // 允许等于窗口上界；超过后不再复用身份，后续会创建新的 logical_id。
    if (gap > config_.reassociation_window_seconds) {
        return false;
    }

    const float overlap = intersection_over_union(object.last_box, detection.box);
    const float normalized_distance = center_distance(object.last_box, detection.box) /
                                      std::max(1.0F, image_diagonal);
    if (object.stable_class_id >= 0 && detection.class_id != object.stable_class_id) {
        // 稳定类别发生变化时，仅当新框仍几乎覆盖同一物理位置才准许复用身份。
        // 否则远处的异类低置信框可能继承旧物体的稳定标签，令一个新目标被错误并入旧身份。
        return overlap >= 0.15F || normalized_distance <= 0.05F;
    }
    // 同类别时中心距离与 IoU 是“满足任一即可”的几何门槛；中心距离是主线索，IoU 为辅助。
    return normalized_distance <= config_.reassociation_center_distance_ratio ||
           overlap >= config_.reassociation_iou_threshold;
}

// 历史类别证据按时间衰减，新观测按置信度累加；切换稳定类别还需优势和持续时间。
void SemanticStabilizer::update_class_fusion(
    LogicalObject& object, const Detection& detection,
    std::chrono::steady_clock::time_point source_timestamp, double dt_seconds)
{
    // 衰减率定义为“每秒保留比例”，dt=0 不衰减；低于 1 时长时间未见的旧证据逐渐淡化。
    const float decay = std::pow(config_.class_evidence_decay_per_second,
                                 static_cast<float>(std::max(0.0, dt_seconds)));
    for (auto& evidence : object.class_evidence) {
        evidence.second *= decay;
    }

    // 置信度以 [0,1] 加入，最低 0.01 保证极低置信观测仍留下非零类别证据。
    object.class_evidence[detection.class_id] +=
        std::max(0.01F, clamp_unit(detection.confidence));

    // 限制证据表最多三个类别，控制长期运行的状态大小；即使稳定类别暂时不在前三，也优先保留它。
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

    // 第一次观测立即确定初始类别；之后类别改变须证据优势达标，并由同一候选持续 hold 时间。
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
    // 置信度平滑与类别票数分开：当前检测质量只占四分之一，减轻单帧置信波动。
    if (!object.has_seen) {
        object.fused_confidence = current_confidence;
    } else {
        object.fused_confidence = object.fused_confidence * 0.75F + current_confidence * 0.25F;
    }
}

void SemanticStabilizer::evict_for_capacity()
{
    // 限额是防止长时间运行时对象表无界增长；优先删已结束对象，活动对象仅作为最后手段。
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

// 只返回本轮实际看到的 Active 对象；内部仍保存候选和短时丢失对象以维护 logical_id。
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

    // 先清过期 Exited，再做容量淘汰；过期比较用 >=，时间恰达保留期限即移除。
    objects_.erase(std::remove_if(objects_.begin(), objects_.end(),
                                  [&](const LogicalObject& object) {
                                      return object.state == LogicalObjectState::Exited &&
                                             object.initialized &&
                                             elapsed_seconds(source_timestamp, object.exited_at) >=
                                                 config_.exited_retention_seconds;
                                  }),
                   objects_.end());
    evict_for_capacity();

    // 归一化位置尺度，以原始帧对角线为单位；框坐标必须与该帧尺寸同一坐标系。
    const float image_diagonal = std::sqrt(static_cast<float>(frame_width * frame_width) +
                                           static_cast<float>(frame_height * frame_height));
    // A detector/NMS pair can briefly return two boxes for one small object
    // when the camera shakes. If both are allowed to create candidates, the
    // second raw track eventually becomes a second logical identity. Collapse
    // only same-class boxes that overlap or touch; separate adjacent objects
    // with a gap remain independent.
    // 输入去重发生在分配之前：同类近重框或几乎完全重叠的异类框只保留置信度较高者。
    // 这避免一次检测抖动为同一实体创建两个候选逻辑身份；只比较与已保留框，不做聚类传递闭包。
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
            // 同类框要有明显重叠，或有轻微重叠且中心距离不超过较小框等效边长的 0.75 倍。
            const bool same_class_shaking_object =
                overlap >= 0.15F ||
                (overlap > 0.0F && smallest_box_scale > 0.0F &&
                 center_distance(candidate.box, kept.box) <= 0.75F * smallest_box_scale);
            // 异类只有高度重合且面积接近时才折叠，避免把相邻或尺度悬殊的真实目标合并。
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

    // 关联分两阶段：先用未过短时窗口的 raw ID 直连，再为剩余检测做几何/时间贪心重关联。
    // object_matched 保证同一个既有 logical object 在本次 update 最多被一个检测认领。
    std::vector<int> assignments(detections.size(), -1);
    std::vector<bool> object_matched(objects_.size(), false);

    // raw ID 在短时窗口内是最强的关联线索，但它会因漏检或 tracker 重置变化，不能作为永久身份。
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

    // raw ID 已变化时仍可通过短时几何证据找回同一物体。对象数较少时采用贪心评分即可，
    // 评分综合中心距离、尺寸相似、IoU 和剩余时间窗口比例；类别不作为硬门槛以允许类别闪烁。
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

    // 对所有旧对象先推进一次时间，包括本轮缺席者；命中对象的 dt 暂存到后面用于正向更新。
    // Advance real-time presence for every live object, including objects
    // missing from this detector update.
    // presence 按真实时间升降，而非简单帧数；检测慢或丢帧时生命周期仍随时间推进。
    std::vector<double> object_dt(objects_.size(), 0.0);
    for (std::size_t object_index = 0U; object_index < objects_.size(); ++object_index) {
        LogicalObject& object = objects_[object_index];
        const double dt = object.initialized
                              ? elapsed_seconds(source_timestamp, object.last_update)
                              : 0.0;
        object_dt[object_index] = dt;
        if (!object_matched[object_index]) {
            // 漏检时 presence 线性按 beta*秒下降，并进入 LostPending；Candidate 也保留以便短时续接。
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
                // 两个条件必须同时成立才终结身份：丢失时间足够长且残余存在分数低于退出门槛。
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
            // 新逻辑身份从首帧建立 Candidate；若本帧再无容量则上面会跳过该检测，不分配部分对象。
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
        // 有效观测按真实 dt 增加存在证据；置信度为零时不会加分，但仍更新“看见”时间和框。
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

        // 已激活对象从短时丢失恢复后沿用 logical_id；新候选需跨越进入门限和稳定时长。
        // 活动身份短时恢复不重新经过 ENTER 门槛，避免在 ROI 端产生重复进入事件。
        if (was_active) {
            // LostPending 恢复为 Active 时沿用原业务身份，不再次制造 ENTER。
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
                    // 静音只针对启动早期已出现的基线对象；后续新对象不受全局启动时间影响。
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

        // 进入前的候选不会向显示/ROI 泄漏；成为 Active 的这一帧立即进入输出。
        if (object.state != LogicalObjectState::Active) {
            continue;
        }
        Detection output = detection;
        output.class_id = object.stable_class_id;
        output.confidence = object.fused_confidence;
        output.logical_id = object.logical_id;
        output.presence_score = object.presence_score;
        output.lifecycle_state = LogicalObjectState::Active;
        // 启动阶段已有的稳定对象仍可显示；ROI 据此跳过它的首次 ENTER。
        output.suppress_enter = object.bootstrap_baseline;
        stabilized.push_back(output);
    }

    return stabilized;
}

}  // namespace edgevision
