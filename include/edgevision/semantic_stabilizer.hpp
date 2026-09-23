#pragma once

#include "edgevision/core_types.hpp"

#include <chrono>
#include <cstddef>
#include <map>
#include <optional>
#include <vector>

namespace edgevision {

struct SemanticStabilizerConfig {
    // 短时身份重关联窗口：raw ID 相同的直连匹配和几何重关联都要求距 last_seen 不超过此秒数。
    // 依据源帧时间计算，不是帧数；推理慢、帧率变化或短暂漏检时仍有相同的实际时间含义。
    double reassociation_window_seconds = 2.5;
    // 只有“距 last_seen 至少此时长”且 presence_score 已降至 exit_threshold 以下，才转为 Exited。
    // 必须不小于重关联窗口，否则对象尚可重新关联时就可能已被标记 Exited。
    double max_lost_time_seconds = 4.0;
    // 中心距离除以整幅图像对角线后的容许比例；用于几何重关联的硬门槛之一。
    float reassociation_center_distance_ratio = 0.20F;
    // 同类别几何重关联的辅助 IoU 门槛；中心距离满足门槛或 IoU 满足门槛即可通过。
    float reassociation_iou_threshold = 0.05F;
    // 每秒被观测时 presence_score 增加 alpha * clamp(confidence) * dt；alpha 必须大于 0。
    float presence_alpha = 3.0F;
    // 每秒未被观测时 presence_score 减少 beta * dt；beta 必须大于 0。
    float presence_beta = 0.5F;
    // 首次进入 Active 的上门槛，取值 [0,1]；分数达到后还须满足 enter_stability_seconds。
    float enter_threshold = 0.45F;
    // 只有超出最大丢失时间且分数不高于此下门槛才 Exited；不得高于 enter_threshold。
    float exit_threshold = 0.20F;
    // 分数达到进入门槛后，连续保持门槛的源时间；掉回门槛下会清除计时起点。
    double enter_stability_seconds = 0.20;
    // 启动后此时间内建立的基线对象可成为 Active 并正常输出，但抑制其首次 ROI ENTER。
    double bootstrap_mute_seconds = 3.0;
    // update 开头清理扫描用于比较 Exited 状态时长的阈值；业务 ID 不复用。
    double exited_retention_seconds = 5.0;
    // 内部对象上限。超限时优先淘汰 Exited、LostPending、Candidate，最后才是 Active。
    std::size_t max_live_objects = 50U;
    // 新类别证据须至少达到当前稳定类别证据乘该比例，才启动候选切换计时；必须 > 1。
    float class_switch_evidence_ratio = 1.30F;
    // 候选类别持续占优的最短源时间；为 0 时下一次满足优势条件的观测即可切换。
    double class_switch_hold_seconds = 0.70;
    // 每秒保留的类别证据比例，以 pow(rate, dt) 衰减；允许 (0,1]，1 表示不随时间衰减。
    float class_evidence_decay_per_second = 0.995F;
};

class SemanticStabilizer {
public:
    explicit SemanticStabilizer(SemanticStabilizerConfig config = {});

    // 输入是本轮 IoU 跟踪结果，source_timestamp 必须与这些结果对应的源帧时间一致且单调推进。
    // frame_width/height 是原始图像尺寸，仅用于图像对角线归一化；必须为正数。
    // 输出只含本轮确实观测到的 Active 对象：Candidate 尚未满足进入迟滞条件，LostPending 是
    // 本轮未观测但仍保留的身份，二者都不输出。短暂缺席期间状态仍按时间衰减，不会假装被看见。
    // 输出会以稳定类别、平滑置信度和 logical_id 覆盖相应字段；raw track_id 仍代表本轮短期轨迹。
    std::vector<Detection> update(
        const std::vector<Detection>& tracked_detections,
        std::chrono::steady_clock::time_point source_timestamp,
        int frame_width = 1280,
        int frame_height = 720);

    // 清空候选、活动、丢失和退出对象以及启动静音计时；下一个逻辑对象从 ID 1 重新分配。
    void reset();

    static float intersection_over_union(const cv::Rect2f& first,
                                         const cv::Rect2f& second);

private:
    struct LogicalObject {
        // 跨 raw track_id 变化的稳定业务身份；在 reset 后重新编号，不由此结构持久化到磁盘。
        int logical_id = -1;
        // 最近一次观察到的短期 ID，仅作强优先关联线索，不能单独证明物理身份相同。
        int raw_track_id = -1;
        // 最近观测框，重关联距离、重叠和尺寸相似度都以它为参照。
        cv::Rect2f last_box;
        // 衰减证据融合后的类别，以及正在等待确认的候选类别；-1 表示尚无值。
        int stable_class_id = -1;
        int pending_class_id = -1;
        // 候选类别开始持续占优的源时间；优势中断或类别改变时清空/重置。
        std::optional<std::chrono::steady_clock::time_point> class_switch_started_at;
        // presence 首次达到 enter_threshold 的源时间；低于门槛就清空，Active 后不再用于进入。
        std::optional<std::chrono::steady_clock::time_point> enter_threshold_reached_at;
        // 生命周期起点、最后一次 update 时间、最后一次真实观测时间和进入 Exited 的时间。
        std::chrono::steady_clock::time_point first_seen{};
        std::chrono::steady_clock::time_point last_update{};
        std::chrono::steady_clock::time_point last_seen{};
        std::chrono::steady_clock::time_point exited_at{};
        // 夹在 [0,1] 的连续存在证据。观测按 confidence 和 dt 增长，缺席按 beta 和 dt 衰减。
        float presence_score = 0.0F;
        // 对外输出的置信度低通值：首次观测直接初始化，之后 75% 旧值 + 25% 当前置信度。
        float fused_confidence = 0.0F;
        // Candidate 或 Active 缺席后可进入 LostPending；短时找回可恢复原身份，
        // 超出丢失时间且分数足够低时进入 Exited。
        LogicalObjectState state = LogicalObjectState::Candidate;
        // 启动基线标记：对象稳定激活后传递给 ROI，抑制历史在场对象的补发 ENTER。
        bool bootstrap_baseline = false;
        // initialized 区分空槽/新建槽与已开始按时间推进的对象；has_seen 区分尚未观测的候选。
        bool initialized = false;
        bool has_seen = false;
        // 记录对象曾经 Active 后暂时丢失；再次观测时可直接恢复 Active，避免二次 ENTER。
        bool active_before_loss = false;
        // 每个类别的一项浮点证据；随时间整体衰减，通常最多保留三个类别并保留稳定类别。
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
