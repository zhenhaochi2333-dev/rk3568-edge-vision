#pragma once

#include "edgevision/core_types.hpp"

#include <cstddef>
#include <vector>

namespace edgevision {

struct IouTrackerConfig {
    // IoU 是当前检测框与上次已匹配框的交并比；它只描述几何重叠，不是运动预测或物体身份概率。
    // 阈值 0.20（含边界）允许相机抖动造成的小目标位移仍沿用 raw track_id；较低阈值也会增加
    // 邻近同类目标互相争用轨迹的机会。匹配还要求 class_id 完全相同，因此类别闪烁会新建 raw ID。
    // 这里的 ID 只在短期关联有效；跨跟踪器 ID 变化的业务身份由 SemanticStabilizer 负责。
    float iou_threshold = 0.20F;
    // 未匹配轨迹在 update() 中逐次计数；只有 missed > max_missed 才删除。
    // 因而默认 6 表示容忍最多 6 次缺席更新，第 7 次缺席后才移除；缺席期间不会出现在输出中。
    std::size_t max_missed = 6U;
};

class IouTracker {
public:
    // 配置范围在构造时校验：IoU 阈值必须位于闭区间 [0, 1]；max_missed 可为 0。
    explicit IouTracker(IouTrackerConfig config = {});

    // 输入是一帧检测，输出仅包含本帧成功匹配或新建的检测，并附上 raw track_id。
    // 对每条既有轨迹贪心挑选同类、未被占用且 IoU 达阈值的最佳框；检测和轨迹是一对一关系。
    // 此法不做全局最优分配、速度估计或遮挡推断，且相同 IoU 时后出现的候选会覆盖先前候选。
    std::vector<Detection> update(const std::vector<Detection>& detections);
    // 清空全部短期状态并将 raw ID 计数重新从 1 开始；调用方应把它视为跟踪会话重启。
    void reset();

private:
    struct Track {
        // 本跟踪器会话中分配的 raw ID，不等同于稳定器分配的 logical_id。
        int id = -1;
        // 最近一次匹配到的完整检测，下一帧以其中的框和类别作关联参考。
        Detection detection;
        // 连续未匹配的 update 次数；匹配即归零。
        std::size_t missed = 0U;
    };

    static float intersection_over_union(const cv::Rect2f& first,
                                         const cv::Rect2f& second);

    IouTrackerConfig config_;
    std::vector<Track> tracks_;
    int next_id_ = 1;
};

}  // namespace edgevision
