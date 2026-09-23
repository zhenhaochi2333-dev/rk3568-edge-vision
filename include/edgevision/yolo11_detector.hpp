#pragma once

#include "edgevision/image_processor.hpp"
#include "edgevision/rknn_model.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace edgevision {

// 完整检测入口：BGR 源帧 -> letterbox/RGB -> RKNN -> YOLO11 DFL/NMS
// -> 源图坐标的 Detection。模型引用不归本类所有。
class Yolo11Detector {
public:
    Yolo11Detector(RknnModel& model, float confidence_threshold, float nms_threshold);

    std::vector<Detection> detect(const cv::Mat& bgr);
    DetectionResult detect_with_metrics(const cv::Mat& bgr);

    // 纯 CPU 解码入口，供真实 RKNN 输出和离线样例共用。当前模型要求
    // 三个尺度、每尺度 [DFL box, 80 类, score_sum] 共 9 张量；输出
    // 视图只借用缓冲，调用者必须维持其有效期。
    static std::vector<Detection> decode_raw(const std::vector<RawTensorView>& outputs,
                                             const LetterboxInfo& letterbox,
                                             int model_width,
                                             int model_height,
                                             float confidence_threshold,
                                             float nms_threshold,
                                             std::size_t* candidate_count = nullptr,
                                             std::size_t* suppressed_count = nullptr);

    // 暴露同一 NMS 逻辑供单元验证，避免测试依赖实际板端 RKNN Runtime。
    static std::vector<Detection> nms_for_test(const std::vector<Detection>& detections,
                                               float threshold);

private:
    RknnModel& model_;
    ImageProcessor processor_;
    float confidence_threshold_;
    float nms_threshold_;
};

}  // namespace edgevision
