#pragma once

#include "edgevision/image_processor.hpp"
#include "edgevision/rknn_model.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace edgevision {

class Yolo11Detector {
public:
    Yolo11Detector(RknnModel& model, float confidence_threshold, float nms_threshold);

    std::vector<Detection> detect(const cv::Mat& bgr);
    DetectionResult detect_with_metrics(const cv::Mat& bgr);

    static std::vector<Detection> decode_raw(const std::vector<RawTensorView>& outputs,
                                             const LetterboxInfo& letterbox,
                                             int model_width,
                                             int model_height,
                                             float confidence_threshold,
                                             float nms_threshold,
                                             std::size_t* candidate_count = nullptr,
                                             std::size_t* suppressed_count = nullptr);

    static std::vector<Detection> nms_for_test(const std::vector<Detection>& detections,
                                               float threshold);

private:
    RknnModel& model_;
    ImageProcessor processor_;
    float confidence_threshold_;
    float nms_threshold_;
};

}  // namespace edgevision
