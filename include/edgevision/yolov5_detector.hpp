#pragma once

#include "edgevision/image_processor.hpp"
#include "edgevision/rknn_model.hpp"

#include <cstddef>
#include <vector>

namespace edgevision {

struct RawTensorView {
    const void* data = nullptr;
    std::size_t size = 0;
    TensorMeta meta;
};

class Yolov5Detector {
public:
    Yolov5Detector(RknnModel& model, float confidence_threshold, float nms_threshold);

    std::vector<Detection> detect(const cv::Mat& bgr);

    static std::vector<Detection> decode_raw(const std::vector<RawTensorView>& outputs,
                                             const LetterboxInfo& letterbox,
                                             int model_width,
                                             int model_height,
                                             float confidence_threshold,
                                             float nms_threshold);

    static std::vector<Detection> nms_for_test(const std::vector<Detection>& detections,
                                               float threshold);

private:
    RknnModel& model_;
    ImageProcessor processor_;
    float confidence_threshold_;
    float nms_threshold_;
};

}  // namespace edgevision
