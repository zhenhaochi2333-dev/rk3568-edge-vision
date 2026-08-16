#pragma once

#include "edgevision/core_types.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace edgevision {

class Visualizer {
public:
    explicit Visualizer(const std::vector<std::string>& labels);

    void draw(cv::Mat& bgr, const std::vector<Detection>& detections,
              const FrameMetrics& metrics, OverlayMode mode) const;

    static cv::Scalar color_for_class(int class_id);

private:
    const std::vector<std::string>& labels_;
};

}  // namespace edgevision
