#pragma once

#include "edgevision/core_types.hpp"

namespace edgevision {

class ImageProcessor {
public:
    ImageProcessor(int model_width, int model_height);

    PreparedInput prepare(const cv::Mat& bgr) const;

    static cv::Rect2f restore_box(float x, float y, float width, float height,
                                  const LetterboxInfo& letterbox);

private:
    int model_width_;
    int model_height_;
};

}  // namespace edgevision
