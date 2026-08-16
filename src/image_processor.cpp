#include "edgevision/image_processor.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace edgevision {

namespace {

int even_down(int value)
{
    return value - (value % 2);
}

float clamp_float(float value, float minimum, float maximum)
{
    return std::max(minimum, std::min(value, maximum));
}

}  // namespace

ImageProcessor::ImageProcessor(int model_width, int model_height)
    : model_width_(model_width), model_height_(model_height)
{
    if (model_width_ <= 0 || model_height_ <= 0) {
        throw std::runtime_error("invalid model dimensions for ImageProcessor");
    }
}

PreparedInput ImageProcessor::prepare(const cv::Mat& bgr) const
{
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::runtime_error("input image must be a non-empty CV_8UC3 BGR image");
    }

    const int source_width = bgr.cols;
    const int source_height = bgr.rows;
    const float scale_width = static_cast<float>(model_width_) / source_width;
    const float scale_height = static_cast<float>(model_height_) / source_height;
    const float scale = std::min(scale_width, scale_height);

    int resize_width = model_width_;
    int resize_height = model_height_;
    if (scale_width < scale_height) {
        resize_height = static_cast<int>(source_height * scale);
    } else {
        resize_width = static_cast<int>(source_width * scale);
    }

    // Match utils/image_utils.c from rknn_model_zoo v1.6.0.
    if (resize_width % 4 != 0) {
        resize_width -= resize_width % 4;
    }
    if (resize_height % 2 != 0) {
        resize_height = even_down(resize_height);
    }
    resize_width = std::max(1, std::min(resize_width, model_width_));
    resize_height = std::max(1, std::min(resize_height, model_height_));

    int pad_x = 0;
    int pad_y = 0;
    if (scale_width < scale_height) {
        pad_y = (model_height_ - resize_height) / 2;
        pad_y = even_down(std::max(0, pad_y));
    } else {
        pad_x = (model_width_ - resize_width) / 2;
        pad_x = even_down(std::max(0, pad_x));
    }

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(resize_width, resize_height), 0.0, 0.0, cv::INTER_LINEAR);
    cv::Mat letterboxed(model_height_, model_width_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(letterboxed(cv::Rect(pad_x, pad_y, resize_width, resize_height)));

    cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
    if (!rgb.isContinuous()) {
        rgb = rgb.clone();
    }

    PreparedInput prepared;
    prepared.rgb_image = rgb;
    prepared.nhwc.resize(static_cast<std::size_t>(rgb.total() * rgb.channels()));
    std::memcpy(prepared.nhwc.data(), rgb.data, prepared.nhwc.size());
    prepared.letterbox.scale = scale;
    prepared.letterbox.pad_x = pad_x;
    prepared.letterbox.pad_y = pad_y;
    prepared.letterbox.original_width = source_width;
    prepared.letterbox.original_height = source_height;
    prepared.letterbox.model_width = model_width_;
    prepared.letterbox.model_height = model_height_;
    return prepared;
}

cv::Rect2f ImageProcessor::restore_box(float x, float y, float width, float height,
                                       const LetterboxInfo& letterbox)
{
    if (letterbox.scale <= 0.0F) {
        throw std::runtime_error("invalid letterbox scale");
    }
    const float x1 = clamp_float(x - static_cast<float>(letterbox.pad_x), 0.0F,
                                 static_cast<float>(letterbox.model_width)) / letterbox.scale;
    const float y1 = clamp_float(y - static_cast<float>(letterbox.pad_y), 0.0F,
                                 static_cast<float>(letterbox.model_height)) / letterbox.scale;
    const float x2 = clamp_float(x + width - static_cast<float>(letterbox.pad_x), 0.0F,
                                 static_cast<float>(letterbox.model_width)) / letterbox.scale;
    const float y2 = clamp_float(y + height - static_cast<float>(letterbox.pad_y), 0.0F,
                                 static_cast<float>(letterbox.model_height)) / letterbox.scale;
    const float right = std::max(x1, std::min(x2, static_cast<float>(letterbox.original_width)));
    const float bottom = std::max(y1, std::min(y2, static_cast<float>(letterbox.original_height)));
    return cv::Rect2f(x1, y1, std::max(0.0F, right - x1), std::max(0.0F, bottom - y1));
}

}  // namespace edgevision
