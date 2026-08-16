#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace edgevision {

struct Detection {
    int class_id = -1;
    float confidence = 0.0F;
    cv::Rect2f box;
};

struct LetterboxInfo {
    float scale = 1.0F;
    int pad_x = 0;
    int pad_y = 0;
    int original_width = 0;
    int original_height = 0;
    int model_width = 0;
    int model_height = 0;
};

struct PreparedInput {
    cv::Mat rgb_image;
    std::vector<std::uint8_t> nhwc;
    LetterboxInfo letterbox;
};

struct AppOptions {
    std::string model_path;
    std::string labels_path;
    std::string input_path;
    std::string output_path;
    float conf_threshold = 0.25F;
    float nms_threshold = 0.45F;
    int max_frames = 0;
    bool show = false;
    bool force = false;
};

struct TensorMeta {
    int index = -1;
    std::string name;
    std::vector<std::uint32_t> dims;
    std::uint32_t n_elems = 0;
    std::uint32_t size = 0;
    int format = 0;
    int type = 0;
    int quantization = 0;
    std::int32_t zero_point = 0;
    float scale = 1.0F;
};

}  // namespace edgevision
