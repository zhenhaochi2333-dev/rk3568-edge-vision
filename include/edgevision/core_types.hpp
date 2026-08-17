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
    int track_id = -1;
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

struct NormalizedRoi {
    float x = 0.0F;
    float y = 0.0F;
    float width = 1.0F;
    float height = 1.0F;
};

struct FrameMetrics {
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    double visualization_ms = 0.0;
    double end_to_end_ms = 0.0;
    double fps = 0.0;
    double display_fps = 0.0;
    double detection_fps = 0.0;
    double ai_latency_ms = 0.0;
    double display_result_age_ms = 0.0;
    double object_count = 0.0;
};

struct DetectionResult {
    std::vector<Detection> detections;
    FrameMetrics metrics;
};

enum class OverlayMode {
    Image,
    Video,
    SmoothVideo,
};

struct AppOptions {
    std::string model_path;
    std::string labels_path;
    std::string input_path;
    std::string camera_path;
    std::string output_path;
    float conf_threshold = 0.25F;
    float nms_threshold = 0.45F;
    int max_frames = 0;
    bool show = false;
    bool fullscreen = false;
    bool smooth_preview = false;
    bool force = false;
    bool roi_enabled = false;
    NormalizedRoi roi;
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

struct RawTensorView {
    const void* data = nullptr;
    std::size_t size = 0;
    TensorMeta meta;
};

}  // namespace edgevision
