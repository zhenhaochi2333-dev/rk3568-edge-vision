#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace edgevision {

enum class LogicalObjectState {
    Candidate,
    Active,
    LostPending,
    Exited,
};

struct Detection {
    int class_id = -1;
    float confidence = 0.0F;
    cv::Rect2f box;
    int track_id = -1;
    int logical_id = -1;
    float presence_score = 0.0F;
    LogicalObjectState lifecycle_state = LogicalObjectState::Candidate;
    // Suppresses the first ROI ENTER for objects that were already stable
    // during the bootstrap mute window. It is metadata-only and is consumed
    // by RegionMonitor; it never changes the logical identity.
    bool suppress_enter = false;
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

enum class InputMode {
    File,
    LocalCamera,
    NetworkCamera,
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
    // Decoder-side evidence kept for diagnostics. detections is already after
    // confidence filtering and NMS; these counters show whether YOLO produced
    // candidates that were later suppressed before tracking.
    std::size_t decoder_candidate_count = 0U;
    std::size_t nms_suppressed_count = 0U;
};

struct AppOptions {
    InputMode input_mode = InputMode::File;
    std::string model_path;
    std::string labels_path;
    std::string input_path;
    std::string camera_path;
    std::string output_path;
    std::string track_log_path;
    // 0.15 keeps low-confidence small/edge-entering objects available to the
    // lifecycle stabilizer. Spatial/class gates still reject duplicate and
    // far-away flicker boxes.
    float conf_threshold = 0.15F;
    float nms_threshold = 0.45F;
    int max_frames = 0;
    bool show = false;
    bool fullscreen = false;
    bool smooth_preview = false;
    bool force = false;
    bool tcp_enabled = false;
    int tcp_port = 9000;
    // The complete camera frame is the default monitoring region; --roi may override it.
    bool roi_enabled = true;
    bool show_roi = false;
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
