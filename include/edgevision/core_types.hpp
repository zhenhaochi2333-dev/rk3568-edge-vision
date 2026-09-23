#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace edgevision {

// Candidate 尚在积累出现证据；Active 可参与显示和 ROI 事件；
// LostPending 暂时失联但可能找回；Exited 是本轮对象生命周期已结束。
enum class LogicalObjectState {
    Candidate,
    Active,
    LostPending,
    Exited,
};

// 一条检测在 YOLO、跟踪和稳定器之间逐步补全：class/confidence/box
// 来自检测器，track_id 来自帧间 IoU 关联，logical_id 和 lifecycle_state
// 来自语义稳定器。box 使用源图像素，不是 640x640 模型画布坐标。
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

// 前处理把源图等比例映射到模型画布；scale 和左/上 pad
// 记录这个变换，后处理按 x_source=(x_model-pad_x)/scale 逆变换。
// original_* 用来裁剪逆变换后的框，model_* 表示输入画布尺寸。
struct LetterboxInfo {
    float scale = 1.0F;
    int pad_x = 0;
    int pad_y = 0;
    int original_width = 0;
    int original_height = 0;
    int model_width = 0;
    int model_height = 0;
};

// rgb_image 是 letterbox 后的 RGB Mat；nhwc 是供 RKNN 输入使用的
// 自有连续 uint8 字节。二者与原始 BGR 帧分离，letterbox 留给框坐标逆变换。
struct PreparedInput {
    cv::Mat rgb_image;
    std::vector<std::uint8_t> nhwc;
    LetterboxInfo letterbox;
};

// 归一化 ROI：x/y 是左上角，width/height 是相对整幅源图的比例。
// 如 (0,0,1,1) 覆盖整帧；它不是直接以像素表示的矩形。
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

// 单位为毫秒的阶段耗时与单位为帧/秒的吞吐量放在同一结构。
// inference_ms 仅统计 rknn_run；pre/post 分别统计 CPU 前后处理。
// display_fps 是显示循环，detection_fps 是已完成推理，两者可以不同；
// display_result_age_ms 是 AI 结果完成后到画面使用它时经过的时间。
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

// CLI 解析后的应用配置。input_mode 决定文件、板载摄像头或网络流分支；
// 阈值参与解码/NMS，roi 只在区域监控路径使用。无参默认值在此集中定义。
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

// rknn_query 返回的张量描述副本。dims/format/type 决定如何遍历字节；
// INT8/UINT8 读值时用 (q-zero_point)*scale 还原浮点，不能把输出字节
// 直接当作 float 或假定所有输出共用同一套量化参数。
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

// 只借用 RKNN 输出 buf，同时保存长度和属性副本以供边界检查。
// data 的有效期不得超过对应 RknnOutputBatch；视图自身不负责 release。
struct RawTensorView {
    const void* data = nullptr;
    std::size_t size = 0;
    TensorMeta meta;
};

}  // namespace edgevision
