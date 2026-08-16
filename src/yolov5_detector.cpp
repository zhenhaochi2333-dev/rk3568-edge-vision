#include "edgevision/yolov5_detector.hpp"

#include "rknn_api.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace edgevision {

namespace {

constexpr int kClassCount = 80;
constexpr int kValuesPerAnchor = 5 + kClassCount;
constexpr int kAnchors[3][6] = {
    {10, 13, 16, 30, 33, 23},
    {30, 61, 62, 45, 59, 119},
    {116, 90, 156, 198, 373, 326},
};

struct Candidate {
    float x = 0.0F;
    float y = 0.0F;
    float width = 0.0F;
    float height = 0.0F;
    float score = 0.0F;
    int class_id = -1;
};

bool is_quantized(const TensorMeta& meta)
{
    return meta.type == static_cast<int>(RKNN_TENSOR_INT8) ||
           meta.type == static_cast<int>(RKNN_TENSOR_UINT8);
}

std::size_t scalar_size(const TensorMeta& meta)
{
    if (is_quantized(meta)) {
        return 1U;
    }
    if (meta.type == static_cast<int>(RKNN_TENSOR_FLOAT32)) {
        return sizeof(float);
    }
    throw std::runtime_error("unsupported RKNN output type in YOLOv5 decoder: " +
                             std::to_string(meta.type));
}

void check_index(const RawTensorView& view, std::size_t index)
{
    if (view.data == nullptr || (index + 1U) * scalar_size(view.meta) > view.size) {
        throw std::runtime_error("RKNN output buffer is smaller than queried tensor metadata");
    }
}

int quantized_value(const RawTensorView& view, std::size_t index)
{
    check_index(view, index);
    if (view.meta.type == static_cast<int>(RKNN_TENSOR_INT8)) {
        return static_cast<const std::int8_t*>(view.data)[index];
    }
    return static_cast<const std::uint8_t*>(view.data)[index];
}

float scalar_value(const RawTensorView& view, std::size_t index)
{
    check_index(view, index);
    if (is_quantized(view.meta)) {
        return (static_cast<float>(quantized_value(view, index)) -
                static_cast<float>(view.meta.zero_point)) * view.meta.scale;
    }
    return static_cast<const float*>(view.data)[index];
}

int quantized_threshold(const TensorMeta& meta, float threshold)
{
    const float raw = threshold / meta.scale + static_cast<float>(meta.zero_point);
    const int minimum = meta.type == static_cast<int>(RKNN_TENSOR_UINT8) ? 0 : -128;
    const int maximum = meta.type == static_cast<int>(RKNN_TENSOR_UINT8) ? 255 : 127;
    return std::max(minimum, std::min(static_cast<int>(raw), maximum));
}

bool passes_threshold(const RawTensorView& view, std::size_t index, float threshold)
{
    if (is_quantized(view.meta)) {
        return quantized_value(view, index) >= quantized_threshold(view.meta, threshold);
    }
    return scalar_value(view, index) >= threshold;
}

std::size_t value_index(const TensorMeta& meta, int channels, int grid_height, int grid_width,
                        int channel, int row, int column)
{
    if (meta.format == static_cast<int>(RKNN_TENSOR_NCHW)) {
        return static_cast<std::size_t>(channel) * static_cast<std::size_t>(grid_height) *
                   static_cast<std::size_t>(grid_width) +
               static_cast<std::size_t>(row) * static_cast<std::size_t>(grid_width) +
               static_cast<std::size_t>(column);
    }
    if (meta.format == static_cast<int>(RKNN_TENSOR_NHWC)) {
        return (static_cast<std::size_t>(row) * static_cast<std::size_t>(grid_width) +
                static_cast<std::size_t>(column)) * static_cast<std::size_t>(channels) +
               static_cast<std::size_t>(channel);
    }
    throw std::runtime_error("unsupported RKNN output format in YOLOv5 decoder");
}

float overlap_iou(const Candidate& a, const Candidate& b)
{
    const float left = std::max(a.x, b.x);
    const float top = std::max(a.y, b.y);
    const float right = std::min(a.x + a.width, b.x + b.width);
    const float bottom = std::min(a.y + a.height, b.y + b.height);
    const float width = std::max(0.0F, right - left + 1.0F);
    const float height = std::max(0.0F, bottom - top + 1.0F);
    const float intersection = width * height;
    const float area_a = (a.width + 1.0F) * (a.height + 1.0F);
    const float area_b = (b.width + 1.0F) * (b.height + 1.0F);
    const float union_area = area_a + area_b - intersection;
    return union_area <= 0.0F ? 0.0F : intersection / union_area;
}

std::vector<Candidate> apply_reference_nms(std::vector<Candidate> candidates, float threshold)
{
    std::vector<int> order(candidates.size());
    for (std::size_t index = 0; index < order.size(); ++index) {
        order[index] = static_cast<int>(index);
    }
    std::stable_sort(order.begin(), order.end(), [&candidates](int lhs, int rhs) {
        return candidates[static_cast<std::size_t>(lhs)].score >
               candidates[static_cast<std::size_t>(rhs)].score;
    });

    std::set<int> class_set;
    for (const Candidate& candidate : candidates) {
        class_set.insert(candidate.class_id);
    }
    for (int filter_id : class_set) {
        for (std::size_t i = 0; i < order.size(); ++i) {
            if (order[i] == -1 || candidates[static_cast<std::size_t>(order[i])].class_id != filter_id) {
                continue;
            }
            const Candidate& current = candidates[static_cast<std::size_t>(order[i])];
            for (std::size_t j = i + 1; j < order.size(); ++j) {
                if (order[j] == -1) {
                    continue;
                }
                // This intentionally mirrors the literal v1.6.0 implementation:
                // the inner class check tests the outer candidate, not candidate[j].
                if (overlap_iou(current, candidates[static_cast<std::size_t>(order[j])]) > threshold) {
                    order[j] = -1;
                }
            }
        }
    }

    std::vector<Candidate> kept;
    kept.reserve(order.size());
    for (int index : order) {
        if (index != -1) {
            kept.push_back(candidates[static_cast<std::size_t>(index)]);
        }
    }
    return kept;
}

}  // namespace

Yolov5Detector::Yolov5Detector(RknnModel& model, float confidence_threshold, float nms_threshold)
    : model_(model),
      processor_(model.model_width(), model.model_height()),
      confidence_threshold_(confidence_threshold),
      nms_threshold_(nms_threshold)
{
    if (!(confidence_threshold_ >= 0.0F && confidence_threshold_ <= 1.0F) ||
        !(nms_threshold_ >= 0.0F && nms_threshold_ <= 1.0F)) {
        throw std::runtime_error("YOLOv5 thresholds must be in [0,1]");
    }
}

std::vector<Detection> Yolov5Detector::detect(const cv::Mat& bgr)
{
    return detect_with_metrics(bgr).detections;
}

DetectionResult Yolov5Detector::detect_with_metrics(const cv::Mat& bgr)
{
    DetectionResult result;
    const auto preprocess_start = std::chrono::steady_clock::now();
    PreparedInput prepared = processor_.prepare(bgr);
    const auto preprocess_end = std::chrono::steady_clock::now();
    result.metrics.preprocess_ms =
        std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();
    RknnOutputBatch output_batch = model_.run(prepared.nhwc.data(), prepared.nhwc.size(),
                                              &result.metrics.inference_ms);
    const std::vector<TensorMeta>& metas = model_.output_metas();
    const std::vector<rknn_output>& outputs = output_batch.outputs();
    std::vector<RawTensorView> views;
    views.reserve(outputs.size());
    for (std::size_t index = 0; index < outputs.size(); ++index) {
        views.push_back(RawTensorView{outputs[index].buf, outputs[index].size, metas[index]});
    }
    const auto postprocess_start = std::chrono::steady_clock::now();
    result.detections = decode_raw(views, prepared.letterbox, model_.model_width(),
                                   model_.model_height(), confidence_threshold_, nms_threshold_);
    const auto postprocess_end = std::chrono::steady_clock::now();
    result.metrics.postprocess_ms =
        std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();
    result.metrics.object_count = static_cast<double>(result.detections.size());
    return result;
}

std::vector<Detection> Yolov5Detector::decode_raw(const std::vector<RawTensorView>& outputs,
                                                  const LetterboxInfo& letterbox,
                                                  int model_width,
                                                  int model_height,
                                                  float confidence_threshold,
                                                  float nms_threshold)
{
    if (outputs.size() != 3U) {
        throw std::runtime_error("YOLOv5 decoder requires exactly three RKNN outputs");
    }

    struct Head {
        const RawTensorView* view;
        int grid_height;
        int grid_width;
        int stride;
    };
    std::vector<Head> heads;
    heads.reserve(outputs.size());
    for (const RawTensorView& view : outputs) {
        if (view.meta.dims.size() != 4U || view.meta.dims[0] != 1U) {
            throw std::runtime_error("YOLOv5 output is not a single four-dimensional tensor");
        }
        int channels = 0;
        int grid_height = 0;
        int grid_width = 0;
        if (view.meta.format == static_cast<int>(RKNN_TENSOR_NCHW)) {
            channels = static_cast<int>(view.meta.dims[1]);
            grid_height = static_cast<int>(view.meta.dims[2]);
            grid_width = static_cast<int>(view.meta.dims[3]);
        } else if (view.meta.format == static_cast<int>(RKNN_TENSOR_NHWC)) {
            grid_height = static_cast<int>(view.meta.dims[1]);
            grid_width = static_cast<int>(view.meta.dims[2]);
            channels = static_cast<int>(view.meta.dims[3]);
        } else {
            throw std::runtime_error("YOLOv5 output format is neither NCHW nor NHWC");
        }
        if (channels != kValuesPerAnchor * 3 || grid_height <= 0 || grid_width <= 0 ||
            model_height % grid_height != 0 || model_width % grid_width != 0) {
            throw std::runtime_error("RKNN output metadata does not describe a YOLOv5 head");
        }
        const int stride = model_height / grid_height;
        if (stride != 8 && stride != 16 && stride != 32) {
            throw std::runtime_error("unexpected YOLOv5 output stride: " + std::to_string(stride));
        }
        heads.push_back(Head{&view, grid_height, grid_width, stride});
    }
    std::sort(heads.begin(), heads.end(), [](const Head& lhs, const Head& rhs) {
        return lhs.stride < rhs.stride;
    });
    if (heads[0].stride != 8 || heads[1].stride != 16 || heads[2].stride != 32) {
        throw std::runtime_error("YOLOv5 output heads do not cover strides 8, 16, and 32");
    }

    std::vector<Candidate> candidates;
    for (std::size_t head_index = 0; head_index < heads.size(); ++head_index) {
        const Head& head = heads[head_index];
        const RawTensorView& view = *head.view;
        const int channels = kValuesPerAnchor * 3;
        const std::size_t grid_size = static_cast<std::size_t>(head.grid_height) *
                                      static_cast<std::size_t>(head.grid_width);
        for (int anchor = 0; anchor < 3; ++anchor) {
            for (int row = 0; row < head.grid_height; ++row) {
                for (int column = 0; column < head.grid_width; ++column) {
                    const int objectness_channel = anchor * kValuesPerAnchor + 4;
                    const std::size_t objectness_index = value_index(view.meta, channels,
                                                                       head.grid_height, head.grid_width,
                                                                       objectness_channel, row, column);
                    if (!passes_threshold(view, objectness_index, confidence_threshold)) {
                        continue;
                    }

                    int best_class = 0;
                    float best_class_value = -std::numeric_limits<float>::infinity();
                    for (int class_id = 0; class_id < kClassCount; ++class_id) {
                        const std::size_t class_index = value_index(view.meta, channels,
                                                                      head.grid_height, head.grid_width,
                                                                      anchor * kValuesPerAnchor + 5 + class_id,
                                                                      row, column);
                        const float value = scalar_value(view, class_index);
                        if (value > best_class_value) {
                            best_class_value = value;
                            best_class = class_id;
                        }
                    }
                    const std::size_t best_class_index = value_index(view.meta, channels,
                                                                       head.grid_height, head.grid_width,
                                                                       anchor * kValuesPerAnchor + 5 + best_class,
                                                                       row, column);
                    if (!passes_threshold(view, best_class_index, confidence_threshold)) {
                        continue;
                    }

                    const int x_channel = anchor * kValuesPerAnchor;
                    const float box_x = scalar_value(view, value_index(view.meta, channels,
                                                                         head.grid_height, head.grid_width,
                                                                         x_channel, row, column)) * 2.0F - 0.5F;
                    const float box_y = scalar_value(view, value_index(view.meta, channels,
                                                                         head.grid_height, head.grid_width,
                                                                         x_channel + 1, row, column)) * 2.0F - 0.5F;
                    const float box_w = scalar_value(view, value_index(view.meta, channels,
                                                                         head.grid_height, head.grid_width,
                                                                         x_channel + 2, row, column)) * 2.0F;
                    const float box_h = scalar_value(view, value_index(view.meta, channels,
                                                                         head.grid_height, head.grid_width,
                                                                         x_channel + 3, row, column)) * 2.0F;
                    Candidate candidate;
                    candidate.x = (box_x + static_cast<float>(column)) * head.stride;
                    candidate.y = (box_y + static_cast<float>(row)) * head.stride;
                    candidate.width = box_w * box_w * static_cast<float>(kAnchors[head_index][anchor * 2]);
                    candidate.height = box_h * box_h * static_cast<float>(kAnchors[head_index][anchor * 2 + 1]);
                    candidate.x -= candidate.width / 2.0F;
                    candidate.y -= candidate.height / 2.0F;
                    candidate.score = scalar_value(view, objectness_index) * best_class_value;
                    candidate.class_id = best_class;
                    candidates.push_back(candidate);
                }
            }
        }
        (void)grid_size;
    }

    const std::vector<Candidate> kept = apply_reference_nms(std::move(candidates), nms_threshold);
    std::vector<Detection> detections;
    detections.reserve(std::min<std::size_t>(kept.size(), 128U));
    for (std::size_t index = 0; index < kept.size() && detections.size() < 128U; ++index) {
        const Candidate& candidate = kept[index];
        Detection detection;
        detection.class_id = candidate.class_id;
        detection.confidence = candidate.score;
        detection.box = ImageProcessor::restore_box(candidate.x, candidate.y, candidate.width,
                                                    candidate.height, letterbox);
        detections.push_back(detection);
    }
    return detections;
}

std::vector<Detection> Yolov5Detector::nms_for_test(const std::vector<Detection>& detections,
                                                    float threshold)
{
    std::vector<Candidate> candidates;
    candidates.reserve(detections.size());
    for (const Detection& detection : detections) {
        candidates.push_back(Candidate{detection.box.x, detection.box.y, detection.box.width,
                                       detection.box.height, detection.confidence, detection.class_id});
    }
    const std::vector<Candidate> kept = apply_reference_nms(std::move(candidates), threshold);
    std::vector<Detection> result;
    result.reserve(kept.size());
    for (const Candidate& candidate : kept) {
        result.push_back(Detection{candidate.class_id, candidate.score,
                                   cv::Rect2f(candidate.x, candidate.y, candidate.width, candidate.height)});
    }
    return result;
}

}  // namespace edgevision
