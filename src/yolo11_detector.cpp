#include "edgevision/yolo11_detector.hpp"

#include "rknn_api.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace edgevision {

namespace {

constexpr int kClassCount = 80;
constexpr int kOutputPerHead = 3;
constexpr int kHeadCount = 3;
constexpr int kExpectedBoxChannels = 64;
constexpr int kExpectedScoreChannels = kClassCount;
constexpr int kExpectedScoreSumChannels = 1;

struct TensorShape {
    int channels = 0;
    int height = 0;
    int width = 0;
};

struct Head {
    const RawTensorView* boxes = nullptr;
    const RawTensorView* scores = nullptr;
    const RawTensorView* score_sum = nullptr;
    TensorShape box_shape;
    TensorShape score_shape;
    TensorShape score_sum_shape;
    int stride = 0;
    int dfl_len = 0;
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
    throw std::runtime_error("unsupported RKNN output type in YOLO11 decoder: " +
                             std::to_string(meta.type));
}

TensorShape tensor_shape(const TensorMeta& meta)
{
    if (meta.dims.size() != 4U || meta.dims[0] != 1U) {
        throw std::runtime_error("YOLO11 output is not a single four-dimensional tensor");
    }

    TensorShape shape;
    if (meta.format == static_cast<int>(RKNN_TENSOR_NCHW)) {
        shape.channels = static_cast<int>(meta.dims[1]);
        shape.height = static_cast<int>(meta.dims[2]);
        shape.width = static_cast<int>(meta.dims[3]);
    } else if (meta.format == static_cast<int>(RKNN_TENSOR_NHWC)) {
        shape.height = static_cast<int>(meta.dims[1]);
        shape.width = static_cast<int>(meta.dims[2]);
        shape.channels = static_cast<int>(meta.dims[3]);
    } else {
        throw std::runtime_error("YOLO11 output format is neither NCHW nor NHWC");
    }
    if (shape.channels <= 0 || shape.height <= 0 || shape.width <= 0) {
        throw std::runtime_error("YOLO11 output has invalid tensor dimensions");
    }
    return shape;
}

std::size_t value_index(const TensorMeta& meta, const TensorShape& shape,
                        int channel, int row, int column)
{
    if (channel < 0 || channel >= shape.channels || row < 0 || row >= shape.height ||
        column < 0 || column >= shape.width) {
        throw std::runtime_error("YOLO11 output index is outside tensor dimensions");
    }
    if (meta.format == static_cast<int>(RKNN_TENSOR_NCHW)) {
        return (static_cast<std::size_t>(channel) * static_cast<std::size_t>(shape.height) +
                static_cast<std::size_t>(row)) * static_cast<std::size_t>(shape.width) +
               static_cast<std::size_t>(column);
    }
    return (static_cast<std::size_t>(row) * static_cast<std::size_t>(shape.width) +
            static_cast<std::size_t>(column)) * static_cast<std::size_t>(shape.channels) +
           static_cast<std::size_t>(channel);
}

float scalar_value(const RawTensorView& view, std::size_t index)
{
    const std::size_t bytes = (index + 1U) * scalar_size(view.meta);
    if (view.data == nullptr || bytes > view.size) {
        throw std::runtime_error("RKNN output buffer is smaller than queried tensor metadata");
    }
    if (view.meta.type == static_cast<int>(RKNN_TENSOR_INT8)) {
        const auto* values = static_cast<const std::int8_t*>(view.data);
        return (static_cast<float>(values[index]) - static_cast<float>(view.meta.zero_point)) *
               view.meta.scale;
    }
    if (view.meta.type == static_cast<int>(RKNN_TENSOR_UINT8)) {
        const auto* values = static_cast<const std::uint8_t*>(view.data);
        return (static_cast<float>(values[index]) - static_cast<float>(view.meta.zero_point)) *
               view.meta.scale;
    }
    return static_cast<const float*>(view.data)[index];
}

float iou(const Candidate& lhs, const Candidate& rhs)
{
    const float left = std::max(lhs.x, rhs.x);
    const float top = std::max(lhs.y, rhs.y);
    const float right = std::min(lhs.x + lhs.width, rhs.x + rhs.width);
    const float bottom = std::min(lhs.y + lhs.height, rhs.y + rhs.height);
    const float overlap_width = std::max(0.0F, right - left + 1.0F);
    const float overlap_height = std::max(0.0F, bottom - top + 1.0F);
    const float intersection = overlap_width * overlap_height;
    const float lhs_area = (lhs.width + 1.0F) * (lhs.height + 1.0F);
    const float rhs_area = (rhs.width + 1.0F) * (rhs.height + 1.0F);
    const float union_area = lhs_area + rhs_area - intersection;
    return union_area <= 0.0F ? 0.0F : intersection / union_area;
}

std::vector<Candidate> classwise_nms(std::vector<Candidate> candidates, float threshold)
{
    std::vector<int> order(candidates.size());
    for (std::size_t index = 0; index < order.size(); ++index) {
        order[index] = static_cast<int>(index);
    }
    std::stable_sort(order.begin(), order.end(), [&candidates](int lhs, int rhs) {
        return candidates[static_cast<std::size_t>(lhs)].score >
               candidates[static_cast<std::size_t>(rhs)].score;
    });

    std::vector<bool> suppressed(order.size(), false);
    std::vector<Candidate> kept;
    kept.reserve(candidates.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        const Candidate& current = candidates[static_cast<std::size_t>(order[i])];
        kept.push_back(current);
        for (std::size_t j = i + 1U; j < order.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }
            const Candidate& candidate = candidates[static_cast<std::size_t>(order[j])];
            if (candidate.class_id == current.class_id && iou(current, candidate) > threshold) {
                suppressed[j] = true;
            }
        }
    }
    return kept;
}

void compute_dfl(const RawTensorView& view, const TensorShape& shape, int row, int column,
                 int dfl_len, std::array<float, 4>& box)
{
    for (int side = 0; side < 4; ++side) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (int bin = 0; bin < dfl_len; ++bin) {
            const std::size_t index = value_index(view.meta, shape, side * dfl_len + bin,
                                                   row, column);
            maximum = std::max(maximum, scalar_value(view, index));
        }
        float sum = 0.0F;
        float weighted = 0.0F;
        for (int bin = 0; bin < dfl_len; ++bin) {
            const std::size_t index = value_index(view.meta, shape, side * dfl_len + bin,
                                                   row, column);
            const float probability = std::exp(scalar_value(view, index) - maximum);
            sum += probability;
            weighted += probability * static_cast<float>(bin);
        }
        box[static_cast<std::size_t>(side)] = sum > 0.0F ? weighted / sum : 0.0F;
    }
}

std::vector<Head> describe_heads(const std::vector<RawTensorView>& outputs,
                                 int model_width, int model_height)
{
    if (outputs.size() != static_cast<std::size_t>(kHeadCount * kOutputPerHead)) {
        throw std::runtime_error("YOLO11 decoder requires exactly nine RKNN outputs");
    }
    if (model_width <= 0 || model_height <= 0) {
        throw std::runtime_error("YOLO11 decoder received invalid model dimensions");
    }

    std::vector<Head> heads;
    heads.reserve(kHeadCount);
    for (int index = 0; index < kHeadCount; ++index) {
        const RawTensorView& boxes = outputs[static_cast<std::size_t>(index * 3)];
        const RawTensorView& scores = outputs[static_cast<std::size_t>(index * 3 + 1)];
        const RawTensorView& score_sum = outputs[static_cast<std::size_t>(index * 3 + 2)];
        Head head;
        head.boxes = &boxes;
        head.scores = &scores;
        head.score_sum = &score_sum;
        head.box_shape = tensor_shape(boxes.meta);
        head.score_shape = tensor_shape(scores.meta);
        head.score_sum_shape = tensor_shape(score_sum.meta);
        if (head.box_shape.channels != kExpectedBoxChannels ||
            head.score_shape.channels != kExpectedScoreChannels ||
            head.score_sum_shape.channels != kExpectedScoreSumChannels ||
            head.box_shape.height != head.score_shape.height ||
            head.box_shape.width != head.score_shape.width ||
            head.box_shape.height != head.score_sum_shape.height ||
            head.box_shape.width != head.score_sum_shape.width) {
            throw std::runtime_error("YOLO11 output group does not match box/score/score-sum layout");
        }
        const int stride_height = model_height / head.box_shape.height;
        const int stride_width = model_width / head.box_shape.width;
        if (model_height % head.box_shape.height != 0 || model_width % head.box_shape.width != 0 ||
            stride_height != stride_width || stride_height <= 0) {
            throw std::runtime_error("YOLO11 output head has invalid stride");
        }
        head.stride = stride_height;
        head.dfl_len = kExpectedBoxChannels / 4;
        heads.push_back(head);
    }
    std::sort(heads.begin(), heads.end(), [](const Head& lhs, const Head& rhs) {
        return lhs.stride < rhs.stride;
    });
    if (heads[0].stride != 8 || heads[1].stride != 16 || heads[2].stride != 32) {
        throw std::runtime_error("YOLO11 output heads do not cover strides 8, 16, and 32");
    }
    return heads;
}

}  // namespace

Yolo11Detector::Yolo11Detector(RknnModel& model, float confidence_threshold, float nms_threshold)
    : model_(model),
      processor_(model.model_width(), model.model_height()),
      confidence_threshold_(confidence_threshold),
      nms_threshold_(nms_threshold)
{
    if (!(confidence_threshold_ >= 0.0F && confidence_threshold_ <= 1.0F) ||
        !(nms_threshold_ >= 0.0F && nms_threshold_ <= 1.0F)) {
        throw std::runtime_error("YOLO11 thresholds must be in [0,1]");
    }
}

std::vector<Detection> Yolo11Detector::detect(const cv::Mat& bgr)
{
    return detect_with_metrics(bgr).detections;
}

DetectionResult Yolo11Detector::detect_with_metrics(const cv::Mat& bgr)
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

std::vector<Detection> Yolo11Detector::decode_raw(const std::vector<RawTensorView>& outputs,
                                                  const LetterboxInfo& letterbox,
                                                  int model_width, int model_height,
                                                  float confidence_threshold, float nms_threshold)
{
    if (!(confidence_threshold >= 0.0F && confidence_threshold <= 1.0F) ||
        !(nms_threshold >= 0.0F && nms_threshold <= 1.0F)) {
        throw std::runtime_error("YOLO11 decode thresholds must be in [0,1]");
    }
    const std::vector<Head> heads = describe_heads(outputs, model_width, model_height);
    std::vector<Candidate> candidates;
    candidates.reserve(8400U);

    for (const Head& head : heads) {
        for (int row = 0; row < head.box_shape.height; ++row) {
            for (int column = 0; column < head.box_shape.width; ++column) {
                const std::size_t sum_index = value_index(
                    head.score_sum->meta, head.score_sum_shape, 0, row, column);
                if (scalar_value(*head.score_sum, sum_index) < confidence_threshold) {
                    continue;
                }

                int best_class = -1;
                float best_score = -std::numeric_limits<float>::infinity();
                for (int class_id = 0; class_id < kClassCount; ++class_id) {
                    const std::size_t score_index = value_index(
                        head.scores->meta, head.score_shape, class_id, row, column);
                    const float score = scalar_value(*head.scores, score_index);
                    if (score > best_score) {
                        best_score = score;
                        best_class = class_id;
                    }
                }
                if (best_class < 0 || !std::isfinite(best_score) || best_score < confidence_threshold) {
                    continue;
                }

                std::array<float, 4> box{};
                compute_dfl(*head.boxes, head.box_shape, row, column, head.dfl_len, box);
                const float x1 = (-box[0] + static_cast<float>(column) + 0.5F) * head.stride;
                const float y1 = (-box[1] + static_cast<float>(row) + 0.5F) * head.stride;
                const float x2 = (box[2] + static_cast<float>(column) + 0.5F) * head.stride;
                const float y2 = (box[3] + static_cast<float>(row) + 0.5F) * head.stride;
                if (!std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(x2) ||
                    !std::isfinite(y2) || x2 <= x1 || y2 <= y1) {
                    continue;
                }
                candidates.push_back(Candidate{x1, y1, x2 - x1, y2 - y1, best_score, best_class});
            }
        }
    }

    const std::vector<Candidate> kept = classwise_nms(std::move(candidates), nms_threshold);
    std::vector<Detection> detections;
    detections.reserve(std::min<std::size_t>(kept.size(), 128U));
    for (const Candidate& candidate : kept) {
        if (detections.size() >= 128U) {
            break;
        }
        detections.push_back(Detection{
            candidate.class_id,
            candidate.score,
            ImageProcessor::restore_box(candidate.x, candidate.y, candidate.width,
                                         candidate.height, letterbox),
            -1});
    }
    return detections;
}

std::vector<Detection> Yolo11Detector::nms_for_test(const std::vector<Detection>& detections,
                                                    float threshold)
{
    std::vector<Candidate> candidates;
    candidates.reserve(detections.size());
    for (const Detection& detection : detections) {
        candidates.push_back(Candidate{detection.box.x, detection.box.y, detection.box.width,
                                       detection.box.height, detection.confidence,
                                       detection.class_id});
    }
    const std::vector<Candidate> kept = classwise_nms(std::move(candidates), threshold);
    std::vector<Detection> result;
    result.reserve(kept.size());
    for (const Candidate& candidate : kept) {
        result.push_back(Detection{candidate.class_id, candidate.score,
                                   cv::Rect2f(candidate.x, candidate.y,
                                              candidate.width, candidate.height), -1});
    }
    return result;
}

}  // namespace edgevision
