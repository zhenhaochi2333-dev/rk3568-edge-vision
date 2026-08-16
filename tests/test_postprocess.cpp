#include "edgevision/yolov5_detector.hpp"

#include "rknn_api.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr float kScale = 0.003922F;
constexpr int kZeroPoint = -128;
constexpr int kChannels = 255;

std::int8_t quantize(float value)
{
    int raw = static_cast<int>(value / kScale + static_cast<float>(kZeroPoint));
    raw = raw < -128 ? -128 : (raw > 127 ? 127 : raw);
    return static_cast<std::int8_t>(raw);
}

edgevision::TensorMeta make_meta(int index, int grid)
{
    edgevision::TensorMeta meta;
    meta.index = index;
    meta.dims = {1U, 255U, static_cast<std::uint32_t>(grid), static_cast<std::uint32_t>(grid)};
    meta.format = static_cast<int>(RKNN_TENSOR_NCHW);
    meta.type = static_cast<int>(RKNN_TENSOR_INT8);
    meta.quantization = static_cast<int>(RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC);
    meta.zero_point = kZeroPoint;
    meta.scale = kScale;
    meta.n_elems = static_cast<std::uint32_t>(kChannels * grid * grid);
    meta.size = meta.n_elems;
    return meta;
}

void set_value(std::vector<std::int8_t>& buffer, int grid, int channel, int row, int column, float value)
{
    const std::size_t offset = static_cast<std::size_t>(channel) * grid * grid +
                               static_cast<std::size_t>(row) * grid + static_cast<std::size_t>(column);
    buffer[offset] = quantize(value);
}

}  // namespace

void run_postprocess_tests()
{
    std::vector<std::int8_t> head_80(static_cast<std::size_t>(kChannels) * 80U * 80U, quantize(0.0F));
    std::vector<std::int8_t> head_40(static_cast<std::size_t>(kChannels) * 40U * 40U, quantize(0.0F));
    std::vector<std::int8_t> head_20(static_cast<std::size_t>(kChannels) * 20U * 20U, quantize(0.0F));

    const int row = 20;
    const int column = 30;
    set_value(head_80, 80, 0, row, column, 0.5F);
    set_value(head_80, 80, 1, row, column, 0.5F);
    set_value(head_80, 80, 2, row, column, 0.5F);
    set_value(head_80, 80, 3, row, column, 0.5F);
    set_value(head_80, 80, 4, row, column, 0.9F);
    set_value(head_80, 80, 5, row, column, 0.9F);

    edgevision::LetterboxInfo letterbox;
    letterbox.scale = 1.0F;
    letterbox.original_width = 640;
    letterbox.original_height = 640;
    letterbox.model_width = 640;
    letterbox.model_height = 640;

    // Deliberately provide runtime tensors out of order; metadata-derived stride mapping must recover them.
    std::vector<edgevision::RawTensorView> views;
    views.push_back(edgevision::RawTensorView{head_20.data(), head_20.size(), make_meta(2, 20)});
    views.push_back(edgevision::RawTensorView{head_80.data(), head_80.size(), make_meta(0, 80)});
    views.push_back(edgevision::RawTensorView{head_40.data(), head_40.size(), make_meta(1, 40)});

    const std::vector<edgevision::Detection> detections = edgevision::Yolov5Detector::decode_raw(
        views, letterbox, 640, 640, 0.25F, 0.45F);
    assert(detections.size() == 1U);
    assert(detections.front().class_id == 0);
    assert(detections.front().confidence > 0.75F && detections.front().confidence < 0.9F);

    std::vector<edgevision::Detection> overlapping{
        edgevision::Detection{0, 0.9F, cv::Rect2f(10.0F, 10.0F, 40.0F, 40.0F)},
        edgevision::Detection{0, 0.8F, cv::Rect2f(12.0F, 12.0F, 40.0F, 40.0F)},
    };
    const std::vector<edgevision::Detection> kept = edgevision::Yolov5Detector::nms_for_test(overlapping, 0.45F);
    assert(kept.size() == 1U);
    assert(std::fabs(kept.front().confidence - 0.9F) < 1e-5F);
}
