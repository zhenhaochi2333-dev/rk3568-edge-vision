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

edgevision::TensorMeta make_meta(int index, int grid_height, int grid_width)
{
    edgevision::TensorMeta meta;
    meta.index = index;
    meta.dims = {1U, 255U, static_cast<std::uint32_t>(grid_height),
                 static_cast<std::uint32_t>(grid_width)};
    meta.format = static_cast<int>(RKNN_TENSOR_NCHW);
    meta.type = static_cast<int>(RKNN_TENSOR_INT8);
    meta.quantization = static_cast<int>(RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC);
    meta.zero_point = kZeroPoint;
    meta.scale = kScale;
    meta.n_elems = static_cast<std::uint32_t>(kChannels * grid_height * grid_width);
    meta.size = meta.n_elems;
    return meta;
}

edgevision::TensorMeta make_meta(int index, int grid)
{
    return make_meta(index, grid, grid);
}

void set_value(std::vector<std::int8_t>& buffer, int grid, int channel, int row, int column, float value)
{
    const std::size_t offset = static_cast<std::size_t>(channel) * grid * grid +
                               static_cast<std::size_t>(row) * grid + static_cast<std::size_t>(column);
    buffer[offset] = quantize(value);
}

void set_value_rect(std::vector<std::int8_t>& buffer, int grid_height, int grid_width,
                    int channel, int row, int column, float value)
{
    const std::size_t offset = static_cast<std::size_t>(channel) *
                                   static_cast<std::size_t>(grid_height) *
                                   static_cast<std::size_t>(grid_width) +
                               static_cast<std::size_t>(row) *
                                   static_cast<std::size_t>(grid_width) +
                               static_cast<std::size_t>(column);
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

    std::vector<std::int8_t> head_68x120(
        static_cast<std::size_t>(kChannels) * 68U * 120U, quantize(0.0F));
    std::vector<std::int8_t> head_34x60(
        static_cast<std::size_t>(kChannels) * 34U * 60U, quantize(0.0F));
    std::vector<std::int8_t> head_17x30(
        static_cast<std::size_t>(kChannels) * 17U * 30U, quantize(0.0F));
    const int rectangular_row = 20;
    const int rectangular_column = 60;
    for (int channel = 0; channel < 6; ++channel) {
        set_value_rect(head_68x120, 68, 120, channel, rectangular_row,
                       rectangular_column, channel == 4 || channel == 5 ? 0.9F : 0.5F);
    }
    edgevision::LetterboxInfo rectangular_letterbox;
    rectangular_letterbox.scale = 0.75F;
    rectangular_letterbox.pad_y = 2;
    rectangular_letterbox.original_width = 1280;
    rectangular_letterbox.original_height = 720;
    rectangular_letterbox.model_width = 960;
    rectangular_letterbox.model_height = 544;
    const std::vector<edgevision::RawTensorView> rectangular_views{
        edgevision::RawTensorView{head_17x30.data(), head_17x30.size(),
                                  make_meta(2, 17, 30)},
        edgevision::RawTensorView{head_68x120.data(), head_68x120.size(),
                                  make_meta(0, 68, 120)},
        edgevision::RawTensorView{head_34x60.data(), head_34x60.size(),
                                  make_meta(1, 34, 60)}};
    const std::vector<edgevision::Detection> rectangular_detections =
        edgevision::Yolov5Detector::decode_raw(rectangular_views, rectangular_letterbox,
                                                960, 544, 0.25F, 0.45F);
    assert(rectangular_detections.size() == 1U);
    assert(rectangular_detections.front().class_id == 0);
}
