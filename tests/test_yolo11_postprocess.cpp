#include "edgevision/yolo11_detector.hpp"

#include "rknn_api.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

edgevision::TensorMeta make_meta(int channels, int grid)
{
    edgevision::TensorMeta meta;
    meta.dims = {1U, static_cast<std::uint32_t>(channels), static_cast<std::uint32_t>(grid),
                 static_cast<std::uint32_t>(grid)};
    meta.format = static_cast<int>(RKNN_TENSOR_NCHW);
    meta.type = static_cast<int>(RKNN_TENSOR_FLOAT32);
    meta.n_elems = static_cast<std::uint32_t>(channels * grid * grid);
    meta.size = meta.n_elems * sizeof(float);
    return meta;
}

void set_value(std::vector<float>& tensor, int channels, int grid, int channel, int row,
               int column, float value)
{
    const std::size_t offset =
        (static_cast<std::size_t>(channel) * static_cast<std::size_t>(grid) +
         static_cast<std::size_t>(row)) * static_cast<std::size_t>(grid) +
        static_cast<std::size_t>(column);
    assert(offset < tensor.size() && channel < channels);
    tensor[offset] = value;
}

}  // namespace

void run_yolo11_postprocess_tests()
{
    std::vector<float> box_80(64U * 80U * 80U, 0.0F);
    std::vector<float> score_80(80U * 80U * 80U, 0.0F);
    std::vector<float> sum_80(80U * 80U, 0.0F);
    std::vector<float> box_40(64U * 40U * 40U, 0.0F);
    std::vector<float> score_40(80U * 40U * 40U, 0.0F);
    std::vector<float> sum_40(40U * 40U, 0.0F);
    std::vector<float> box_20(64U * 20U * 20U, 0.0F);
    std::vector<float> score_20(80U * 20U * 20U, 0.0F);
    std::vector<float> sum_20(20U * 20U, 0.0F);

    const int row = 20;
    const int column = 30;
    set_value(score_80, 80, 80, 0, row, column, 0.9F);
    set_value(sum_80, 1, 80, 0, row, column, 0.9F);

    edgevision::LetterboxInfo letterbox;
    letterbox.scale = 1.0F;
    letterbox.original_width = 640;
    letterbox.original_height = 640;
    letterbox.model_width = 640;
    letterbox.model_height = 640;
    const std::vector<edgevision::RawTensorView> outputs{
        {box_80.data(), box_80.size() * sizeof(float), make_meta(64, 80)},
        {score_80.data(), score_80.size() * sizeof(float), make_meta(80, 80)},
        {sum_80.data(), sum_80.size() * sizeof(float), make_meta(1, 80)},
        {box_40.data(), box_40.size() * sizeof(float), make_meta(64, 40)},
        {score_40.data(), score_40.size() * sizeof(float), make_meta(80, 40)},
        {sum_40.data(), sum_40.size() * sizeof(float), make_meta(1, 40)},
        {box_20.data(), box_20.size() * sizeof(float), make_meta(64, 20)},
        {score_20.data(), score_20.size() * sizeof(float), make_meta(80, 20)},
        {sum_20.data(), sum_20.size() * sizeof(float), make_meta(1, 20)},
    };

    const std::vector<edgevision::Detection> detections =
        edgevision::Yolo11Detector::decode_raw(outputs, letterbox, 640, 640, 0.25F, 0.45F);
    assert(detections.size() == 1U);
    assert(detections.front().class_id == 0);
    assert(std::fabs(detections.front().confidence - 0.9F) < 1e-5F);
    assert(detections.front().box.x >= 0.0F && detections.front().box.y >= 0.0F);
    assert(detections.front().box.width > 0.0F && detections.front().box.height > 0.0F);

    const std::vector<edgevision::Detection> overlapping{
        {0, 0.9F, cv::Rect2f(10.0F, 10.0F, 40.0F, 40.0F)},
        {0, 0.8F, cv::Rect2f(12.0F, 12.0F, 40.0F, 40.0F)},
        {1, 0.7F, cv::Rect2f(12.0F, 12.0F, 40.0F, 40.0F)},
    };
    const std::vector<edgevision::Detection> kept =
        edgevision::Yolo11Detector::nms_for_test(overlapping, 0.45F);
    assert(kept.size() == 1U);
    assert(kept.front().class_id == 0 && std::fabs(kept.front().confidence - 0.9F) < 1e-5F);
}
