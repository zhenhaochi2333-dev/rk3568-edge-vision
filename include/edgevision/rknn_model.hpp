#pragma once

#include "edgevision/core_types.hpp"

#include "rknn_api.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace edgevision {

class RknnOutputBatch {
public:
    RknnOutputBatch() = default;
    RknnOutputBatch(rknn_context context, std::vector<rknn_output> outputs);
    ~RknnOutputBatch();

    RknnOutputBatch(const RknnOutputBatch&) = delete;
    RknnOutputBatch& operator=(const RknnOutputBatch&) = delete;
    RknnOutputBatch(RknnOutputBatch&& other) noexcept;
    RknnOutputBatch& operator=(RknnOutputBatch&& other) noexcept;

    const std::vector<rknn_output>& outputs() const { return outputs_; }

private:
    void release() noexcept;

    rknn_context context_ = 0;
    std::vector<rknn_output> outputs_;
    bool active_ = false;
};

class RknnModel {
public:
    explicit RknnModel(const std::string& model_path);
    ~RknnModel();

    RknnModel(const RknnModel&) = delete;
    RknnModel& operator=(const RknnModel&) = delete;
    RknnModel(RknnModel&&) = delete;
    RknnModel& operator=(RknnModel&&) = delete;

    int model_width() const { return model_width_; }
    int model_height() const { return model_height_; }
    int model_channels() const { return model_channels_; }
    const std::vector<TensorMeta>& input_metas() const { return input_metas_; }
    const std::vector<TensorMeta>& output_metas() const { return output_metas_; }

    RknnOutputBatch run(const std::uint8_t* input_data, std::size_t input_size);

private:
    void query_metadata();
    static TensorMeta convert_attr(const rknn_tensor_attr& attr);
    static std::string describe(const TensorMeta& meta);

    rknn_context context_ = 0;
    std::vector<TensorMeta> input_metas_;
    std::vector<TensorMeta> output_metas_;
    int model_width_ = 0;
    int model_height_ = 0;
    int model_channels_ = 0;
};

}  // namespace edgevision
