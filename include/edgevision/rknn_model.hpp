#pragma once

#include "edgevision/core_types.hpp"

#include "rknn_api.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace edgevision {

// move-only 输出所有权：同一批 rknn_outputs_get 的 buf 必须交还给
// 取得它们的 context。析构统一释放，禁止复制以免重复归还。
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

// 模型资源边界：构造时读入 .rknn 并查询真实张量属性；析构销毁 context。
// Detector 借用此对象，因此其寿命必须覆盖每次 detect 和输出解码。
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

    // 传入一帧紧密排列的 RGB uint8 NHWC 字节；函数验证长度并执行
    // inputs_set -> rknn_run -> outputs_get。返回值持有输出，调用者需
    // 在它析构前完成 CPU 解码；inference_ms 仅覆盖 rknn_run 调用。
    RknnOutputBatch run(const std::uint8_t* input_data, std::size_t input_size,
                        double* inference_ms = nullptr);

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
