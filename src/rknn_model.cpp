#include "edgevision/rknn_model.hpp"

#include "edgevision/logger.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace edgevision {

RknnOutputBatch::RknnOutputBatch(rknn_context context, std::vector<rknn_output> outputs)
    : context_(context), outputs_(std::move(outputs)), active_(true)
{
}

RknnOutputBatch::~RknnOutputBatch()
{
    release();
}

RknnOutputBatch::RknnOutputBatch(RknnOutputBatch&& other) noexcept
    : context_(other.context_), outputs_(std::move(other.outputs_)), active_(other.active_)
{
    other.context_ = 0;
    other.active_ = false;
}

RknnOutputBatch& RknnOutputBatch::operator=(RknnOutputBatch&& other) noexcept
{
    if (this != &other) {
        release();
        context_ = other.context_;
        outputs_ = std::move(other.outputs_);
        active_ = other.active_;
        other.context_ = 0;
        other.active_ = false;
    }
    return *this;
}

void RknnOutputBatch::release() noexcept
{
    if (!active_ || context_ == 0 || outputs_.empty()) {
        return;
    }
    const int ret = rknn_outputs_release(context_, static_cast<std::uint32_t>(outputs_.size()), outputs_.data());
    if (ret != RKNN_SUCC) {
        log_warn("rknn_outputs_release failed with ret=" + std::to_string(ret));
    }
    active_ = false;
    outputs_.clear();
}

RknnModel::RknnModel(const std::string& model_path)
{
    std::ifstream file(model_path.c_str(), std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("cannot open RKNN model: " + model_path);
    }
    const std::streamoff model_size = file.tellg();
    if (model_size <= 0 || static_cast<unsigned long long>(model_size) > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("invalid RKNN model size: " + model_path);
    }
    file.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> model(static_cast<std::size_t>(model_size));
    if (!file.read(reinterpret_cast<char*>(model.data()), model_size)) {
        throw std::runtime_error("failed to read RKNN model: " + model_path);
    }

    const int init_ret = rknn_init(&context_, model.data(), static_cast<std::uint32_t>(model.size()), 0, nullptr);
    if (init_ret != RKNN_SUCC) {
        context_ = 0;
        throw std::runtime_error("rknn_init failed with ret=" + std::to_string(init_ret));
    }
    try {
        query_metadata();
    } catch (...) {
        rknn_destroy(context_);
        context_ = 0;
        throw;
    }
}

RknnModel::~RknnModel()
{
    if (context_ != 0) {
        rknn_destroy(context_);
        context_ = 0;
    }
}

TensorMeta RknnModel::convert_attr(const rknn_tensor_attr& attr)
{
    TensorMeta meta;
    meta.index = static_cast<int>(attr.index);
    meta.name = attr.name;
    meta.dims.assign(attr.dims, attr.dims + attr.n_dims);
    meta.n_elems = attr.n_elems;
    meta.size = attr.size;
    meta.format = static_cast<int>(attr.fmt);
    meta.type = static_cast<int>(attr.type);
    meta.quantization = static_cast<int>(attr.qnt_type);
    meta.zero_point = attr.zp;
    meta.scale = attr.scale;
    return meta;
}

std::string RknnModel::describe(const TensorMeta& meta)
{
    std::ostringstream stream;
    stream << "index=" << meta.index << " name=" << meta.name << " dims=[";
    for (std::size_t i = 0; i < meta.dims.size(); ++i) {
        if (i != 0U) {
            stream << ',';
        }
        stream << meta.dims[i];
    }
    stream << "] elems=" << meta.n_elems << " size=" << meta.size
           << " format=" << meta.format << " type=" << meta.type
           << " qnt=" << meta.quantization << " zp=" << meta.zero_point
           << " scale=" << meta.scale;
    return stream.str();
}

void RknnModel::query_metadata()
{
    rknn_input_output_num io_num{};
    int ret = rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("RKNN_QUERY_IN_OUT_NUM failed with ret=" + std::to_string(ret));
    }
    if (io_num.n_input != 1U || io_num.n_output != 3U) {
        throw std::runtime_error("expected one input and three YOLOv5 outputs, got " +
                                 std::to_string(io_num.n_input) + " and " + std::to_string(io_num.n_output));
    }

    input_metas_.reserve(io_num.n_input);
    for (std::uint32_t index = 0; index < io_num.n_input; ++index) {
        rknn_tensor_attr attr{};
        attr.index = index;
        ret = rknn_query(context_, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
        if (ret != RKNN_SUCC) {
            throw std::runtime_error("RKNN_QUERY_INPUT_ATTR failed with ret=" + std::to_string(ret));
        }
        input_metas_.push_back(convert_attr(attr));
        log_info("input tensor " + describe(input_metas_.back()));
    }

    output_metas_.reserve(io_num.n_output);
    for (std::uint32_t index = 0; index < io_num.n_output; ++index) {
        rknn_tensor_attr attr{};
        attr.index = index;
        ret = rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
        if (ret != RKNN_SUCC) {
            throw std::runtime_error("RKNN_QUERY_OUTPUT_ATTR failed with ret=" + std::to_string(ret));
        }
        output_metas_.push_back(convert_attr(attr));
        log_info("output tensor " + describe(output_metas_.back()));
    }

    const TensorMeta& input = input_metas_.front();
    if (input.format != static_cast<int>(RKNN_TENSOR_NHWC) || input.dims.size() != 4U ||
        input.dims[0] != 1U || input.dims[3] != 3U) {
        throw std::runtime_error("model input metadata is not the expected single RGB NHWC tensor");
    }
    model_height_ = static_cast<int>(input.dims[1]);
    model_width_ = static_cast<int>(input.dims[2]);
    model_channels_ = static_cast<int>(input.dims[3]);
}

RknnOutputBatch RknnModel::run(const std::uint8_t* input_data, std::size_t input_size,
                               double* inference_ms)
{
    if (input_data == nullptr || input_size == 0U || input_size > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("invalid RKNN input buffer");
    }

    rknn_input input{};
    input.index = 0;
    input.buf = const_cast<std::uint8_t*>(input_data);
    input.size = static_cast<std::uint32_t>(input_size);
    input.pass_through = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = RKNN_TENSOR_NHWC;

    int ret = rknn_inputs_set(context_, 1, &input);
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("rknn_inputs_set failed with ret=" + std::to_string(ret));
    }
    const auto inference_start = std::chrono::steady_clock::now();
    ret = rknn_run(context_, nullptr);
    const auto inference_end = std::chrono::steady_clock::now();
    if (inference_ms != nullptr) {
        *inference_ms = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
    }
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("rknn_run failed with ret=" + std::to_string(ret));
    }

    std::vector<rknn_output> outputs(output_metas_.size());
    for (std::size_t index = 0; index < outputs.size(); ++index) {
        outputs[index].index = static_cast<std::uint32_t>(index);
        const int type = output_metas_[index].type;
        outputs[index].want_float = (type == static_cast<int>(RKNN_TENSOR_INT8) ||
                                     type == static_cast<int>(RKNN_TENSOR_UINT8)) ? 0U : 1U;
    }
    ret = rknn_outputs_get(context_, static_cast<std::uint32_t>(outputs.size()), outputs.data(), nullptr);
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("rknn_outputs_get failed with ret=" + std::to_string(ret));
    }
    return RknnOutputBatch(context_, std::move(outputs));
}

}  // namespace edgevision
