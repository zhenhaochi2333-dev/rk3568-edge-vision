/*
 * 模型文件 -> rknn_context -> 输入/输出属性查询 -> 每帧推理。
 * RknnModel 拥有 context；RknnOutputBatch 拥有本帧 rknn_outputs_get 返回的缓冲。
 * 先释放输出、再销毁 context，才能让后处理读取的 RawTensorView 保持有效。
 */
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

// Runtime 创建的输出缓冲不由 std::vector<rknn_output> 自行释放。
// 析构统一调用 rknn_outputs_release；即使检测流程抛异常也要归还这一批输出。
RknnOutputBatch::~RknnOutputBatch()
{
    release();
}

// 移动后只有新对象继续负责 release；将源对象置为 inactive 防止同一批输出
// 在两个析构函数里重复归还给 RKNN Runtime。
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

// release 对空批次和重复调用安全。释放必须使用取得输出时的同一个 context
// 与完整 output 数量；这里不把 buf 交给 C++ delete/free。
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

// 按二进制读取整个 .rknn 文件，并拒绝空文件或超过 RKNN API 长度字段的文件。
// 临时 vector 向 rknn_init 提供模型字节；初始化后本类通过 context 与 Runtime 交互。
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
    // 初始化成功后仍可能因为不支持的输入/输出属性而拒绝模型；异常路径
    // 显式销毁 context，避免构造函数失败时析构函数不会运行造成资源泄漏。
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

// 把 RKNN 的 C 结构复制成可长期保存的 metadata；包括 dims 的副本、
// layout、dtype 以及 INT8/UINT8 反量化所需的 zero point 和 scale。
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

// 模型转换可改变 tensor 的维度顺序、dtype 和量化参数；以 rknn_query
// 当前加载模型的属性为准，并在进入推理循环前验证本解码器的输入契约。
void RknnModel::query_metadata()
{
    rknn_input_output_num io_num{};
    int ret = rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("RKNN_QUERY_IN_OUT_NUM failed with ret=" + std::to_string(ret));
    }
    if (io_num.n_input != 1U || io_num.n_output == 0U) {
        throw std::runtime_error("expected one input and at least one output, got " +
                                 std::to_string(io_num.n_input) + " and " +
                                 std::to_string(io_num.n_output));
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

    // 当前前处理会生成单批次、3 通道、紧密排列的 RGB 字节。只接受相应
    // NHWC 输入；模型宽高来自 dims[2]/dims[1]，不依赖 README 的 640 常量。
    const TensorMeta& input = input_metas_.front();
    if (input.format != static_cast<int>(RKNN_TENSOR_NHWC) || input.dims.size() != 4U ||
        input.dims[0] != 1U || input.dims[3] != 3U) {
        throw std::runtime_error("model input metadata is not the supported single RGB NHWC tensor");
    }
    model_height_ = static_cast<int>(input.dims[1]);
    model_width_ = static_cast<int>(input.dims[2]);
    model_channels_ = static_cast<int>(input.dims[3]);
    const std::size_t expected_elements = static_cast<std::size_t>(model_width_) *
                                          static_cast<std::size_t>(model_height_) *
                                          static_cast<std::size_t>(model_channels_);
    if (input.n_elems != expected_elements) {
        throw std::runtime_error("model input metadata element count does not match NHWC shape");
    }
}

// input_data 是调用者的 RGB uint8 NHWC 缓冲，大小必须等于查询到的 W*H*3。
// 本函数只借用输入指针；返回的 move-only 对象负责 Runtime 输出的归还。
RknnOutputBatch RknnModel::run(const std::uint8_t* input_data, std::size_t input_size,
                               double* inference_ms)
{
    const std::size_t expected_size = static_cast<std::size_t>(model_width_) *
                                      static_cast<std::size_t>(model_height_) *
                                      static_cast<std::size_t>(model_channels_);
    if (input_data == nullptr || input_size != expected_size ||
        input_size > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("invalid RKNN input buffer");
    }

    rknn_input input{};
    input.index = 0;
    // RKNN C API 用 void* 表示输入，即使这里只读也需去掉 const。
    // 这不转移输入 buffer 所有权，prepared.nhwc 在 detect_with_metrics 中保持存活。
    input.buf = const_cast<std::uint8_t*>(input_data);
    input.size = static_cast<std::uint32_t>(input_size);
    // pass_through=0：将下述 UINT8/NHWC 声明交给 Runtime 处理输入格式；
    // 不把此 RGB 字节数组直接解释成模型内部的量化张量。
    input.pass_through = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = RKNN_TENSOR_NHWC;

    int ret = rknn_inputs_set(context_, 1, &input);
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("rknn_inputs_set failed with ret=" + std::to_string(ret));
    }
    // steady_clock 的两个时刻只包住 rknn_run：inference_ms 是该 API 的调用耗时，
    // 与整帧预处理、输出获取、CPU 解码和可视化耗时分别记录。
    const auto inference_start = std::chrono::steady_clock::now();
    ret = rknn_run(context_, nullptr);
    const auto inference_end = std::chrono::steady_clock::now();
    if (inference_ms != nullptr) {
        *inference_ms = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
    }
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("rknn_run failed with ret=" + std::to_string(ret));
    }

    // INT8/UINT8 输出要求原始整数，后处理按每张量的 zp/scale 自行反量化；
    // 其他输出向 Runtime 请求 float。输出数组只保存描述符，buf 在 outputs_get 后有效。
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
    // 取得 buf 后立即交给 RAII 包装；调用者读取完并退出作用域时才 release。
    return RknnOutputBatch(context_, std::move(outputs));
}

}  // namespace edgevision
