/*
 * 检测前处理的两套坐标系：输入是源帧 BGR 像素坐标，模型读取固定尺寸 RGB 画布。
 * 等比例缩放和 114 填充保留目标形状；LetterboxInfo 记录逆变换所需的比例与偏移。
 * prepare 返回的 nhwc 是自有连续字节，调用者须让它活到 rknn_inputs_set 完成。
 */
#include "edgevision/image_processor.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace edgevision {

namespace {

int even_down(int value)
{
    return value - (value % 2);
}

float clamp_float(float value, float minimum, float maximum)
{
    return std::max(minimum, std::min(value, maximum));
}

}  // namespace

ImageProcessor::ImageProcessor(int model_width, int model_height)
    : model_width_(model_width), model_height_(model_height)
{
    if (model_width_ <= 0 || model_height_ <= 0) {
        throw std::runtime_error("invalid model dimensions for ImageProcessor");
    }
}

// 输入必须是非空 CV_8UC3 BGR；函数既保留 RGB Mat 供观察，也复制出交给 RKNN
// 的 uint8 NHWC 字节数组。PreparedInput 拥有两者，不借用原始帧的内存。
PreparedInput ImageProcessor::prepare(const cv::Mat& bgr) const
{
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::runtime_error("input image must be a non-empty CV_8UC3 BGR image");
    }

    const int source_width = bgr.cols;
    const int source_height = bgr.rows;
    const float scale_width = static_cast<float>(model_width_) / source_width;
    const float scale_height = static_cast<float>(model_height_) / source_height;
    // 两轴缩放率取较小值，使整幅源图落入模型画布。否则非正方形画面会被
    // 拉成正方形，检测框的几何含义也会随之改变。
    const float scale = std::min(scale_width, scale_height);

    // 一轴铺满画布，另一轴按原始宽高比缩小；下方还会按模型 zoo 的
    // 对齐约定调整实际拷贝尺寸和 padding 位置。
    int resize_width = model_width_;
    int resize_height = model_height_;
    if (scale_width < scale_height) {
        resize_height = static_cast<int>(source_height * scale);
    } else {
        resize_width = static_cast<int>(source_width * scale);
    }

    // 宽度按 4、 高度按 2 对齐，填充起点再取偶数；这是当前项目沿用的
    // RKNN model zoo 预处理几何，不是 OpenCV resize 本身的要求。
    // Match utils/image_utils.c from rknn_model_zoo v1.6.0.
    if (resize_width % 4 != 0) {
        resize_width -= resize_width % 4;
    }
    if (resize_height % 2 != 0) {
        resize_height = even_down(resize_height);
    }
    resize_width = std::max(1, std::min(resize_width, model_width_));
    resize_height = std::max(1, std::min(resize_height, model_height_));

    int pad_x = 0;
    int pad_y = 0;
    if (scale_width < scale_height) {
        pad_y = (model_height_ - resize_height) / 2;
        pad_y = even_down(std::max(0, pad_y));
    } else {
        pad_x = (model_width_ - resize_width) / 2;
        pad_x = even_down(std::max(0, pad_x));
    }

    // 画布先以 114 填满，再把缩放图像放进指定 ROI；未覆盖的边带也会作为
    // 模型输入像素，因此不能只缩放图像却省略相同的填充规则。
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(resize_width, resize_height), 0.0, 0.0, cv::INTER_LINEAR);
    cv::Mat letterboxed(model_height_, model_width_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(letterboxed(cv::Rect(pad_x, pad_y, resize_width, resize_height)));

    // OpenCV 的 IMREAD_COLOR / V4L2 转换产出 BGR；模型输入按 RGB 排列。
    // isContinuous 保护后面的逐字节复制，不依赖 Mat 的行 stride 恰好等于宽*3。
    cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
    if (!rgb.isContinuous()) {
        rgb = rgb.clone();
    }

    PreparedInput prepared;
    prepared.rgb_image = rgb;
    // rgb 是 HWC 内存顺序，单批次时可直接作为 NHWC 的 H/W/C 部分。
    // 复制进 vector 后，推理输入不受 Mat 引用计数或原始摄像头缓冲区复用影响。
    prepared.nhwc.resize(static_cast<std::size_t>(rgb.total() * rgb.channels()));
    std::memcpy(prepared.nhwc.data(), rgb.data, prepared.nhwc.size());
    prepared.letterbox.scale = scale;
    prepared.letterbox.pad_x = pad_x;
    prepared.letterbox.pad_y = pad_y;
    prepared.letterbox.original_width = source_width;
    prepared.letterbox.original_height = source_height;
    prepared.letterbox.model_width = model_width_;
    prepared.letterbox.model_height = model_height_;
    return prepared;
}

// YOLO 解码给出模型画布坐标。先减 letterbox 的左/上偏移，再除缩放比例，
// 将两角分别限制在可见范围内，构造源图像素框供跟踪、ROI 和显示共同使用。
cv::Rect2f ImageProcessor::restore_box(float x, float y, float width, float height,
                                       const LetterboxInfo& letterbox)
{
    if (letterbox.scale <= 0.0F) {
        throw std::runtime_error("invalid letterbox scale");
    }
    const float x1 = clamp_float(x - static_cast<float>(letterbox.pad_x), 0.0F,
                                 static_cast<float>(letterbox.model_width)) / letterbox.scale;
    const float y1 = clamp_float(y - static_cast<float>(letterbox.pad_y), 0.0F,
                                 static_cast<float>(letterbox.model_height)) / letterbox.scale;
    const float x2 = clamp_float(x + width - static_cast<float>(letterbox.pad_x), 0.0F,
                                 static_cast<float>(letterbox.model_width)) / letterbox.scale;
    const float y2 = clamp_float(y + height - static_cast<float>(letterbox.pad_y), 0.0F,
                                 static_cast<float>(letterbox.model_height)) / letterbox.scale;
    // 用两角而非直接缩放宽高做裁剪，避免越过图像边缘的预测框保留原宽高。
    const float right = std::max(x1, std::min(x2, static_cast<float>(letterbox.original_width)));
    const float bottom = std::max(y1, std::min(y2, static_cast<float>(letterbox.original_height)));
    return cv::Rect2f(x1, y1, std::max(0.0F, right - x1), std::max(0.0F, bottom - y1));
}

}  // namespace edgevision
