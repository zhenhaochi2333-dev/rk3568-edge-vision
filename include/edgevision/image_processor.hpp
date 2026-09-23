#pragma once

#include "edgevision/core_types.hpp"

namespace edgevision {

// 前处理和几何逆变换成对使用：准备模型的 RGB/NHWC 输入，同时记录
// 源图到 letterbox 画布的比例与填充；检测后把画布框映回源图像素。
class ImageProcessor {
public:
    ImageProcessor(int model_width, int model_height);

    // 输入须为非空 CV_8UC3 BGR；返回值拥有模型尺寸 RGB 图像和字节数组，
    // 不依赖原始 Mat 的后续生命周期。
    PreparedInput prepare(const cv::Mat& bgr) const;

    // 输入的 x/y/width/height 属于模型画布；输出裁剪到源图边界，
    // 供 tracker、ROI 判定和绘制统一使用。
    static cv::Rect2f restore_box(float x, float y, float width, float height,
                                  const LetterboxInfo& letterbox);

private:
    int model_width_;
    int model_height_;
};

}  // namespace edgevision
