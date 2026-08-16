#include "edgevision/image_processor.hpp"

#include <opencv2/core.hpp>

#include <cassert>
#include <cmath>

void run_geometry_tests()
{
    edgevision::ImageProcessor processor(640, 640);

    const cv::Mat wide(100, 200, CV_8UC3, cv::Scalar(10, 20, 30));
    const edgevision::PreparedInput wide_input = processor.prepare(wide);
    assert(wide_input.rgb_image.cols == 640 && wide_input.rgb_image.rows == 640);
    assert(wide_input.letterbox.pad_x == 0);
    assert(wide_input.letterbox.pad_y == 160);
    assert(std::fabs(wide_input.letterbox.scale - 3.2F) < 1e-5F);
    const cv::Vec3b wide_pixel = wide_input.rgb_image.at<cv::Vec3b>(160, 0);
    assert(wide_pixel[0] == 30 && wide_pixel[1] == 20 && wide_pixel[2] == 10);

    const cv::Mat tall(200, 100, CV_8UC3, cv::Scalar(1, 2, 3));
    const edgevision::PreparedInput tall_input = processor.prepare(tall);
    assert(tall_input.letterbox.pad_x == 160);
    assert(tall_input.letterbox.pad_y == 0);
    assert(tall_input.nhwc.size() == 640U * 640U * 3U);

    const cv::Mat square(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    const edgevision::PreparedInput square_input = processor.prepare(square);
    assert(square_input.letterbox.pad_x == 0 && square_input.letterbox.pad_y == 0);

    const cv::Rect2f restored = edgevision::ImageProcessor::restore_box(
        160.0F + 32.0F, 16.0F, 64.0F, 32.0F, wide_input.letterbox);
    assert(restored.x >= 9.9F && restored.x <= 10.1F);
    assert(restored.y >= 4.9F && restored.y <= 5.1F);
    assert(restored.width >= 19.9F && restored.width <= 20.1F);
    assert(restored.height >= 9.9F && restored.height <= 10.1F);
}
