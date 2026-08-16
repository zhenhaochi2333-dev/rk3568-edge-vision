#include "edgevision/visualizer.hpp"

#include <opencv2/core.hpp>

#include <cassert>
#include <string>
#include <vector>

void run_visualizer_tests()
{
    std::vector<std::string> labels(80, "class");
    labels[0] = "person";
    edgevision::Visualizer visualizer(labels);

    const cv::Scalar first = edgevision::Visualizer::color_for_class(0);
    const cv::Scalar second = edgevision::Visualizer::color_for_class(0);
    assert(first == second);

    cv::Mat image(64, 96, CV_8UC3, cv::Scalar(20, 30, 40));
    const std::vector<edgevision::Detection> detections{
        edgevision::Detection{0, 0.836F, cv::Rect2f(-3.0F, -2.0F, 40.0F, 32.0F)}};
    edgevision::FrameMetrics metrics;
    metrics.inference_ms = 4.0;
    metrics.end_to_end_ms = 7.0;
    metrics.object_count = 1.0;
    visualizer.draw(image, detections, metrics, edgevision::OverlayMode::Image);
    assert(image.cols == 96 && image.rows == 64);
    assert(image.at<cv::Vec3b>(0, 0) != cv::Vec3b(20, 30, 40));

    cv::Mat empty(32, 32, CV_8UC3, cv::Scalar(1, 2, 3));
    visualizer.draw(empty, {}, metrics, edgevision::OverlayMode::Video);
    assert(empty.cols == 32 && empty.rows == 32);

    metrics.display_fps = 28.0;
    metrics.detection_fps = 7.5;
    metrics.ai_latency_ms = 70.0;
    visualizer.draw(empty, {}, metrics, edgevision::OverlayMode::SmoothVideo);
    assert(empty.cols == 32 && empty.rows == 32);
}
