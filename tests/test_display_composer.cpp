#include "edgevision/display_composer.hpp"

#include <opencv2/core.hpp>

#include <cassert>
#include <chrono>
#include <string>
#include <vector>

void run_display_composer_tests()
{
    std::vector<std::string> labels(80, "class");
    labels[0] = "person";
    edgevision::DisplayComposer composer(labels);
    const cv::Mat camera(720, 1280, CV_8UC3, cv::Scalar(12, 24, 36));
    edgevision::Detection person{0, 0.86F, cv::Rect2f(100.0F, 120.0F, 200.0F, 300.0F)};
    person.track_id = 1;
    edgevision::FrameMetrics metrics;
    metrics.display_fps = 15.0;
    metrics.detection_fps = 7.5;
    metrics.inference_ms = 62.0;
    metrics.ai_latency_ms = 88.0;
    const edgevision::NormalizedRoi roi{0.25F, 0.20F, 0.50F, 0.60F};
    const auto timestamp = std::chrono::steady_clock::time_point{};
    const std::vector<edgevision::RegionEvent> events{
        edgevision::RegionEvent{edgevision::RegionEventType::Enter, 1, 0, timestamp}};
    const cv::Mat& dashboard = composer.compose(camera, {person}, metrics, 1U, &roi, events);
    assert(dashboard.cols == 800 && dashboard.rows == 1280);
    assert(dashboard.type() == CV_8UC3);
    assert(dashboard.at<cv::Vec3b>(10, 10) != cv::Vec3b(28, 34, 44));
}
