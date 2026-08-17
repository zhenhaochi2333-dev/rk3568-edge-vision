#include "edgevision/display_composer.hpp"

#include <opencv2/core.hpp>

#include <cassert>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

void run_display_composer_tests()
{
    const edgevision::DisplayGeometry fill = edgevision::DisplayComposer::compute_geometry(
        cv::Size(1280, 720), cv::Size(1280, 800), edgevision::DisplayPolicy::Fill);
    assert(fill.source_crop == cv::Rect(64, 0, 1152, 720));
    assert(std::fabs(fill.scale_x - (1280.0 / 1152.0)) < 1e-9);
    assert(std::fabs(fill.scale_y - (800.0 / 720.0)) < 1e-9);

    const cv::Rect2f visible = edgevision::DisplayComposer::map_source_box(
        cv::Rect2f(0.0F, 100.0F, 100.0F, 200.0F), fill);
    assert(std::fabs(visible.x) < 1e-5F);
    assert(std::fabs(visible.width - 40.0F) < 1e-4F);
    const cv::Rect2f hidden = edgevision::DisplayComposer::map_source_box(
        cv::Rect2f(0.0F, 0.0F, 32.0F, 100.0F), fill);
    assert(hidden.width == 0.0F && hidden.height == 0.0F);

    const edgevision::DisplayGeometry fit = edgevision::DisplayComposer::compute_geometry(
        cv::Size(1280, 720), cv::Size(1280, 800), edgevision::DisplayPolicy::Fit);
    assert(fit.source_crop == cv::Rect(0, 0, 1280, 720));
    assert(fit.resized_size == cv::Size(1280, 720));
    assert(fit.offset == cv::Point(0, 40));

    std::vector<std::string> labels(80, "class");
    labels[0] = "person";
    edgevision::DisplayComposer composer(labels);
    const cv::Mat camera(720, 1280, CV_8UC3, cv::Scalar(12, 24, 36));
    edgevision::Detection person{0, 0.86F, cv::Rect2f(100.0F, 120.0F, 200.0F, 300.0F)};
    person.track_id = 1;
    const auto timestamp = std::chrono::steady_clock::now();
    const std::vector<edgevision::RegionEvent> events{
        edgevision::RegionEvent{edgevision::RegionEventType::Enter, 1, 0, timestamp}};
    const cv::Mat& output = composer.compose(camera, {person}, events, nullptr, false);
    assert(output.cols == 1280 && output.rows == 800);
    assert(output.type() == CV_8UC3);
    assert(composer.last_timings().crop_resize_ms >= 0.0);
    assert(output.at<cv::Vec3b>(400, 640) != cv::Vec3b(0, 0, 0));
}
