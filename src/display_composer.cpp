#include "edgevision/display_composer.hpp"
#include "edgevision/visualizer.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace edgevision {

namespace {

const cv::Scalar kBackground(28, 34, 44);
const cv::Scalar kSection(38, 46, 58);
const cv::Scalar kBorder(76, 91, 110);
const cv::Scalar kPrimary(238, 242, 247);
const cv::Scalar kSecondary(177, 191, 209);
const cv::Scalar kRoi(34, 211, 238);

}  // namespace

DisplayComposer::DisplayComposer(const std::vector<std::string>& labels)
    : labels_(labels)
{
    canvas_.create(kCanvasHeight, kCanvasWidth, CV_8UC3);
    camera_view_.create(kCameraHeight, kCameraWidth, CV_8UC3);
}

std::string DisplayComposer::label_for(int class_id) const
{
    if (class_id >= 0 && static_cast<std::size_t>(class_id) < labels_.size()) {
        return labels_[static_cast<std::size_t>(class_id)];
    }
    return "class_" + std::to_string(class_id);
}

std::string DisplayComposer::event_type_name(RegionEventType type)
{
    switch (type) {
    case RegionEventType::Enter:
        return "ENTER";
    case RegionEventType::Exit:
        return "EXIT";
    case RegionEventType::Dwell:
        return "DWELL";
    }
    return "EVENT";
}

void DisplayComposer::draw_text(const std::string& text, int x, int y, double scale,
                                const cv::Scalar& color, int thickness)
{
    cv::putText(canvas_, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv::LINE_AA);
}

void DisplayComposer::draw_section(int top, int bottom, const std::string& title)
{
    cv::rectangle(canvas_, cv::Rect(18, top, kCanvasWidth - 36, bottom - top), kSection,
                  cv::FILLED);
    cv::rectangle(canvas_, cv::Rect(18, top, kCanvasWidth - 36, bottom - top), kBorder, 1);
    draw_text(title, 34, top + 34, 0.72, kPrimary, 1);
}

void DisplayComposer::draw_camera(const cv::Mat& bgr,
                                  const std::vector<Detection>& detections,
                                  const NormalizedRoi* roi)
{
    const cv::Rect camera_rect(0, 80, kCameraWidth, kCameraHeight);
    canvas_(camera_rect).setTo(cv::Scalar(12, 16, 22));

    const double scale = std::min(static_cast<double>(camera_rect.width) / bgr.cols,
                                  static_cast<double>(camera_rect.height) / bgr.rows);
    const int view_width = std::max(1, static_cast<int>(std::lround(bgr.cols * scale)));
    const int view_height = std::max(1, static_cast<int>(std::lround(bgr.rows * scale)));
    camera_view_.create(view_height, view_width, CV_8UC3);
    cv::resize(bgr, camera_view_, camera_view_.size(), 0.0, 0.0, cv::INTER_AREA);
    const int offset_x = camera_rect.x + (camera_rect.width - view_width) / 2;
    const int offset_y = camera_rect.y + (camera_rect.height - view_height) / 2;
    camera_view_.copyTo(canvas_(cv::Rect(offset_x, offset_y, view_width, view_height)));

    if (roi != nullptr) {
        const int left = offset_x + static_cast<int>(std::lround(roi->x * view_width));
        const int top = offset_y + static_cast<int>(std::lround(roi->y * view_height));
        const int right = offset_x + static_cast<int>(std::lround((roi->x + roi->width) * view_width));
        const int bottom = offset_y + static_cast<int>(std::lround((roi->y + roi->height) * view_height));
        cv::rectangle(canvas_, cv::Rect(cv::Point(left, top), cv::Point(right, bottom)), kRoi, 2);
        draw_text("ROI", std::max(8, left + 6), std::max(72, top - 8), 0.55, kRoi, 1);
    }

    for (const Detection& detection : detections) {
        const int left = offset_x + static_cast<int>(std::floor(detection.box.x * scale));
        const int top = offset_y + static_cast<int>(std::floor(detection.box.y * scale));
        const int right = offset_x + static_cast<int>(std::ceil((detection.box.x + detection.box.width) * scale));
        const int bottom = offset_y + static_cast<int>(std::ceil((detection.box.y + detection.box.height) * scale));
        const int clipped_left = std::max(offset_x, std::min(left, offset_x + view_width - 1));
        const int clipped_top = std::max(offset_y, std::min(top, offset_y + view_height - 1));
        const int clipped_right = std::max(clipped_left + 1, std::min(right, offset_x + view_width - 1));
        const int clipped_bottom = std::max(clipped_top + 1, std::min(bottom, offset_y + view_height - 1));
        const cv::Scalar color = Visualizer::color_for_class(detection.class_id);
        cv::rectangle(canvas_, cv::Rect(cv::Point(clipped_left, clipped_top),
                                       cv::Point(clipped_right, clipped_bottom)),
                      color, 2, cv::LINE_AA);

        std::ostringstream label;
        label << label_for(detection.class_id);
        if (detection.track_id >= 0) {
            label << " #" << detection.track_id;
        }
        label << ' ' << std::fixed << std::setprecision(0) << detection.confidence * 100.0F << '%';
        draw_text(label.str(), clipped_left + 4, std::max(98, clipped_top - 8), 0.52, color, 1);
    }
}

const cv::Mat& DisplayComposer::compose(const cv::Mat& bgr,
                                        const std::vector<Detection>& detections,
                                        const FrameMetrics& metrics,
                                        std::size_t roi_occupancy,
                                        const NormalizedRoi* roi,
                                        const std::vector<RegionEvent>& recent_events)
{
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::runtime_error("DisplayComposer requires a non-empty CV_8UC3 BGR image");
    }

    canvas_.setTo(kBackground);
    cv::rectangle(canvas_, cv::Rect(0, 0, kCanvasWidth, 72), cv::Scalar(22, 28, 37), cv::FILLED);
    draw_text("RK3568 EdgeVision", 24, 31, 0.82, kPrimary, 1);
    draw_text("YOLO11s INT8 | RK3568 NPU", 24, 58, 0.54, kSecondary, 1);
    draw_camera(bgr, detections, roi);

    draw_section(558, 735, "LIVE METRICS");
    std::ostringstream line;
    line << std::fixed << std::setprecision(1) << "Display FPS       " << metrics.display_fps;
    draw_text(line.str(), 36, 610, 0.62);
    line.str("");
    line << "Detection FPS     " << metrics.detection_fps;
    draw_text(line.str(), 36, 642, 0.62);
    line.str("");
    line << std::setprecision(1) << "Inference         " << metrics.inference_ms << " ms";
    draw_text(line.str(), 410, 610, 0.62);
    line.str("");
    line << "AI Latency        " << metrics.ai_latency_ms << " ms";
    draw_text(line.str(), 410, 642, 0.62);
    line.str("");
    line << "ROI Occupancy     " << roi_occupancy;
    draw_text(line.str(), 36, 690, 0.62, roi == nullptr ? kSecondary : kPrimary);
    line.str("");
    line << "Objects           " << detections.size();
    draw_text(line.str(), 410, 690, 0.62);

    draw_section(755, 1015, "TRACKED OBJECTS");
    const std::size_t object_limit = 8U;
    int object_y = 805;
    const std::size_t object_count = std::min(object_limit, detections.size());
    for (std::size_t index = 0U; index < object_count; ++index) {
        const Detection& detection = detections[index];
        std::ostringstream object;
        object << label_for(detection.class_id);
        if (detection.track_id >= 0) {
            object << " #" << detection.track_id;
        }
        object << "    " << std::fixed << std::setprecision(0) << detection.confidence * 100.0F << '%';
        draw_text(object.str(), 40, object_y, 0.64);
        object_y += 28;
    }
    if (detections.empty()) {
        draw_text("No tracked objects", 40, object_y, 0.64, kSecondary);
    } else if (detections.size() > object_limit) {
        draw_text("+" + std::to_string(detections.size() - object_limit) + " more",
                  40, object_y, 0.60, kSecondary);
    }

    draw_section(1035, 1265, "RECENT EVENTS");
    const std::size_t event_limit = 7U;
    int event_y = 1085;
    const std::size_t event_count = std::min(event_limit, recent_events.size());
    for (std::size_t offset = 0U; offset < event_count; ++offset) {
        const RegionEvent& event = recent_events[recent_events.size() - 1U - offset];
        std::ostringstream text;
        text << event_type_name(event.type) << "  " << label_for(event.class_id)
             << " #" << event.track_id;
        draw_text(text.str(), 40, event_y, 0.58, kSecondary);
        event_y += 25;
    }
    if (recent_events.empty()) {
        draw_text("No recent events", 40, event_y, 0.60, kSecondary);
    }

    return canvas_;
}

}  // namespace edgevision
