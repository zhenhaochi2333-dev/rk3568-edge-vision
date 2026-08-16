#include "edgevision/visualizer.hpp"

#include "edgevision/label_loader.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace edgevision {

namespace {

const std::array<cv::Scalar, 10> kClassColors = {
    cv::Scalar(56, 189, 248),  cv::Scalar(52, 211, 153),  cv::Scalar(251, 146, 60),
    cv::Scalar(167, 139, 250), cv::Scalar(244, 114, 182), cv::Scalar(250, 204, 21),
    cv::Scalar(45, 212, 191),  cv::Scalar(248, 113, 113),  cv::Scalar(129, 140, 248),
    cv::Scalar(163, 230, 53)};

double frame_scale(const cv::Mat& image)
{
    return std::max(0.5, std::min(image.cols, image.rows) / 640.0);
}

std::string milliseconds(double value)
{
    std::ostringstream text;
    text << std::fixed << std::setprecision(1) << value << " ms";
    return text.str();
}

std::string object_count(double value)
{
    std::ostringstream text;
    text << static_cast<int>(value);
    return text.str();
}

void draw_panel(cv::Mat& image, const FrameMetrics& metrics, OverlayMode mode, double scale)
{
    const int margin = std::max(8, static_cast<int>(12.0 * scale));
    const int panel_width = std::min(image.cols - 2 * margin, std::max(230, static_cast<int>(286.0 * scale)));
    const int line_height = std::max(15, static_cast<int>(20.0 * scale));
    const int panel_lines = mode == OverlayMode::Video ? 6 : 5;
    const int panel_height = std::min(image.rows - 2 * margin,
                                      std::max(line_height * panel_lines + margin * 2,
                                               static_cast<int>(126.0 * scale)));
    if (panel_width <= 0 || panel_height <= 0) {
        return;
    }

    const cv::Rect panel_rect(margin, margin, panel_width, panel_height);
    cv::Mat overlay = image.clone();
    cv::rectangle(overlay, panel_rect, cv::Scalar(18, 24, 38), cv::FILLED);
    cv::addWeighted(overlay, 0.76, image, 0.24, 0.0, image);

    const double font = std::max(0.38, 0.50 * scale);
    const int thickness = std::max(1, static_cast<int>(std::lround(scale)));
    const cv::Scalar primary(242, 247, 255);
    const cv::Scalar secondary(185, 198, 216);
    int x = panel_rect.x + std::max(7, static_cast<int>(10.0 * scale));
    int y = panel_rect.y + std::max(15, static_cast<int>(19.0 * scale));
    cv::putText(image, "RK3568 EdgeVision", cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX,
                font, primary, thickness, cv::LINE_AA);
    y += line_height;
    cv::putText(image, "YOLOv5s INT8 | RK3568 NPU",
                cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, font * 0.84, secondary,
                thickness, cv::LINE_AA);
    y += line_height;

    std::ostringstream inference;
    inference << "Inference  " << milliseconds(metrics.inference_ms);
    cv::putText(image, inference.str(), cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX,
                font * 0.84, primary, thickness, cv::LINE_AA);
    y += line_height;

    std::ostringstream end_to_end;
    end_to_end << "E2E         " << milliseconds(metrics.end_to_end_ms);
    cv::putText(image, end_to_end.str(), cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX,
                font * 0.84, primary, thickness, cv::LINE_AA);
    y += line_height;

    std::ostringstream objects;
    objects << "Objects     " << object_count(metrics.object_count);
    cv::putText(image, objects.str(), cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX,
                font * 0.84, primary, thickness, cv::LINE_AA);
    if (mode == OverlayMode::Video) {
        y += line_height;
        std::ostringstream fps;
        fps << "FPS         " << std::fixed << std::setprecision(1) << metrics.fps;
        cv::putText(image, fps.str(), cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX,
                    font * 0.84, primary, thickness, cv::LINE_AA);
    }
}

}  // namespace

Visualizer::Visualizer(const std::vector<std::string>& labels)
    : labels_(labels)
{
}

cv::Scalar Visualizer::color_for_class(int class_id)
{
    const std::size_t index = class_id < 0
                                  ? 0U
                                  : static_cast<std::size_t>(class_id) % kClassColors.size();
    return kClassColors[index];
}

void Visualizer::draw(cv::Mat& bgr, const std::vector<Detection>& detections,
                      const FrameMetrics& metrics, OverlayMode mode) const
{
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::runtime_error("Visualizer requires a non-empty CV_8UC3 BGR image");
    }

    const double scale = frame_scale(bgr);
    const int thickness = std::max(1, static_cast<int>(std::lround(scale * 2.0)));
    const double font = std::max(0.45, 0.52 * scale);
    const int baseline = std::max(1, static_cast<int>(std::lround(4.0 * scale)));
    for (const Detection& detection : detections) {
        const int left = std::max(0, static_cast<int>(std::floor(detection.box.x)));
        const int top = std::max(0, static_cast<int>(std::floor(detection.box.y)));
        const int right = std::min(bgr.cols - 1,
                                   static_cast<int>(std::ceil(detection.box.x + detection.box.width)));
        const int bottom = std::min(bgr.rows - 1,
                                    static_cast<int>(std::ceil(detection.box.y + detection.box.height)));
        if (left >= bgr.cols || top >= bgr.rows || right <= left || bottom <= top) {
            continue;
        }

        const cv::Scalar color = color_for_class(detection.class_id);
        cv::rectangle(bgr, cv::Rect(cv::Point(left, top), cv::Point(right, bottom)), color,
                      thickness, cv::LINE_AA);

        std::string class_name;
        if (detection.class_id >= 0 && static_cast<std::size_t>(detection.class_id) < labels_.size()) {
            class_name = labels_[static_cast<std::size_t>(detection.class_id)];
        } else {
            class_name = "class_" + std::to_string(detection.class_id);
        }
        std::ostringstream label;
        label << class_name << ' ' << std::fixed << std::setprecision(1)
              << detection.confidence * 100.0F << '%';
        int text_baseline = 0;
        const cv::Size text_size = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX,
                                                    font, thickness, &text_baseline);
        const int label_x = std::max(0, std::min(left, bgr.cols - text_size.width - 2));
        int label_y = top - text_baseline - 6;
        if (label_y < text_size.height + text_baseline + 2) {
            label_y = std::min(bgr.rows - text_baseline - 2, top + text_size.height + baseline + 4);
        }
        const cv::Rect label_rect(label_x,
                                  std::max(0, label_y - text_size.height - text_baseline - 3),
                                  std::min(text_size.width + 8, bgr.cols - label_x),
                                  std::min(text_size.height + text_baseline + 6,
                                           bgr.rows - std::max(0, label_y - text_size.height - text_baseline - 3)));
        if (label_rect.width > 0 && label_rect.height > 0) {
            cv::rectangle(bgr, label_rect, color, cv::FILLED);
        }
        cv::putText(bgr, label.str(), cv::Point(label_x + 4, label_y), cv::FONT_HERSHEY_SIMPLEX,
                    font, cv::Scalar(20, 24, 32), thickness, cv::LINE_AA);
    }

    draw_panel(bgr, metrics, mode, scale);
}

}  // namespace edgevision
