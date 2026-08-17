#pragma once

#include "edgevision/core_types.hpp"
#include "edgevision/region_monitor.hpp"

#include <opencv2/core.hpp>

#include <chrono>
#include <cstddef>
#include <deque>
#include <string>
#include <vector>

namespace edgevision {

enum class DisplayPolicy {
    Fit,
    Fill,
};

struct DisplayGeometry {
    cv::Size source_size;
    cv::Size display_size;
    cv::Rect source_crop;
    cv::Size resized_size;
    cv::Point offset;
    double scale_x = 1.0;
    double scale_y = 1.0;
};

struct DisplayComposeTimings {
    double crop_resize_ms = 0.0;
    double overlay_ms = 0.0;
    double toast_ms = 0.0;
};

class DisplayComposer {
public:
    static constexpr int kDefaultDisplayWidth = 1280;
    static constexpr int kDefaultDisplayHeight = 800;

    explicit DisplayComposer(const std::vector<std::string>& labels,
                              int display_width = kDefaultDisplayWidth,
                              int display_height = kDefaultDisplayHeight,
                              DisplayPolicy policy = DisplayPolicy::Fill);

    const cv::Mat& compose(const cv::Mat& bgr,
                           const std::vector<Detection>& detections,
                           const std::vector<RegionEvent>& new_events = {},
                           const NormalizedRoi* roi = nullptr,
                           bool show_roi = false);

    const DisplayGeometry& geometry() const { return geometry_; }
    const DisplayComposeTimings& last_timings() const { return last_timings_; }
    DisplayPolicy policy() const { return policy_; }

    static DisplayGeometry compute_geometry(const cv::Size& source_size,
                                            const cv::Size& display_size,
                                            DisplayPolicy policy);
    static cv::Rect2f map_source_box(const cv::Rect2f& source_box,
                                     const DisplayGeometry& geometry);
    static cv::Scalar color_for_class(int class_id);

private:
    struct UiToast {
        RegionEvent event;
        std::chrono::steady_clock::time_point shown_at{};
    };

    void ensure_geometry(const cv::Size& source_size);
    void draw_detection(const Detection& detection,
                        std::vector<cv::Rect>& occupied_label_rects);
    void draw_label(const std::string& text, const cv::Rect& box,
                    const cv::Scalar& color,
                    std::vector<cv::Rect>& occupied_label_rects);
    void draw_objects_badge(std::size_t object_count);
    void draw_roi(const NormalizedRoi& roi);
    void update_and_draw_toasts(const std::vector<RegionEvent>& new_events);
    void draw_text(const std::string& text, const cv::Point& origin,
                   double scale, const cv::Scalar& color, int thickness);
    std::string label_for(int class_id) const;
    static std::string event_type_name(RegionEventType type);

    const std::vector<std::string>& labels_;
    const int display_width_;
    const int display_height_;
    const DisplayPolicy policy_;
    DisplayGeometry geometry_;
    bool geometry_ready_ = false;
    cv::Mat canvas_;
    cv::Mat scaled_view_;
    std::deque<UiToast> toasts_;
    DisplayComposeTimings last_timings_;
};

}  // namespace edgevision
