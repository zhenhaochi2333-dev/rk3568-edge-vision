#pragma once

#include "edgevision/core_types.hpp"
#include "edgevision/region_monitor.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace edgevision {

class DisplayComposer {
public:
    static constexpr int kCanvasWidth = 800;
    static constexpr int kCanvasHeight = 1280;
    static constexpr int kCameraWidth = 800;
    static constexpr int kCameraHeight = 450;

    explicit DisplayComposer(const std::vector<std::string>& labels);

    const cv::Mat& compose(const cv::Mat& bgr,
                           const std::vector<Detection>& detections,
                           const FrameMetrics& metrics,
                           std::size_t roi_occupancy,
                           const NormalizedRoi* roi,
                           const std::vector<RegionEvent>& recent_events);

private:
    void draw_camera(const cv::Mat& bgr, const std::vector<Detection>& detections,
                     const NormalizedRoi* roi);
    void draw_text(const std::string& text, int x, int y, double scale = 0.7,
                   const cv::Scalar& color = cv::Scalar(226, 232, 240), int thickness = 1);
    void draw_section(int top, int bottom, const std::string& title);
    std::string label_for(int class_id) const;
    static std::string event_type_name(RegionEventType type);

    const std::vector<std::string>& labels_;
    cv::Mat canvas_;
    cv::Mat camera_view_;
};

}  // namespace edgevision
