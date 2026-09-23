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

// Fit 保留整幅源图并在画布两侧或上下留黑边；Fill 让画布无空边，但会居中裁掉超出部分。
enum class DisplayPolicy {
    Fit,
    Fill,
};

// 一帧源图到显示画布的几何变换快照。检测与 ROI 始终以源图像素为坐标基准，
// source_crop 描述实际被显示的源区域；缩放和 offset 再把该区域映射到画布。
// scale_x/scale_y 单独保存，是因为取整后的宽高可能让两个轴的最终比例略有不同。
struct DisplayGeometry {
    cv::Size source_size;
    cv::Size display_size;
    // 源图中可见的半开矩形 [x,x+w)×[y,y+h)。Fill 通常小于整幅源图。
    cv::Rect source_crop;
    // 可见源区域缩放后的整数像素尺寸；Fit 时小于画布的一轴。
    cv::Size resized_size;
    // 缩放结果在画布中的左上角；Fit 用它居中，Fill 为 (0,0)。
    cv::Point offset;
    double scale_x = 1.0;
    double scale_y = 1.0;
};

// compose() 内部分段的墙钟耗时，单位毫秒；每次 compose 都覆盖为该次调用的测量值。
// 这些值只描述裁剪/缩放、检测与 ROI 叠加、事件 toast 绘制，不包含相机采集或推理。
struct DisplayComposeTimings {
    double crop_resize_ms = 0.0;
    double overlay_ms = 0.0;
    double toast_ms = 0.0;
};

// 把源 BGR 帧和源图坐标中的业务结果合成为固定尺寸的显示画布。
// 对象复用 canvas_ 与中间缓冲区以避免每帧重新分配；因此不是线程安全的，
// 同一实例应由一个合成线程顺序调用，调用者也不能把返回引用当成独立帧长期保存。
class DisplayComposer {
public:
    static constexpr int kDefaultDisplayWidth = 1280;
    static constexpr int kDefaultDisplayHeight = 800;

    explicit DisplayComposer(const std::vector<std::string>& labels,
                              int display_width = kDefaultDisplayWidth,
                              int display_height = kDefaultDisplayHeight,
                              DisplayPolicy policy = DisplayPolicy::Fill);

    // 返回对象内部画布的只读引用。OpenCV Mat 的引用计数不会在这里生成一份像素副本；
    // 下一次 compose 会覆写同一存储。需要跨帧排队、异步写盘或跨线程保留时，调用方须 clone()。
    const cv::Mat& compose(const cv::Mat& bgr,
                           const std::vector<Detection>& detections,
                           const std::vector<RegionEvent>& new_events = {},
                           const NormalizedRoi* roi = nullptr,
                           bool show_roi = false);

    // geometry() 在首次 compose 确认输入尺寸后有效；源尺寸变化时会按当前策略重算。
    const DisplayGeometry& geometry() const { return geometry_; }
    const DisplayComposeTimings& last_timings() const { return last_timings_; }
    DisplayPolicy policy() const { return policy_; }

    // 纯几何计算入口，供测试/调用方在不创建画布的情况下检查 Fit/Fill 变换。
    // 对非法的零或负尺寸抛出异常；浮点缩放最终会取整到实际 OpenCV 像素尺寸。
    static DisplayGeometry compute_geometry(const cv::Size& source_size,
                                            const cv::Size& display_size,
                                            DisplayPolicy policy);
    // 将源坐标框先裁到 source_crop，再乘比例并加画布偏移；完全在裁剪外时返回空框。
    static cv::Rect2f map_source_box(const cv::Rect2f& source_box,
                                     const DisplayGeometry& geometry);
    static cv::Scalar color_for_class(int class_id);

private:
    // toast 的过期依据本机 steady_clock，而非墙钟：系统时间校准不会让提示突然变长或倒退。
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
