/* 检测框保持源图坐标；本模块独立计算显示裁剪/缩放，将框、ROI 和事件提示映射到画布。 */
#include "edgevision/display_composer.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace edgevision {

namespace {

using Clock = std::chrono::steady_clock;

constexpr double kToastLifetimeSeconds = 2.8;
constexpr std::size_t kMaxToasts = 3U;
const cv::Scalar kCanvasBlack(0, 0, 0);
const cv::Scalar kOverlayBackground(18, 22, 28);
const cv::Scalar kOverlayText(245, 247, 250);
const cv::Scalar kRoiColor(90, 210, 220);
const std::array<cv::Scalar, 10> kClassColors = {
    cv::Scalar(56, 189, 248),  cv::Scalar(52, 211, 153),  cv::Scalar(251, 146, 60),
    cv::Scalar(167, 139, 250), cv::Scalar(244, 114, 182), cv::Scalar(250, 204, 21),
    cv::Scalar(45, 212, 191),  cv::Scalar(248, 113, 113), cv::Scalar(129, 140, 248),
    cv::Scalar(163, 230, 53)};

bool intersects(const cv::Rect& first, const cv::Rect& second)
{
    return (first & second).area() > 0;
}

}  // namespace

DisplayComposer::DisplayComposer(const std::vector<std::string>& labels,
                                 int display_width,
                                 int display_height,
                                 DisplayPolicy policy)
    : labels_(labels), display_width_(display_width), display_height_(display_height), policy_(policy)
{
    if (display_width_ <= 0 || display_height_ <= 0) {
        throw std::runtime_error("DisplayComposer requires a positive display size");
    }
    geometry_.display_size = cv::Size(display_width_, display_height_);
}

// Fill 居中裁掉源图边缘以铺满画布；Fit 保留完整源图并留边，二者映射参数不同。
DisplayGeometry DisplayComposer::compute_geometry(const cv::Size& source_size,
                                                  const cv::Size& display_size,
                                                  DisplayPolicy policy)
{
    if (source_size.width <= 0 || source_size.height <= 0 || display_size.width <= 0 ||
        display_size.height <= 0) {
        throw std::runtime_error("display geometry requires positive source and display sizes");
    }

    DisplayGeometry geometry;
    geometry.source_size = source_size;
    geometry.display_size = display_size;
    // Fill 采用两轴所需缩放比例中的较大值：缩放后画面一定盖住画布，
    // 因而必须先在源图上裁出与画布宽高比相符的中央区域。
    if (policy == DisplayPolicy::Fill) {
        // 例如 1280x720 填进 1280x800：高度比例更大，换算得源图只保留中央 1152x720，
        // 左右各裁 64 像素；source_crop 的整数取整就是最终显示几何的依据。
        const double scale = std::max(static_cast<double>(display_size.width) / source_size.width,
                                      static_cast<double>(display_size.height) / source_size.height);
        const int crop_width = std::max(1, std::min(source_size.width,
                                                    static_cast<int>(std::lround(display_size.width / scale))));
        const int crop_height = std::max(1, std::min(source_size.height,
                                                     static_cast<int>(std::lround(display_size.height / scale))));
        geometry.source_crop = cv::Rect((source_size.width - crop_width) / 2,
                                        (source_size.height - crop_height) / 2,
                                        crop_width, crop_height);
        geometry.resized_size = display_size;
        geometry.offset = cv::Point(0, 0);
        geometry.scale_x = static_cast<double>(display_size.width) / crop_width;
        geometry.scale_y = static_cast<double>(display_size.height) / crop_height;
    } else {
        // Fit 取较小比例使整幅图都落在画布内。缩放尺寸取整后再由 offset 居中，
        // 留边像素若遇奇数差值会在左右/上下相差一个像素。
        const double scale = std::min(static_cast<double>(display_size.width) / source_size.width,
                                      static_cast<double>(display_size.height) / source_size.height);
        const int resized_width = std::max(1, static_cast<int>(std::lround(source_size.width * scale)));
        const int resized_height = std::max(1, static_cast<int>(std::lround(source_size.height * scale)));
        geometry.source_crop = cv::Rect(0, 0, source_size.width, source_size.height);
        geometry.resized_size = cv::Size(resized_width, resized_height);
        geometry.offset = cv::Point((display_size.width - resized_width) / 2,
                                    (display_size.height - resized_height) / 2);
        geometry.scale_x = static_cast<double>(resized_width) / source_size.width;
        geometry.scale_y = static_cast<double>(resized_height) / source_size.height;
    }
    return geometry;
}

// 先与实际显示的源图裁剪区相交，再按缩放与偏移变成画布坐标。
cv::Rect2f DisplayComposer::map_source_box(const cv::Rect2f& source_box,
                                           const DisplayGeometry& geometry)
{
    const cv::Rect2f crop(static_cast<float>(geometry.source_crop.x),
                          static_cast<float>(geometry.source_crop.y),
                          static_cast<float>(geometry.source_crop.width),
                          static_cast<float>(geometry.source_crop.height));
    // Rect2f 按左上角加宽高表达。裁剪交集必须先求出：否则 Fill 时画面边缘被裁掉，
    // 但检测框仍会落在画布上，造成“框住了看不见的物体”的错位。
    const float left = std::max(source_box.x, crop.x);
    const float top = std::max(source_box.y, crop.y);
    const float right = std::min(source_box.x + source_box.width, crop.x + crop.width);
    const float bottom = std::min(source_box.y + source_box.height, crop.y + crop.height);
    if (right <= left || bottom <= top) {
        return cv::Rect2f();
    }
    return cv::Rect2f(
        static_cast<float>(geometry.offset.x) + (left - crop.x) * static_cast<float>(geometry.scale_x),
        static_cast<float>(geometry.offset.y) + (top - crop.y) * static_cast<float>(geometry.scale_y),
        (right - left) * static_cast<float>(geometry.scale_x),
        (bottom - top) * static_cast<float>(geometry.scale_y));
}

cv::Scalar DisplayComposer::color_for_class(int class_id)
{
    // 颜色只为视觉区分而确定性循环复用；class_id 越界不会访问标签表或颜色表越界。
    const std::size_t index = class_id < 0
                                  ? 0U
                                  : static_cast<std::size_t>(class_id) % kClassColors.size();
    return kClassColors[index];
}

void DisplayComposer::ensure_geometry(const cv::Size& source_size)
{
    if (geometry_ready_ && geometry_.source_size == source_size) {
        return;
    }
    // 源分辨率变化才重算几何和缓冲区。常见固定分辨率视频只付首次成本；
    // canvas_ 自己的像素内存会由 create() 按目标大小复用。
    geometry_ = compute_geometry(source_size, cv::Size(display_width_, display_height_), policy_);
    canvas_.create(display_height_, display_width_, CV_8UC3);
    if (policy_ == DisplayPolicy::Fit) {
        scaled_view_.create(geometry_.resized_size, CV_8UC3);
    } else {
        scaled_view_.release();
    }
    geometry_ready_ = true;
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

void DisplayComposer::draw_text(const std::string& text, const cv::Point& origin,
                                double scale, const cv::Scalar& color, int thickness)
{
    cv::putText(canvas_, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                cv::LINE_AA);
}

void DisplayComposer::draw_label(const std::string& text, const cv::Rect& box,
                                 const cv::Scalar& color,
                                 std::vector<cv::Rect>& occupied_label_rects)
{
    // 标签优先放框上方，再尝试框内、框下方；先测文字尺寸并维护已占用区域，
    // 让相邻目标尽量不互相遮挡。若三个位置都冲突，则保留首选位置并裁入画布。
    const double font = 0.52;
    const int thickness = 1;
    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font,
                                               thickness, &baseline);
    const int width = text_size.width + 8;
    const int height = text_size.height + baseline + 6;
    const int x = std::max(0, std::min(box.x, canvas_.cols - width));
    const int above_y = box.y - height - 3;
    const int inside_y = box.y + 3;
    const int below_y = box.y + box.height + 3;
    const std::array<cv::Rect, 3> candidates = {
        cv::Rect(x, above_y, width, height), cv::Rect(x, inside_y, width, height),
        cv::Rect(x, below_y, width, height)};

    cv::Rect chosen = candidates.front();
    for (const cv::Rect& candidate : candidates) {
        const cv::Rect clipped = candidate & cv::Rect(0, 0, canvas_.cols, canvas_.rows);
        if (clipped.width == width && clipped.height == height) {
            bool overlaps = false;
            for (const cv::Rect& occupied : occupied_label_rects) {
                if (intersects(clipped, occupied)) {
                    overlaps = true;
                    break;
                }
            }
            if (!overlaps) {
                chosen = clipped;
                break;
            }
        }
    }
    chosen &= cv::Rect(0, 0, canvas_.cols, canvas_.rows);
    if (chosen.width <= 0 || chosen.height <= 0) {
        return;
    }
    cv::rectangle(canvas_, chosen, kOverlayBackground, cv::FILLED);
    cv::rectangle(canvas_, chosen, color, 1, cv::LINE_AA);
    cv::putText(canvas_, text, cv::Point(chosen.x + 4, chosen.y + text_size.height + 1),
                cv::FONT_HERSHEY_SIMPLEX, font, kOverlayText, thickness, cv::LINE_AA);
    occupied_label_rects.push_back(chosen);
}

void DisplayComposer::draw_detection(const Detection& detection,
                                     std::vector<cv::Rect>& occupied_label_rects)
{
    // Detection.box 仍是推理输入对应的源图坐标。这里才映射到输出画布，
    // 以免显示裁剪反向影响检测、跟踪或区域事件的业务坐标。
    const cv::Rect2f mapped = map_source_box(detection.box, geometry_);
    if (mapped.width <= 0.0F || mapped.height <= 0.0F) {
        return;
    }
    const cv::Rect display_bounds(0, 0, canvas_.cols, canvas_.rows);
    const cv::Rect box = cv::Rect(static_cast<int>(std::floor(mapped.x)),
                                  static_cast<int>(std::floor(mapped.y)),
                                  std::max(1, static_cast<int>(std::ceil(mapped.width))),
                                  std::max(1, static_cast<int>(std::ceil(mapped.height)))) &
                         display_bounds;
    if (box.width <= 0 || box.height <= 0) {
        return;
    }

    const cv::Scalar color = color_for_class(detection.class_id);
    cv::rectangle(canvas_, box, color, 2, cv::LINE_AA);
    std::ostringstream label;
    label << label_for(detection.class_id);
    const int display_id = detection.logical_id >= 0 ? detection.logical_id : detection.track_id;
    if (display_id >= 0) {
        label << " #" << display_id;
    }
    label << ' ' << std::fixed << std::setprecision(0) << detection.confidence * 100.0F << '%';
    draw_label(label.str(), box, color, occupied_label_rects);
}

void DisplayComposer::draw_objects_badge(std::size_t object_count)
{
    std::ostringstream text;
    text << "Objects " << object_count;
    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(text.str(), cv::FONT_HERSHEY_SIMPLEX, 0.62, 1,
                                               &baseline);
    const int width = text_size.width + 18;
    const int height = text_size.height + baseline + 12;
    const cv::Rect badge(canvas_.cols - width - 16, 16, width, height);
    cv::rectangle(canvas_, badge, kOverlayBackground, cv::FILLED);
    cv::putText(canvas_, text.str(), cv::Point(badge.x + 9, badge.y + text_size.height + 4),
                cv::FONT_HERSHEY_SIMPLEX, 0.62, kOverlayText, 1, cv::LINE_AA);
}

void DisplayComposer::draw_roi(const NormalizedRoi& roi)
{
    const cv::Rect2f source_roi(roi.x * geometry_.source_size.width,
                                roi.y * geometry_.source_size.height,
                                roi.width * geometry_.source_size.width,
                                roi.height * geometry_.source_size.height);
    const cv::Rect2f mapped = map_source_box(source_roi, geometry_);
    if (mapped.width <= 0.0F || mapped.height <= 0.0F) {
        return;
    }
    const cv::Rect box = cv::Rect(static_cast<int>(std::lround(mapped.x)),
                                  static_cast<int>(std::lround(mapped.y)),
                                  std::max(1, static_cast<int>(std::lround(mapped.width))),
                                  std::max(1, static_cast<int>(std::lround(mapped.height)))) &
                         cv::Rect(0, 0, canvas_.cols, canvas_.rows);
    if (box.width > 0 && box.height > 0) {
        cv::rectangle(canvas_, box, kRoiColor, 1, cv::LINE_AA);
    }
}

void DisplayComposer::update_and_draw_toasts(const std::vector<RegionEvent>& new_events)
{
    // 本轮传入的所有事件共用“开始绘制时刻”作为显示起点。按最早事件先淘汰，
    // 同时把队列限制为少量条目，避免事件突发令提示占满画布。
    const Clock::time_point now = Clock::now();
    for (const RegionEvent& event : new_events) {
        toasts_.push_back(UiToast{event, now});
    }
    while (toasts_.size() > kMaxToasts) {
        toasts_.pop_front();
    }
    while (!toasts_.empty() &&
           std::chrono::duration<double>(now - toasts_.front().shown_at).count() >
               kToastLifetimeSeconds) {
        toasts_.pop_front();
    }

    int y = canvas_.rows - 22;
    for (auto it = toasts_.rbegin(); it != toasts_.rend(); ++it) {
        std::ostringstream text;
        text << event_type_name(it->event.type) << " | " << label_for(it->event.class_id);
        const int display_id = it->event.logical_id >= 0 ? it->event.logical_id
                                                          : it->event.track_id;
        if (display_id >= 0) {
            text << " #" << display_id;
        }
        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(text.str(), cv::FONT_HERSHEY_SIMPLEX, 0.58, 1,
                                                   &baseline);
        const int width = text_size.width + 18;
        const int height = text_size.height + baseline + 12;
        const int x = std::max(10, (canvas_.cols - width) / 2);
        const int top = y - height;
        cv::rectangle(canvas_, cv::Rect(x, top, width, height), kOverlayBackground, cv::FILLED);
        cv::putText(canvas_, text.str(), cv::Point(x + 9, top + text_size.height + 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.58, kOverlayText, 1, cv::LINE_AA);
        y = top - 8;
    }
}

// 返回内部 canvas_ 的引用，只在下一次 compose 前有效；调用端应立即显示或复制。
const cv::Mat& DisplayComposer::compose(const cv::Mat& bgr,
                                        const std::vector<Detection>& detections,
                                        const std::vector<RegionEvent>& new_events,
                                        const NormalizedRoi* roi,
                                        bool show_roi)
{
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::runtime_error("DisplayComposer requires a non-empty CV_8UC3 BGR image");
    }

    // 几何、裁剪和缩放只作用于图像；后续叠加仍使用同一画布。
    // 分段计时用 steady_clock 的单调时钟，适合耗时差值，不代表墙钟时间戳。
    const Clock::time_point compose_start = Clock::now();
    ensure_geometry(bgr.size());
    // Fill 直接把源图 ROI 缩放到整张 canvas；Fit 先清黑底，再把完整缩放图复制到居中 ROI。
    // OpenCV 的 bgr(...) 是共享源像素的视图，resize 在本调用内读取它并写入独立 canvas。
    if (policy_ == DisplayPolicy::Fill) {
        cv::resize(bgr(geometry_.source_crop), canvas_, geometry_.display_size, 0.0, 0.0,
                   cv::INTER_LINEAR);
    } else {
        canvas_.setTo(kCanvasBlack);
        cv::resize(bgr, scaled_view_, geometry_.resized_size, 0.0, 0.0, cv::INTER_LINEAR);
        scaled_view_.copyTo(canvas_(cv::Rect(geometry_.offset, geometry_.resized_size)));
    }
    const Clock::time_point after_resize = Clock::now();
    last_timings_.crop_resize_ms =
        std::chrono::duration<double, std::milli>(after_resize - compose_start).count();

    // overlay 计时区间只包含本帧 detections、可选 ROI 和对象数角标；事件 toast 单独计时。
    const Clock::time_point overlay_start = Clock::now();
    std::vector<cv::Rect> occupied_label_rects;
    occupied_label_rects.reserve(detections.size());
    // 只排序指向本轮检测结果的指针，绘制顺序不改动调用者的检测列表。
    std::vector<const Detection*> ordered;
    ordered.reserve(detections.size());
    for (const Detection& detection : detections) {
        ordered.push_back(&detection);
    }
    // 用源检测对象地址排序而非复制 Detection：坐标和置信度不变，标签位置也不会改变
    // 调用方 vector 的顺序；上到下、左到右的次序让重叠标签更容易预测。
    std::sort(ordered.begin(), ordered.end(), [this](const Detection* first, const Detection* second) {
        const cv::Rect2f first_box = map_source_box(first->box, geometry_);
        const cv::Rect2f second_box = map_source_box(second->box, geometry_);
        if (first_box.y != second_box.y) {
            return first_box.y < second_box.y;
        }
        return first_box.x < second_box.x;
    });
    for (const Detection* detection : ordered) {
        draw_detection(*detection, occupied_label_rects);
    }
    if (show_roi && roi != nullptr) {
        draw_roi(*roi);
    }
    draw_objects_badge(detections.size());
    const Clock::time_point after_overlay = Clock::now();
    last_timings_.overlay_ms =
        std::chrono::duration<double, std::milli>(after_overlay - overlay_start).count();

    const Clock::time_point toast_start = Clock::now();
    update_and_draw_toasts(new_events);
    last_timings_.toast_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - toast_start).count();
    return canvas_;
}

}  // namespace edgevision
