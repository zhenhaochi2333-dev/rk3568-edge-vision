#include "edgevision/application.hpp"

#include "edgevision/display_composer.hpp"
#include "edgevision/iou_tracker.hpp"
#include "edgevision/label_loader.hpp"
#include "edgevision/logger.hpp"
#include "edgevision/perf_monitor.hpp"
#include "edgevision/region_monitor.hpp"
#include "edgevision/rknn_model.hpp"
#include "edgevision/visualizer.hpp"
#include "edgevision/yolo11_detector.hpp"
#if EDGEVISION_WITH_VIDEO
#include "edgevision/camera_source.hpp"
#include "edgevision/video_io.hpp"
#endif

#include <opencv2/imgcodecs.hpp>
#if EDGEVISION_WITH_VIDEO
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include <limits.h>
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace edgevision {

namespace {

using Clock = std::chrono::steady_clock;
constexpr const char* kDisplayWindowTitle = "RK3568 EdgeVision";

bool is_regular_file(const std::string& path)
{
    struct stat info{};
    return stat(path.c_str(), &info) == 0 && S_ISREG(info.st_mode);
}

std::string real_path_if_possible(const std::string& path)
{
    char resolved[PATH_MAX]{};
    if (realpath(path.c_str(), resolved) != nullptr) {
        return resolved;
    }
    return path;
}

void validate_options(const AppOptions& options)
{
    if (!is_regular_file(options.model_path)) {
        throw std::runtime_error("missing model file: " + options.model_path);
    }
    if (!is_regular_file(options.labels_path)) {
        throw std::runtime_error("missing labels file: " + options.labels_path);
    }
    const bool camera_mode = !options.camera_path.empty();
    if (!camera_mode && !is_regular_file(options.input_path)) {
        throw std::runtime_error("missing input file: " + options.input_path);
    }
    if (camera_mode && options.output_path.empty() && !options.show) {
        throw std::runtime_error("camera mode requires --show or --output");
    }
    if (options.fullscreen && !options.show) {
        throw std::runtime_error("--fullscreen requires --show");
    }
    if (options.smooth_preview && options.camera_path.empty()) {
        throw std::runtime_error("--smooth-preview requires --camera");
    }
    if (options.smooth_preview && !options.show) {
        throw std::runtime_error("--smooth-preview requires --show");
    }
    if (options.smooth_preview && !options.output_path.empty()) {
        throw std::runtime_error("--smooth-preview does not support --output");
    }
    if (!options.output_path.empty() && !camera_mode &&
        real_path_if_possible(options.input_path) == real_path_if_possible(options.output_path)) {
        throw std::runtime_error("input and output paths must be different");
    }

    if (options.output_path.empty()) {
        return;
    }
    struct stat output_info{};
    if (stat(options.output_path.c_str(), &output_info) == 0) {
        if (!S_ISREG(output_info.st_mode)) {
            throw std::runtime_error("output path exists but is not a regular file: " + options.output_path);
        }
        if (!options.force) {
            throw std::runtime_error("output already exists; pass --force to replace it: " + options.output_path);
        }
    }
}

std::string format_detection(const Detection& detection, const std::vector<std::string>& labels)
{
    std::ostringstream line;
    line << LabelLoader::name(labels, detection.class_id) << " @ ("
         << static_cast<int>(detection.box.x) << ' '
         << static_cast<int>(detection.box.y) << ' '
         << static_cast<int>(detection.box.x + detection.box.width) << ' '
         << static_cast<int>(detection.box.y + detection.box.height) << ") "
         << std::fixed << std::setprecision(3) << detection.confidence;
    return line.str();
}

bool gui_available()
{
    return std::getenv("DISPLAY") != nullptr || std::getenv("WAYLAND_DISPLAY") != nullptr;
}

#if EDGEVISION_WITH_VIDEO

constexpr int kPreviewWidth = 640;
constexpr int kPreviewHeight = 640;
constexpr int kCameraPreviewWidth = 720;
constexpr int kCameraPreviewHeight = 405;
constexpr int kPreviewX = 20;
constexpr int kPreviewY = 20;
// The X11 desktop is temporarily rotated to the physical panel's landscape
// logical size. Keep the camera's 16:9 aspect ratio inside this canvas.
constexpr int kFullscreenWidth = 1280;
constexpr int kFullscreenHeight = 800;

cv::Mat fit_frame_to_display(const cv::Mat& frame, int width, int height)
{
    if (frame.empty() || width <= 0 || height <= 0) {
        return cv::Mat();
    }

    const double scale = std::min(static_cast<double>(width) / frame.cols,
                                  static_cast<double>(height) / frame.rows);
    const int fitted_width = std::max(1, static_cast<int>(std::lround(frame.cols * scale)));
    const int fitted_height = std::max(1, static_cast<int>(std::lround(frame.rows * scale)));

    cv::Mat fitted;
    cv::resize(frame, fitted, cv::Size(fitted_width, fitted_height), 0.0, 0.0,
               cv::INTER_LINEAR);
    cv::Mat canvas(height, width, frame.type(), cv::Scalar::all(0));
    const int offset_x = (width - fitted_width) / 2;
    const int offset_y = (height - fitted_height) / 2;
    fitted.copyTo(canvas(cv::Rect(offset_x, offset_y, fitted_width, fitted_height)));
    return canvas;
}

void configure_display_window(int width, int height, bool fullscreen)
{
    cv::namedWindow(kDisplayWindowTitle, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    if (fullscreen) {
        cv::setWindowProperty(kDisplayWindowTitle, cv::WND_PROP_FULLSCREEN,
                              cv::WINDOW_FULLSCREEN);
        cv::resizeWindow(kDisplayWindowTitle, kFullscreenWidth, kFullscreenHeight);
        return;
    }
    cv::resizeWindow(kDisplayWindowTitle, width, height);
    cv::moveWindow(kDisplayWindowTitle, kPreviewX, kPreviewY);
}

void wait_for_image_preview()
{
    for (;;) {
        const int key = cv::waitKey(50);
        if (key == 27 || key == 10 || key == 13 || key == 'q' || key == 'Q') {
            break;
        }
    }
}

#endif

void log_image_metrics(const FrameMetrics& metrics)
{
    log_perf("preprocess=" + std::to_string(metrics.preprocess_ms) + " ms inference=" +
             std::to_string(metrics.inference_ms) + " ms postprocess=" +
             std::to_string(metrics.postprocess_ms) + " ms visualization=" +
             std::to_string(metrics.visualization_ms) + " ms e2e=" +
             std::to_string(metrics.end_to_end_ms) + " ms");
}

int run_image(const AppOptions& options, const std::vector<std::string>& labels,
              Yolo11Detector& detector, const Visualizer& visualizer, const cv::Mat& image)
{
    const auto e2e_start = Clock::now();
    const DetectionResult result = detector.detect_with_metrics(image);
    FrameMetrics metrics = result.metrics;
    metrics.object_count = static_cast<double>(result.detections.size());

    const auto visualization_start = Clock::now();
    cv::Mat output = image.clone();
    visualizer.draw(output, result.detections, metrics, OverlayMode::Image);
    const auto visualization_end = Clock::now();
    metrics.visualization_ms =
        std::chrono::duration<double, std::milli>(visualization_end - visualization_start).count();
    metrics.end_to_end_ms =
        std::chrono::duration<double, std::milli>(visualization_end - e2e_start).count();

    for (const Detection& detection : result.detections) {
        log_info(format_detection(detection, labels));
    }
    log_info("detections=" + std::to_string(result.detections.size()));
    log_image_metrics(metrics);
    if (!cv::imwrite(options.output_path, output)) {
        throw std::runtime_error("failed to write output image: " + options.output_path);
    }
    log_info("wrote output image: " + options.output_path);

    if (options.show) {
#if EDGEVISION_WITH_VIDEO
        if (!gui_available()) {
            log_warn("Local GUI session unavailable; display disabled");
        } else {
            try {
                configure_display_window(kPreviewWidth, kPreviewHeight, options.fullscreen);
                cv::imshow(kDisplayWindowTitle, output);
                wait_for_image_preview();
                cv::destroyWindow(kDisplayWindowTitle);
            } catch (const cv::Exception& error) {
                log_warn(std::string("Local GUI session unavailable; display disabled: ") + error.what());
            }
        }
#else
        log_warn("Local GUI session unavailable; display disabled");
#endif
    }
    return 0;
}

#if EDGEVISION_WITH_VIDEO

constexpr std::size_t kCameraWarmupFrames = 30U;

struct CameraProfileTotals {
    double camera_read_ms = 0.0;
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    double visualization_ms = 0.0;
    double display_ms = 0.0;
    double writer_ms = 0.0;
    double other_ms = 0.0;
    double loop_ms = 0.0;
    double legacy_e2e_ms = 0.0;
    std::size_t frames = 0U;

    void add(double camera_read, double preprocess, double inference, double postprocess,
             double visualization, double display, double writer, double other, double loop,
             double legacy_e2e)
    {
        camera_read_ms += camera_read;
        preprocess_ms += preprocess;
        inference_ms += inference;
        postprocess_ms += postprocess;
        visualization_ms += visualization;
        display_ms += display;
        writer_ms += writer;
        other_ms += other;
        loop_ms += loop;
        legacy_e2e_ms += legacy_e2e;
        ++frames;
    }
};

volatile std::sig_atomic_t g_stop_requested = 0;

void on_sigint(int)
{
    g_stop_requested = 1;
}

class SignalGuard {
public:
    SignalGuard()
        : previous_(std::signal(SIGINT, on_sigint))
    {
        g_stop_requested = 0;
    }

    ~SignalGuard()
    {
        std::signal(SIGINT, previous_);
    }

private:
    void (*previous_)(int) = SIG_DFL;
};

struct SmoothDetectionSnapshot {
    std::uint64_t generation = 0U;
    std::uint64_t source_frame_id = 0U;
    std::vector<Detection> detections;
    FrameMetrics metrics;
    RegionSnapshot region;
    Clock::time_point captured_at{};
    Clock::time_point finished_at{};
};

class SmoothAiWorker {
public:
    SmoothAiWorker(Yolo11Detector& detector, const NormalizedRoi* roi)
        : detector_(detector),
          region_monitor_(roi == nullptr
                              ? nullptr
                              : std::unique_ptr<RegionMonitor>(new RegionMonitor(*roi))),
          worker_(&SmoothAiWorker::run, this)
    {
    }

    ~SmoothAiWorker()
    {
        request_stop();
        join();
    }

    SmoothAiWorker(const SmoothAiWorker&) = delete;
    SmoothAiWorker& operator=(const SmoothAiWorker&) = delete;

    bool submit(const cv::Mat& frame, std::uint64_t source_frame_id,
                Clock::time_point captured_at)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_requested_ || busy_ || pending_) {
                return false;
            }
        }

        cv::Mat frame_copy = frame.clone();
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_requested_ || busy_ || pending_) {
            return false;
        }
        pending_frame_ = std::move(frame_copy);
        pending_source_frame_id_ = source_frame_id;
        pending_captured_at = captured_at;
        pending_ = true;
        condition_.notify_one();
        return true;
    }

    bool copy_latest(SmoothDetectionSnapshot& snapshot) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (latest_.generation == 0U) {
            return false;
        }
        snapshot = latest_;
        return true;
    }

    bool failed(std::string& message) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!failed_) {
            return false;
        }
        message = error_message_;
        return true;
    }

    void request_stop()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_requested_ = true;
            pending_frame_.release();
            pending_ = false;
        }
        condition_.notify_all();
    }

    void join()
    {
        if (worker_.joinable()) {
            worker_.join();
        }
    }

private:
    void run()
    {
        for (;;) {
            cv::Mat frame;
            std::uint64_t source_frame_id = 0U;
            Clock::time_point captured_at{};
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [this] { return stop_requested_ || pending_; });
                if (stop_requested_ && !pending_) {
                    return;
                }
                frame = std::move(pending_frame_);
                source_frame_id = pending_source_frame_id_;
                captured_at = pending_captured_at;
                pending_ = false;
                busy_ = true;
            }

            try {
                const Clock::time_point ai_start = Clock::now();
                const DetectionResult result = detector_.detect_with_metrics(frame);
                const Clock::time_point finished_at = Clock::now();
                std::vector<Detection> tracked_detections = tracker_.update(result.detections);
                RegionSnapshot region;
                if (region_monitor_ != nullptr) {
                    region = region_monitor_->update(tracked_detections, captured_at,
                                                     frame.cols, frame.rows);
                }
                FrameMetrics metrics = result.metrics;
                metrics.object_count = static_cast<double>(tracked_detections.size());
                metrics.ai_latency_ms =
                    std::chrono::duration<double, std::milli>(finished_at - ai_start).count();

                std::lock_guard<std::mutex> lock(mutex_);
                latest_.generation += 1U;
                latest_.source_frame_id = source_frame_id;
                latest_.detections = std::move(tracked_detections);
                latest_.metrics = metrics;
                latest_.region = std::move(region);
                latest_.captured_at = captured_at;
                latest_.finished_at = finished_at;
                busy_ = false;
            } catch (const std::exception& error) {
                std::lock_guard<std::mutex> lock(mutex_);
                failed_ = true;
                error_message_ = error.what();
                busy_ = false;
                stop_requested_ = true;
                condition_.notify_all();
                return;
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex_);
                failed_ = true;
                error_message_ = "smooth-preview AI worker failed with an unknown exception";
                busy_ = false;
                stop_requested_ = true;
                condition_.notify_all();
                return;
            }
        }
    }

    Yolo11Detector& detector_;
    IouTracker tracker_;
    std::unique_ptr<RegionMonitor> region_monitor_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::thread worker_;
    cv::Mat pending_frame_;
    std::uint64_t pending_source_frame_id_ = 0U;
    Clock::time_point pending_captured_at{};
    SmoothDetectionSnapshot latest_;
    bool pending_ = false;
    bool busy_ = false;
    bool stop_requested_ = false;
    bool failed_ = false;
    std::string error_message_;
};

void verify_video_output(const VideoWriterInfo& info)
{
    cv::VideoCapture verification(info.actual_path);
    if (!verification.isOpened()) {
        throw std::runtime_error("written video cannot be reopened: " + info.actual_path);
    }
    cv::Mat frame;
    if (!verification.read(frame) || frame.empty()) {
        throw std::runtime_error("written video contains no readable frames: " + info.actual_path);
    }
    if (frame.size() != info.resolution) {
        throw std::runtime_error("written video resolution differs from requested resolution");
    }
}

int run_smooth_camera(const AppOptions& options, Yolo11Detector& detector,
                      const std::vector<std::string>& labels)
{
    if (!gui_available()) {
        throw std::runtime_error("smooth-preview requires a usable X11 display");
    }

    CameraSource camera(options.camera_path);
    camera.open();
    const CameraSourceInfo& source = camera.info();
    log_info("smooth camera device=" + camera.device() + " backend=" + source.backend +
             " resolution=" + std::to_string(source.width) + "x" +
             std::to_string(source.height) + " fps=" + std::to_string(source.fps));
    log_info("smooth camera pipeline=" + camera.pipeline());

    bool window_ready = false;
    try {
        configure_display_window(options.fullscreen ? DisplayComposer::kCanvasWidth : 400,
                                 options.fullscreen ? DisplayComposer::kCanvasHeight : 640,
                                 options.fullscreen);
        window_ready = true;
    } catch (const cv::Exception& error) {
        camera.release();
        throw std::runtime_error(std::string("smooth-preview window setup failed: ") + error.what());
    }

    SignalGuard signal_guard;
    SmoothAiWorker worker(detector, options.roi_enabled ? &options.roi : nullptr);
    DisplayComposer composer(labels);
    const NormalizedRoi* active_roi = options.roi_enabled ? &options.roi : nullptr;
    cv::Mat frame;
    cv::Mat preview_frame;
    std::vector<Detection> latest_detections;
    RegionSnapshot latest_region;
    FrameMetrics latest_metrics;
    Clock::time_point latest_result_finished{};
    std::uint64_t frame_id = 0U;
    std::uint64_t last_generation = 0U;
    std::size_t displayed_frames = 0U;
    std::size_t submitted_frames = 0U;
    std::size_t skipped_frames = 0U;
    std::size_t completed_inferences = 0U;
    std::size_t consecutive_read_failures = 0U;
    std::size_t result_age_samples = 0U;
    double result_age_sum_ms = 0.0;
    double result_age_max_ms = 0.0;
    std::string error_message;
    const auto wall_start = Clock::now();

    const auto cleanup = [&] {
        worker.request_stop();
        worker.join();
        camera.release();
        if (window_ready) {
            cv::destroyAllWindows();
        }
    };

    try {
        while (!g_stop_requested) {
            if (worker.failed(error_message)) {
                throw std::runtime_error("smooth-preview AI worker failed: " + error_message);
            }
            if (!camera.read(frame)) {
                ++consecutive_read_failures;
                if (displayed_frames == 0U) {
                    throw std::runtime_error("smooth-preview failed to read the first camera frame");
                }
                if (consecutive_read_failures >= 5U) {
                    throw std::runtime_error("smooth-preview camera read failed for 5 consecutive frames");
                }
                continue;
            }
            consecutive_read_failures = 0U;
            if (frame.cols != 1280 || frame.rows != 720 || frame.type() != CV_8UC3) {
                throw std::runtime_error("smooth-preview camera frame must be 1280x720 BGR CV_8UC3");
            }

            ++frame_id;
            const Clock::time_point captured_at = Clock::now();
            SmoothDetectionSnapshot snapshot;
            if (worker.copy_latest(snapshot) && snapshot.generation != last_generation) {
                completed_inferences +=
                    static_cast<std::size_t>(snapshot.generation - last_generation);
                last_generation = snapshot.generation;
                latest_detections = snapshot.detections;
                latest_region = snapshot.region;
                latest_metrics = snapshot.metrics;
                latest_result_finished = snapshot.finished_at;
            }
            if (worker.failed(error_message)) {
                throw std::runtime_error("smooth-preview AI worker failed: " + error_message);
            }

            if (worker.submit(frame, frame_id, captured_at)) {
                ++submitted_frames;
            } else {
                ++skipped_frames;
            }

            const Clock::time_point display_now = Clock::now();
            const double elapsed_s =
                std::chrono::duration<double>(display_now - wall_start).count();
            FrameMetrics display_metrics = latest_metrics;
            display_metrics.object_count = static_cast<double>(latest_detections.size());
            display_metrics.display_fps =
                elapsed_s > 0.0 ? static_cast<double>(displayed_frames + 1U) / elapsed_s : 0.0;
            display_metrics.detection_fps =
                elapsed_s > 0.0 ? static_cast<double>(completed_inferences) / elapsed_s : 0.0;
            if (latest_result_finished != Clock::time_point{}) {
                display_metrics.display_result_age_ms =
                    std::chrono::duration<double, std::milli>(display_now - latest_result_finished)
                        .count();
                result_age_sum_ms += display_metrics.display_result_age_ms;
                result_age_max_ms = std::max(result_age_max_ms, display_metrics.display_result_age_ms);
                ++result_age_samples;
            }

            const auto visualization_start = Clock::now();
            const cv::Mat& dashboard = composer.compose(
                frame, latest_detections, display_metrics, latest_region.occupancy,
                active_roi, latest_region.recent_events);
            const auto visualization_end = Clock::now();
            display_metrics.visualization_ms =
                std::chrono::duration<double, std::milli>(visualization_end - visualization_start)
                    .count();

            try {
                if (options.fullscreen) {
                    cv::imshow(kDisplayWindowTitle, dashboard);
                } else {
                    cv::resize(dashboard, preview_frame, cv::Size(400, 640), 0.0, 0.0,
                               cv::INTER_AREA);
                    cv::imshow(kDisplayWindowTitle, preview_frame);
                }
                const int key = cv::waitKey(1);
                if (key == 27 || key == 'q' || key == 'Q') {
                    g_stop_requested = 1;
                }
            } catch (const cv::Exception& error) {
                throw std::runtime_error(std::string("smooth-preview display failed: ") + error.what());
            }
            ++displayed_frames;

            if (options.max_frames > 0 &&
                displayed_frames >= static_cast<std::size_t>(options.max_frames)) {
                break;
            }
        }
    } catch (...) {
        cleanup();
        throw;
    }

    cleanup();
    if (displayed_frames == 0U) {
        throw std::runtime_error("smooth-preview produced no displayable frames");
    }
    if (g_stop_requested != 0) {
        log_warn("smooth-preview stopped by Ctrl+C or preview request");
    }

    const double elapsed_s = std::chrono::duration<double>(Clock::now() - wall_start).count();
    const double display_fps = elapsed_s > 0.0
                                   ? static_cast<double>(displayed_frames) / elapsed_s
                                   : 0.0;
    const double detection_fps = elapsed_s > 0.0
                                     ? static_cast<double>(completed_inferences) / elapsed_s
                                     : 0.0;
    const double average_result_age = result_age_samples > 0U
                                          ? result_age_sum_ms /
                                                static_cast<double>(result_age_samples)
                                          : 0.0;
    log_info("smooth camera summary displayed_frames=" + std::to_string(displayed_frames) +
             " submitted_frames=" + std::to_string(submitted_frames) +
             " skipped_frames=" + std::to_string(skipped_frames) +
             " completed_inferences=" + std::to_string(completed_inferences) +
             " display_fps=" + std::to_string(display_fps) +
             " detection_fps=" + std::to_string(detection_fps) +
             " display_result_age_avg=" + std::to_string(average_result_age) +
             " ms display_result_age_max=" + std::to_string(result_age_max_ms) + " ms");
    log_perf("smooth camera inference=" + std::to_string(latest_metrics.inference_ms) +
             " ms ai_latency=" + std::to_string(latest_metrics.ai_latency_ms) +
             " ms display_fps=" + std::to_string(display_fps) +
             " detection_fps=" + std::to_string(detection_fps));
    return 0;
}

int run_video(const AppOptions& options, Yolo11Detector& detector, const Visualizer& visualizer)
{
    VideoIO video;
    video.open_input(options.input_path);
    const VideoSourceInfo source = video.source_info();
    log_info("video input=" + options.input_path + " backend=" + source.backend +
             " resolution=" + std::to_string(source.width) + "x" + std::to_string(source.height) +
             " fps=" + std::to_string(source.fps) + " frames=" + std::to_string(source.frame_count));

    bool preview = options.show;
    if (preview && !gui_available()) {
        log_warn("Local GUI session unavailable; display disabled");
        preview = false;
    }

    if (preview) {
        try {
            configure_display_window(kPreviewWidth, kPreviewHeight, options.fullscreen);
        } catch (const cv::Exception& error) {
            log_warn(std::string("Local GUI session unavailable; display disabled: ") + error.what());
            preview = false;
        }
    }

    SignalGuard signal_guard;
    PerfMonitor monitor;
    cv::Mat frame;
    std::size_t processed = 0U;
    const auto wall_start = Clock::now();
    FrameMetrics previous_overlay_metrics;
    bool have_previous_overlay_metrics = false;
    while (!g_stop_requested && video.read(frame)) {
        if (frame.empty()) {
            continue;
        }
        if (!video.output_open()) {
            video.open_output(options.output_path, source.fps > 0.0 ? source.fps : 30.0,
                              frame.size(), options.force);
            const VideoWriterInfo& info = video.writer_info();
            log_info("video output requested=" + info.requested_path + " actual=" + info.actual_path +
                     " codec=" + info.codec + " fps=" + std::to_string(info.fps) +
                     " resolution=" + std::to_string(info.resolution.width) + "x" +
                     std::to_string(info.resolution.height));
        }

        const auto e2e_start = Clock::now();
        const DetectionResult result = detector.detect_with_metrics(frame);
        FrameMetrics metrics = result.metrics;
        metrics.object_count = static_cast<double>(result.detections.size());
        FrameMetrics overlay_metrics = metrics;
        if (have_previous_overlay_metrics) {
            overlay_metrics.visualization_ms = previous_overlay_metrics.visualization_ms;
            overlay_metrics.end_to_end_ms = previous_overlay_metrics.end_to_end_ms;
        }
        const auto visualization_start = Clock::now();
        const double before_draw_s =
            std::chrono::duration<double>(visualization_start - wall_start).count();
        metrics.fps = before_draw_s > 0.0 ? static_cast<double>(processed + 1U) / before_draw_s : 0.0;
        cv::Mat output = frame.clone();
        visualizer.draw(output, result.detections, overlay_metrics, OverlayMode::Video);
        const auto visualization_end = Clock::now();
        metrics.visualization_ms =
            std::chrono::duration<double, std::milli>(visualization_end - visualization_start).count();
        metrics.end_to_end_ms =
            std::chrono::duration<double, std::milli>(visualization_end - e2e_start).count();
        previous_overlay_metrics = metrics;
        have_previous_overlay_metrics = true;
        video.write(output);
        ++processed;
        const double elapsed = std::chrono::duration<double>(Clock::now() - wall_start).count();
        monitor.set_throughput_fps(elapsed > 0.0 ? static_cast<double>(processed) / elapsed : 0.0);
        monitor.add_frame(metrics);

        if (preview) {
            try {
                cv::imshow(kDisplayWindowTitle, output);
                const int key = cv::waitKey(1);
                if (key == 27 || key == 'q' || key == 'Q') {
                    g_stop_requested = 1;
                }
            } catch (const cv::Exception& error) {
                log_warn(std::string("Local GUI session unavailable; display disabled: ") + error.what());
                preview = false;
                cv::destroyAllWindows();
            }
        }
        if (options.max_frames > 0 && processed >= static_cast<std::size_t>(options.max_frames)) {
            break;
        }
    }

    if (preview) {
        cv::destroyAllWindows();
    }
    if (processed == 0U) {
        throw std::runtime_error("video input produced no readable frames: " + options.input_path);
    }
    if (g_stop_requested != 0) {
        log_warn("video processing stopped by Ctrl+C or preview request");
    }

    video.close_output();
    const double elapsed = std::chrono::duration<double>(Clock::now() - wall_start).count();
    const double actual_fps = elapsed > 0.0 ? static_cast<double>(processed) / elapsed : 0.0;
    monitor.set_throughput_fps(actual_fps);
    verify_video_output(video.writer_info());
    const FrameMetrics average = monitor.final_average();
    log_info("video summary frames=" + std::to_string(processed) + " actual_fps=" +
             std::to_string(actual_fps) + " output=" + video.writer_info().actual_path);
    log_perf("average preprocess=" + std::to_string(average.preprocess_ms) +
             " ms inference=" + std::to_string(average.inference_ms) +
             " ms postprocess=" + std::to_string(average.postprocess_ms) +
             " ms visualization=" + std::to_string(average.visualization_ms) +
             " ms e2e=" + std::to_string(average.end_to_end_ms) +
             " ms throughput=" + std::to_string(monitor.throughput_fps()) + " fps");
    return 0;
}

int run_camera(const AppOptions& options, Yolo11Detector& detector, const Visualizer& visualizer)
{
    bool preview = options.show;
    if (preview && !gui_available()) {
        log_warn("Local GUI session unavailable; camera preview disabled");
        preview = false;
    }
    if (!preview && options.output_path.empty()) {
        throw std::runtime_error("camera mode requires a usable --show session or --output");
    }

    CameraSource camera(options.camera_path);
    camera.open();
    const CameraSourceInfo& source = camera.info();
    log_info("camera device=" + camera.device() + " backend=" + source.backend +
             " resolution=" + std::to_string(source.width) + "x" +
             std::to_string(source.height) + " fps=" + std::to_string(source.fps));
    log_info("camera pipeline=" + camera.pipeline());

    if (preview) {
        try {
            configure_display_window(kCameraPreviewWidth, kCameraPreviewHeight, options.fullscreen);
        } catch (const cv::Exception& error) {
            log_warn(std::string("Local GUI session unavailable; camera preview disabled: ") +
                     error.what());
            preview = false;
            if (options.output_path.empty()) {
                camera.release();
                throw std::runtime_error("camera mode requires a usable --show session or --output");
            }
        }
    }

    VideoIO output;
    const bool record_output = !options.output_path.empty();
    SignalGuard signal_guard;
    PerfMonitor monitor;
    cv::Mat frame;
    std::size_t processed = 0U;
    int consecutive_read_failures = 0;
    const auto wall_start = Clock::now();
    CameraProfileTotals profile;
    bool profile_started = false;
    Clock::time_point profile_wall_start{};
    Clock::time_point profile_wall_end{};
    FrameMetrics previous_overlay_metrics;
    bool have_previous_overlay_metrics = false;

    while (!g_stop_requested) {
        const auto loop_start = Clock::now();
        const auto camera_read_start = Clock::now();
        if (!camera.read(frame)) {
            ++consecutive_read_failures;
            if (processed == 0U) {
                throw std::runtime_error("camera failed to read the first frame from " +
                                         camera.device());
            }
            if (consecutive_read_failures >= 5) {
                throw std::runtime_error("camera read failed for 5 consecutive frames after " +
                                         std::to_string(processed) + " processed frames");
            }
            continue;
        }
        const double camera_read_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - camera_read_start).count();
        consecutive_read_failures = 0;
        if (frame.cols != 1280 || frame.rows != 720 || frame.type() != CV_8UC3) {
            throw std::runtime_error("camera frame must be 1280x720 BGR CV_8UC3; got " +
                                     std::to_string(frame.cols) + "x" +
                                     std::to_string(frame.rows) + " type=" +
                                     std::to_string(frame.type()));
        }

        if (record_output && !output.output_open()) {
            output.open_output(options.output_path, source.fps > 0.0 ? source.fps : 30.0,
                               frame.size(), options.force);
            const VideoWriterInfo& info = output.writer_info();
            log_info("camera output requested=" + info.requested_path +
                     " actual=" + info.actual_path + " codec=" + info.codec +
                     " fps=" + std::to_string(info.fps) + " resolution=" +
                     std::to_string(info.resolution.width) + "x" +
                     std::to_string(info.resolution.height));
        }

        const auto e2e_start = Clock::now();
        const DetectionResult result = detector.detect_with_metrics(frame);
        FrameMetrics metrics = result.metrics;
        metrics.object_count = static_cast<double>(result.detections.size());
        FrameMetrics overlay_metrics = metrics;
        if (have_previous_overlay_metrics) {
            overlay_metrics.visualization_ms = previous_overlay_metrics.visualization_ms;
            overlay_metrics.end_to_end_ms = previous_overlay_metrics.end_to_end_ms;
        }
        const auto visualization_start = Clock::now();
        const double before_draw_s =
            std::chrono::duration<double>(visualization_start - wall_start).count();
        metrics.fps = before_draw_s > 0.0 ? static_cast<double>(processed + 1U) / before_draw_s : 0.0;
        cv::Mat output_frame = frame.clone();
        visualizer.draw(output_frame, result.detections, overlay_metrics, OverlayMode::Video);
        const auto visualization_end = Clock::now();
        metrics.visualization_ms =
            std::chrono::duration<double, std::milli>(visualization_end - visualization_start).count();
        metrics.end_to_end_ms =
            std::chrono::duration<double, std::milli>(visualization_end - e2e_start).count();
        previous_overlay_metrics = metrics;
        have_previous_overlay_metrics = true;
        double writer_ms = 0.0;
        if (record_output) {
            const auto writer_start = Clock::now();
            output.write(output_frame);
            writer_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - writer_start).count();
        }
        ++processed;
        const double elapsed = std::chrono::duration<double>(Clock::now() - wall_start).count();
        monitor.set_throughput_fps(elapsed > 0.0 ? static_cast<double>(processed) / elapsed : 0.0);
        monitor.add_frame(metrics);

        double display_ms = 0.0;
        if (preview) {
            try {
                cv::Mat preview_frame;
                if (!options.fullscreen) {
                    cv::resize(output_frame, preview_frame,
                               cv::Size(kCameraPreviewWidth, kCameraPreviewHeight), 0.0, 0.0,
                               cv::INTER_AREA);
                } else {
                    preview_frame = fit_frame_to_display(output_frame, kFullscreenWidth,
                                                         kFullscreenHeight);
                }
                const auto display_start = Clock::now();
                cv::imshow(kDisplayWindowTitle, preview_frame);
                const int key = cv::waitKey(1);
                display_ms =
                    std::chrono::duration<double, std::milli>(Clock::now() - display_start).count();
                if (key == 27 || key == 'q' || key == 'Q') {
                    g_stop_requested = 1;
                }
            } catch (const cv::Exception& error) {
                log_warn(std::string("Local GUI session unavailable; camera preview disabled: ") +
                         error.what());
                preview = false;
                cv::destroyAllWindows();
                if (!record_output) {
                    throw std::runtime_error("camera preview failed and no --output was provided");
                }
            }
        }

        const auto loop_end = Clock::now();
        const double loop_ms =
            std::chrono::duration<double, std::milli>(loop_end - loop_start).count();
        if (processed > kCameraWarmupFrames) {
            if (!profile_started) {
                profile_started = true;
                profile_wall_start = loop_start;
            }
            const double other_ms = std::max(
                0.0, loop_ms - camera_read_ms - metrics.preprocess_ms - metrics.inference_ms -
                          metrics.postprocess_ms - metrics.visualization_ms - display_ms - writer_ms);
            profile.add(camera_read_ms, metrics.preprocess_ms, metrics.inference_ms,
                        metrics.postprocess_ms, metrics.visualization_ms, display_ms, writer_ms,
                        other_ms, loop_ms, metrics.end_to_end_ms);
            profile_wall_end = loop_end;
        }

        if (options.max_frames > 0 && processed >= static_cast<std::size_t>(options.max_frames)) {
            break;
        }
    }

    if (preview) {
        cv::destroyAllWindows();
    }
    camera.release();
    if (processed == 0U) {
        throw std::runtime_error("camera produced no readable frames: " + camera.device());
    }
    if (g_stop_requested != 0) {
        log_warn("camera processing stopped by Ctrl+C or preview request");
    }

    if (record_output) {
        output.close_output();
        verify_video_output(output.writer_info());
    }
    const double elapsed = std::chrono::duration<double>(Clock::now() - wall_start).count();
    const double all_frames_fps = elapsed > 0.0 ? static_cast<double>(processed) / elapsed : 0.0;
    double measured_actual_fps = 0.0;
    if (profile.frames > 0U) {
        const double measured_elapsed =
            std::chrono::duration<double>(profile_wall_end - profile_wall_start).count();
        measured_actual_fps = measured_elapsed > 0.0
                                  ? static_cast<double>(profile.frames) / measured_elapsed
                                  : 0.0;
    }
    const double actual_fps = profile.frames > 0U ? measured_actual_fps : all_frames_fps;
    monitor.set_throughput_fps(actual_fps);
    log_info("camera summary frames=" + std::to_string(processed) +
             " warmup=" + std::to_string(std::min(processed, kCameraWarmupFrames)) +
             " measured=" + std::to_string(profile.frames) +
             " actual_fps=" + std::to_string(actual_fps) +
             (record_output ? " output=" + output.writer_info().actual_path :
                              " output=none"));
    if (profile.frames > 0U) {
        const double count = static_cast<double>(profile.frames);
        log_perf("camera profile warmup_frames=" + std::to_string(std::min(processed, kCameraWarmupFrames)) +
                 " measured_frames=" + std::to_string(profile.frames) +
                 " camera_read=" + std::to_string(profile.camera_read_ms / count) +
                 " ms preprocess=" + std::to_string(profile.preprocess_ms / count) +
                 " ms inference=" + std::to_string(profile.inference_ms / count) +
                 " ms postprocess=" + std::to_string(profile.postprocess_ms / count) +
                 " ms visualization=" + std::to_string(profile.visualization_ms / count) +
                 " ms display=" + std::to_string(profile.display_ms / count) +
                 " ms writer=" + std::to_string(profile.writer_ms / count) +
                 " ms other=" + std::to_string(profile.other_ms / count) +
                 " ms full_loop=" + std::to_string(profile.loop_ms / count) +
                 " ms legacy_e2e=" + std::to_string(profile.legacy_e2e_ms / count) +
                 " ms actual_fps=" + std::to_string(measured_actual_fps));
    } else {
        log_warn("camera profiling has no measured frames; increase --max-frames beyond warmup");
    }
    return 0;
}

#endif

}  // namespace

int run_application(const AppOptions& options)
{
    validate_options(options);
    const std::vector<std::string> labels = LabelLoader::load(options.labels_path);
    RknnModel model(options.model_path);
    Yolo11Detector detector(model, options.conf_threshold, options.nms_threshold);
    const Visualizer visualizer(labels);

    if (!options.camera_path.empty()) {
#if EDGEVISION_WITH_VIDEO
        if (options.smooth_preview) {
            return run_smooth_camera(options, detector, labels);
        }
        return run_camera(options, detector, visualizer);
#else
        throw std::runtime_error("camera mode requires video support in the build");
#endif
    }

    const cv::Mat image = cv::imread(options.input_path, cv::IMREAD_COLOR);
    if (!image.empty()) {
        return run_image(options, labels, detector, visualizer, image);
    }

#if EDGEVISION_WITH_VIDEO
    return run_video(options, detector, visualizer);
#else
    throw std::runtime_error("input is not a readable image and video support is not built: " +
                             options.input_path);
#endif
}

}  // namespace edgevision
