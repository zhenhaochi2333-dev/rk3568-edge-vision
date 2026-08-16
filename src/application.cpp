#include "edgevision/application.hpp"

#include "edgevision/label_loader.hpp"
#include "edgevision/logger.hpp"
#include "edgevision/perf_monitor.hpp"
#include "edgevision/rknn_model.hpp"
#include "edgevision/visualizer.hpp"
#include "edgevision/yolov5_detector.hpp"
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

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
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

void configure_display_window(int width, int height)
{
    cv::namedWindow(kDisplayWindowTitle, cv::WINDOW_NORMAL);
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
              Yolov5Detector& detector, const Visualizer& visualizer, const cv::Mat& image)
{
    const auto e2e_start = Clock::now();
    const DetectionResult result = detector.detect_with_metrics(image);
    FrameMetrics metrics = result.metrics;
    metrics.object_count = static_cast<double>(result.detections.size());

    const auto visualization_start = Clock::now();
    cv::Mat timing_image = image.clone();
    visualizer.draw(timing_image, result.detections, metrics, OverlayMode::Image);
    const auto visualization_end = Clock::now();
    metrics.visualization_ms =
        std::chrono::duration<double, std::milli>(visualization_end - visualization_start).count();
    metrics.end_to_end_ms =
        std::chrono::duration<double, std::milli>(visualization_end - e2e_start).count();

    cv::Mat output = image.clone();
    visualizer.draw(output, result.detections, metrics, OverlayMode::Image);
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
                configure_display_window(kPreviewWidth, kPreviewHeight);
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

int run_video(const AppOptions& options, Yolov5Detector& detector, const Visualizer& visualizer)
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
            configure_display_window(kPreviewWidth, kPreviewHeight);
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
        const auto visualization_start = Clock::now();
        const double before_draw_s =
            std::chrono::duration<double>(visualization_start - wall_start).count();
        metrics.fps = before_draw_s > 0.0 ? static_cast<double>(processed + 1U) / before_draw_s : 0.0;
        cv::Mat timing_image = frame.clone();
        visualizer.draw(timing_image, result.detections, metrics, OverlayMode::Video);
        const auto visualization_end = Clock::now();
        metrics.visualization_ms =
            std::chrono::duration<double, std::milli>(visualization_end - visualization_start).count();
        metrics.end_to_end_ms =
            std::chrono::duration<double, std::milli>(visualization_end - e2e_start).count();

        cv::Mat output = frame.clone();
        visualizer.draw(output, result.detections, metrics, OverlayMode::Video);
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

int run_camera(const AppOptions& options, Yolov5Detector& detector, const Visualizer& visualizer)
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
            configure_display_window(kCameraPreviewWidth, kCameraPreviewHeight);
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

    while (!g_stop_requested) {
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
        const auto visualization_start = Clock::now();
        const double before_draw_s =
            std::chrono::duration<double>(visualization_start - wall_start).count();
        metrics.fps = before_draw_s > 0.0 ? static_cast<double>(processed + 1U) / before_draw_s : 0.0;
        cv::Mat timing_image = frame.clone();
        visualizer.draw(timing_image, result.detections, metrics, OverlayMode::Video);
        const auto visualization_end = Clock::now();
        metrics.visualization_ms =
            std::chrono::duration<double, std::milli>(visualization_end - visualization_start).count();
        metrics.end_to_end_ms =
            std::chrono::duration<double, std::milli>(visualization_end - e2e_start).count();

        cv::Mat output_frame = frame.clone();
        visualizer.draw(output_frame, result.detections, metrics, OverlayMode::Video);
        if (record_output) {
            output.write(output_frame);
        }
        ++processed;
        const double elapsed = std::chrono::duration<double>(Clock::now() - wall_start).count();
        monitor.set_throughput_fps(elapsed > 0.0 ? static_cast<double>(processed) / elapsed : 0.0);
        monitor.add_frame(metrics);

        if (preview) {
            try {
                cv::Mat preview_frame;
                cv::resize(output_frame, preview_frame,
                           cv::Size(kCameraPreviewWidth, kCameraPreviewHeight), 0.0, 0.0,
                           cv::INTER_AREA);
                cv::imshow(kDisplayWindowTitle, preview_frame);
                const int key = cv::waitKey(1);
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
    const double actual_fps = elapsed > 0.0 ? static_cast<double>(processed) / elapsed : 0.0;
    monitor.set_throughput_fps(actual_fps);
    const FrameMetrics average = monitor.final_average();
    log_info("camera summary frames=" + std::to_string(processed) +
             " actual_fps=" + std::to_string(actual_fps) +
             (record_output ? " output=" + output.writer_info().actual_path :
                              " output=none"));
    log_perf("camera average preprocess=" + std::to_string(average.preprocess_ms) +
             " ms inference=" + std::to_string(average.inference_ms) +
             " ms postprocess=" + std::to_string(average.postprocess_ms) +
             " ms visualization=" + std::to_string(average.visualization_ms) +
             " ms e2e=" + std::to_string(average.end_to_end_ms) +
             " ms throughput=" + std::to_string(monitor.throughput_fps()) + " fps");
    return 0;
}

#endif

}  // namespace

int run_application(const AppOptions& options)
{
    validate_options(options);
    const std::vector<std::string> labels = LabelLoader::load(options.labels_path);
    RknnModel model(options.model_path);
    Yolov5Detector detector(model, options.conf_threshold, options.nms_threshold);
    const Visualizer visualizer(labels);

    if (!options.camera_path.empty()) {
#if EDGEVISION_WITH_VIDEO
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
