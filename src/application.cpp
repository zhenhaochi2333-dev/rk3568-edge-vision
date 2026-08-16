#include "edgevision/application.hpp"

#include "edgevision/label_loader.hpp"
#include "edgevision/logger.hpp"
#include "edgevision/perf_monitor.hpp"
#include "edgevision/rknn_model.hpp"
#include "edgevision/visualizer.hpp"
#include "edgevision/yolov5_detector.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <limits.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace edgevision {

namespace {

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
    if (!is_regular_file(options.input_path)) {
        throw std::runtime_error("missing input image: " + options.input_path);
    }
    if (real_path_if_possible(options.input_path) == real_path_if_possible(options.output_path)) {
        throw std::runtime_error("input and output paths must be different");
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

}  // namespace

int run_application(const AppOptions& options)
{
    validate_options(options);
    const std::vector<std::string> labels = LabelLoader::load(options.labels_path);
    if (options.show) {
        log_warn("--show is accepted for CLI compatibility; preview presentation is deferred to Phase 6");
    }

    const cv::Mat image = cv::imread(options.input_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("input is not a readable image: " + options.input_path);
    }

    RknnModel model(options.model_path);
    Yolov5Detector detector(model, options.conf_threshold, options.nms_threshold);
    const auto end_to_end_start = std::chrono::steady_clock::now();
    const DetectionResult result = detector.detect_with_metrics(image);
    FrameMetrics metrics = result.metrics;
    const auto visualization_start = std::chrono::steady_clock::now();
    const Visualizer visualizer(labels);
    cv::Mat timing_image = image.clone();
    visualizer.draw(timing_image, result.detections, metrics, OverlayMode::Image);
    const auto visualization_end = std::chrono::steady_clock::now();
    metrics.visualization_ms =
        std::chrono::duration<double, std::milli>(visualization_end - visualization_start).count();
    metrics.end_to_end_ms =
        std::chrono::duration<double, std::milli>(visualization_end - end_to_end_start).count();
    metrics.object_count = static_cast<double>(result.detections.size());

    cv::Mat output = image.clone();
    visualizer.draw(output, result.detections, metrics, OverlayMode::Image);

    PerfMonitor monitor;
    monitor.add_frame(metrics);

    for (const Detection& detection : result.detections) {
        const std::string& label = LabelLoader::name(labels, detection.class_id);
        std::ostringstream line;
        line << label << " @ (" << static_cast<int>(detection.box.x) << ' '
             << static_cast<int>(detection.box.y) << ' '
             << static_cast<int>(detection.box.x + detection.box.width) << ' '
             << static_cast<int>(detection.box.y + detection.box.height) << ") "
             << std::fixed << std::setprecision(3) << detection.confidence;
        log_info(line.str());
    }
    log_info("detections=" + std::to_string(result.detections.size()));
    log_perf("preprocess=" + std::to_string(metrics.preprocess_ms) + " ms inference=" +
             std::to_string(metrics.inference_ms) + " ms postprocess=" +
             std::to_string(metrics.postprocess_ms) + " ms visualization=" +
             std::to_string(metrics.visualization_ms) + " ms e2e=" +
             std::to_string(metrics.end_to_end_ms) + " ms");
    if (!cv::imwrite(options.output_path, output)) {
        throw std::runtime_error("failed to write output image: " + options.output_path);
    }
    log_info("wrote output image: " + options.output_path);
    return 0;
}

}  // namespace edgevision
