#include "edgevision/application.hpp"

#include "edgevision/label_loader.hpp"
#include "edgevision/logger.hpp"
#include "edgevision/rknn_model.hpp"
#include "edgevision/yolov5_detector.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <limits.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdlib>
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

void save_minimal_debug_image(const cv::Mat& source, const std::vector<Detection>& detections,
                              const std::vector<std::string>& labels, const std::string& path)
{
    cv::Mat output = source.clone();
    for (const Detection& detection : detections) {
        const int left = std::max(0, static_cast<int>(detection.box.x));
        const int top = std::max(0, static_cast<int>(detection.box.y));
        if (left >= output.cols || top >= output.rows) {
            continue;
        }
        const int width = std::min(std::max(1, static_cast<int>(detection.box.width)), output.cols - left);
        const int height = std::min(std::max(1, static_cast<int>(detection.box.height)), output.rows - top);
        cv::rectangle(output, cv::Rect(left, top, width, height), cv::Scalar(0, 255, 0), 2);
        std::ostringstream text;
        text << LabelLoader::name(labels, detection.class_id) << ' '
             << std::fixed << std::setprecision(1) << detection.confidence * 100.0F << '%';
        cv::putText(output, text.str(), cv::Point(left, std::max(15, top - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }
    if (!cv::imwrite(path, output)) {
        throw std::runtime_error("failed to write output image: " + path);
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
    const std::vector<Detection> detections = detector.detect(image);

    for (const Detection& detection : detections) {
        const std::string& label = LabelLoader::name(labels, detection.class_id);
        std::ostringstream line;
        line << label << " @ (" << static_cast<int>(detection.box.x) << ' '
             << static_cast<int>(detection.box.y) << ' '
             << static_cast<int>(detection.box.x + detection.box.width) << ' '
             << static_cast<int>(detection.box.y + detection.box.height) << ") "
             << std::fixed << std::setprecision(3) << detection.confidence;
        log_info(line.str());
    }
    log_info("detections=" + std::to_string(detections.size()));
    save_minimal_debug_image(image, detections, labels, options.output_path);
    log_info("wrote output image: " + options.output_path);
    return 0;
}

}  // namespace edgevision
