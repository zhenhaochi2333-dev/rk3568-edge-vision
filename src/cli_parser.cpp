#include "edgevision/cli_parser.hpp"

#include <cerrno>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace edgevision {

namespace {

std::string require_value(int& index, int argc, char** argv, const char* option)
{
    if (index + 1 >= argc || argv[index + 1] == nullptr || argv[index + 1][0] == '\0') {
        throw std::runtime_error(std::string("missing value for ") + option);
    }
    ++index;
    return argv[index];
}

float parse_float(const std::string& text, const char* option)
{
    char* end = nullptr;
    errno = 0;
    const float value = std::strtof(text.c_str(), &end);
    if (errno != 0 || end == text.c_str() || *end != '\0') {
        throw std::runtime_error(std::string("invalid value for ") + option + ": " + text);
    }
    return value;
}

int parse_int(const std::string& text, const char* option)
{
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0' ||
        value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("invalid value for ") + option + ": " + text);
    }
    return static_cast<int>(value);
}

NormalizedRoi parse_roi(const std::string& text)
{
    std::stringstream stream(text);
    NormalizedRoi roi;
    char first = '\0';
    char second = '\0';
    char third = '\0';
    if (!(stream >> roi.x >> first >> roi.y >> second >> roi.width >> third >> roi.height) ||
        first != ',' || second != ',' || third != ',') {
        throw std::runtime_error("invalid value for --roi: " + text);
    }
    stream >> std::ws;
    if (!stream.eof()) {
        throw std::runtime_error("invalid value for --roi: " + text);
    }
    if (roi.x < 0.0F || roi.y < 0.0F || roi.width <= 0.0F || roi.height <= 0.0F ||
        roi.x + roi.width > 1.0F || roi.y + roi.height > 1.0F) {
        throw std::runtime_error("--roi must stay within normalized [0,1] coordinates");
    }
    return roi;
}

}  // namespace

CliParseResult CliParser::parse(int argc, char** argv)
{
    CliParseResult result;
    bool roi_specified = false;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index] == nullptr ? std::string() : argv[index];
        if (option == "--help" || option == "-h") {
            result.show_help = true;
        } else if (option == "--model") {
            result.options.model_path = require_value(index, argc, argv, "--model");
        } else if (option == "--labels") {
            result.options.labels_path = require_value(index, argc, argv, "--labels");
        } else if (option == "--input") {
            result.options.input_path = require_value(index, argc, argv, "--input");
        } else if (option == "--camera") {
            result.options.camera_path = require_value(index, argc, argv, "--camera");
        } else if (option == "--output") {
            result.options.output_path = require_value(index, argc, argv, "--output");
        } else if (option == "--conf") {
            result.options.conf_threshold = parse_float(require_value(index, argc, argv, "--conf"), "--conf");
        } else if (option == "--nms") {
            result.options.nms_threshold = parse_float(require_value(index, argc, argv, "--nms"), "--nms");
        } else if (option == "--max-frames") {
            result.options.max_frames = parse_int(require_value(index, argc, argv, "--max-frames"), "--max-frames");
        } else if (option == "--show") {
            result.options.show = true;
        } else if (option == "--fullscreen") {
            result.options.fullscreen = true;
        } else if (option == "--smooth-preview") {
            result.options.smooth_preview = true;
        } else if (option == "--roi") {
            result.options.roi = parse_roi(require_value(index, argc, argv, "--roi"));
            result.options.roi_enabled = true;
            roi_specified = true;
        } else if (option == "--show-roi") {
            result.options.show_roi = true;
        } else if (option == "--force") {
            result.options.force = true;
        } else {
            throw std::runtime_error("unknown option: " + option);
        }
    }

    if (result.show_help) {
        return result;
    }
    if (result.options.model_path.empty() || result.options.labels_path.empty()) {
        throw std::runtime_error("--model and --labels are required");
    }
    if (!result.options.input_path.empty() && !result.options.camera_path.empty()) {
        throw std::runtime_error("--input and --camera are mutually exclusive");
    }
    if (result.options.input_path.empty() && result.options.camera_path.empty()) {
        throw std::runtime_error("provide exactly one of --input or --camera");
    }
    if (!result.options.camera_path.empty() && result.options.output_path.empty() &&
        !result.options.show) {
        throw std::runtime_error("camera mode requires --show or --output");
    }
    if (!result.options.input_path.empty() && result.options.output_path.empty()) {
        throw std::runtime_error("--output is required with --input");
    }
    if (!(result.options.conf_threshold >= 0.0F && result.options.conf_threshold <= 1.0F)) {
        throw std::runtime_error("--conf must be in [0,1]");
    }
    if (!(result.options.nms_threshold >= 0.0F && result.options.nms_threshold <= 1.0F)) {
        throw std::runtime_error("--nms must be in [0,1]");
    }
    if (result.options.max_frames < 0) {
        throw std::runtime_error("--max-frames must be non-negative");
    }
    if (result.options.fullscreen && !result.options.show) {
        throw std::runtime_error("--fullscreen requires --show");
    }
    if (result.options.smooth_preview && result.options.camera_path.empty()) {
        throw std::runtime_error("--smooth-preview requires --camera");
    }
    if (result.options.smooth_preview && !result.options.show) {
        throw std::runtime_error("--smooth-preview requires --show");
    }
    if (result.options.smooth_preview && !result.options.output_path.empty()) {
        throw std::runtime_error("--smooth-preview does not support --output");
    }
    if (roi_specified && !result.options.smooth_preview) {
        throw std::runtime_error("--roi requires --smooth-preview");
    }
    if (result.options.show_roi && !result.options.smooth_preview) {
        throw std::runtime_error("--show-roi requires --smooth-preview");
    }
    return result;
}

const char* CliParser::usage()
{
    return "Usage: edge_vision --model MODEL --labels LABELS "
           "(--input INPUT --output OUTPUT | --camera DEVICE [--output OUTPUT]) "
           "[--conf FLOAT] [--nms FLOAT] [--max-frames N] [--show] [--fullscreen] "
           "[--smooth-preview] [--roi X,Y,W,H] [--show-roi] [--force] [--help]";
}

}  // namespace edgevision
