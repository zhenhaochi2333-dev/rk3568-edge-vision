#include "edgevision/cli_parser.hpp"

#include <cerrno>
#include <cstdlib>
#include <limits>
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

}  // namespace

CliParseResult CliParser::parse(int argc, char** argv)
{
    CliParseResult result;
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
    return result;
}

const char* CliParser::usage()
{
    return "Usage: edge_vision --model MODEL --labels LABELS "
           "(--input INPUT --output OUTPUT | --camera DEVICE [--output OUTPUT]) "
           "[--conf FLOAT] [--nms FLOAT] [--max-frames N] [--show] [--force] [--help]";
}

}  // namespace edgevision
