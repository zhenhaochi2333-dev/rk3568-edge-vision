#include "edgevision/rknn_model.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string model_path;
    std::size_t warmup = 30U;
    std::size_t runs = 300U;
};

void print_usage(const char* program)
{
    std::cout << "Usage: " << program
              << " --model <path> [--warmup <count>] [--runs <count>]\n";
}

std::size_t parse_count(const std::string& value, const char* option)
{
    std::size_t parsed = 0U;
    try {
        parsed = static_cast<std::size_t>(std::stoul(value));
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid value for ") + option);
    }
    if (parsed == 0U) {
        throw std::runtime_error(std::string(option) + " must be greater than zero");
    }
    return parsed;
}

Options parse_options(int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (argument == "--model" || argument == "--warmup" || argument == "--runs") {
            if (index + 1 >= argc) {
                throw std::runtime_error("missing value for " + argument);
            }
            const std::string value = argv[++index];
            if (argument == "--model") {
                options.model_path = value;
            } else if (argument == "--warmup") {
                options.warmup = parse_count(value, "--warmup");
            } else {
                options.runs = parse_count(value, "--runs");
            }
            continue;
        }
        throw std::runtime_error("unknown argument: " + argument);
    }
    if (options.model_path.empty()) {
        throw std::runtime_error("--model is required");
    }
    return options;
}

double percentile95(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const std::size_t index = (values.size() * 95U + 99U) / 100U - 1U;
    return values[index];
}

void print_stats(const std::vector<double>& samples)
{
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    double total = 0.0;
    for (const double sample : samples) {
        total += sample;
    }
    const std::size_t median_index = sorted.size() / 2U;
    const double median = sorted.size() % 2U == 0U
                              ? (sorted[median_index - 1U] + sorted[median_index]) / 2.0
                              : sorted[median_index];
    std::cout << std::fixed << std::setprecision(3)
              << "runs=" << samples.size() << '\n'
              << "average_ms=" << total / static_cast<double>(samples.size()) << '\n'
              << "median_ms=" << median << '\n'
              << "p95_ms=" << percentile95(sorted) << '\n'
              << "min_ms=" << sorted.front() << '\n'
              << "max_ms=" << sorted.back() << '\n';
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Options options = parse_options(argc, argv);
        edgevision::RknnModel model(options.model_path);
        const std::size_t input_size = static_cast<std::size_t>(model.model_width()) *
                                       static_cast<std::size_t>(model.model_height()) *
                                       static_cast<std::size_t>(model.model_channels());
        std::vector<std::uint8_t> input(input_size, 114U);

        std::cout << "model=" << options.model_path << '\n'
                  << "input=" << model.model_width() << 'x' << model.model_height() << 'x'
                  << model.model_channels() << '\n'
                  << "warmup=" << options.warmup << '\n';
        for (std::size_t index = 0U; index < options.warmup; ++index) {
            double inference_ms = 0.0;
            auto outputs = model.run(input.data(), input.size(), &inference_ms);
            (void)outputs;
        }

        std::vector<double> samples;
        samples.reserve(options.runs);
        for (std::size_t index = 0U; index < options.runs; ++index) {
            double inference_ms = 0.0;
            auto outputs = model.run(input.data(), input.size(), &inference_ms);
            (void)outputs;
            samples.push_back(inference_ms);
        }
        print_stats(samples);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_error=" << error.what() << '\n';
        return 1;
    }
}
