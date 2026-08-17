#include "edgevision/display_composer.hpp"

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::string> load_labels(const std::string& path)
{
    std::ifstream file(path.c_str());
    if (!file) {
        throw std::runtime_error("cannot open labels: " + path);
    }
    std::vector<std::string> labels;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            labels.push_back(line);
        }
    }
    return labels;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 4) {
        std::cerr << "Usage: make_final_ui_preview BUS_IMAGE LABELS OUTPUT_PNG\n";
        return 2;
    }

    const cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Cannot read input image: " << argv[1] << "\n";
        return 1;
    }

    const std::vector<std::string> labels = load_labels(argv[2]);
    edgevision::DisplayComposer composer(labels);
    const std::vector<edgevision::Detection> detections{
        edgevision::Detection{0, 0.836F, cv::Rect2f(211.0F, 240.0F, 72.0F, 278.0F), 1},
        edgevision::Detection{0, 0.798F, cv::Rect2f(475.0F, 231.0F, 85.0F, 289.0F), 2},
        edgevision::Detection{0, 0.796F, cv::Rect2f(114.0F, 235.0F, 93.0F, 308.0F), 3},
        edgevision::Detection{5, 0.783F, cv::Rect2f(90.0F, 133.0F, 463.0F, 328.0F), 4},
    };
    const auto now = std::chrono::steady_clock::now();
    const std::vector<edgevision::RegionEvent> events{
        edgevision::RegionEvent{edgevision::RegionEventType::Enter, 1, 0, now},
        edgevision::RegionEvent{edgevision::RegionEventType::Exit, 2, 0, now}};
    const cv::Mat& preview = composer.compose(image, detections, events);
    if (!cv::imwrite(argv[3], preview)) {
        std::cerr << "Cannot write output image: " << argv[3] << "\n";
        return 1;
    }
    std::cout << "Wrote final UI preview: " << argv[3] << " ("
              << preview.cols << "x" << preview.rows << ")\n";
    return 0;
}
