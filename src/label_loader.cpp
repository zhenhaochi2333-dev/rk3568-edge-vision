#include "edgevision/label_loader.hpp"

#include <fstream>
#include <stdexcept>

namespace edgevision {

std::vector<std::string> LabelLoader::load(const std::string& path)
{
    std::ifstream input(path.c_str());
    if (!input) {
        throw std::runtime_error("cannot open labels: " + path);
    }

    std::vector<std::string> labels;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            labels.push_back(line);
        }
    }
    if (labels.size() < 80U) {
        throw std::runtime_error("labels file contains fewer than 80 COCO classes: " + path);
    }
    return labels;
}

const std::string& LabelLoader::name(const std::vector<std::string>& labels, int class_id)
{
    if (class_id < 0 || static_cast<std::size_t>(class_id) >= labels.size()) {
        throw std::runtime_error("detector returned an invalid class id: " + std::to_string(class_id));
    }
    return labels[static_cast<std::size_t>(class_id)];
}

}  // namespace edgevision
