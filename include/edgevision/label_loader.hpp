#pragma once

#include <string>
#include <vector>

namespace edgevision {

class LabelLoader {
public:
    static std::vector<std::string> load(const std::string& path);
    static const std::string& name(const std::vector<std::string>& labels, int class_id);
};

}  // namespace edgevision
