#pragma once

#include "edgevision/app_options.hpp"

namespace edgevision {

class CliParser {
public:
    static CliParseResult parse(int argc, char** argv);
    static const char* usage();
};

}  // namespace edgevision
