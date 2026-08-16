#pragma once

#include <string>

namespace edgevision {

void log_info(const std::string& message);
void log_warn(const std::string& message);
void log_error(const std::string& message);
void log_perf(const std::string& message);

}  // namespace edgevision
