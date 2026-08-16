#include "edgevision/logger.hpp"

#include <iostream>

namespace edgevision {

namespace {

void write_log(const char* level, const std::string& message)
{
    std::cerr << level << ' ' << message << '\n';
}

}  // namespace

void log_info(const std::string& message)
{
    write_log("[INFO]", message);
}

void log_warn(const std::string& message)
{
    write_log("[WARN]", message);
}

void log_error(const std::string& message)
{
    write_log("[ERROR]", message);
}

void log_perf(const std::string& message)
{
    write_log("[PERF]", message);
}

}  // namespace edgevision
