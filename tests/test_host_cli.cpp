#include "edgevision/cli_parser.hpp"

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

edgevision::CliParseResult parse(std::initializer_list<std::string> arguments)
{
    std::vector<std::string> values(arguments);
    std::vector<char*> argv;
    argv.reserve(values.size());
    for (std::string& value : values) {
        argv.push_back(&value[0]);
    }
    return edgevision::CliParser::parse(static_cast<int>(argv.size()), argv.data());
}

template <typename Function>
void expect_error(Function function)
{
    bool failed = false;
    try {
        function();
    } catch (const std::runtime_error&) {
        failed = true;
    }
    assert(failed);
}

}  // namespace

int main()
{
    const edgevision::CliParseResult result = parse({
        "edge_vision", "--model", "model.rknn", "--labels", "labels.txt",
        "--camera", "/dev/video0", "--show", "--smooth-preview",
        "--roi", "0.25,0.20,0.50,0.60"});
    assert(result.options.roi_enabled);
    assert(result.options.roi.width == 0.50F);
    assert(result.options.roi.height == 0.60F);
    expect_error([] {
        parse({"edge_vision", "--model", "m", "--labels", "l", "--camera", "/dev/video0",
               "--show", "--smooth-preview", "--roi", "0.9,0.1,0.2,0.2"});
    });
    expect_error([] {
        parse({"edge_vision", "--model", "m", "--labels", "l", "--input", "i",
               "--output", "o", "--roi", "0.1,0.1,0.5,0.5"});
    });
    return 0;
}
