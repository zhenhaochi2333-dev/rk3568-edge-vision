#include "edgevision/cli_parser.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void run_geometry_tests();
void run_postprocess_tests();

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

void run_cli_tests()
{
    const edgevision::CliParseResult valid = parse({
        "edge_vision", "--model", "model.rknn", "--labels", "labels.txt",
        "--input", "bus.jpg", "--output", "out.png", "--conf", "0.25",
        "--nms", "0.45", "--max-frames", "0", "--force"});
    assert(!valid.show_help);
    assert(valid.options.conf_threshold == 0.25F);
    assert(valid.options.force);
    expect_error([] { parse({"edge_vision", "--unknown"}); });
    expect_error([] { parse({"edge_vision", "--model"}); });
    expect_error([] { parse({"edge_vision", "--model", "m", "--labels", "l", "--input", "i", "--output", "o", "--conf", "1.1"}); });
    expect_error([] { parse({"edge_vision", "--model", "m", "--labels", "l", "--input", "i", "--output", "o", "--max-frames", "-1"}); });
}

}  // namespace

int main()
{
    run_cli_tests();
    run_geometry_tests();
    run_postprocess_tests();
    std::cout << "edgevision_tests: PASS\n";
    return 0;
}
