#include "edgevision/cli_parser.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void run_geometry_tests();
void run_display_composer_tests();
void run_yolo11_postprocess_tests();
void run_iou_tracker_tests();
void run_region_monitor_tests();
void run_perf_monitor_tests();
void run_camera_source_tests();
void run_video_io_tests();

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
    assert(valid.options.roi_enabled);
    assert(valid.options.roi.x == 0.0F);
    assert(valid.options.roi.y == 0.0F);
    assert(valid.options.roi.width == 1.0F);
    assert(valid.options.roi.height == 1.0F);
    const edgevision::CliParseResult camera = parse({
        "edge_vision", "--model", "model.rknn", "--labels", "labels.txt",
        "--camera", "/dev/video0", "--show", "--fullscreen", "--smooth-preview",
        "--roi", "0.25,0.20,0.50,0.60", "--show-roi"});
    assert(camera.options.camera_path == "/dev/video0");
    assert(camera.options.input_path.empty());
    assert(camera.options.fullscreen);
    assert(camera.options.smooth_preview);
    assert(camera.options.roi_enabled);
    assert(camera.options.show_roi);
    assert(camera.options.roi.x == 0.25F);
    assert(camera.options.roi.height == 0.60F);
    const edgevision::CliParseResult default_roi_debug = parse({
        "edge_vision", "--model", "model.rknn", "--labels", "labels.txt",
        "--camera", "/dev/video0", "--show", "--smooth-preview", "--show-roi"});
    assert(default_roi_debug.options.roi_enabled);
    assert(default_roi_debug.options.show_roi);
    assert(default_roi_debug.options.roi.width == 1.0F);
    expect_error([] { parse({"edge_vision", "--unknown"}); });
    expect_error([] { parse({"edge_vision", "--model"}); });
    expect_error([] { parse({"edge_vision", "--model", "m", "--labels", "l", "--input", "i",
                             "--camera", "/dev/video0", "--output", "o"}); });
    expect_error([] { parse({"edge_vision", "--model", "m", "--labels", "l", "--camera",
                             "/dev/video0"}); });
    expect_error([] { parse({"edge_vision", "--model", "m", "--labels", "l", "--camera",
                             "/dev/video0", "--fullscreen"}); });
    expect_error([] { parse({"edge_vision", "--model", "m", "--labels", "l", "--camera",
                             "/dev/video0", "--show", "--smooth-preview", "--output", "o"}); });
    expect_error([] { parse({"edge_vision", "--model", "m", "--labels", "l", "--input", "i",
                             "--output", "o", "--smooth-preview"}); });
    expect_error([] { parse({"edge_vision", "--model", "m", "--labels", "l", "--input", "i",
                             "--output", "o", "--roi", "0.1,0.1,0.5,0.5"}); });
    expect_error([] { parse({"edge_vision", "--model", "m", "--labels", "l", "--camera",
                             "/dev/video0", "--show", "--smooth-preview", "--roi",
                             "0.8,0.1,0.3,0.3"}); });
    expect_error([] { parse({"edge_vision", "--model", "m", "--labels", "l", "--camera",
                             "/dev/video0", "--show", "--smooth-preview", "--roi",
                             "0.1,0.1,0.5"}); });
    expect_error([] { parse({"edge_vision", "--model", "m", "--labels", "l", "--input", "i", "--output", "o", "--conf", "1.1"}); });
    expect_error([] { parse({"edge_vision", "--model", "m", "--labels", "l", "--input", "i", "--output", "o", "--max-frames", "-1"}); });
}

}  // namespace

int main()
{
    run_cli_tests();
    run_display_composer_tests();
    run_geometry_tests();
    run_yolo11_postprocess_tests();
    run_iou_tracker_tests();
    run_region_monitor_tests();
    run_perf_monitor_tests();
#if EDGEVISION_WITH_VIDEO
    run_camera_source_tests();
    run_video_io_tests();
#endif
    std::cout << "edgevision_tests: PASS\n";
    return 0;
}
