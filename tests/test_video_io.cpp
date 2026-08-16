#include "edgevision/video_io.hpp"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>

void run_video_io_tests()
{
    const std::string input_path = "edgevision_video_io_input.avi";
    const std::string requested_output = "edgevision_video_io_output.mp4";
    const std::string fallback_output = "edgevision_video_io_output_mjpg.avi";
    const cv::Size size(96, 64);
    const cv::Mat frame(size, CV_8UC3, cv::Scalar(20, 40, 80));

    cv::VideoWriter seed(input_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 5.0, size, true);
    assert(seed.isOpened());
    seed.write(frame);
    seed.write(frame);
    seed.release();

    edgevision::VideoIO input;
    input.open_input(input_path);
    assert(input.source_info().width == size.width);
    assert(input.source_info().height == size.height);
    cv::Mat read_frame;
    assert(input.read(read_frame));
    assert(!read_frame.empty());

    bool invalid_failed = false;
    try {
        edgevision::VideoIO invalid;
        invalid.open_input("edgevision_video_io_missing.avi");
    } catch (const std::runtime_error&) {
        invalid_failed = true;
    }
    assert(invalid_failed);

    edgevision::VideoIO output;
    output.open_output(requested_output, 5.0, size, true, true);
    assert(output.writer_info().codec == "MJPG");
    assert(output.writer_info().actual_path == fallback_output);
    output.write(read_frame);
    output.close_output();
    cv::VideoCapture verification(fallback_output);
    assert(verification.isOpened());
    assert(verification.read(read_frame));
    assert(read_frame.size() == size);
    verification.release();

    std::remove(input_path.c_str());
    std::remove(requested_output.c_str());
    std::remove(fallback_output.c_str());
}
