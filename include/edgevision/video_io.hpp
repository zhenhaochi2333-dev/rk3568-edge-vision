#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <string>

namespace edgevision {

struct VideoSourceInfo {
    int width = 0;
    int height = 0;
    double fps = 0.0;
    long long frame_count = 0;
    std::string backend;
};

struct VideoWriterInfo {
    std::string requested_path;
    std::string actual_path;
    std::string codec;
    double fps = 0.0;
    cv::Size resolution;
};

class VideoIO {
public:
    VideoIO() = default;
    ~VideoIO();

    VideoIO(const VideoIO&) = delete;
    VideoIO& operator=(const VideoIO&) = delete;

    void open_input(const std::string& path);
    bool read(cv::Mat& frame);
    void open_output(const std::string& requested_path, double fps, cv::Size resolution,
                     bool force_overwrite, bool force_mjpg_for_test = false);
    void write(const cv::Mat& frame);
    void close_output();

    bool input_open() const { return capture_.isOpened(); }
    bool output_open() const { return writer_.isOpened(); }
    const VideoSourceInfo& source_info() const { return source_info_; }
    const VideoWriterInfo& writer_info() const { return writer_info_; }

private:
    static bool exists(const std::string& path);
    static std::string lowercase_extension(const std::string& path);
    static std::string fallback_path(const std::string& path);
    static void reject_existing(const std::string& path, bool force_overwrite);

    cv::VideoCapture capture_;
    cv::VideoWriter writer_;
    VideoSourceInfo source_info_;
    VideoWriterInfo writer_info_;
};

}  // namespace edgevision
