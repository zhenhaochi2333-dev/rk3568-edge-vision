#include "edgevision/video_io.hpp"

#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <sys/stat.h>

namespace edgevision {

VideoIO::~VideoIO()
{
    close_output();
    capture_.release();
}

bool VideoIO::exists(const std::string& path)
{
    struct stat info{};
    return stat(path.c_str(), &info) == 0;
}

std::string VideoIO::lowercase_extension(const std::string& path)
{
    const std::size_t slash = path.find_last_of("/\\");
    const std::size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        return std::string();
    }
    std::string extension = path.substr(dot);
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return extension;
}

std::string VideoIO::fallback_path(const std::string& path)
{
    const std::size_t slash = path.find_last_of("/\\");
    const std::size_t dot = path.find_last_of('.');
    const std::size_t end = dot == std::string::npos || (slash != std::string::npos && dot < slash)
                                ? path.size()
                                : dot;
    return path.substr(0, end) + "_mjpg.avi";
}

void VideoIO::reject_existing(const std::string& path, bool force_overwrite)
{
    if (exists(path) && !force_overwrite) {
        throw std::runtime_error("output already exists; pass --force to replace it: " + path);
    }
}

void VideoIO::open_input(const std::string& path)
{
    capture_.open(path);
    if (!capture_.isOpened()) {
        throw std::runtime_error("cannot open video input: " + path);
    }
    source_info_.width = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
    source_info_.height = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
    source_info_.fps = capture_.get(cv::CAP_PROP_FPS);
    source_info_.frame_count = static_cast<long long>(capture_.get(cv::CAP_PROP_FRAME_COUNT));
    source_info_.backend = capture_.getBackendName();
}

bool VideoIO::read(cv::Mat& frame)
{
    return capture_.read(frame);
}

void VideoIO::open_output(const std::string& requested_path, double fps, cv::Size resolution,
                          bool force_overwrite, bool force_mjpg_for_test)
{
    if (fps <= 0.0) {
        fps = 30.0;
    }
    if (resolution.width <= 0 || resolution.height <= 0) {
        throw std::runtime_error("invalid video output resolution");
    }
    reject_existing(requested_path, force_overwrite);

    const bool requested_mp4 = lowercase_extension(requested_path) == ".mp4";
    if (requested_mp4 && !force_mjpg_for_test) {
        writer_.open(requested_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                     resolution, true);
        if (writer_.isOpened()) {
            writer_info_ = VideoWriterInfo{requested_path, requested_path, "mp4v", fps, resolution};
            return;
        }
        const std::string fallback = fallback_path(requested_path);
        reject_existing(fallback, force_overwrite);
        writer_.open(fallback, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, resolution, true);
        if (writer_.isOpened()) {
            writer_info_ = VideoWriterInfo{requested_path, fallback, "MJPG", fps, resolution};
            return;
        }
    } else {
        const std::string output_path = requested_mp4 ? fallback_path(requested_path) : requested_path;
        if (output_path != requested_path) {
            reject_existing(output_path, force_overwrite);
        }
        writer_.open(output_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, resolution, true);
        if (writer_.isOpened()) {
            writer_info_ = VideoWriterInfo{requested_path, output_path, "MJPG", fps, resolution};
            return;
        }
    }

    throw std::runtime_error("cannot open video writer for requested output: " + requested_path);
}

void VideoIO::write(const cv::Mat& frame)
{
    if (!writer_.isOpened()) {
        throw std::runtime_error("video writer is not open");
    }
    writer_.write(frame);
}

void VideoIO::close_output()
{
    if (writer_.isOpened()) {
        writer_.release();
    }
}

}  // namespace edgevision
