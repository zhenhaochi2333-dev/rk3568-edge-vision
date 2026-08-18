#include "edgevision/network_camera_source.hpp"

#include <stdexcept>

namespace edgevision {

NetworkCameraSource::NetworkCameraSource(int port)
    : port_(port), pipeline_(make_pipeline(port))
{
    if (port_ < 1 || port_ > 65535) {
        throw std::runtime_error("network camera UDP port must be within [1,65535]");
    }
}

NetworkCameraSource::~NetworkCameraSource()
{
    release();
}

std::string NetworkCameraSource::make_pipeline(int port)
{
    return "udpsrc port=" + std::to_string(port) +
           " caps=\"application/x-rtp,media=video,clock-rate=90000,"
           "encoding-name=H264,payload=96\" ! rtph264depay ! h264parse ! "
           "mppvideodec format=BGR ! video/x-raw,format=BGR ! "
           "appsink sync=false max-buffers=1 drop=true";
}

void NetworkCameraSource::open()
{
    if (capture_.isOpened()) {
        return;
    }

    if (!capture_.open(pipeline_, cv::CAP_GSTREAMER)) {
        throw std::runtime_error("cannot open network camera pipeline: " + pipeline_);
    }

    info_.width = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
    info_.height = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
    info_.fps = capture_.get(cv::CAP_PROP_FPS);
    info_.backend = "GStreamer/MPP";
    info_.pixel_format = "BGR";
    info_.plane_count = 1;
    info_.bytes_per_line = static_cast<std::size_t>(info_.width) * 3U;
}

bool NetworkCameraSource::read(cv::Mat& frame)
{
    frame.release();
    if (!capture_.isOpened() || !capture_.read(frame)) {
        return false;
    }
    if (frame.empty() || frame.cols != 1280 || frame.rows != 720 || frame.type() != CV_8UC3) {
        throw std::runtime_error("network camera frame must be 1280x720 BGR CV_8UC3; got " +
                                 std::to_string(frame.cols) + "x" +
                                 std::to_string(frame.rows) + " type=" +
                                 std::to_string(frame.type()));
    }
    return true;
}

void NetworkCameraSource::release()
{
    if (capture_.isOpened()) {
        capture_.release();
    }
}

}  // namespace edgevision
