#include "edgevision/camera_source.hpp"

#include <stdexcept>

namespace edgevision {

CameraSource::CameraSource(const std::string& device)
    : device_(device), pipeline_(make_pipeline(device))
{
    if (device_.empty()) {
        throw std::runtime_error("camera device path must not be empty");
    }
}

CameraSource::~CameraSource()
{
    release();
}

std::string CameraSource::make_pipeline(const std::string& device)
{
    return "v4l2src device=" + device + " io-mode=mmap ! "
           "video/x-raw,format=NV12,width=1280,height=720,framerate=30/1 ! "
           "videoconvert ! video/x-raw,format=BGR ! "
           "appsink drop=true max-buffers=1 sync=false";
}

void CameraSource::open()
{
    if (capture_.isOpened()) {
        return;
    }
    capture_.open(pipeline_, cv::CAP_GSTREAMER);
    if (!capture_.isOpened()) {
        throw std::runtime_error("cannot open camera " + device_ +
                                 " with GStreamer pipeline: " + pipeline_);
    }

    info_.width = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
    info_.height = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
    info_.fps = capture_.get(cv::CAP_PROP_FPS);
    info_.backend = capture_.getBackendName();
}

bool CameraSource::read(cv::Mat& frame)
{
    frame.release();
    if (!capture_.isOpened() || !capture_.read(frame)) {
        return false;
    }
    return !frame.empty();
}

void CameraSource::release()
{
    if (capture_.isOpened()) {
        capture_.release();
    }
}

}  // namespace edgevision
