#pragma once

#include "edgevision/camera_source.hpp"

#include <opencv2/videoio.hpp>

#include <string>

namespace edgevision {

class NetworkCameraSource {
public:
    explicit NetworkCameraSource(int port = 5600);
    ~NetworkCameraSource();

    NetworkCameraSource(const NetworkCameraSource&) = delete;
    NetworkCameraSource& operator=(const NetworkCameraSource&) = delete;

    void open();
    bool read(cv::Mat& frame);
    void release();

    bool is_opened() const { return capture_.isOpened(); }
    int port() const { return port_; }
    const std::string& pipeline() const { return pipeline_; }
    const CameraSourceInfo& info() const { return info_; }

private:
    static std::string make_pipeline(int port);

    const int port_;
    const std::string pipeline_;
    CameraSourceInfo info_;
    cv::VideoCapture capture_;
};

}  // namespace edgevision
