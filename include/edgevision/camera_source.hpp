#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <string>

namespace edgevision {

struct CameraSourceInfo {
    int width = 0;
    int height = 0;
    double fps = 0.0;
    std::string backend;
};

class CameraSource {
public:
    explicit CameraSource(const std::string& device);
    ~CameraSource();

    CameraSource(const CameraSource&) = delete;
    CameraSource& operator=(const CameraSource&) = delete;

    void open();
    bool read(cv::Mat& frame);
    void release();

    bool is_opened() const { return capture_.isOpened(); }
    const std::string& device() const { return device_; }
    const std::string& pipeline() const { return pipeline_; }
    const CameraSourceInfo& info() const { return info_; }

private:
    static std::string make_pipeline(const std::string& device);

    std::string device_;
    std::string pipeline_;
    cv::VideoCapture capture_;
    CameraSourceInfo info_;
};

}  // namespace edgevision
