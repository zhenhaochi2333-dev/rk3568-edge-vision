#pragma once

#include "edgevision/camera_source.hpp"

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

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

    bool is_opened() const
    {
        std::lock_guard<std::mutex> lock(lifecycle_mutex_);
        return server_fd_ >= 0;
    }
    int port() const { return port_; }
    const std::string& pipeline() const { return pipeline_; }
    const CameraSourceInfo& info() const { return info_; }

private:
    static std::string make_pipeline(int port);
    bool extract_latest_jpeg(std::vector<unsigned char>& jpeg);
    void close_client_locked();

    const int port_;
    const std::string pipeline_;
    mutable std::mutex lifecycle_mutex_;
    std::atomic<bool> stop_requested_{false};
    int server_fd_ = -1;
    int client_fd_ = -1;
    std::vector<unsigned char> stream_buffer_;
    CameraSourceInfo info_;
};

}  // namespace edgevision
