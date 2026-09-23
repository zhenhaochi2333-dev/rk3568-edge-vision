#pragma once

#include "edgevision/camera_source.hpp"

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

namespace edgevision {

// 板端 TCP :5600 MJPEG 接收器。TCP 只提供字节流；类内按 JPEG
// SOI/EOI 拼帧、保留跨 recv 的残片，并用 OpenCV 解码成 CV_8UC3 BGR。
class NetworkCameraSource {
public:
    explicit NetworkCameraSource(int port = 5600);
    ~NetworkCameraSource();

    NetworkCameraSource(const NetworkCameraSource&) = delete;
    NetworkCameraSource& operator=(const NetworkCameraSource&) = delete;

    // open 建立非阻塞监听而不等待 PC；read 在短轮询周期里接受连接并
    // 试取最新完整 JPEG，暂未得到有效帧时返回 false。release 关闭 fd
    // 并设置停止标志，允许采集线程的读循环及时退出。
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
    // 仅由 read 的接收/拼帧路径维护；缓存未闭合 JPEG 尾部。
    // 超过 16 MiB 时在实现中尝试重新同步，防止坏流无限占内存。
    std::vector<unsigned char> stream_buffer_;
    CameraSourceInfo info_;
};

}  // namespace edgevision
