#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

struct _GMainLoop;
struct _GstAppSrc;
struct _GstBuffer;
struct _GstRTSPMedia;
struct _GstRTSPMediaFactory;

namespace edgevision {

class RtspStreamer {
public:
    explicit RtspStreamer(int port = 8554,
                          std::string advertised_host = "192.168.77.2");
    ~RtspStreamer();

    RtspStreamer(const RtspStreamer&) = delete;
    RtspStreamer& operator=(const RtspStreamer&) = delete;

    void start();
    void publish(const cv::Mat& annotated_bgr);
    void stop();

    bool is_running() const;
    int port() const { return port_; }
    const std::string& url() const { return url_; }
    const std::string& pipeline() const { return pipeline_; }

private:
    static std::string make_pipeline();

    static void on_media_configure(_GstRTSPMediaFactory* factory,
                                   _GstRTSPMedia* media,
                                   void* user_data);
    static void on_need_data(_GstAppSrc* appsrc, unsigned int length, void* user_data);

    void configure_media(_GstRTSPMedia* media);
    void push_latest(_GstAppSrc* appsrc);
    void run_server();

    const int port_;
    const std::string url_;
    const std::string pipeline_;

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::thread server_thread_;
    _GMainLoop* loop_ = nullptr;
    _GstBuffer* latest_buffer_ = nullptr;
    std::uint64_t next_pts_ = 0U;
    bool ready_ = false;
    bool running_ = false;
    bool stop_requested_ = false;
    bool failed_ = false;
    std::string error_message_;
};

}  // namespace edgevision
