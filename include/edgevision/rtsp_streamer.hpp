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
    // 构造配置发布端 URL 和编码管线；GStreamer 对象在 start() 才创建。
    explicit RtspStreamer(int port = 8554,
                          std::string advertised_host = "192.168.77.2");
    ~RtspStreamer();

    RtspStreamer(const RtspStreamer&) = delete;
    RtspStreamer& operator=(const RtspStreamer&) = delete;

    // 启动独立 GLib 主循环线程，等待其报告就绪或初始化失败。
    void start();
    // 发布调用者画面的自有 GstBuffer 副本；仅保留最新帧以限制延迟和内存。
    // 输入须为固定尺寸的连续 CV_8UC3 BGR Mat；实现仅校验尺寸/类型，未校验连续性。
    void publish(const cv::Mat& annotated_bgr);
    // 退出 GLib 循环、回收服务线程并释放最后缓存的帧。
    void stop();

    bool is_running() const;
    int port() const { return port_; }
    const std::string& url() const { return url_; }
    const std::string& pipeline() const { return pipeline_; }

private:
    static std::string make_pipeline();

    // 工厂创建媒体时的回调入口，将配置工作转交给此实例。
    static void on_media_configure(_GstRTSPMediaFactory* factory,
                                   _GstRTSPMedia* media,
                                   void* user_data);
    // appsrc 请求数据时取最新画面；静态回调再转交给拥有状态的实例。
    static void on_need_data(_GstAppSrc* appsrc, unsigned int length, void* user_data);

    void configure_media(_GstRTSPMedia* media);
    void push_latest(_GstAppSrc* appsrc);
    void run_server();

    const int port_;
    const std::string url_;
    const std::string pipeline_;

    // 互斥保护服务标志、GLib loop 指针、帧缓冲指针和媒体时间戳。
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::thread server_thread_;
    // loop_ 由服务线程创建运行，stop() 取得指针后在锁外请求其退出。
    _GMainLoop* loop_ = nullptr;
    // 持有当前最新 GstBuffer；替换或服务退出时释放本对象持有的引用。
    _GstBuffer* latest_buffer_ = nullptr;
    // appsrc 媒体时间轴的下一个 PTS，按推送帧的固定时长向前推进。
    std::uint64_t next_pts_ = 0U;
    bool ready_ = false;
    bool running_ = false;
    bool stop_requested_ = false;
    bool failed_ = false;
    std::string error_message_;
};

}  // namespace edgevision
