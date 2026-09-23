#pragma once

#include "edgevision/region_monitor.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

namespace edgevision {

struct TcpStatusSnapshot {
    // 多字段状态作为一个快照整体发布，避免序列化过程中读到新旧字段混合的值。
    std::size_t objects = 0U;
    double camera_fps = 0.0;
    double display_fps = 0.0;
    double detection_fps = 0.0;
};

class TcpServer {
public:
    // port=0 可由系统分配临时端口；队列容量限制未发送事件的内存占用。
    explicit TcpServer(std::uint16_t port, std::size_t max_event_queue = 64U);
    ~TcpServer();

    TcpServer(const TcpServer&) = delete;
    TcpServer& operator=(const TcpServer&) = delete;

    // 创建监听 socket 并启动网络工作线程；实际端口通过 port() 查询。
    void start();
    // 请求线程退出并等待其关闭连接与监听 socket，避免析构时遗留工作线程。
    void stop();

    std::uint16_t port() const;
    // 业务线程只更新内存快照，socket I/O 始终留给服务线程执行。
    void update_status(const TcpStatusSnapshot& status);
    // 仅订阅期间入队；队列满时淘汰最早事件，保持内存有界并尽量保留新鲜状态。
    void publish_event(const RegionEvent& event, const std::string& class_name);
    std::uint64_t dropped_event_count() const { return dropped_events_.load(); }

private:
    // 保存事件值与类名副本，调用方返回后可立即复用自己的对象和字符串。
    struct QueuedEvent {
        RegionEvent event;
        std::string class_name;
    };

    void run();
    void accept_client();
    void close_client();
    void receive_client_data();
    void handle_command(const std::string& command);
    void flush_events();

    bool send_line(const std::string& line);
    bool send_status();
    bool send_error(const std::string& message);

    static std::string json_escape(const std::string& value);
    static const char* event_name(RegionEventType type);

    const std::uint16_t requested_port_;
    const std::size_t max_event_queue_;
    // 状态与事件队列分锁保护，网络事件排队不阻塞状态快照的更新。
    mutable std::mutex status_mutex_;
    TcpStatusSnapshot status_;
    std::mutex event_mutex_;
    std::deque<QueuedEvent> event_queue_;
    // 队列满时淘汰事件的累计计数；原子读取无需与队列锁竞争。
    std::atomic<std::uint64_t> dropped_events_{0U};
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> subscribed_{false};
    mutable std::mutex port_mutex_;
    std::uint16_t bound_port_ = 0U;
    std::chrono::steady_clock::time_point started_at_{};
    // -1 表示 socket 已关闭；运行期间由服务生命周期和工作线程协作管理。
    int listen_fd_ = -1;
    int client_fd_ = -1;
    // 保存 TCP 字节流末尾未凑齐换行的片段，支持命令跨 recv 边界到达。
    std::string command_buffer_;
    std::thread worker_;
};

}  // namespace edgevision
