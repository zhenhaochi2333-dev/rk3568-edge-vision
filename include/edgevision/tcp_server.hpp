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
    std::size_t objects = 0U;
    double camera_fps = 0.0;
    double display_fps = 0.0;
    double detection_fps = 0.0;
};

class TcpServer {
public:
    explicit TcpServer(std::uint16_t port, std::size_t max_event_queue = 64U);
    ~TcpServer();

    TcpServer(const TcpServer&) = delete;
    TcpServer& operator=(const TcpServer&) = delete;

    void start();
    void stop();

    std::uint16_t port() const;
    void update_status(const TcpStatusSnapshot& status);
    void publish_event(const RegionEvent& event, const std::string& class_name);
    std::uint64_t dropped_event_count() const { return dropped_events_.load(); }

private:
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
    mutable std::mutex status_mutex_;
    TcpStatusSnapshot status_;
    std::mutex event_mutex_;
    std::deque<QueuedEvent> event_queue_;
    std::atomic<std::uint64_t> dropped_events_{0U};
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> subscribed_{false};
    mutable std::mutex port_mutex_;
    std::uint16_t bound_port_ = 0U;
    std::chrono::steady_clock::time_point started_at_{};
    int listen_fd_ = -1;
    int client_fd_ = -1;
    std::string command_buffer_;
    std::thread worker_;
};

}  // namespace edgevision
