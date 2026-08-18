#include "edgevision/tcp_server.hpp"

#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#if defined(__linux__)
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace edgevision {

namespace {

constexpr std::size_t kMaxCommandLength = 1024U;
constexpr int kPollTimeoutMs = 100;

#if defined(__linux__)
void close_fd(int& fd)
{
    if (fd >= 0) {
        close(fd);
        fd = -1;
    }
}
#endif

}  // namespace

TcpServer::TcpServer(std::uint16_t port, std::size_t max_event_queue)
    : requested_port_(port), max_event_queue_(max_event_queue)
{
    if (max_event_queue_ == 0U) {
        throw std::runtime_error("TCP server requires a non-empty event queue");
    }
}

TcpServer::~TcpServer()
{
    stop();
}

void TcpServer::start()
{
#if !defined(__linux__)
    throw std::runtime_error("TCP server requires Linux POSIX sockets");
#else
    if (worker_.joinable()) {
        throw std::runtime_error("TCP server is already running");
    }

    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error("TCP socket creation failed");
    }
    int reuse = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) != 0) {
        close(fd);
        throw std::runtime_error("TCP SO_REUSEADDR setup failed");
    }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(requested_port_);
    if (bind(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
        const std::string message = "TCP bind failed on port " + std::to_string(requested_port_);
        close(fd);
        throw std::runtime_error(message);
    }
    if (listen(fd, 1) != 0) {
        close(fd);
        throw std::runtime_error("TCP listen failed");
    }

    sockaddr_in bound_address{};
    socklen_t bound_length = sizeof(bound_address);
    if (getsockname(fd, reinterpret_cast<sockaddr*>(&bound_address), &bound_length) != 0) {
        close(fd);
        throw std::runtime_error("TCP could not query bound port");
    }

    stop_requested_.store(false);
    subscribed_.store(false);
    {
        std::lock_guard<std::mutex> lock(event_mutex_);
        event_queue_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(port_mutex_);
        bound_port_ = ntohs(bound_address.sin_port);
    }
    started_at_ = std::chrono::steady_clock::now();
    listen_fd_ = fd;
    worker_ = std::thread(&TcpServer::run, this);
#endif
}

void TcpServer::stop()
{
#if defined(__linux__)
    stop_requested_.store(true);
    if (worker_.joinable()) {
        worker_.join();
    }
    subscribed_.store(false);
#endif
}

std::uint16_t TcpServer::port() const
{
    std::lock_guard<std::mutex> lock(port_mutex_);
    return bound_port_;
}

void TcpServer::update_status(const TcpStatusSnapshot& status)
{
    std::lock_guard<std::mutex> lock(status_mutex_);
    status_ = status;
}

void TcpServer::publish_event(const RegionEvent& event, const std::string& class_name)
{
    if (!subscribed_.load()) {
        return;
    }
    std::lock_guard<std::mutex> lock(event_mutex_);
    if (event_queue_.size() >= max_event_queue_) {
        event_queue_.pop_front();
        dropped_events_.fetch_add(1U);
    }
    event_queue_.push_back(QueuedEvent{event, class_name});
}

#if defined(__linux__)

void TcpServer::run()
{
    while (!stop_requested_.load()) {
        pollfd descriptors[2]{};
        descriptors[0].fd = listen_fd_;
        descriptors[0].events = POLLIN;
        descriptors[1].fd = client_fd_;
        descriptors[1].events = client_fd_ >= 0 ? POLLIN : 0;

        const int result = poll(descriptors, 2, kPollTimeoutMs);
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        if (result > 0) {
            if ((descriptors[0].revents & POLLIN) != 0) {
                accept_client();
            }
            if (client_fd_ >= 0 &&
                (descriptors[1].revents & (POLLIN | POLLERR | POLLHUP | POLLNVAL)) != 0) {
                if ((descriptors[1].revents & POLLIN) != 0) {
                    receive_client_data();
                } else {
                    close_client();
                }
            }
        }
        if (client_fd_ >= 0) {
            flush_events();
        }
    }

    close_client();
    close_fd(listen_fd_);
}

void TcpServer::accept_client()
{
    sockaddr_in address{};
    socklen_t length = sizeof(address);
    const int accepted = accept(listen_fd_, reinterpret_cast<sockaddr*>(&address), &length);
    if (accepted < 0) {
        return;
    }

    close_client();
    const int flags = fcntl(accepted, F_GETFL, 0);
    if (flags >= 0) {
        fcntl(accepted, F_SETFL, flags | O_NONBLOCK);
    }
    client_fd_ = accepted;
    command_buffer_.clear();
    subscribed_.store(false);
}

void TcpServer::close_client()
{
    close_fd(client_fd_);
    command_buffer_.clear();
    subscribed_.store(false);
    std::lock_guard<std::mutex> lock(event_mutex_);
    event_queue_.clear();
}

void TcpServer::receive_client_data()
{
    char buffer[4096]{};
    const ssize_t received = recv(client_fd_, buffer, sizeof(buffer), 0);
    if (received == 0) {
        close_client();
        return;
    }
    if (received < 0) {
        if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
            close_client();
        }
        return;
    }

    command_buffer_.append(buffer, static_cast<std::size_t>(received));
    for (;;) {
        const std::size_t newline = command_buffer_.find('\n');
        if (newline == std::string::npos) {
            if (command_buffer_.size() > kMaxCommandLength) {
                command_buffer_.clear();
                send_error("command_too_long");
            }
            break;
        }

        std::string command = command_buffer_.substr(0U, newline);
        command_buffer_.erase(0U, newline + 1U);
        if (!command.empty() && command.back() == '\r') {
            command.pop_back();
        }
        if (command.size() > kMaxCommandLength) {
            send_error("command_too_long");
            continue;
        }
        handle_command(command);
        if (client_fd_ < 0) {
            break;
        }
    }
}

void TcpServer::handle_command(const std::string& command)
{
    if (command == "PING") {
        send_line("{\"type\":\"pong\"}\n");
    } else if (command == "GET_STATUS") {
        send_status();
    } else if (command == "SUBSCRIBE_EVENTS") {
        subscribed_.store(true);
        send_status();
    } else if (command == "UNSUBSCRIBE_EVENTS") {
        subscribed_.store(false);
        std::lock_guard<std::mutex> lock(event_mutex_);
        event_queue_.clear();
        send_status();
    } else if (!command.empty()) {
        send_error("unknown_command");
    }
}

void TcpServer::flush_events()
{
    if (!subscribed_.load()) {
        return;
    }
    for (;;) {
        QueuedEvent queued;
        {
            std::lock_guard<std::mutex> lock(event_mutex_);
            if (event_queue_.empty()) {
                return;
            }
            queued = std::move(event_queue_.front());
            event_queue_.pop_front();
        }

        const auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                       queued.event.source_timestamp.time_since_epoch())
                                       .count();
        std::ostringstream message;
        const int logical_id = queued.event.logical_id >= 0 ? queued.event.logical_id
                                                              : queued.event.track_id;
        message << "{\"type\":\"event\",\"event\":\""
                << event_name(queued.event.type) << "\",\"class\":\""
                << json_escape(queued.class_name) << "\",\"logical_id\":"
                << logical_id << ",\"track_id\":" << logical_id
                << ",\"confidence\":" << std::fixed
                << std::setprecision(3) << queued.event.confidence << ",\"timestamp_ms\":"
                << timestamp_ms << "}\n";
        if (!send_line(message.str())) {
            close_client();
            return;
        }
    }
}

bool TcpServer::send_line(const std::string& line)
{
    if (client_fd_ < 0) {
        return false;
    }
    std::size_t offset = 0U;
    while (offset < line.size() && !stop_requested_.load()) {
        const ssize_t sent = send(client_fd_, line.data() + offset, line.size() - offset,
                                  MSG_NOSIGNAL);
        if (sent > 0) {
            offset += static_cast<std::size_t>(sent);
            continue;
        }
        if (sent < 0 && errno == EINTR) {
            continue;
        }
        if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            pollfd descriptor{client_fd_, POLLOUT, 0};
            if (poll(&descriptor, 1, kPollTimeoutMs) > 0 &&
                (descriptor.revents & POLLOUT) != 0) {
                continue;
            }
        }
        return false;
    }
    return offset == line.size();
}

bool TcpServer::send_status()
{
    TcpStatusSnapshot status;
    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        status = status_;
    }
    std::ostringstream message;
    const auto uptime_ms = started_at_ == std::chrono::steady_clock::time_point{}
                               ? 0LL
                               : std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now() - started_at_)
                                     .count();
    message << "{\"type\":\"status\",\"objects\":" << status.objects
            << ",\"camera_fps\":" << std::fixed << std::setprecision(3) << status.camera_fps
            << ",\"display_fps\":" << status.display_fps
            << ",\"detection_fps\":" << status.detection_fps
            << ",\"uptime_ms\":" << uptime_ms << ",\"subscribed\":"
            << (subscribed_.load() ? "true" : "false") << "}\n";
    return send_line(message.str());
}

bool TcpServer::send_error(const std::string& message)
{
    return send_line("{\"type\":\"error\",\"message\":\"" + json_escape(message) +
                     "\"}\n");
}

#else

void TcpServer::run() {}
void TcpServer::accept_client() {}
void TcpServer::close_client() {}
void TcpServer::receive_client_data() {}
void TcpServer::handle_command(const std::string&) {}
void TcpServer::flush_events() {}
bool TcpServer::send_line(const std::string&) { return false; }
bool TcpServer::send_status() { return false; }
bool TcpServer::send_error(const std::string&) { return false; }

#endif

std::string TcpServer::json_escape(const std::string& value)
{
    std::string escaped;
    escaped.reserve(value.size());
    for (const char character : value) {
        switch (character) {
        case '"': escaped += "\\\""; break;
        case '\\': escaped += "\\\\"; break;
        case '\n': escaped += "\\n"; break;
        case '\r': escaped += "\\r"; break;
        case '\t': escaped += "\\t"; break;
        default: escaped += character; break;
        }
    }
    return escaped;
}

const char* TcpServer::event_name(RegionEventType type)
{
    switch (type) {
    case RegionEventType::Enter: return "ENTER";
    case RegionEventType::Dwell: return "DWELL";
    case RegionEventType::Exit: return "EXIT";
    }
    return "EVENT";
}

}  // namespace edgevision
