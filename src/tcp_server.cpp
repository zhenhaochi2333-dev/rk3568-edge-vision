/* 独立 POSIX TCP 服务：换行命令请求状态或订阅事件；响应是逐行 JSON。
 * 它与传 JPEG 的 :5600 输入 socket 分离，默认监听 :9000。
 */
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
// poll 周期性醒来观察停止标志，stop() 不必从另一线程关闭正在轮询的 fd。
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

    // 监听 socket 在调用线程创建和配置；配置完整后才交给工作线程使用。
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error("TCP socket creation failed");
    }
    int reuse = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) != 0) {
        close(fd);
        throw std::runtime_error("TCP SO_REUSEADDR setup failed");
    }

    // 监听所有本机网卡地址；请求端口为 0 时由 getsockname 读取系统分配值。
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

    // 每轮 start 建立新会话状态，清除前轮残留事件，避免发给后来连接的客户端。
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
    // worker 的 poll 最多等待一个短周期，随后检查此标志并负责关闭自己使用的 fd。
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

// AI/显示线程只把已订阅事件放进有界队列；socket 写入由服务线程完成。
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

    // 同时只保留一个客户端；新连接替代旧连接时，旧会话队列和订阅状态作废。
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
    // 对端断开意味着它不再消费事件，因此清除积压，让下次订阅从新事件开始。
    close_fd(client_fd_);
    command_buffer_.clear();
    subscribed_.store(false);
    std::lock_guard<std::mutex> lock(event_mutex_);
    event_queue_.clear();
}

// TCP 也是字节流；一个 recv 可能只含半条命令或多条命令，因此按换行拆包。
void TcpServer::receive_client_data()
{
    // recv 返回任意长度的字节片段：既可能是半条命令，也可能含多条完整命令。
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

    // 累积到成员缓存后按换行拆包，保留末尾不足一行的内容等待下次 recv。
    command_buffer_.append(buffer, static_cast<std::size_t>(received));
    for (;;) {
        const std::size_t newline = command_buffer_.find('\n');
        if (newline == std::string::npos) {
            // 即使客户端始终不发换行，未完成命令也受长度上限约束，避免缓存无限增长。
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
    // 协议命令为换行分隔的大写文本；非空未知命令得到一个独立错误响应。
    if (command == "PING") {
        send_line("{\"type\":\"pong\"}\n");
    } else if (command == "GET_STATUS") {
        send_status();
    } else if (command == "SUBSCRIBE_EVENTS") {
        // 先开启订阅，再返回快照确认；后续新事件由服务线程异步推送。
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

// 把事件队列转换为业务 JSON；兼容字段 track_id 当前承载同一个 logical_id。
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
            // 只在锁内移动出队首，之后在锁外格式化并写 socket，避免慢客户端阻塞生产者。
            queued = std::move(event_queue_.front());
            event_queue_.pop_front();
        }

        // 事件时间来自 steady_clock 的采集时间点；毫秒值位于单调时钟时间轴，
        // 不能当作可换算为日历日期的 Unix epoch 时间戳。
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

// send 可能只写入部分字节；offset 循环尝试发完一行，失败则返回 false。
bool TcpServer::send_line(const std::string& line)
{
    if (client_fd_ < 0) {
        return false;
    }
    // SOCK_STREAM 不保留消息边界；send 可能只接受部分数据，offset 记录已接受前缀。
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
        // 非阻塞发送缓冲满时等到可写，再从原 offset 续发，避免截断一行 JSON。
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
    // 先复制后解锁；构造 JSON 和发送期间，业务线程可继续更新新的完整状态快照。
    TcpStatusSnapshot status;
    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        status = status_;
    }
    std::ostringstream message;
    // uptime 使用单调时钟计时；它不是 Unix 墙上时间戳。
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
    // 错误消息同样转义 JSON 特殊字符，保持响应可被逐行 JSON 解析器读取。
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
