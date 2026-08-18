#include "edgevision/network_camera_source.hpp"

#include <opencv2/imgcodecs.hpp>

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#if defined(__linux__)
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace edgevision {

namespace {

constexpr std::size_t kReceiveChunkSize = 64U * 1024U;
constexpr std::size_t kMaximumBufferedJpegBytes = 16U * 1024U * 1024U;
constexpr int kPollTimeoutMs = 100;

#if defined(__linux__)
int set_nonblocking(int fd)
{
    const int flags = fcntl(fd, F_GETFL, 0);
    return flags < 0 ? -1 : fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}
#endif

}  // namespace

NetworkCameraSource::NetworkCameraSource(int port)
    : port_(port), pipeline_(make_pipeline(port))
{
    if (port_ < 1 || port_ > 65535) {
        throw std::runtime_error("network camera TCP port must be within [1,65535]");
    }
}

NetworkCameraSource::~NetworkCameraSource()
{
    release();
}

std::string NetworkCameraSource::make_pipeline(int port)
{
    return "tcp-server-mjpeg port=" + std::to_string(port) +
           " ! jpeg SOI/EOI framing ! OpenCV imdecode(BGR)";
}

void NetworkCameraSource::open()
{
#if !defined(__linux__)
    throw std::runtime_error("JPEG/TCP network input requires the Linux socket backend");
#else
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    if (server_fd_ >= 0) {
        return;
    }

    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error(std::string("cannot create network camera TCP socket: ") +
                                 std::strerror(errno));
    }

    int reuse = 1;
    (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    if (set_nonblocking(fd) < 0) {
        const std::string error = std::strerror(errno);
        close(fd);
        throw std::runtime_error("cannot make network camera TCP socket non-blocking: " +
                                 error);
    }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(static_cast<std::uint16_t>(port_));
    if (bind(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) < 0) {
        const std::string error = std::strerror(errno);
        close(fd);
        throw std::runtime_error("cannot bind network camera TCP port " +
                                 std::to_string(port_) + ": " + error);
    }
    if (listen(fd, 1) < 0) {
        const std::string error = std::strerror(errno);
        close(fd);
        throw std::runtime_error("cannot listen on network camera TCP port " + error);
    }

    stop_requested_.store(false);
    server_fd_ = fd;
    client_fd_ = -1;
    stream_buffer_.clear();
    info_.width = 1280;
    info_.height = 720;
    info_.fps = 15.0;
    info_.backend = "POSIX TCP/MJPEG/OpenCV";
    info_.pixel_format = "BGR";
    info_.plane_count = 1;
    info_.bytes_per_line = static_cast<std::size_t>(info_.width) * 3U;
#endif
}

bool NetworkCameraSource::extract_latest_jpeg(std::vector<unsigned char>& jpeg)
{
    const auto find_marker = [this](std::size_t start, unsigned char first,
                                     unsigned char second) {
        for (std::size_t index = start; index + 1U < stream_buffer_.size(); ++index) {
            if (stream_buffer_[index] == first && stream_buffer_[index + 1U] == second) {
                return index;
            }
        }
        return std::string::size_type(std::string::npos);
    };

    std::size_t search_from = 0U;
    std::size_t latest_start = std::string::npos;
    std::size_t latest_end = std::string::npos;
    while (search_from + 1U < stream_buffer_.size()) {
        const std::size_t start = find_marker(search_from, 0xFFU, 0xD8U);
        if (start == std::string::npos) {
            break;
        }
        const std::size_t end = find_marker(start + 2U, 0xFFU, 0xD9U);
        if (end == std::string::npos) {
            break;
        }
        latest_start = start;
        latest_end = end + 2U;
        search_from = latest_end;
    }

    if (latest_end == std::string::npos) {
        if (stream_buffer_.size() > kMaximumBufferedJpegBytes) {
            const std::size_t last_start = find_marker(
                stream_buffer_.size() - kMaximumBufferedJpegBytes, 0xFFU, 0xD8U);
            if (last_start == std::string::npos) {
                stream_buffer_.clear();
            } else if (last_start > 0U) {
                stream_buffer_.erase(stream_buffer_.begin(),
                                     stream_buffer_.begin() +
                                         static_cast<std::ptrdiff_t>(last_start));
            }
        }
        return false;
    }

    jpeg.assign(stream_buffer_.begin() + static_cast<std::ptrdiff_t>(latest_start),
                stream_buffer_.begin() + static_cast<std::ptrdiff_t>(latest_end));
    stream_buffer_.erase(stream_buffer_.begin(),
                         stream_buffer_.begin() + static_cast<std::ptrdiff_t>(latest_end));
    return true;
}

void NetworkCameraSource::close_client_locked()
{
#if defined(__linux__)
    if (client_fd_ >= 0) {
        (void)shutdown(client_fd_, SHUT_RDWR);
        close(client_fd_);
        client_fd_ = -1;
    }
#endif
    stream_buffer_.clear();
}

bool NetworkCameraSource::read(cv::Mat& frame)
{
    frame.release();
#if !defined(__linux__)
    return false;
#else
    std::vector<unsigned char> jpeg;
    std::array<unsigned char, kReceiveChunkSize> receive_buffer{};

    for (;;) {
        if (stop_requested_.load()) {
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(lifecycle_mutex_);
            if (extract_latest_jpeg(jpeg)) {
                frame = cv::imdecode(jpeg, cv::IMREAD_COLOR);
                if (frame.empty()) {
                    return false;
                }
                if (frame.cols != 1280 || frame.rows != 720 || frame.type() != CV_8UC3) {
                    throw std::runtime_error("network camera JPEG must decode to 1280x720 "
                                             "BGR CV_8UC3; got " +
                                             std::to_string(frame.cols) + "x" +
                                             std::to_string(frame.rows) + " type=" +
                                             std::to_string(frame.type()));
                }
                return true;
            }
        }

        int server_fd = -1;
        int client_fd = -1;
        {
            std::lock_guard<std::mutex> lock(lifecycle_mutex_);
            server_fd = server_fd_;
            client_fd = client_fd_;
        }
        if (server_fd < 0) {
            return false;
        }

        if (client_fd < 0) {
            pollfd server_poll{};
            server_poll.fd = server_fd;
            server_poll.events = POLLIN;
            const int result = poll(&server_poll, 1, kPollTimeoutMs);
            if (result <= 0) {
                return false;
            }
            if ((server_poll.revents & POLLIN) == 0) {
                return false;
            }

            const int accepted = accept(server_fd, nullptr, nullptr);
            if (accepted < 0) {
                return false;
            }
            if (set_nonblocking(accepted) < 0) {
                close(accepted);
                return false;
            }
            std::lock_guard<std::mutex> lock(lifecycle_mutex_);
            if (stop_requested_.load() || server_fd_ != server_fd) {
                shutdown(accepted, SHUT_RDWR);
                close(accepted);
                return false;
            }
            close_client_locked();
            client_fd_ = accepted;
            continue;
        }

        pollfd client_poll{};
        client_poll.fd = client_fd;
        client_poll.events = POLLIN;
        const int result = poll(&client_poll, 1, kPollTimeoutMs);
        if (result == 0) {
            return false;
        }
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if ((client_poll.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
            std::lock_guard<std::mutex> lock(lifecycle_mutex_);
            if (client_fd_ == client_fd) {
                close_client_locked();
            }
            return false;
        }

        const ssize_t received = recv(client_fd, receive_buffer.data(), receive_buffer.size(), 0);
        if (received > 0) {
            std::lock_guard<std::mutex> lock(lifecycle_mutex_);
            stream_buffer_.insert(stream_buffer_.end(), receive_buffer.begin(),
                                  receive_buffer.begin() + received);
            continue;
        }
        if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) {
            continue;
        }

        std::lock_guard<std::mutex> lock(lifecycle_mutex_);
        if (client_fd_ == client_fd) {
            close_client_locked();
        }
        return false;
    }
#endif
}

void NetworkCameraSource::release()
{
    stop_requested_.store(true);
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    close_client_locked();
#if defined(__linux__)
    if (server_fd_ >= 0) {
        close(server_fd_);
        server_fd_ = -1;
    }
#else
    server_fd_ = -1;
#endif
}

}  // namespace edgevision
