/*
 * 网络摄像头输入是 FFmpeg 写入 TCP :5600 的连续 MJPEG 字节，而非带长度的消息。
 * 每个 recv 的边界都与 JPEG 帧无关：从流中寻找 SOI/EOI，保留残片，优先取
 * 最新完整图像，然后由 imdecode 得到 1280x720 CV_8UC3 BGR 帧。
 * 这个 BGR 契约与板载 V4L2 输入一致，应用层无需知道采集来源。
 */
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

// 接收块大小只影响一次读的上限，不定义一帧大小；未闭合 JPEG 的
// 累计缓冲区另设 16 MiB 恢复上限，避免失去 EOI 后无限积累旧字节。
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

// 创建非阻塞监听 socket。真正的客户端连接可能在板端先启动很久后才到来；
// open 不等待发送端，read 用短 poll 周期让外层采集线程仍能响应停止。
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

    // 重启程序或重新监听同一端口时允许尽快 bind；socket 的所有权在
    // server_fd_ 接管前仍属于局部 fd，失败分支自行 close。
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
    // 此处是本输入适配器接受的固定协议参数，不是从 JPEG 或 TCP 协商出的值；
    // read 在每次 imdecode 后还会验证实际尺寸和 CV_8UC3 类型。
    info_.width = 1280;
    info_.height = 720;
    info_.fps = 15.0;
    info_.backend = "POSIX TCP/MJPEG/OpenCV";
    info_.pixel_format = "BGR";
    info_.plane_count = 1;
    info_.bytes_per_line = static_cast<std::size_t>(info_.width) * 3U;
#endif
}

// 在累计字节中逐个寻找 JPEG SOI(FF D8) 与 EOI(FF D9)。
// 若一次 recv 到了多张完整图像，只返回最后一张，旧帧被丢弃以控制显示时延；
// 最新图像之后的不完整尾巴仍留在 stream_buffer_ 等下次 recv 补齐。
bool NetworkCameraSource::extract_latest_jpeg(std::vector<unsigned char>& jpeg)
{
    // 标记搜索只在当前累计缓冲区进行；返回索引用于区分已完成帧前缀
    // 与跨 recv 尚未完成的尾部。
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

    // 没有完整 EOI 就不能调用 imdecode。超过缓冲上限时尝试从靠后的
    // SOI 重新同步；若找不到则清空旧字节，等待新的 JPEG 起点。
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

    // jpeg 获得独立字节副本，可安全传入 imdecode；从 stream_buffer_ 清除
    // 到 EOI 为止的所有前缀，包括未交付的旧完整帧。
    jpeg.assign(stream_buffer_.begin() + static_cast<std::ptrdiff_t>(latest_start),
                stream_buffer_.begin() + static_cast<std::ptrdiff_t>(latest_end));
    stream_buffer_.erase(stream_buffer_.begin(),
                         stream_buffer_.begin() + static_cast<std::ptrdiff_t>(latest_end));
    return true;
}

// 在 lifecycle_mutex_ 保护下关闭当前连接并丢弃残帧；新客户端必须
// 从自己的 SOI 开始，不能与上一条 TCP 连接的尾部拼接。
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

// read 一次最多交付一张完整图像。短时 poll 超时或断线返回 false；
// CameraCaptureThread 统计连续失败并负责重开，而这里保存跨 recv 的字节状态。
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
            // imdecode 只看抽出的完整 JPEG 副本。解码后的 BGR 像素缓冲由
            // frame(cv::Mat) 管理，不依赖下面的接收数组或压缩字节 vector。
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

        // 尚无发送端时监听 fd 等待 accept；已有连接时转为轮询客户端。
        // 两种等待都有 100 ms 上限，供外层定期检查停止/失败状态。
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
            // 重连后清除旧客户端及其残留 JPEG 前缀，再接管新 fd。
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

        // 64 KiB 只是本次读取容量。recv 可能得到任意正长度；追加到
        // stream_buffer_ 后重新检查完整 SOI/EOI，而非把本次返回当作一帧。
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

// release 可由 CameraCaptureThread 的停止路径调用：先立停止标志，
// 再在锁内 shutdown/close 连接及监听 fd，避免 read 继续等待网络数据。
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
