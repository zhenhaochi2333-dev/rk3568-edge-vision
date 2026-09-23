/* 可选板载 V4L2 路径：1280x720 NV12 单平面 MMAP -> 去 stride 拷贝 -> OpenCV BGR。
 * 与正式演示的 PC JPEG/TCP 输入不同，但提供相同的 BGR 帧契约。
 */
#include "edgevision/camera_source.hpp"

#include <opencv2/imgproc.hpp>

#include <array>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <utility>

#ifdef __linux__
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace edgevision {

namespace {

#ifdef __linux__

constexpr std::uint32_t kCameraWidth = 1280U;
constexpr std::uint32_t kCameraHeight = 720U;
constexpr std::uint32_t kRequestedBufferCount = 4U;
constexpr std::uint32_t kMaxPlanes = VIDEO_MAX_PLANES;

int xioctl(int fd, unsigned long request, void* argument)
{
    // 信号中断不会让设备请求失效，因此只在 errno 为 EINTR 时原样重试。
    int result = 0;
    do {
        result = ::ioctl(fd, request, argument);
    } while (result < 0 && errno == EINTR);
    return result;
}

std::string errno_message(const std::string& operation)
{
    return operation + ": " + std::strerror(errno);
}

#endif

}  // namespace

CameraSource::CameraSource(const std::string& device)
    : device_(device), pipeline_(make_pipeline(device))
{
    if (device_.empty()) {
        throw std::runtime_error("camera device path must not be empty");
    }
}

CameraSource::~CameraSource()
{
    release();
}

std::string CameraSource::make_pipeline(const std::string& device)
{
    return "v4l2-mmap device=" + device +
           " api=VIDEO_CAPTURE_MPLANE format=NV12 width=1280 height=720 buffers=4";
}

// 协商多平面采集 API 中的单个 NV12 内存平面，并把驱动缓冲区映射到进程地址空间。
void CameraSource::open()
{
#ifndef __linux__
    throw std::runtime_error("Direct V4L2 camera requires Linux");
#else
    // 已完成初始化时直接返回，避免重复申请队列或重复 STREAMON。
    if (fd_ >= 0 && streaming_) {
        return;
    }

    // 非阻塞设备在暂时没有帧时会以 EAGAIN 返回，等待逻辑由 dequeue_buffer 处理。
    fd_ = ::open(device_.c_str(), O_RDWR | O_NONBLOCK);
    if (fd_ < 0) {
        throw std::runtime_error(errno_message("open " + device_));
    }

    try {
        v4l2_capability capability{};
        if (xioctl(fd_, VIDIOC_QUERYCAP, &capability) < 0) {
            throw std::runtime_error(errno_message("VIDIOC_QUERYCAP"));
        }
        // 新驱动把有效能力放在 device_caps；旧驱动则直接使用 capabilities 位图。
        const std::uint32_t device_caps =
            (capability.capabilities & V4L2_CAP_DEVICE_CAPS) != 0U
                ? capability.device_caps
                : capability.capabilities;
        if ((device_caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE) == 0U ||
            (device_caps & V4L2_CAP_STREAMING) == 0U) {
            throw std::runtime_error(
                "camera does not provide multi-planar streaming capture");
        }

        v4l2_format format{};
        format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        format.fmt.pix_mp.width = kCameraWidth;
        format.fmt.pix_mp.height = kCameraHeight;
        format.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
        format.fmt.pix_mp.field = V4L2_FIELD_NONE;
        if (xioctl(fd_, VIDIOC_S_FMT, &format) < 0) {
            throw std::runtime_error(errno_message("VIDIOC_S_FMT NV12"));
        }
        if (format.fmt.pix_mp.width != kCameraWidth ||
            format.fmt.pix_mp.height != kCameraHeight ||
            format.fmt.pix_mp.pixelformat != V4L2_PIX_FMT_NV12) {
            throw std::runtime_error("camera did not keep requested 1280x720 NV12 format");
        }
        // 必须以 S_FMT 返回值为准，因为驱动可以调整请求；后续 stride/size 均来自协商结果。
        plane_count_ = format.fmt.pix_mp.num_planes;
        if (plane_count_ != 1U) {
            throw std::runtime_error("camera NV12 format is not single-plane");
        }
        // stride 可大于可见宽度；Y 与交错 UV 行均按 stride 存放。
        bytes_per_line_ = format.fmt.pix_mp.plane_fmt[0].bytesperline;
        if (bytes_per_line_ < kCameraWidth) {
            throw std::runtime_error("camera NV12 stride is smaller than image width");
        }
        // NV12 的 UV 交错数据只有 Y 平面的一半高度；每行仍按 stride 存储，
        // 因此存放 720+360 行的缓冲最小需要 stride * height * 3/2 字节。
        const std::size_t required_bytes =
            bytes_per_line_ * static_cast<std::size_t>(kCameraHeight) * 3U / 2U;
        if (format.fmt.pix_mp.plane_fmt[0].sizeimage < required_bytes) {
            throw std::runtime_error("camera NV12 plane is smaller than its stride geometry");
        }

        v4l2_requestbuffers request{};
        request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        request.memory = V4L2_MEMORY_MMAP;
        request.count = kRequestedBufferCount;
        if (xioctl(fd_, VIDIOC_REQBUFS, &request) < 0) {
            throw std::runtime_error(errno_message("VIDIOC_REQBUFS"));
        }
        if (request.count < 2U) {
            throw std::runtime_error("camera returned too few MMAP buffers");
        }
        // 使用驱动实际返回的 count 建立槽位表；每个 index 后续对应同一映射缓冲。
        buffers_.resize(request.count);
        for (std::uint32_t index = 0U; index < request.count; ++index) {
            v4l2_buffer buffer{};
            std::array<v4l2_plane, kMaxPlanes> planes{};
            buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            buffer.memory = V4L2_MEMORY_MMAP;
            buffer.index = index;
            buffer.length = kMaxPlanes;
            buffer.m.planes = planes.data();
            if (xioctl(fd_, VIDIOC_QUERYBUF, &buffer) < 0) {
                throw std::runtime_error(errno_message("VIDIOC_QUERYBUF"));
            }
            if (buffer.length != plane_count_) {
                throw std::runtime_error("camera buffer plane count differs from format");
            }

            // QUERYBUF 返回队列槽位的 plane 长度和 mmap 偏移；MAP_SHARED 让进程
            // 直接读驱动缓冲。后续仍逐行复制可见 NV12 字节以去掉硬件 stride。
            MappedBuffer mapped;
            mapped.addresses.resize(buffer.length, nullptr);
            mapped.lengths.resize(buffer.length, 0U);
            for (std::uint32_t plane = 0U; plane < buffer.length; ++plane) {
                void* address = ::mmap(nullptr, planes[plane].length,
                                       PROT_READ | PROT_WRITE, MAP_SHARED, fd_,
                                       planes[plane].m.mem_offset);
                if (address == MAP_FAILED) {
                    throw std::runtime_error(errno_message("mmap camera plane"));
                }
                mapped.addresses[plane] = address;
                mapped.lengths[plane] = planes[plane].length;
            }
            buffers_[index] = std::move(mapped);
        }
        // 将所有 MMAP buffer 先交回驱动，STREAMON 后才能循环 DQBUF/QBUF。
        for (std::uint32_t index = 0U; index < buffers_.size(); ++index) {
            queue_buffer(index);
        }

        // Mat 行宽只取可见宽度；read() 会逐行去掉映射中的 stride padding。
        nv12_buffer_.create(static_cast<int>(kCameraHeight * 3U / 2U),
                            static_cast<int>(kCameraWidth), CV_8UC1);
        info_.width = static_cast<int>(kCameraWidth);
        info_.height = static_cast<int>(kCameraHeight);
        info_.fps = 30.0;
        info_.backend = "V4L2-MMAP";
        info_.pixel_format = "NV12";
        info_.plane_count = static_cast<int>(plane_count_);
        info_.bytes_per_line = bytes_per_line_;

        // 所有资源和初始队列准备成功后才启动硬件；启动失败由 catch 统一回滚。
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        if (xioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
            throw std::runtime_error(errno_message("VIDIOC_STREAMON"));
        }
        streaming_ = true;
    } catch (...) {
        release();
        throw;
    }
#endif
}

// DQBUF 借用一块驱动内存；拷贝可见 NV12 字节并转 BGR 后立即 QBUF 归还。
bool CameraSource::read(cv::Mat& frame)
{
    // 失败时调用方不能误把上一次的图像当成本次采集结果。
    frame.release();
#ifndef __linux__
    return false;
#else
    if (fd_ < 0 || !streaming_ || plane_count_ != 1U || nv12_buffer_.empty()) {
        return false;
    }

    std::uint32_t index = 0U;
    std::size_t bytes_used = 0U;
    if (!dequeue_buffer(index, bytes_used)) {
        return false;
    }

    try {
        const std::size_t required_bytes =
            bytes_per_line_ * static_cast<std::size_t>(kCameraHeight) * 3U / 2U;
        if (index >= buffers_.size() || buffers_[index].addresses.empty() ||
            buffers_[index].lengths[0] < required_bytes ||
            (bytes_used != 0U && bytes_used < required_bytes)) {
            throw std::runtime_error("camera returned an incomplete NV12 frame");
        }
        // DQBUF 后该槽位暂归 CPU 使用；完成拷贝前驱动不会重写它的图像内容。
        const auto* source = static_cast<const std::uint8_t*>(buffers_[index].addresses[0]);
        // 去掉每行硬件 padding，形成 OpenCV 期望的紧凑 720*3/2 行 NV12 Mat。
        for (std::uint32_t row = 0U; row < kCameraHeight * 3U / 2U; ++row) {
            std::memcpy(nv12_buffer_.ptr(static_cast<int>(row)),
                        source + static_cast<std::size_t>(row) * bytes_per_line_, kCameraWidth);
        }
        cv::cvtColor(nv12_buffer_, frame, cv::COLOR_YUV2BGR_NV12);
    } catch (...) {
        try {
            queue_buffer(index);
        } catch (...) {
        }
        throw;
    }
    // 颜色转换只读取紧凑副本，不再引用映射地址，此时可以把槽位归还给驱动。
    queue_buffer(index);
    return !frame.empty();
#endif
}

// 先停视频流，再解除映射、释放驱动队列和 fd；析构时也走相同路径。
void CameraSource::release()
{
#ifdef __linux__
    // 先停止硬件写入，再解除用户态映射，避免释放仍处于采集队列中的内存。
    if (streaming_) {
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        xioctl(fd_, VIDIOC_STREAMOFF, &type);
        streaming_ = false;
    }

    // 逐 plane 解除成功建立的 mmap；未成功映射的空地址会被安全跳过。
    for (MappedBuffer& buffer : buffers_) {
        for (std::size_t plane = 0U; plane < buffer.addresses.size(); ++plane) {
            if (buffer.addresses[plane] != nullptr && buffer.addresses[plane] != MAP_FAILED) {
                ::munmap(buffer.addresses[plane], buffer.lengths[plane]);
            }
            buffer.addresses[plane] = nullptr;
            buffer.lengths[plane] = 0U;
        }
    }
    if (fd_ >= 0) {
        // count=0 通知驱动释放已申请的 MMAP 队列，随后关闭本次设备会话。
        v4l2_requestbuffers request{};
        request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        request.memory = V4L2_MEMORY_MMAP;
        request.count = 0U;
        xioctl(fd_, VIDIOC_REQBUFS, &request);
        ::close(fd_);
        fd_ = -1;
    }
#else
    fd_ = -1;
    streaming_ = false;
#endif
    buffers_.clear();
    plane_count_ = 0U;
    bytes_per_line_ = 0U;
    nv12_buffer_.release();
}

#ifdef __linux__

void CameraSource::queue_buffer(std::uint32_t index)
{
    if (fd_ < 0 || index >= buffers_.size()) {
        throw std::runtime_error("cannot queue invalid camera buffer");
    }
    // QBUF 把槽位所有权交回驱动；从此到下一次 DQBUF 之间应用不应读取它。
    v4l2_buffer buffer{};
    std::array<v4l2_plane, kMaxPlanes> planes{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.index = index;
    buffer.length = plane_count_;
    buffer.m.planes = planes.data();
    if (xioctl(fd_, VIDIOC_QBUF, &buffer) < 0) {
        throw std::runtime_error(errno_message("VIDIOC_QBUF"));
    }
}

bool CameraSource::dequeue_buffer(std::uint32_t& index, std::size_t& bytes_used)
{
    for (;;) {
        // DQBUF 与 QBUF 必须使用匹配的队列、内存类型和 plane 数。
        v4l2_buffer buffer{};
        std::array<v4l2_plane, kMaxPlanes> planes{};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.length = plane_count_;
        buffer.m.planes = planes.data();
        if (xioctl(fd_, VIDIOC_DQBUF, &buffer) == 0) {
            if (buffer.index >= buffers_.size() || buffer.length != plane_count_) {
                throw std::runtime_error("camera returned invalid dequeued buffer");
            }
            index = buffer.index;
            bytes_used = 0U;
            for (std::uint32_t plane = 0U; plane < buffer.length; ++plane) {
                bytes_used += planes[plane].bytesused;
            }
            return true;
        }
        if (errno != EAGAIN) {
            throw std::runtime_error(errno_message("VIDIOC_DQBUF"));
        }
        // 非阻塞 fd 在队列暂空时 EAGAIN；短暂休眠避免空转占满 CPU。
        ::usleep(1000);
    }
}

#else

void CameraSource::queue_buffer(std::uint32_t)
{
    throw std::runtime_error("Direct V4L2 camera requires Linux");
}

bool CameraSource::dequeue_buffer(std::uint32_t&, std::size_t&)
{
    return false;
}

#endif

}  // namespace edgevision
