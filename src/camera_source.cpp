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

void CameraSource::open()
{
#ifndef __linux__
    throw std::runtime_error("Direct V4L2 camera requires Linux");
#else
    if (fd_ >= 0 && streaming_) {
        return;
    }

    fd_ = ::open(device_.c_str(), O_RDWR | O_NONBLOCK);
    if (fd_ < 0) {
        throw std::runtime_error(errno_message("open " + device_));
    }

    try {
        v4l2_capability capability{};
        if (xioctl(fd_, VIDIOC_QUERYCAP, &capability) < 0) {
            throw std::runtime_error(errno_message("VIDIOC_QUERYCAP"));
        }
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
        plane_count_ = format.fmt.pix_mp.num_planes;
        if (plane_count_ != 1U) {
            throw std::runtime_error("camera NV12 format is not single-plane");
        }
        bytes_per_line_ = format.fmt.pix_mp.plane_fmt[0].bytesperline;
        if (bytes_per_line_ < kCameraWidth) {
            throw std::runtime_error("camera NV12 stride is smaller than image width");
        }
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
        for (std::uint32_t index = 0U; index < buffers_.size(); ++index) {
            queue_buffer(index);
        }

        nv12_buffer_.create(static_cast<int>(kCameraHeight * 3U / 2U),
                            static_cast<int>(kCameraWidth), CV_8UC1);
        info_.width = static_cast<int>(kCameraWidth);
        info_.height = static_cast<int>(kCameraHeight);
        info_.fps = 30.0;
        info_.backend = "V4L2-MMAP";
        info_.pixel_format = "NV12";
        info_.plane_count = static_cast<int>(plane_count_);
        info_.bytes_per_line = bytes_per_line_;

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

bool CameraSource::read(cv::Mat& frame)
{
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
        const auto* source = static_cast<const std::uint8_t*>(buffers_[index].addresses[0]);
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
    queue_buffer(index);
    return !frame.empty();
#endif
}

void CameraSource::release()
{
#ifdef __linux__
    if (streaming_) {
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        xioctl(fd_, VIDIOC_STREAMOFF, &type);
        streaming_ = false;
    }

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
