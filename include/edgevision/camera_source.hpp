#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace edgevision {

struct CameraSourceInfo {
    int width = 0;
    int height = 0;
    double fps = 0.0;
    std::string backend;
    std::string pixel_format;
    int plane_count = 0;
    std::size_t bytes_per_line = 0U;
};

class CameraSource {
public:
    explicit CameraSource(const std::string& device);
    ~CameraSource();

    CameraSource(const CameraSource&) = delete;
    CameraSource& operator=(const CameraSource&) = delete;

    void open();
    bool read(cv::Mat& frame);
    void release();

    bool is_opened() const { return fd_ >= 0 && streaming_; }
    const std::string& device() const { return device_; }
    const std::string& pipeline() const { return pipeline_; }
    const CameraSourceInfo& info() const { return info_; }

private:
    struct MappedBuffer {
        std::vector<void*> addresses;
        std::vector<std::size_t> lengths;
    };

    static std::string make_pipeline(const std::string& device);

    void queue_buffer(std::uint32_t index);
    bool dequeue_buffer(std::uint32_t& index, std::size_t& bytes_used);

    std::string device_;
    std::string pipeline_;
    CameraSourceInfo info_;
    int fd_ = -1;
    bool streaming_ = false;
    std::uint32_t plane_count_ = 0U;
    std::size_t bytes_per_line_ = 0U;
    std::vector<MappedBuffer> buffers_;
    cv::Mat nv12_buffer_;
};

}  // namespace edgevision
