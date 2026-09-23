#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace edgevision {

struct CameraSourceInfo {
    // 记录驱动最终协商出的采集参数，便于状态接口或日志说明帧的来源与布局。
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
    // 构造只保存设备路径；设备句柄和驱动资源由 open() 延迟申请。
    explicit CameraSource(const std::string& device);
    // 析构走统一 release() 路径，确保对象生命周期结束时资源归还给内核。
    ~CameraSource();

    CameraSource(const CameraSource&) = delete;
    CameraSource& operator=(const CameraSource&) = delete;

    // 协商格式、申请并映射队列、预入队缓冲，最后启动视频流。
    void open();
    // 取得一帧并复制/转换为独立的 BGR Mat；无有效帧时返回 false。
    bool read(cv::Mat& frame);
    // 停止视频流、解除映射、释放 V4L2 队列并关闭设备，可用于失败回滚。
    void release();

    bool is_opened() const { return fd_ >= 0 && streaming_; }
    const std::string& device() const { return device_; }
    const std::string& pipeline() const { return pipeline_; }
    const CameraSourceInfo& info() const { return info_; }

private:
    // 每个驱动缓冲槽保存其 plane 对应的用户态映射地址和映射长度。
    // 当前 NV12 契约只有一个 plane，但用向量与 VIDEO_CAPTURE_MPLANE ioctl 一致。
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
    // fd_ 表示设备已打开；streaming_ 仅在 VIDIOC_STREAMON 成功后置为 true。
    int fd_ = -1;
    bool streaming_ = false;
    std::uint32_t plane_count_ = 0U;
    std::size_t bytes_per_line_ = 0U;
    // 映射在整个采集期间有效；DQBUF 只暂时把某个槽位的读取权交给应用。
    std::vector<MappedBuffer> buffers_;
    // 去掉硬件 stride padding 后的紧凑 NV12 暂存区，供 OpenCV 转换颜色格式。
    cv::Mat nv12_buffer_;
};

}  // namespace edgevision
