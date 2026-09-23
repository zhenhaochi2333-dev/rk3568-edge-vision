#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <string>

namespace edgevision {

// 输入容器/后端报告的元数据快照。帧宽高、FPS 和总帧数来自 VideoCapture 属性，
// 某些直播流或不完整文件可能报告 0；它们是描述信息，不保证能作为精确播放时钟。
struct VideoSourceInfo {
    int width = 0;
    int height = 0;
    double fps = 0.0;
    long long frame_count = 0;
    std::string backend;
};

// 实际打开输出后的记录：requested_path 是用户请求，actual_path/codec 是最终采用值。
// 当 mp4v 打不开而回退到 MJPG 时，调用方应展示 actual_path，避免误报文件位置或编码。
struct VideoWriterInfo {
    std::string requested_path;
    std::string actual_path;
    std::string codec;
    double fps = 0.0;
    cv::Size resolution;
};

// 对 OpenCV VideoCapture/VideoWriter 的轻量 RAII 封装。输入和输出句柄各自持有资源；
// 对象不可复制，析构时释放句柄，显式 close_output() 可让容器尾部索引及时写完。
class VideoIO {
public:
    VideoIO() = default;
    ~VideoIO();

    VideoIO(const VideoIO&) = delete;
    VideoIO& operator=(const VideoIO&) = delete;

    // 打开文件/媒体源并采集后端报告的元数据；失败时抛异常，不能把“未打开”当作 EOF。
    void open_input(const std::string& path);
    // 成功时由 OpenCV 将解码帧写入 frame（通常为 BGR）；false 表示当前读取未得到帧，
    // 文件模式下常见于 EOF。复用同一个 Mat 可让 OpenCV 重用其像素缓冲区。
    bool read(cv::Mat& frame);
    // 输出尺寸必须为正；非正 FPS 按 30 帧/秒处理。除 force_overwrite 为真外，
    // 会先拒绝覆盖目标；请求 .mp4 时先尝试 mp4v，失败才尝试同基名 *_mjpg.avi。
    // force_mjpg_for_test 只用于可重复测试回退分支，生产调用应保持默认 false。
    void open_output(const std::string& requested_path, double fps, cv::Size resolution,
                     bool force_overwrite, bool force_mjpg_for_test = false);
    // 把一帧交给已打开的编码器；帧尺寸/通道应与 open_output 的参数一致。
    // VideoWriter::write 通常不提供逐帧编码失败返回值，容器最终状态要结合关闭/重读检查。
    void write(const cv::Mat& frame);
    // 释放编码器并完成容器收尾；在读取生成文件或报告处理成功前应先关闭输出。
    void close_output();

    bool input_open() const { return capture_.isOpened(); }
    bool output_open() const { return writer_.isOpened(); }
    const VideoSourceInfo& source_info() const { return source_info_; }
    const VideoWriterInfo& writer_info() const { return writer_info_; }

private:
    static bool exists(const std::string& path);
    static std::string lowercase_extension(const std::string& path);
    static std::string fallback_path(const std::string& path);
    static void reject_existing(const std::string& path, bool force_overwrite);

    // 句柄拥有各自底层 OpenCV/系统后端资源；Mat 帧通过 read 输出给调用方管理。
    cv::VideoCapture capture_;
    cv::VideoWriter writer_;
    VideoSourceInfo source_info_;
    VideoWriterInfo writer_info_;
};

}  // namespace edgevision
