#include "edgevision/video_io.hpp"

#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <sys/stat.h>

namespace edgevision {

// 即使调用方忘记显式关闭，析构也释放容器/设备资源；先收尾 writer 以写出索引，
// 再释放 capture。显式 close_output 仍适用于需要立刻验证文件的路径。
VideoIO::~VideoIO()
{
    close_output();
    capture_.release();
}

bool VideoIO::exists(const std::string& path)
{
    struct stat info{};
    return stat(path.c_str(), &info) == 0;
}

std::string VideoIO::lowercase_extension(const std::string& path)
{
    const std::size_t slash = path.find_last_of("/\\");
    const std::size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        return std::string();
    }
    std::string extension = path.substr(dot);
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return extension;
}

// 回退文件沿用请求路径的目录和主文件名，只把扩展名替换成 _mjpg.avi；
// 没有扩展名时则追加后缀。这样请求 output.mp4 对应 output_mjpg.avi。
std::string VideoIO::fallback_path(const std::string& path)
{
    const std::size_t slash = path.find_last_of("/\\");
    const std::size_t dot = path.find_last_of('.');
    const std::size_t end = dot == std::string::npos || (slash != std::string::npos && dot < slash)
                                ? path.size()
                                : dot;
    return path.substr(0, end) + "_mjpg.avi";
}

// 这是防止意外覆盖的前置检查，不是原子性的“创建新文件”操作；
// 在检查和 VideoWriter 打开之间若有别的进程创建同一路径，仍可能发生竞争。
void VideoIO::reject_existing(const std::string& path, bool force_overwrite)
{
    if (exists(path) && !force_overwrite) {
        throw std::runtime_error("output already exists; pass --force to replace it: " + path);
    }
}

void VideoIO::open_input(const std::string& path)
{
    // OpenCV 选择可用的媒体后端并尝试打开路径；打开失败要与 read() 到达文件末尾区分。
    capture_.open(path);
    if (!capture_.isOpened()) {
        throw std::runtime_error("cannot open video input: " + path);
    }
    // 属性由后端报告，容器损坏或直播输入时可能不精确；调用方仍应以读出的 Mat.size()
    // 作为实际帧尺寸。frame_count/FPS 也不能替代逐帧读取结果判断 EOF。
    source_info_.width = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
    source_info_.height = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
    source_info_.fps = capture_.get(cv::CAP_PROP_FPS);
    source_info_.frame_count = static_cast<long long>(capture_.get(cv::CAP_PROP_FRAME_COUNT));
    source_info_.backend = capture_.getBackendName();
}

bool VideoIO::read(cv::Mat& frame)
{
    // VideoCapture 在 frame 上写入解码结果；返回 false 通常意味着 EOF 或输入暂时不可读，
    // 具体语义取决于文件/设备后端，VideoIO 不在这里合成或缓存额外帧。
    return capture_.read(frame);
}

void VideoIO::open_output(const std::string& requested_path, double fps, cv::Size resolution,
                          bool force_overwrite, bool force_mjpg_for_test)
{
    if (fps <= 0.0) {
        fps = 30.0;
    }
    if (resolution.width <= 0 || resolution.height <= 0) {
        throw std::runtime_error("invalid video output resolution");
    }
    reject_existing(requested_path, force_overwrite);

    // OpenCV 的编码器可用性取决于当前构建所带的后端/插件。先检查请求名，再按实际
    // 打开结果决定 writer_info；不能仅凭扩展名推断编码成功。
    const bool requested_mp4 = lowercase_extension(requested_path) == ".mp4";
    // 对 MP4 首选常见的 MPEG-4 Part 2 fourcc。若后端不支持，尝试 AVI/MJPEG；
    // 回退目标也需遵守覆盖保护，避免“MP4 失败”时静默覆盖已有 AVI。
    if (requested_mp4 && !force_mjpg_for_test) {
        writer_.open(requested_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                     resolution, true);
        if (writer_.isOpened()) {
            writer_info_ = VideoWriterInfo{requested_path, requested_path, "mp4v", fps, resolution};
            return;
        }
        const std::string fallback = fallback_path(requested_path);
        reject_existing(fallback, force_overwrite);
        writer_.open(fallback, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, resolution, true);
        if (writer_.isOpened()) {
            writer_info_ = VideoWriterInfo{requested_path, fallback, "MJPG", fps, resolution};
            return;
        }
    } else {
        // 非 MP4 直接尝试 MJPG；测试开关强制 MP4 请求进入此支路，以验证回退文件名、
        // codec 元数据及可重新读取性，而不依赖测试机器是否安装 mp4v 编码器。
        const std::string output_path = requested_mp4 ? fallback_path(requested_path) : requested_path;
        if (output_path != requested_path) {
            reject_existing(output_path, force_overwrite);
        }
        writer_.open(output_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, resolution, true);
        if (writer_.isOpened()) {
            writer_info_ = VideoWriterInfo{requested_path, output_path, "MJPG", fps, resolution};
            return;
        }
    }

    throw std::runtime_error("cannot open video writer for requested output: " + requested_path);
}

// VideoWriter 通常按 open 时确定的宽高、颜色模式编码；此封装只检查句柄状态，
// 输入帧是否符合约定由调用方保证，避免逐帧复制或转换隐藏成本。
void VideoIO::write(const cv::Mat& frame)
{
    if (!writer_.isOpened()) {
        throw std::runtime_error("video writer is not open");
    }
    writer_.write(frame);
}

void VideoIO::close_output()
{
    if (writer_.isOpened()) {
        writer_.release();
    }
}

}  // namespace edgevision
