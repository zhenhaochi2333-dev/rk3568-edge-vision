/* Windows 侧启动三个 FFmpeg 子进程：JPEG/TCP 发送、原始预览、RTSP 检测预览。
 * 发送与两个窗口分别运行；PC 侧不会直接执行 RKNN 推理。
 */
#ifdef _WIN32

#include <windows.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

std::atomic<bool> stop_requested{false};

BOOL WINAPI console_handler(DWORD event)
{
    switch (event) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
    case CTRL_LOGOFF_EVENT:
    case CTRL_SHUTDOWN_EVENT:
        stop_requested.store(true);
        return TRUE;
    default:
        return FALSE;
    }
}

std::string quote(const std::string& value)
{
    std::string result = "\"";
    for (const char character : value) {
        if (character == '"') {
            result += "\\\"";
        } else {
            result += character;
        }
    }
    result += "\"";
    return result;
}

// 保存 CreateProcess 返回的进程/线程内核句柄。started 代表句柄仍由本结构负责关闭，
// 不是“进程一定还活着”；has_exited 会向系统查询实时状态。
struct ChildProcess {
    std::string name;
    PROCESS_INFORMATION process{};
    bool started = false;
};

// 每条媒体支路作为独立 FFmpeg 进程启动，SDL 窗口生命周期因此不会直接销毁其他支路。
// CreateProcess 需要可修改的命令行缓冲区，所以这里构造本地 string 并传 data()。
bool start_child(ChildProcess& child, const std::string& executable,
                 const std::string& arguments)
{
    std::string command_line = quote(executable) + " " + arguments;
    STARTUPINFOA startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESHOWWINDOW;
    startup.wShowWindow = SW_SHOWNORMAL;

    if (!CreateProcessA(nullptr, command_line.data(), nullptr, nullptr, FALSE,
                        CREATE_NO_WINDOW, nullptr, nullptr, &startup,
                        &child.process)) {
        std::cerr << "cannot start " << child.name << ": Win32 error "
                  << GetLastError() << "\n";
        return false;
    }
    child.started = true;
    return true;
}

// 零超时轮询，不等待子进程；WAIT_OBJECT_0 表示已结束，便于主循环定期响应 Ctrl+C。
bool has_exited(const ChildProcess& child)
{
    if (!child.started) {
        return true;
    }
    return WaitForSingleObject(child.process.hProcess, 0) == WAIT_OBJECT_0;
}

// 若仍运行则先结束并等待最多 3 秒，随后无论自然退出还是强制结束都关闭句柄。
// CloseHandle 只释放父进程持有的句柄，不会在子进程运行时自动终止它。
void stop_child(ChildProcess& child)
{
    if (!child.started) {
        return;
    }
    if (!has_exited(child)) {
        TerminateProcess(child.process.hProcess, 0);
        WaitForSingleObject(child.process.hProcess, 3000);
    }
    CloseHandle(child.process.hThread);
    CloseHandle(child.process.hProcess);
    child.started = false;
}

// PC 输入采集、板端 JPEG/TCP 入口和板端 RTSP 预览的默认连接参数。
// 摄像头发送与原始预览各自打开 DirectShow 设备，因此驱动需允许两个并发采集句柄；
// 检测预览从板端 RTSP 读取，不会再打开本机摄像头。
struct Options {
    std::string ffmpeg = R"(D:\EVCapture\ffmpeg.exe)";
    std::string camera = "Integrated Camera";
    std::string board = "192.168.77.2";
    int input_port = 5600;
    int rtsp_port = 8554;
};

bool read_option(int& index, int argc, char** argv, const char* name,
                 std::string& target)
{
    if (std::string(argv[index]) != name || index + 1 >= argc) {
        return false;
    }
    target = argv[++index];
    return true;
}

int parse_int(const std::string& value, const char* name)
{
    char* end = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0' || parsed < 1 || parsed > 65535) {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return static_cast<int>(parsed);
}

Options parse_options(int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; ++index) {
        if (read_option(index, argc, argv, "--ffmpeg", options.ffmpeg) ||
            read_option(index, argc, argv, "--camera", options.camera) ||
            read_option(index, argc, argv, "--board", options.board)) {
            continue;
        }
        std::string value;
        if (read_option(index, argc, argv, "--input-port", value)) {
            options.input_port = parse_int(value, "--input-port");
        } else if (read_option(index, argc, argv, "--rtsp-port", value)) {
            options.rtsp_port = parse_int(value, "--rtsp-port");
        } else if (std::string(argv[index]) == "--help" ||
                   std::string(argv[index]) == "-h") {
            std::cout << "edgevision_pc_bridge [--ffmpeg PATH] [--camera NAME] "
                         "[--board IP] [--input-port PORT] [--rtsp-port PORT]\n";
            std::exit(0);
        } else {
            throw std::runtime_error(std::string("unknown option: ") + argv[index]);
        }
    }
    return options;
}

// DirectShow 以 1280x720 NV12/30 FPS 打开摄像头；发送和原始预览
// 各自创建采集进程，发送支路再通过 fps 滤镜降为 15 FPS MJPEG。
// 两次采集不共用单一视频流，不能假设原始与检测画面逐帧同步。
std::string camera_input(const Options& options)
{
    return "-hide_banner -loglevel warning -rtbufsize 64M "
           "-f dshow -video_size 1280x720 -framerate 30 -pixel_format nv12 -i " +
           quote("video=" + options.camera);
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Options options = parse_options(argc, argv);
        if (!SetConsoleCtrlHandler(console_handler, TRUE)) {
            throw std::runtime_error("cannot install console handler");
        }

        const std::string camera = camera_input(options);
        const std::string sender_url =
            "tcp://" + options.board + ":" + std::to_string(options.input_port) +
            "?tcp_nodelay=1";
        const std::string rtsp_url =
            "rtsp://" + options.board + ":" + std::to_string(options.rtsp_port) + "/live";

        // 下面三条支路各自启动 FFmpeg：发送 MJPEG 到板端；本机显示原始采集；
        // 从板端 RTSP 解码显示检测后画面。前两路分别打开摄像头，检测预览只连接 RTSP。
        // These are deliberately independent processes. Closing either SDL
        // preview cannot close the camera sender or the board application.
        ChildProcess sender{"camera sender"};
        ChildProcess raw_preview{"raw camera preview"};
        ChildProcess detection_preview{"board detection preview"};

        // -f mjpeg 把连续 JPEG 帧写到板端 TCP :5600；接收端按 JPEG 标记恢复帧边界。
        // sender_started/raw_started 只表示 CreateProcess 创建成功；FFmpeg 也可能随后
        // 因设备占用、网络断开或参数错误退出，故下面还要轮询检测预览，并在主循环看 sender。
        const bool sender_started = start_child(
            sender, options.ffmpeg,
            camera + " -vf fps=15 -q:v 3 -an -f mjpeg " + quote(sender_url));
        const bool raw_started = start_child(
            raw_preview, options.ffmpeg,
            camera + " -vf format=yuv420p -f sdl " + quote("EdgeVision Raw Camera"));
        // The board needs a few encoded frames before RTSP can advertise the
        // H.264 dimensions. Starting the viewer at the same instant as the
        // sender makes FFmpeg occasionally exit with "unspecified size".
        // 等待板端编码器至少产出可探测尺寸的 H.264 数据。固定等待是启动时序缓冲，
        // 并不构成服务就绪握手；网络或推理启动更慢时仍可能需要后续重试。
        std::this_thread::sleep_for(std::chrono::seconds(3));
        // 预览读取的是板端重新编码的 RTSP H.264 结果，与原始发送支路独立。
        bool detection_started = false;
        // RTSP 输入探测可能在流尚无 SPS/PPS 或分辨率信息时失败。若进程在 1.5 秒观察期
        // 内存活就视为启动成功；最多尝试四次，并在每次失败间隔一秒。
        for (int attempt = 0; attempt < 4 && !stop_requested.load(); ++attempt) {
            if (start_child(
                    detection_preview, options.ffmpeg,
                    "-rtsp_transport tcp -analyzeduration 5M -probesize 32M -i " +
                        quote(rtsp_url) + " -vf format=yuv420p -f sdl " +
                        quote("EdgeVision PC Detection"))) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1500));
                if (!has_exited(detection_preview)) {
                    detection_started = true;
                    break;
                }
                stop_child(detection_preview);
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        if (!sender_started || !raw_started || !detection_started) {
            stop_child(detection_preview);
            stop_child(raw_preview);
            stop_child(sender);
            return 1;
        }

        std::cout << "PC bridge started. Raw and detection windows are independent.\n"
                  << "Press Ctrl+C to stop all three processes.\n";
        // 发送支路决定整个桥接器是否仍有采集/传输任务；两个预览提前退出不会停止发送。
        // Ctrl+C 等控制台事件只设置原子标志，进程清理由此主线程串行完成。
        while (!stop_requested.load() && !has_exited(sender)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        if (has_exited(raw_preview)) {
            std::cout << "raw preview closed; sender remains active.\n";
        }
        if (has_exited(detection_preview)) {
            std::cout << "detection preview closed; sender remains active.\n";
        }
        stop_child(detection_preview);
        stop_child(raw_preview);
        stop_child(sender);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "edgevision_pc_bridge: " << error.what() << "\n";
        return 1;
    }
}

#else

int main()
{
    return 1;
}

#endif
