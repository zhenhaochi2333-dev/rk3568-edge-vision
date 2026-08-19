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

struct ChildProcess {
    std::string name;
    PROCESS_INFORMATION process{};
    bool started = false;
};

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

bool has_exited(const ChildProcess& child)
{
    if (!child.started) {
        return true;
    }
    return WaitForSingleObject(child.process.hProcess, 0) == WAIT_OBJECT_0;
}

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

        // These are deliberately independent processes. Closing either SDL
        // preview cannot close the camera sender or the board application.
        ChildProcess sender{"camera sender"};
        ChildProcess raw_preview{"raw camera preview"};
        ChildProcess detection_preview{"board detection preview"};

        const bool sender_started = start_child(
            sender, options.ffmpeg,
            camera + " -vf fps=15 -q:v 3 -an -f mjpeg " + quote(sender_url));
        const bool raw_started = start_child(
            raw_preview, options.ffmpeg,
            camera + " -vf format=yuv420p -f sdl " + quote("EdgeVision Raw Camera"));
        // The board needs a few encoded frames before RTSP can advertise the
        // H.264 dimensions. Starting the viewer at the same instant as the
        // sender makes FFmpeg occasionally exit with "unspecified size".
        std::this_thread::sleep_for(std::chrono::seconds(3));
        bool detection_started = false;
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
