/* Windows 侧订阅板端 :9000 的逐行 JSON 事件，只把业务字段写为本地 CSV。 */
#ifdef _WIN32

#include <winsock2.h>
#include <ws2tcpip.h>

#include <windows.h>

#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

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

struct Options {
    std::string host = "192.168.77.2";
    int port = 9000;
    std::string output = "event_log.csv";
};

// 端口只接受十进制 1..65535；strtol 的 end 检查可拒绝空值、尾随字符和溢出范围值。
int parse_port(const std::string& text)
{
    char* end = nullptr;
    const long value = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0' || value < 1 || value > 65535) {
        throw std::runtime_error("invalid --port: " + text);
    }
    return static_cast<int>(value);
}

Options parse_options(int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index] == nullptr ? "" : argv[index];
        if (argument == "--host" && index + 1 < argc) {
            options.host = argv[++index];
        } else if (argument == "--port" && index + 1 < argc) {
            options.port = parse_port(argv[++index]);
        } else if (argument == "--output" && index + 1 < argc) {
            options.output = argv[++index];
        } else if (argument == "--help" || argument == "-h") {
            std::cout << "edgevision_event_logger [--host IP] [--port PORT] "
                         "[--output CSV]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown or incomplete option: " + argument);
        }
    }
    return options;
}

// 轻量读取服务端当前协议中的字符串字段，支持常见反斜线转义的“去反斜线”处理。
// 这不是完整 JSON 解析器：它依赖固定事件消息形状，不处理嵌套对象、Unicode 转义解码，
// 也不区分字符串内容中恰好出现的同名字段；协议扩展时应换成正式 JSON 库。
std::string json_string(const std::string& line, const std::string& key, bool& found)
{
    const std::string marker = "\"" + key + "\"";
    const std::size_t key_position = line.find(marker);
    if (key_position == std::string::npos) {
        found = false;
        return {};
    }
    const std::size_t colon = line.find(':', key_position + marker.size());
    const std::size_t quote = colon == std::string::npos ? std::string::npos
                                                          : line.find('"', colon + 1U);
    if (quote == std::string::npos) {
        throw std::runtime_error("malformed JSON string field: " + key);
    }
    std::string value;
    for (std::size_t index = quote + 1U; index < line.size(); ++index) {
        if (line[index] == '"') {
            found = true;
            return value;
        }
        if (line[index] == '\\' && index + 1U < line.size()) {
            value.push_back(line[++index]);
        } else {
            value.push_back(line[index]);
        }
    }
    throw std::runtime_error("unterminated JSON string field: " + key);
}

// 读取十进制整数（含可选负号）；缺少键时以 found=false 区分默认值 -1。
// stoi 对过大整数会抛异常，由本行的事件解析捕获并报告。
int json_integer(const std::string& line, const std::string& key, bool& found)
{
    const std::string marker = "\"" + key + "\"";
    const std::size_t key_position = line.find(marker);
    if (key_position == std::string::npos) {
        found = false;
        return -1;
    }
    const std::size_t colon = line.find(':', key_position + marker.size());
    if (colon == std::string::npos) {
        throw std::runtime_error("malformed JSON integer field: " + key);
    }
    std::size_t begin = colon + 1U;
    while (begin < line.size() && std::isspace(static_cast<unsigned char>(line[begin]))) {
        ++begin;
    }
    std::size_t end = begin;
    while (end < line.size() &&
           (std::isdigit(static_cast<unsigned char>(line[end])) || line[end] == '-')) {
        ++end;
    }
    if (end == begin) {
        throw std::runtime_error("malformed JSON integer field: " + key);
    }
    found = true;
    return std::stoi(line.substr(begin, end - begin));
}

// RFC 风格 CSV 字段转义：包含逗号、引号或换行时整个字段加双引号，字段内引号加倍。
// 时间戳和数字字段由调用处单独输出；字符串类名和事件名经过此函数。
std::string csv_field(const std::string& value)
{
    if (value.find_first_of(",\"\r\n") == std::string::npos) {
        return value;
    }
    std::string escaped = "\"";
    for (const char character : value) {
        escaped += character;
        if (character == '"') {
            escaped += '"';
        }
    }
    escaped += '"';
    return escaped;
}

// 记录 PC 本地墙钟接收时间到毫秒。它用于人工对照日志，不是板端事件发生时间，
// 也不适合计算跨设备延迟：两端时钟未同步，而且数据还经历网络与缓冲。
std::string local_timestamp()
{
    const auto now = std::chrono::system_clock::now();
    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
    localtime_s(&local, &time);
    std::ostringstream output;
    output << std::put_time(&local, "%Y-%m-%d %H:%M:%S") << '.'
           << std::setfill('0') << std::setw(3) << milliseconds.count();
    return output.str();
}

// TCP send 允许短写；循环推进偏移直到订阅命令的全部字节写入，
// 这里发送以换行结尾的 ASCII 命令，换行是服务端命令解析的帧终止符。
void send_all(SOCKET socket, const std::string& message)
{
    std::size_t offset = 0U;
    while (offset < message.size()) {
        const int sent = send(socket, message.data() + offset,
                              static_cast<int>(message.size() - offset), 0);
        if (sent <= 0) {
            throw std::runtime_error("failed to send SUBSCRIBE_EVENTS");
        }
        offset += static_cast<std::size_t>(sent);
    }
}

// 先解析 IPv4/IPv6 候选地址并依次连接；getaddrinfo 返回的列表必须在成功或失败前释放。
// 接收超时只让 recv 周期性返回，以便检查 Ctrl+C；它不是连接/事件的重试策略。
SOCKET connect_to(const Options& options)
{
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* addresses = nullptr;
    const std::string port = std::to_string(options.port);
    const int lookup = getaddrinfo(options.host.c_str(), port.c_str(), &hints, &addresses);
    if (lookup != 0) {
        throw std::runtime_error("cannot resolve event host: " + options.host);
    }

    SOCKET connected = INVALID_SOCKET;
    for (addrinfo* address = addresses; address != nullptr; address = address->ai_next) {
        connected = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
        if (connected == INVALID_SOCKET) {
            continue;
        }
        if (connect(connected, address->ai_addr, static_cast<int>(address->ai_addrlen)) == 0) {
            break;
        }
        closesocket(connected);
        connected = INVALID_SOCKET;
    }
    freeaddrinfo(addresses);
    if (connected == INVALID_SOCKET) {
        throw std::runtime_error("cannot connect to " + options.host + ":" + port);
    }

    const DWORD timeout_ms = 500U;
    setsockopt(connected, SOL_SOCKET, SO_RCVTIMEO,
               reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));
    return connected;
}

// status 等非事件消息不落盘；CSV 时间是 PC 接收时刻，不是板端源帧单调时钟。
void record_event(const std::string& line, std::ofstream& output)
{
    bool type_found = false;
    const std::string type = json_string(line, "type", type_found);
    if (!type_found || type != "event") {
        return;
    }
    bool event_found = false;
    bool logical_id_found = false;
    bool class_found = false;
    const std::string event = json_string(line, "event", event_found);
    const int logical_id = json_integer(line, "logical_id", logical_id_found);
    const std::string class_name = json_string(line, "class", class_found);
    if (!event_found || !logical_id_found || !class_found ||
        (event != "ENTER" && event != "DWELL" && event != "EXIT")) {
        throw std::runtime_error("event JSON is missing required business fields");
    }
    // 每个有效事件立即 flush，降低进程异常退出时丢失最近记录的窗口；代价是每条事件
    // 都触发流刷新。CSV 记录与控制台显示分别取本地时钟，二者毫秒值可能略有不同。
    output << local_timestamp() << ',' << csv_field(event) << ',' << logical_id << ','
           << csv_field(class_name) << '\n';
    output.flush();
    std::cout << local_timestamp() << ' ' << event << " logical_id=" << logical_id
              << " class=" << class_name << '\n';
}

}  // namespace

int main(int argc, char** argv)
{
    WSADATA wsa_data{};
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
        std::cerr << "cannot initialize Winsock\n";
        return 1;
    }

    try {
        const Options options = parse_options(argc, argv);
        if (!SetConsoleCtrlHandler(console_handler, TRUE)) {
            throw std::runtime_error("cannot install console handler");
        }
        // 以 trunc 打开意味着本次运行会重新建立日志文件，而非向旧 CSV 追加；
        // 标题行固定对应 record_event 的四列输出。
        std::ofstream output(options.output, std::ios::trunc);
        if (!output.is_open()) {
            throw std::runtime_error("cannot open output CSV: " + options.output);
        }
        output << "timestamp,event,logical_id,class\n";

        const SOCKET socket = connect_to(options);
        send_all(socket, "SUBSCRIBE_EVENTS\n");
        std::cout << "Subscribed to " << options.host << ':' << options.port
                  << ", writing " << options.output << "\n";

        // TCP 是字节流：一次 recv 可能只有半行，也可能合并多行。pending 跨 recv 保留
        // 尾部残片，直到 LF 才交给解析器；内存没有长度上限，异常超长且永不换行的数据会累积。
        std::string pending;
        char buffer[4096]{};
        while (!stop_requested.load()) {
            const int received = recv(socket, buffer, sizeof(buffer), 0);
            if (received == 0) {
                std::cout << "event server closed the connection\n";
                break;
            }
            if (received < 0) {
                const int error = WSAGetLastError();
                if (error == WSAETIMEDOUT || error == WSAEWOULDBLOCK) {
                    continue;
                }
                throw std::runtime_error("event receive failed");
            }
            // 一次 recv 的边界与 JSON 行无关；pending 保留尚未收到换行的残片。
            pending.append(buffer, static_cast<std::size_t>(received));
            // 一次读取可能含多条完整 JSON 行，所以反复消费 LF；最后不足一行的尾部
            // 留在 pending，等下一次 recv 拼接。连接断开时未完成的尾行会被丢弃。
            for (;;) {
                const std::size_t newline = pending.find('\n');
                if (newline == std::string::npos) {
                    break;
                }
                const std::string line = pending.substr(0U, newline);
                pending.erase(0U, newline + 1U);
                if (line.empty()) {
                    continue;
                }
                // 单条格式错误只记录诊断并继续处理后续行；网络读取错误则离开外层循环。
                try {
                    record_event(line, output);
                } catch (const std::exception& error) {
                    std::cerr << "event JSON parse error: " << error.what() << '\n';
                }
            }
        }
        closesocket(socket);
        WSACleanup();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "edgevision_event_logger: " << error.what() << '\n';
        WSACleanup();
        return 1;
    }
}

#else

int main()
{
    return 1;
}

#endif
