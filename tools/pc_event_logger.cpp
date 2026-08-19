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
        std::ofstream output(options.output, std::ios::trunc);
        if (!output.is_open()) {
            throw std::runtime_error("cannot open output CSV: " + options.output);
        }
        output << "timestamp,event,logical_id,class\n";

        const SOCKET socket = connect_to(options);
        send_all(socket, "SUBSCRIBE_EVENTS\n");
        std::cout << "Subscribed to " << options.host << ':' << options.port
                  << ", writing " << options.output << "\n";

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
            pending.append(buffer, static_cast<std::size_t>(received));
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
