#include "edgevision/tcp_server.hpp"

#include <arpa/inet.h>
#include <cassert>
#include <chrono>
#include <cstring>
#include <netinet/in.h>
#include <poll.h>
#include <stdexcept>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int connect_client(std::uint16_t port)
{
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    require(fd >= 0, "TCP test socket creation failed");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    require(inet_pton(AF_INET, "127.0.0.1", &address.sin_addr) == 1,
            "TCP test address setup failed");
    for (int attempt = 0; attempt < 20; ++attempt) {
        if (connect(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0) {
            return fd;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    close(fd);
    throw std::runtime_error("TCP test client connection timed out");
}

void send_text(int fd, const std::string& text)
{
    std::size_t offset = 0U;
    while (offset < text.size()) {
        const ssize_t sent = send(fd, text.data() + offset, text.size() - offset, 0);
        require(sent > 0, "TCP test send failed");
        offset += static_cast<std::size_t>(sent);
    }
}

std::string read_line(int fd)
{
    std::string result;
    for (;;) {
        pollfd descriptor{fd, POLLIN, 0};
        require(poll(&descriptor, 1, 2000) > 0, "TCP test read timed out");
        char character = '\0';
        require(recv(fd, &character, 1, 0) == 1, "TCP test connection closed while reading");
        result += character;
        if (character == '\n') {
            return result;
        }
        require(result.size() <= 4096U, "TCP test response exceeded line limit");
    }
}

}  // namespace

void run_tcp_server_tests()
{
    edgevision::TcpServer server(0U);
    server.start();
    require(server.port() != 0U, "TCP test server did not bind an ephemeral port");
    const int client = connect_client(server.port());

    send_text(client, "PING\nGET_STATUS\n");
    require(read_line(client).find("\"pong\"") != std::string::npos,
            "TCP PING response mismatch");
    const std::string status = read_line(client);
    require(status.find("\"type\":\"status\"") != std::string::npos,
            "TCP GET_STATUS response type mismatch");
    require(status.find("uptime_ms") != std::string::npos,
            "TCP GET_STATUS response missing uptime");

    send_text(client, "SUBSCRIBE_EVENTS\n");
    require(read_line(client).find("\"subscribed\":true") != std::string::npos,
            "TCP subscribe response mismatch");
    edgevision::RegionEvent event;
    event.type = edgevision::RegionEventType::Enter;
    event.track_id = 7;
    event.class_id = 0;
    event.confidence = 0.91F;
    event.source_timestamp = std::chrono::steady_clock::now();
    server.publish_event(event, "person\"demo");
    const std::string event_line = read_line(client);
    require(event_line.find("\"event\":\"ENTER\"") != std::string::npos,
            "TCP event type mismatch");
    require(event_line.find("person\\\"demo") != std::string::npos,
            "TCP event JSON escaping mismatch");
    require(event_line.find("\"track_id\":7") != std::string::npos,
            "TCP event track id mismatch");

    send_text(client, "UNSUBSCRIBE_EVENTS\n");
    require(read_line(client).find("\"subscribed\":false") != std::string::npos,
            "TCP unsubscribe response mismatch");
    close(client);
    server.stop();

    edgevision::TcpServer restarted(server.port());
    restarted.start();
    restarted.stop();
}
