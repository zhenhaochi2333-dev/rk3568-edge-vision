#include "edgevision/rtsp_streamer.hpp"

#include <cassert>
#include <stdexcept>
#include <string>

void run_rtsp_streamer_tests()
{
    const edgevision::RtspStreamer streamer(8554);
    assert(streamer.port() == 8554);
    assert(streamer.url() == "rtsp://192.168.77.2:8554/live");
    assert(streamer.pipeline().find("appsrc name=src") != std::string::npos);
    assert(streamer.pipeline().find("format=NV12") != std::string::npos);
    assert(streamer.pipeline().find("mpph264enc") != std::string::npos);
    assert(streamer.pipeline().find("h264parse") != std::string::npos);
    assert(streamer.pipeline().find("rtph264pay name=pay0") != std::string::npos);

    bool invalid_port_failed = false;
    try {
        const edgevision::RtspStreamer invalid(0);
        (void)invalid;
    } catch (const std::runtime_error&) {
        invalid_port_failed = true;
    }
    assert(invalid_port_failed);
}
