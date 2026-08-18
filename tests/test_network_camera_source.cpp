#include "edgevision/network_camera_source.hpp"

#include <cassert>

void run_network_camera_source_tests()
{
    const edgevision::NetworkCameraSource source(5600);
    assert(source.port() == 5600);
    assert(source.pipeline().find("tcp-server-mjpeg port=5600") != std::string::npos);
    assert(source.pipeline().find("OpenCV imdecode(BGR)") != std::string::npos);
    assert(source.pipeline().find("SOI/EOI framing") != std::string::npos);
    assert(source.pipeline().find("tcpserversrc") == std::string::npos);
    assert(source.pipeline().find("mppvideodec") == std::string::npos);
    assert(source.pipeline().find("rtph264depay") == std::string::npos);
    assert(source.pipeline().find("appsink") == std::string::npos);
}
