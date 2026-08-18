#include "edgevision/network_camera_source.hpp"

#include <cassert>

void run_network_camera_source_tests()
{
    const edgevision::NetworkCameraSource source(5600);
    assert(source.port() == 5600);
    assert(source.pipeline().find("udpsrc port=5600") != std::string::npos);
    assert(source.pipeline().find("rtph264depay ! h264parse") != std::string::npos);
    assert(source.pipeline().find("mppvideodec format=BGR") != std::string::npos);
    assert(source.pipeline().find("appsink sync=false max-buffers=1 drop=true") !=
           std::string::npos);
}
