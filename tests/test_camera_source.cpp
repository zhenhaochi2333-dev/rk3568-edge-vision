#include "edgevision/camera_source.hpp"

#include <cassert>
#include <string>

void run_camera_source_tests()
{
    const edgevision::CameraSource camera("/dev/video0");
    assert(camera.device() == "/dev/video0");
    assert(camera.pipeline().find("v4l2-mmap device=/dev/video0") != std::string::npos);
    assert(camera.pipeline().find("api=VIDEO_CAPTURE_MPLANE") != std::string::npos);
    assert(camera.pipeline().find("format=NV12 width=1280 height=720 buffers=4") !=
           std::string::npos);
}
