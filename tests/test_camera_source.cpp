#include "edgevision/camera_source.hpp"

#include <cassert>
#include <string>

void run_camera_source_tests()
{
    const edgevision::CameraSource camera("/dev/video0");
    assert(camera.device() == "/dev/video0");
    assert(camera.pipeline().find("v4l2src device=/dev/video0 io-mode=mmap") != std::string::npos);
    assert(camera.pipeline().find("format=NV12,width=1280,height=720,framerate=30/1") !=
           std::string::npos);
    assert(camera.pipeline().find("video/x-raw,format=BGR") != std::string::npos);
    assert(camera.pipeline().find("appsink drop=true max-buffers=1 sync=false") != std::string::npos);
}
