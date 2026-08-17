# Phase 5 RTSP Prerequisites

Status: **BLOCKED / NOT IMPLEMENTED**.

Previously recorded RK3568 runtime probe:

- GStreamer 1.18.5: available
- `appsrc`: available
- `rtph264pay`: available
- Rockchip `mpph264enc`: available
- `gstreamer-app-1.0` development package: missing
- `gstreamer-rtsp-server-1.0` development package: missing
- `h264parse`: missing

The future streaming path is:

```text
EdgeVision latest frame
  -> appsrc
  -> required format conversion
  -> Rockchip hardware H.264 encoder
  -> RTP
  -> RTSP server
  -> PC viewer
```

GStreamer will be used only on the streaming branch. It must not become a
second camera owner; `v4l2src` will not be added to replace or duplicate the
existing Direct V4L2 camera path.

Formal `RtspStreamer` implementation waits until the actual board development
headers, libraries, caps, and encoder properties are available for a real
GStreamer 1.18 build and test.
