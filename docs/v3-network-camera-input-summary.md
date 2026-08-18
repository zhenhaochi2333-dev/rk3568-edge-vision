# Network Camera Input: JPEG/TCP

The formal network input is now a single-owner JPEG transport:

```text
PC Integrated Camera
  -> one FFmpeg dshow capture
  -> MJPEG byte stream over TCP :5600
  -> POSIX TCP server socket
  -> JPEG SOI/EOI frame parser
  -> OpenCV imdecode(BGR)
  -> existing EdgeVision capture/latest-frame/YOLO11 chain
```

The PC sender keeps the raw preview on the same camera capture and emits a
fixed 1280x720, approximately 15 FPS MJPEG stream at fixed `q:v 3` quality.
The RK3568 input uses software JPEG decoding; no H.264/RTP/MPP decoder or
GStreamer input pipeline is in the formal camera input path.

The receiver keeps only the newest complete JPEG found in its receive buffer,
so slow consumers do not create an unbounded decoded-frame queue. The existing
upper layers are unchanged: YOLO11, Tracker, RegionMonitor, TCP, local display,
and RTSP output.

The RTSP output remains a separate branch and continues to use the validated
Rockchip `mpph264enc` path.
