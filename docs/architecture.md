# Final architecture

## Runtime chain

```text
Windows Integrated Camera
  -> tools/pc_two_window_bridge.cpp
  -> FFmpeg DirectShow capture
  -> MJPEG byte stream over TCP :5600
  -> src/network_camera_source.cpp
  -> JPEG SOI/EOI framing + OpenCV imdecode(BGR)
  -> src/application.cpp
  -> YOLO11 RKNN -> IouTracker -> SemanticStabilizer
  -> RegionMonitor
  -> RK local display / RTSP / TCP status and events
```

The PC bridge owns one camera capture and starts three independent workers: the
MJPEG sender, the raw camera preview, and the RTSP detection preview. Closing a
preview does not intentionally close the sender or board process.

## Core modules

- `src/application.cpp`: capture, inference, tracking, stabilization, output
  coordination, and optional diagnostic trace.
- `src/network_camera_source.cpp`: POSIX TCP receiver, newest-complete-JPEG
  buffering, and BGR decode.
- `src/yolo11_detector.cpp`: YOLO11 RKNN tensor decode, confidence filtering,
  and NMS.
- `src/iou_tracker.cpp`: short-term frame-to-frame raw tracking.
- `src/semantic_stabilizer.cpp`: temporal logical identity and class fusion.
- `src/region_monitor.cpp`: ROI lifecycle and ENTER/DWELL/EXIT production.
- `src/tcp_server.cpp`: newline-delimited commands and event JSON.
- `src/rtsp_streamer.cpp`: annotated BGR to Rockchip H.264 RTSP output.

## Identity and events

`track_id` is a short-term tracker identity. `logical_id` is created and
maintained by `SemanticStabilizer` and is the business identity used by the
display and TCP events. A short detector gap may enter `LOST_PENDING` and
recover without a new ENTER. After the configured lost window, EXIT ends the
lifecycle.

The TCP event payload contains `event`, `class`, `logical_id`, and a
compatibility `track_id`. The PC logger keeps only the four business CSV
fields; it does not join frames, detections, or trace rows.

## Video boundary

The PC-to-board input is deliberately not H.264/RTP/MPP. It is a framed MJPEG
TCP byte stream decoded by OpenCV. The board-to-PC RTSP output is separate and
continues to use `mpph264enc`, `h264parse`, and `rtph264pay`.
