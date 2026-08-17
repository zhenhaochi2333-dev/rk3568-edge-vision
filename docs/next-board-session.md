# Next Board Session Checklist

This is a short board-side follow-up list. Do not treat any item as passed
until it is executed on the RK3568.

## Camera recovery

1. Restore the existing Direct V4L2 camera path to `1280x720 NV12`.
2. Confirm the active node and current format with read-only V4L2 checks.

The exact historical restoration command is currently unknown. Requires
board-side diagnosis. Do not add an `800x600` compatibility path to the
Windows source.

## Phase 4 real-board integration

After the camera format is restored:

1. Confirm the existing YOLO11s EdgeVision process starts with TCP enabled.
2. From the PC, run `PING`, `GET_STATUS`, `SUBSCRIBE_EVENTS`, and
   `UNSUBSCRIBE_EVENTS`.
3. Confirm disconnect and reconnect behavior.
4. Confirm Ctrl+C releases the TCP port, camera node, and worker threads.

Physical ROI ENTER/DWELL/EXIT validation remains
`DEFERRED / NOT VALIDATED` until a separately confirmed field test.

## Phase 5 dependency gate

Verify on board, without guessing package names or encoder properties:

1. `appsrc`
2. `rtph264pay`
3. `mpph264enc`
4. `gstreamer-app-1.0` development headers and library
5. `gstreamer-rtsp-server-1.0` development headers and library
6. `h264parse`

Only after all required dependencies are genuinely available:

1. Build the smallest Rockchip H.264 encoder pipeline.
2. Connect the latest EdgeVision frame through `appsrc`.
3. Add RTP and the RTSP server.
4. Validate the stream from a PC tool.

GStreamer is for streaming only. It must not become a second camera owner;
do not add `v4l2src` to the camera path.

## Phase 6 board validation

After TCP and RTSP exist:

- Run a 10–15 minute combined Camera/Display/Detection/TCP/RTSP check.
- Disconnect and reconnect the PC TCP client and RTSP client.
- Verify a network/PC disconnect does not stop local vision processing.
- Run a final 30–60 minute stability soak only after all features are complete.
- Record camera, display, detection, TCP, RTSP, thread, and resource-release
  failures separately.
