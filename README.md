# RK3568 EdgeVision

## Project overview

EdgeVision is a C++17 real-time vision pipeline for an RK3568 board. A Windows
PC camera is sent to the board as MJPEG over TCP; YOLO11 INT8 RKNN performs
detection, then tracking, semantic stabilization, lifecycle monitoring, local
display, RTSP output, and lightweight TCP events complete the demo chain.

The formal model is `YOLO11s INT8 RKNN`, with a 640x640 detector input. Only
the YOLO11 detector is part of the active source tree and runtime path.

## Hardware and software

- PC: Windows, Integrated Camera, FFmpeg with DirectShow support.
- Board: RK3568 Linux, ARM64, Rockchip RKNN runtime and OpenCV.
- Board address used by the formal launcher: `192.168.77.2`.
- PC tools and launcher: C++ bridge/event logger plus small PowerShell process
  orchestration. No Python runtime is required for the formal demo.

## Architecture and data flow

```text
Integrated Camera
  -> C++ PC bridge / FFmpeg capture
  -> MJPEG over TCP :5600
  -> NetworkCameraSource: JPEG framing + OpenCV imdecode
  -> YOLO11 RKNN
  -> IouTracker
  -> SemanticStabilizer (logical_id)
  -> RegionMonitor (ENTER / DWELL / EXIT)
  -> RK local display, RTSP :8554, TCP :9000
  -> PC RTSP preview and event_log.csv
```

The input and output video codecs are intentionally different: the input is
JPEG/TCP, while the validated RTSP output uses Rockchip H.264 encoding.

## Repository structure

```text
include/edgevision/   public C++ interfaces
src/                  board application and pipeline implementation
tests/                C++ regression tests
tools/                PC C++ tools and launcher scripts
scripts/              build, dependency staging, and source sync helpers
models/               active YOLO11 model deployment notes
assets/               labels and small regression input
docs/                 final architecture, validation, deployment, and design notes
```

Important modules are `src/application.cpp`,
`src/network_camera_source.cpp`, `src/yolo11_detector.cpp`,
`src/iou_tracker.cpp`, `src/semantic_stabilizer.cpp`,
`src/region_monitor.cpp`, `src/tcp_server.cpp`, and
`src/rtsp_streamer.cpp`.

## Quick start

Build the Windows C++ tools once from the repository root:

```powershell
cmake -S . -B build-pc-tools -DEDGEVISION_BUILD_PC_BRIDGE=ON -DEDGEVISION_BUILD_PC_EVENT_LOGGER=ON
cmake --build build-pc-tools --config Release
```

The board executable, model, labels, FFmpeg, and camera must already be
available. The launcher does not install dependencies or deploy files.

Start the complete demo:

```powershell
cd 'D:\秋招\embedded-job-prep\11-projects\rk3568-edge-vision'
powershell -ExecutionPolicy Bypass -File .\tools\start_edgevision.ps1
```

Enter the SSH password interactively when prompted; it is never stored in the
repository. The launcher starts the RK local display, PC Raw Preview, PC RTSP
Detection Preview, and the event subscriber. Stop with Ctrl+C, then run:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\stop_edgevision.ps1
```

Do not start a second copy while one is running.

## Inputs and outputs

- Network input: `192.168.77.2:5600`, JPEG SOI/EOI framed TCP stream.
- Board local display: annotated detection window on the RK3568 X11 display.
- PC detection preview: `rtsp://192.168.77.2:8554/live`.
- Status/events: `192.168.77.2:9000`.
- Event CSV: one runtime file with `timestamp,event,logical_id,class`.

The event logger sends `SUBSCRIBE_EVENTS` and accepts `ENTER`, `DWELL`, and
`EXIT`. It intentionally does not store per-frame detection or tracker data.

## Performance and validation

The final launcher smoke measurement on 2026-08-19 reported approximately:

| Metric | Measured value |
| --- | ---: |
| Capture FPS | 13.954 |
| Display FPS | 15.178 |
| Detection FPS | 5.074 |

The same smoke run started the full chain twice, received a real `DWELL` event,
and passed Ctrl+C, normal stop, and restart checks. See
`docs/validation.md` for the evidence boundary and historical measurements.

## Engineering decisions and limits

The board input formerly explored H.264/RTP/MPP decoding. The formal input is
now MJPEG/TCP plus OpenCV JPEG decode because it is simpler and stable for the
demo. H.264 remains only on the validated RTSP output branch.

`logical_id` identifies one continuous presence lifecycle. If an object fully
exits and later re-enters, a new logical ID is expected. Persistent identity
across separate lifecycles would require appearance embeddings or lightweight
ReID; this is future work and is not implemented.
