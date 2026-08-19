# RK3568 EdgeVision

## Quick Start

The formal demo chain is:

```text
Integrated Camera
  → PC C++ bridge
  → MJPEG/TCP :5600
  → RK3568 YOLO11 detection
  → RK local detection window and RTSP :8554
  → PC event subscriber :9000
```

Build the Windows PC tools once from the repository root:

```powershell
cmake -S . -B build-pc-tools -DEDGEVISION_BUILD_PC_BRIDGE=ON -DEDGEVISION_BUILD_PC_EVENT_LOGGER=ON
cmake --build build-pc-tools --config Release
```

The launcher expects OpenSSH, FFmpeg, the deployed RK3568 program/model, and
the Integrated Camera to already exist. It does not install dependencies or
modify the board.

Start the complete demo:

```powershell
.\tools\start_edgevision.ps1
```

This starts the RK local detection window, PC Raw Preview, PC RTSP Detection
Preview, and `event_log.csv` TCP event recording. SSH password input remains
interactive unless passwordless SSH is configured.

Stop the PC helpers and board normally:

```powershell
.\tools\stop_edgevision.ps1
```
