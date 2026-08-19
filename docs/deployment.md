# Deployment and operation

## Prerequisites

The Windows PC needs OpenSSH, CMake, a C++ compiler, FFmpeg with DirectShow
and SDL support, and an `Integrated Camera`. The RK3568 needs the already
deployed `edge_vision` executable, YOLO11 RKNN model, labels, RKNN runtime,
X11 display, and the RTSP-enabled build.

The formal launcher expects these board paths:

```text
/root/edgevision_minimal/build-native-rtsp/edge_vision
/root/edgevision_minimal/models/yolo11s_rk3568_i8.rknn
/root/edgevision_minimal/assets/coco_80_labels_list.txt
```

The SSH password is entered interactively. It is not a source file value.

## Build PC tools

```powershell
cmake -S . -B build-pc-tools -DEDGEVISION_BUILD_PC_BRIDGE=ON -DEDGEVISION_BUILD_PC_EVENT_LOGGER=ON
cmake --build build-pc-tools --config Release
```

The two formal C++ executables are `edgevision_pc_bridge.exe` and
`edgevision_event_logger.exe`.

## Run and stop

```powershell
.\tools\start_edgevision.ps1
# enter the SSH password when prompted

# after Ctrl+C in the launcher window:
.\tools\stop_edgevision.ps1
```

The launcher writes its transient process state and `event_log.csv` under the
Windows temporary `edgevision-runtime` directory. These are runtime outputs,
not repository assets.

## Board build and sync

The Windows repository is the source of truth. The existing scripts
`scripts/build_native_rk3568.sh`, `scripts/build_cross.sh`,
`scripts/stage_rknn_deps.sh`, and `scripts/sync_to_ubuntu.ps1` are helpers for
an explicitly configured build/deployment environment. They do not change the
formal runtime architecture and must be run only with the intended remote
host and SDK paths.
