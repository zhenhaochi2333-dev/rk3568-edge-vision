# PC two-window bridge

The final PC path is a single C++ launcher with three independent FFmpeg
workers:

1. `EdgeVision Raw Camera` displays the PC camera.
2. The sender encodes the same camera as JPEG/MJPEG and sends it to the board
   at TCP port 5600.
3. `EdgeVision PC Detection` displays the board RTSP result at port 8554.

The windows are independent from the sender. Closing either preview does not
close the camera TCP connection or stop the board. Stop the bridge with
`Ctrl+C`.

Build only this Windows tool from the repository root:

```powershell
cmake -S . -B build-pc-bridge -DEDGEVISION_BUILD_PC_BRIDGE=ON
cmake --build build-pc-bridge --config Release
```

The default FFmpeg path is `D:\EVCapture\ffmpeg.exe`; override it with
`--ffmpeg PATH` when needed.
