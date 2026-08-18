# EdgeVision Network Camera Input 阶段性工作总结

## 1. 阶段目标

本阶段在 `feature/v3-network-camera-input` 分支中，为 EdgeVision 增加基于 RTP/H.264/UDP 的网络摄像头输入。

目标：

- 在 RK3568 上使用 Rockchip MPP 完成 H.264 硬件解码；
- 将解码结果直接以 BGR 图像交给现有 OpenCV/AI/UI 主链；
- 保留原有 Direct V4L2 本地摄像头输入；
- 复用现有 Capture Thread、Latest Frame、YOLO11、Tracker、RegionMonitor、TCP 和显示流程；
- 完成正式 EdgeVision 应用验收，而不是只验证独立 probe。

本阶段不涉及 RTSP、RGA 优化、640×384 输入、摄像头分辨率调整、H.265 或完整系统重构。

## 2. 最终输入后端

最终采用：

```text
OpenCV VideoCapture
  + CAP_GSTREAMER
  + GStreamer RTP/H.264 pipeline
  + Rockchip mppvideodec
  + BGR output
```

正式 pipeline：

```text
udpsrc port=5600
  caps="application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96"
! rtph264depay
! h264parse
! mppvideodec format=BGR
! video/x-raw,format=BGR
! appsink sync=false max-buffers=1 drop=true
```

设计取舍：

1. `rtph264depay → mppvideodec` 直接连接时，MPP sink 要求 `parsed=true`，所以 `h264parse` 是必需依赖。
2. `mppvideodec format=BGR` 已被实测为可用的直接 BGR 输出路径。
3. `videoconvert` 实测只有约 1.8 FPS，因此没有加入正式 pipeline。
4. `appsink max-buffers=1 drop=true` 优先保留最新帧，避免旧帧持续积压。

## 3. GStreamer 与硬件解码 Gate

### 3.1 h264parse

RK3568 上离线安装的 ARM64 包：

```text
gstreamer1.0-plugins-bad 1.16.3-0ubuntu1.1 arm64
```

parser plugin：

```text
/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstvideoparsersbad.so
```

当时系统 GStreamer core 为 1.18.5。安装没有升级 GStreamer core、kernel、OpenCV、RKNN，也没有覆盖 Rockchip MPP plugin。

### 3.2 RTP/MPP PoC

```text
PC Webcam
  → NVIDIA H.264 Encoder MFT
  → RTP/UDP 192.168.77.2:5600
  → rtph264depay
  → h264parse
  → mppvideodec
  → fakesink
```

代表性结果：

- 运行时间约 74.8 秒；
- decoded frame 1062；
- decoded FPS 约 15.02；
- MPP raw output 为 1280×720 NV12；
- dropped frame 为 0；
- 未使用 `avdec_h264`、`openh264` 或 software FFmpeg decoder fallback。

### 3.3 OpenCV BGR Gate

使用 `cv::VideoCapture(..., cv::CAP_GSTREAMER)` 验证正式后端：

- 运行时间约 65.1 秒；
- 895 frames；
- 实际约 13.76 FPS；
- 输出 1280×720、BGR、`CV_8UC3`；
- OpenCV backend 为 GStreamer；
- 无连续 read failure。

## 4. 正式代码设计

### 4.1 `NetworkCameraSource`

新增：

```text
include/edgevision/network_camera_source.hpp
src/network_camera_source.cpp
```

该类封装网络输入的生命周期、pipeline 和格式约束：

```cpp
NetworkCameraSource source(5600);
source.open();
source.read(frame);
source.release();
```

主要接口：

- `open()`：使用 OpenCV `VideoCapture` 打开 GStreamer pipeline；
- `read(cv::Mat&)`：读取 BGR 帧；
- `release()`：释放 VideoCapture/GStreamer pipeline；
- `is_opened()`、`pipeline()`、`info()`：提供状态和输入元数据。

`read()` 明确检查 `1280×720`、BGR、`CV_8UC3`，不符合时立即报错，避免错误格式进入 YOLO11 预处理。

### 4.2 Capture Thread 接入

没有复制新的网络线程，而是在 `application.cpp` 中为现有 Capture Thread 增加最小输入适配：

```text
CaptureInput
├── LocalCaptureInput   → CameraSource
└── NetworkCaptureInput → NetworkCameraSource
```

现有 Capture Thread 继续负责：

- 独立读取线程；
- Latest Frame snapshot；
- UI/AI worker 共享最新完整帧；
- 采集、覆盖、读取耗时和 FPS 统计；
- 退出时 release 和 join。

后续主链保持不变：

```text
Capture Thread
  → Latest Frame Snapshot
  → SmoothAiWorker
  → YOLO11
  → Tracker / RegionMonitor
  → DisplayComposer
  → TCP status/events
```

没有新增 Plugin system、EventBus、复杂 Frame framework、watchdog thread 或 reconnect manager。

### 4.3 停止与恢复

`CameraCaptureThread::request_stop()` 会先调用输入的 `release()`，用于中断可能阻塞的 appsink read，然后等待线程退出。

连续 read failure 时使用简单节流恢复：

```text
连续 5 次 read 失败
  → release()
  → 等待约 500 ms
  → open()
  → 成功后继续读取
```

恢复逻辑位于现有 capture thread 内，没有新增后台 watchdog。Ctrl+C 到达时，条件变量等待会被唤醒，不会 busy-loop。

## 5. CLI 设计

新增：

```cpp
enum class InputMode {
    File,
    LocalCamera,
    NetworkCamera,
};
```

### Network Camera

```bash
edge_vision \
  --model validation/yolo11s_rk3568_i8.rknn \
  --labels assets/coco_80_labels_list.txt \
  --input network \
  --show
```

网络模式固定使用 UDP 5600，并进入 smooth camera 主链；不支持 `--output`，也不会尝试打开 `/dev/video0`。

### Local Camera

```bash
edge_vision --model MODEL --labels LABELS --input local --show
```

`--input local` 是 `/dev/video0` 的简洁别名。原有形式继续保留：

```bash
edge_vision --model MODEL --labels LABELS --camera /dev/video0 --show
```

### File Input

原有 `--input FILE --output OUTPUT` 语义继续保留。只有值完全匹配 `local` 或 `network` 时才解释为摄像头模式，普通文件路径不会被误判。

## 6. 正式应用验收

### 6.1 正常网络输入

正式应用实际运行了：

```text
NetworkCameraSource
  → GStreamer/MPP BGR
  → DisplayComposer
  → YOLO11
  → Tracker
  → RegionMonitor
  → TCP
```

代表性结果：

- Network capture：约 15.12 FPS；
- Display：约 15.10 FPS；
- YOLO11 completed inference：80 次；
- Detection FPS：约 6.71 FPS；
- 输出：1280×720 BGR `CV_8UC3`；
- TCP server：成功监听 9100；
- ROI/RegionMonitor 路径：正常运行；
- BBox/overlay 显示路径：正常运行；
- 无应用级持续 ERROR。

### 6.2 Sender 停止与恢复

在正式 EdgeVision 运行期间停止 PC RTP sender：

- EdgeVision 没有 crash；
- UI/主线程没有永久卡死；
- sender 重启后继续获得网络帧；
- 同一进程最终正常响应 Ctrl+C。

包含中断窗口的长时间统计：

- display frames：904；
- capture frames：909；
- 整体 display FPS：约 13.37；
- sender 重启后网络输入继续工作。

### 6.3 应用重启

停止上一进程后重新启动 EdgeVision，网络 pipeline、MPP 解码、YOLO11、显示和 TCP 均再次初始化成功。

代表性重启结果：

- display frames：120；
- capture frames：126；
- completed inference：29；
- detection FPS：约 5.94；
- 正常退出，未留下占用 UDP/TCP 资源。

## 7. 构建与测试

RK3568 使用原生配置：

```text
EDGEVISION_WITH_VIDEO=ON
EDGEVISION_BUILD_TESTS=ON
```

验证结果：

- `edge_vision`：build PASS；
- `edgevision_tests`：PASS；
- CLI network/local 模式测试：PASS；
- `NetworkCameraSource` pipeline 配置测试：PASS；
- Direct V4L2 相关源文件仍参与编译；
- 一次性 probe 未加入 CMake。

测试中的 OpenCV invalid-input warning 来自既有 `test_video_io` 的故意错误输入，不是网络输入失败。

正式应用运行期间的主要 warning 为 OpenCV live video position warning 和 Rockchip RGA compatibility mode 提示；没有导致 H.264 parser、MPP decoder 或 BGR pipeline 失败。

## 8. 本阶段正式修改文件

```text
CMakeLists.txt
include/edgevision/core_types.hpp
include/edgevision/network_camera_source.hpp
src/application.cpp
src/cli_parser.cpp
src/network_camera_source.cpp
tests/test_cli.cpp
tests/test_network_camera_source.cpp
```

一次性 Gate 工具已删除：

```text
D:\rk3568\network_camera_probe.cpp
D:\rk3568\network_camera_recovery_probe.cpp
```

## 9. Git 状态

```text
Branch: feature/v3-network-camera-input
Implementation commit: f226d0f feat: add RTP H264 network camera input
Documentation: this summary is a separate follow-up commit
Push: not performed
```

## 10. 阶段结论

```text
NETWORK CAMERA SOURCE       PASS
RTP/UDP INPUT               PASS
H264 PARSER                 PASS
RK3568 MPP HARDWARE DECODE PASS
OPENCV BGR OUTPUT           PASS
EDGEVISION NETWORK INPUT   PASS
STOP/RESTART RECOVERY       PASS
APPLICATION RESTART         PASS
```

当前状态：

```text
READY FOR NEXT PHASE = YES
```

下一阶段可以在此稳定输入后端基础上继续进行，但不应把本阶段扩展为 RTSP、RGA、640×384 或其他未要求的优化工作。
