# EdgeVision V3 Phase 2 阶段工作总结

日期：2026-08-17  
分支：`test/v3-phase2-yolo11-validation`  
范围：YOLO11s 真机定型验收、性能基线、摄像头验证和工程状态确认

> 本文是本阶段的中途总结，不属于最终发布文档。本阶段文档只保存在本地，未提交或推送到 GitHub。

## 1. 阶段目标与完成情况

本阶段围绕已经完成的 YOLOv5s → YOLO11s 迁移，验证正式 `edge_vision` 是否能够在 RK3568 上稳定完成：

```text
YOLO11s INT8 RKNN
    → RknnModel
    → ImageProcessor
    → Yolo11Detector
    → 检测结果 / 摄像头显示
```

已完成：

- 正式 `edge_vision` 使用 `Yolo11Detector`。
- YOLO11s INT8 RKNN 模型在 RK3568 上成功加载并运行 `rknn_run`。
- ARM64 原生编译、CTest、`bus.jpg` 回归验证通过。
- 完成摄像头打开、关闭、重新打开和有限帧采集验证。
- 完成平滑预览模式验证，确认单 AI worker + 最新帧槽位的工作方式。
- 完成若干真实物体场景采集，记录模型识别结果和误检情况。
- 保留 YOLOv5 源码、模型和回滚分支，没有删除旧实现。

未完成或暂不纳入本阶段：

- 长时间性能稳定性仍未确认。
- Tracker、ROI 业务逻辑尚未完成完整真机验收。
- 摄像头画面传输到 PC 的网络流功能尚未实现。
- 视频录制、编码器选择、RTSP/RTP/OBS 集成暂未开始。
- 未修改系统 RKNN Runtime、驱动、Kernel 或 Device Tree。

## 2. Git 与工程变更

当前正式源目录：

```text
D:\秋招\embedded-job-prep\11-projects\rk3568-edge-vision
```

当前分支：

```text
test/v3-phase2-yolo11-validation
```

本阶段已创建的本地提交：

```text
7b7da10 feat: complete YOLO11s formal detector migration
c7afe99 test: add YOLO11 RKNN run benchmark harness
8fa99ac fix: clarify smooth preview result age metric
```

本阶段没有执行 GitHub push。工作区仍保留此前未提交的中间记录文件和采集脚本，未自动纳入本阶段提交。

## 3. 新增和调整的代码模块

### 3.1 正式 YOLO11 检测器

新增或启用：

- `include/edgevision/yolo11_detector.hpp`
- `src/yolo11_detector.cpp`

主要职责：

- 解析 YOLO11 的 9 个 RKNN 输出张量。
- 按 stride 8、16、32 对输出进行分组。
- 处理 INT8/UINT8 输出和 scale/zp 反量化。
- 执行 DFL 解码和 anchor-free 检测框恢复。
- 计算 objectness/class score。
- 执行 class-wise NMS。
- 将模型坐标恢复到原始摄像头图像坐标并进行边界裁剪。

YOLOv5 detector 仍保留在 legacy 测试库中，仅用于回滚和旧测试，不再链接到正式应用目标。

### 3.2 RKNN 模型封装调整

调整：

- `src/rknn_model.cpp`
- `include/edgevision/rknn_model.hpp`

`RknnModel` 继续只负责 RKNN Runtime 生命周期、输入、运行、输出元数据和输出生命周期，不包含 YOLO11 解码或 UI 逻辑。

输出仍采用简单的 RAII lease：

```text
rknn_outputs_get
    → 在输出仍有效时同步解码
    → 生成拥有数据的 Detection 结果
    → 自动 rknn_outputs_release
```

没有保留 `output.buf`，也没有引入大型模板化内存框架。

### 3.3 应用与预览路径

调整：

- `src/application.cpp`
- `CMakeLists.txt`

应用层现在正式连接：

```text
Application
    → ImageProcessor
    → RknnModel
    → Yolo11Detector
```

平滑预览使用单个最新帧槽位和单个 AI worker：

- 摄像头线程持续保留最新画面。
- AI worker 只处理最新可用帧。
- 不建立无限队列。
- 旧帧被跳过，以避免延迟持续累积。

已有的横屏显示适配继续保留：摄像头画面保持比例显示，允许黑边，不直接拉伸到屏幕尺寸。

### 3.4 Benchmark 工具

新增：

- `tools/rknn_run_benchmark.cpp`

该工具只测量 `rknn_run()` 边界附近的推理耗时，不把完整摄像头循环、显示或网络传输混入推理指标。

默认参数：

```text
warmup: 30
samples: 300
```

## 4. YOLO11s 技术合同与运行参数

模型：

```text
YOLO11s INT8 RKNN
target: RK3568
model size: 11,934,219 bytes
SHA-256: caf30c2c21333ebbbbc2369dcab0af0aa672c3bdd03ddcd348954f6fd470ee5a
```

RKNN Runtime：

```text
Toolkit: RKNN-Toolkit2 2.3.2
Board runtime: 1.6.0
RKNPU driver: 0.7.2
```

输入：

```text
[1,640,640,3]
UINT8 / NHWC / RGB
zero point: -128
scale: 0.00392157
```

输出：

```text
stride 8:
  [1,64,80,80], [1,80,80,80], [1,1,80,80]

stride 16:
  [1,64,40,40], [1,80,40,40], [1,1,40,40]

stride 32:
  [1,64,20,20], [1,80,20,20], [1,1,20,20]
```

默认检测参数：

```text
confidence threshold: 0.25
NMS threshold: 0.45
```

摄像头参数：

```text
device: /dev/video0
capture: 1280x720
format: NV12 through GStreamer, then BGR/RGB processing
```

显示参数：

```text
logical display: 1280x800
camera aspect ratio: 16:9
display policy: aspect-fit with black bars, no forced stretch
```

## 5. 编译与基础验证

### 5.1 ARM64 原生编译

在 RK3568 上完成带视频支持的原生 ARM64 编译：

- `edge_vision` 构建成功。
- `edgevision_tests` 构建成功。
- `edgevision_rknn_benchmark` 构建成功。

CTest：

```text
1/1 test passed
```

### 5.2 可执行文件和依赖

验证结果：

- 可执行文件为 AArch64 ELF。
- 使用项目局部 `lib/librknnrt.so`。
- 动态库通过 `$ORIGIN/lib` / `LD_LIBRARY_PATH` 加载。
- 没有覆盖 `/lib/librknnrt.so` 或 `/usr/lib/librknnrt.so`。
- 没有执行系统级 Runtime、驱动或 Kernel 修改。

## 6. bus.jpg Golden Regression

命令使用 YOLO11 正式模型、COCO labels、`conf=0.25`、`nms=0.45`，返回码为 `0`。

记录到的检测结果：

| class | confidence | bbox `(x, y, w, h)` |
|---|---:|---|
| bus | 0.942 | `(91, 133, 550, 437)` |
| person | 0.903 | `(475, 231, 559, 521)` |
| person | 0.896 | `(211, 239, 283, 508)` |
| person | 0.896 | `(108, 237, 225, 536)` |
| person | 0.641 | `(79, 326, 126, 519)` |

输出图片已生成并确认可读，检测类别和框的位置合理。该结果证明 YOLO11s 正式路径可以在 RK3568 上完成一次完整的：

```text
图片读取 → 预处理 → RKNN 推理 → YOLO11 解码 → NMS → 输出结果
```

## 7. RKNN 推理耗时基线

使用 30 次 warm-up、300 次采样，连续执行了两轮 `rknn_run` 测量。

| round | average | median | P95 | min | max |
|---|---:|---:|---:|---:|---:|
| 1 | 121.638 ms | 106.968 ms | 152.383 ms | 104.181 ms | 166.318 ms |
| 2 | 145.751 ms | 142.952 ms | 165.742 ms | 107.662 ms | 205.482 ms |

结论：

- 两轮均成功返回。
- 推理耗时当前存在明显波动。
- 现阶段不能把 60 ms 或某一个单次结果宣称为 YOLO11s 稳定基线。
- 本阶段没有为了降低耗时修改模型、Runtime、驱动、频率或系统配置。

测试期间观察到的系统信息没有显示明确的故障性温度、驱动或负载异常，但仍不足以解释全部推理波动，后续需要单独安排性能分析。

## 8. 平滑预览验证

执行了 450 帧平滑预览，过程正常结束，返回码为 `0`，没有残留 EdgeVision 进程。

实测记录：

```text
displayed_frames: 450
submitted_frames: 181
skipped_frames: 269
completed_inferences: 180
display_fps: 14.761736
detection_fps: 5.904695
inference_avg: 116.196374 ms
ai_latency_avg: 131.663228 ms
display_result_age_avg: 79.673733 ms
display_result_age_max: 220.869044 ms
```

这里的 `display_result_age` 指：

```text
显示时刻 - AI 结果完成时刻
```

它不是摄像头源帧从采集到显示的完整年龄。

平滑预览的主要验证结论是：丢弃旧帧后，系统可以保持显示连续性，不会因为 AI 推理速度低于摄像头帧率而无限积压待处理帧。

## 9. 摄像头关闭与重新打开

在平滑预览结束后，补做了静态图片回归和摄像头重新打开验证。

结果：

- 摄像头重新打开成功。
- `/dev/video0` 可以再次被应用使用。
- 1280×720 摄像头管线正常工作。
- 35 帧 MP4 输出成功生成并可读。
- 没有残留摄像头占用进程。

重新打开测试的有限采样结果：

```text
warmup: 30 frames
measured: 5 frames
inference: 106.704804 ms
full_loop: 197.514091 ms
actual_fps: 5.062795
return code: 0
```

## 10. 真实物体场景记录

本阶段采用用户确认后逐场景单帧采集的方式。每个场景保存 raw frame、YOLO11 输出图和检测文本，避免连续自动采图影响对比。

### 场景 1：鼠标单独、距离较近

记录结果：

```text
person 0.884  bbox=(276, 2, 1154, 524)
microwave 0.295  bbox=(1, 247, 101, 436)
```

鼠标没有被正确识别，`microwave` 属于 COCO 类别名称，是本场景中的误检结果。

### 场景 2：手机和水杯

记录结果：

```text
remote 0.579       bbox=(247, 484, 1278, 711)
person 0.479       bbox=(336, 161, 1201, 501)
cell phone 0.290   bbox=(240, 488, 1271, 712)
person 0.262       bbox=(339, 147, 542, 504)
```

水杯没有被识别；`remote` 和部分 `person` 为误检或错误分类，手机仅以较低置信度被识别。

### 场景 3：水杯单独

记录结果：

```text
tv 0.367  bbox=(374, 2, 1204, 620)
```

水杯没有被识别，`tv` 为明显误检。

### 场景 4：现场确认后采集的杯子和鼠标画面

记录结果：

```text
cup 0.803    bbox=(835, 2, 1185, 595)
mouse 0.710  bbox=(247, 487, 614, 714)
```

该画面中杯子和鼠标均被识别。不同场景之间的结果差异说明当前模型对物体大小、背景、遮挡、姿态和画面位置较敏感，不能仅凭单个场景判断整体识别能力。

## 11. 当前已经具备的功能

当前正式 YOLO11 版本已经具备：

- RK3568 上 YOLO11s INT8 模型加载和单次推理。
- `bus.jpg` 图片检测和结构化终端输出。
- 1280×720 OV5695 摄像头输入。
- YOLO11 摄像头检测与结果叠加显示。
- 横屏 1280×800 逻辑桌面适配。
- 保持 16:9 画面比例的黑边显示。
- 平滑预览模式和最新帧丢弃策略。
- 有限帧 MP4 输出验证。
- 运行结束后摄像头资源释放和重新打开。
- 项目局部 RKNN Runtime 依赖加载。
- YOLOv5 legacy 回滚路径保留。

## 12. 当前问题与技术债务

- YOLO11s `rknn_run` 耗时波动较大，尚未形成稳定性能基线。
- 真实场景中存在漏检和误检，尤其是鼠标、水杯等小目标或非 COCO 标准类别目标。
- `cup`、`mouse` 等标签是否属于当前 labels/model 的正式类别，需要在后续产品化阶段统一标签和展示策略。
- Tracker 和 ROI 还没有完成本阶段要求的完整真机验收。
- 目前没有将开发板摄像头画面实时传到 PC 的功能。
- 目前没有实现 PC 端显示、录制、RTSP/RTP 或 OBS 输入适配。
- YOLOv5 旧模型和板端旧验证目录暂时不能删除，需等 YOLO11 性能和功能验收明确后再处理。

## 13. 本阶段停止点

按照当前用户指令，本阶段在确认“开发板摄像头画面后续可以传输到 PC”这一方案可行后停止。

尚未执行：

- 摄像头网络推流实现。
- 硬件 H.264 编码器插件探测。
- RTP/RTSP 命令验证。
- PC VLC、ffplay 或 OBS 接收验证。
- 录像文件生成和长时间稳定性测试。

后续如果开始该功能，建议保持以下边界：

```text
RK3568 /dev/video0
    → 一路供 YOLO11
    → 一路硬件编码
    → LAN RTP/RTSP
    → PC 显示或录制
```

本阶段没有修改代码来实现网络传输，也没有继续进行新的长时间测试。
