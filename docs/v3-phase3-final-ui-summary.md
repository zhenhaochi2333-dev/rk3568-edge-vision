# EdgeVision V3 Phase 3 阶段总结

## 1. 阶段目标

本阶段围绕 YOLO11s 正式主线完成最终显示界面替换和 RK3568 真机显示验证：

- YOLO11s 作为正式 detector；
- 移除运行路径中的 YOLOv5 和旧 Dashboard；
- 建立 1280×800 横屏全屏检测界面；
- 保持摄像头原始 1280×720 输入比例；
- 为后续 ENTER、EXIT、DWELL 事件测试保留 ROI 和跟踪接口。

本阶段没有加入网络传输、RGA、零拷贝、新模型或视频录制功能。

## 2. 新增和调整的主要模块

### DisplayComposer

新增最终显示合成模块：

- `include/edgevision/display_composer.hpp`
- `src/display_composer.cpp`

主要职责：

- 将摄像头画面合成为 1280×800 显示帧；
- 绘制检测框、类别、置信度和跟踪 ID；
- 绘制右上角 Objects 数量标识；
- 绘制临时 ENTER、EXIT、DWELL 提示；
- 可选绘制 ROI；
- 统一计算 crop、缩放和坐标映射；
- 复用显示缓冲，避免重复 resize 和重复整帧绘制。

### Application 显示路径

`src/application.cpp` 已统一接入 DisplayComposer：

- 图片模式；
- 普通视频模式；
- 摄像头模式；
- smooth-preview 平滑摄像头模式。

smooth-preview 继续采用一个 AI worker 和最新帧槽位，避免堆积旧帧。主线程负责摄像头显示和窗口事件处理。

### YOLOv5 旧路径清理

已从正式构建路径删除：

- `Yolov5Detector` 头文件和实现；
- 旧 Visualizer；
- YOLOv5 postprocess 测试；
- 旧 Dashboard 预览工具；
- 对应 CMake target 和测试引用。

YOLOv5 仍保留在历史提交和 rollback 参考中，没有恢复为正式 detector。

### CLI 和测试

新增或保留：

- `--show-roi`；
- `--roi X,Y,W,H`；
- DisplayComposer 几何和坐标映射测试；
- CLI 参数和 ROI 参数校验；
- RegionMonitor 的 ENTER、EXIT、DWELL 单元测试。

## 3. 显示技术方案

摄像头输入为 1280×720，物理屏横屏逻辑分辨率为 1280×800。

采用显示层 aspect-fill：

- AI 输入仍使用完整的 1280×720 摄像头帧；
- 不旋转 camera frame；
- 不裁剪 AI 输入；
- 不拉伸 camera frame；
- 仅在最终显示阶段进行居中裁剪和缩放。

对于 1280×720 → 1280×800：

- 横向裁剪源图约 64 px 两侧；
- 显示帧为 1280×800；
- 检测框先在原始图坐标中生成，再映射到显示坐标；
- ROI 判断仍基于完整原始摄像头帧。

这样可以让画面占据屏幕大部分，同时保持 16:9 画面比例。

## 4. 事件功能现状

RegionMonitor 支持三类事件：

- `ENTER`：跟踪目标中心点进入 ROI；
- `EXIT`：跟踪目标中心点离开 ROI；
- `DWELL`：目标持续位于 ROI 内约 3 秒。

事件当前通过短时 toast 显示在画面下方，最多保留 3 条，单条显示约 2.8 秒。

当前正在运行的全屏预览没有传入 `--roi`，因此事件监控尚未启用。正式测试需要使用类似命令：

```text
--roi 0.25,0.20,0.50,0.60 --show-roi
```

测试时应让物体缓慢进入 ROI、停留超过 3 秒，再缓慢移出。事件判断使用检测框中心点，而不是检测框与 ROI 的部分重叠。

## 5. 构建和真机验证

### Ubuntu VM

- ARM64 交叉构建通过；
- `EDGEVISION_WITH_VIDEO=OFF` 构建通过；
- 无新增编译警告。

### RK3568

- 使用项目本地 RKNN 1.6.0 头文件和运行库；
- 未覆盖 `/lib/librknnrt.so` 或 `/usr/lib/librknnrt.so`；
- native RK3568 构建通过；
- ARM64 可执行文件生成成功；
- CTest：1/1 通过；
- YOLO11s RKNN 模型成功加载并完成推理；
- 摄像头成功打开，输入为 1280×720，GStreamer 管线正常；
- 全屏横屏预览已在物理屏运行，用户确认“画面正常”。

### bus.jpg 回归结果

本阶段自有代码在 RK3568 上返回码为 0，主要结果为：

| 类别 | 置信度 | bbox `(x, y, w, h)` |
|---|---:|---|
| bus | 0.942 | `(91, 133, 550, 437)` |
| person | 0.903 | `(475, 231, 559, 521)` |
| person | 0.896 | `(211, 239, 283, 508)` |
| person | 0.896 | `(108, 237, 225, 536)` |
| person | 0.641 | `(79, 326, 126, 519)` |

静态最终 UI 图片已生成并确认可读，尺寸为 1280×800。

## 6. Git 状态

当前分支：

```text
feature/v3-phase3-final-ui
```

本阶段主要提交：

```text
db0c3e0 refactor: remove legacy YOLOv5 runtime code
797c50d feat: replace dashboard with fullscreen detection UI
```

本阶段提交尚未推送到 GitHub。本总结文档也是本地阶段记录，不作为本次推送内容。

## 7. 当前功能状态

已完成：

- YOLO11s 正式 detector 路径；
- 全屏横屏检测 UI；
- 1280×720 摄像头画面比例保持；
- 检测框、标签、置信度、跟踪 ID；
- Objects 数量显示；
- ROI 和事件 toast 接口；
- RK3568 真机摄像头显示验证；
- bus.jpg 回归验证。

尚未完成或暂缓：

- 600 帧或 30 秒显示管线性能基线；
- 现场 ENTER、EXIT、DWELL 测试；
- 视频 pipeline 和 codec fallback；
- PC 端摄像头画面传输；
- 最终 README、发布和许可证审计。

## 8. 当前停止点

V3 Phase 3 的最终 UI 已完成物理屏显示确认。下一步应先启动带 ROI 的实时预览，再由用户逐步摆放和移动物体，验证 ENTER、DWELL、EXIT 事件；在此之前不进行自动连续场景采集。
