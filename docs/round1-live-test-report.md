# 第一轮真人检测测试报告

日期：2026-08-19
测试方式：PC 摄像头 → TCP/MJPEG → RK3568 → YOLO11 → Tracker → SemanticStabilizer → 板端显示/RTSP 回传

## 1. 结论

就“目标检测稳定性”和“事件身份连续性”而言，本轮已经达到阶段目标：

- 没有发现同一个目标在同一画面中出现两个近同位置的框。
- 没有发现同一个 cup 因轻微晃动持续生成新 logical ID。
- person 目标的不同 logical ID 对应不同的连续进出阶段，而不是每帧刷新。
- Tracker 和 SemanticStabilizer 的计算开销很小，不是当前主要性能瓶颈。

但本轮还不能称为最终严格验收，原因是没有人工标注的 ground truth，也没有单独记录 ENTER/EXIT 事件报文的数量与时间。因此本报告可以确认稳定性和链路行为，不能替代 precision、recall 和事件准确率测试。

## 2. 测试数据

- 持续时间：约 163 秒
- YOLO 推理帧数：680 帧
- YOLO 实测检测速率：约 4.17 FPS
- PC 输入速率：约 14.0–14.4 FPS
- 板端显示速率：约 15.0 FPS
- 运行中检测对象数：约 2–6 个
- 原始 trace：`D:/rk3568/edgevision-live-round1.csv`
- 板端运行日志：`D:/rk3568/edgevision-live-round1.log`

## 3. 检测性能

| 项目 | 结果 |
|---|---:|
| YOLO 解码候选框平均数量 | 36.3/frame |
| NMS 平均抑制数量 | 30.8/frame |
| raw 完全空帧 | 0 |
| stabilized 空帧 | 2 |
| YOLO 加后处理平均耗时 | 200.55 ms |
| IoU Tracker 平均耗时 | 0.02 ms |
| SemanticStabilizer 平均耗时 | 0.10 ms |
| RTSP 输出延迟 | 平均 44.27 ms |
| 板端本地显示延迟 | 平均 54.38 ms |

当前主要瓶颈是 YOLO11 RKNN 推理，不是稳定器。稳定器对整体性能影响可以忽略。

## 4. 重复框和 ID 稳定性

对每一帧的 raw 和 stabilized 框进行了近同框检查，条件为：

- IoU ≥ 0.80
- 框面积相似度 ≥ 0.75

结果：

- raw 阶段近同框重复对：0
- stabilized 阶段近同框重复对：0

主要类别的 logical ID 表现：

| 类别 | logical ID 情况 | 判断 |
|---|---|---|
| person（class 0） | ID 1、27、38，分别对应约 0.7–41.5s、49.1–71.9s、78.9–162.9s | 不同连续进出阶段使用不同 ID，未出现帧级飙升 |
| cup（class 41） | 全程主要保持 logical ID 6 | 稳定 |
| class 62 | 主要保持 logical ID 3 | 稳定 |
| class 73 | 主要保持 logical ID 5 | 稳定 |

person 出现多个 ID 本身不是问题，因为它们对应三个明显分开的持续时间段；真正的问题是同一次进入期间 ID 是否不断变化，本轮没有观察到这种现象。

## 5. 事件判断

当前事件链使用 `logical_id`，而不是 YOLO 或 IoU Tracker 的原始 `track_id`：

```text
YOLO11
  → 同帧重复框清理
  → IoU Tracker
  → 原始 ID 重新关联
  → presence 分数和生命周期
  → logical_id
  → RegionMonitor 事件
```

因此短暂漏检或原始 track ID 变化不会立即产生新的物体，也不会立刻触发 EXIT/ENTER 抖动。

本轮 trace 能确认 logical ID 连续性，但板端日志中没有单独输出 ENTER/EXIT 报文，因此本轮无法给出事件数量、事件延迟和事件误报率的严格统计。事件逻辑本身已经接入稳定后的 logical object，但下一轮最好增加独立的事件 CSV 或事件计数日志。

## 6. 最终判断

### 阶段目标

**达到。**

当前已经解决了本轮最主要的两个问题：

1. 同一物体重复画两个近似框。
2. 同一物体因为晃动或原始 ID 变化而不断生成新身份。

### 最终验收

**暂不宣称完全验收。**

还需要带有明确人工动作标记和人工真值的测试，才能确认：

- person/cup 的漏检率；
- ENTER/EXIT 是否一一对应真实进出；
- 事件触发延迟；
- 不同物体靠近时是否会被错误合并。

本轮测试结果支持继续进入下一阶段的事件专项测试。
