# EdgeVision V3 Phase 1 — YOLO11s migration checkpoint

Date: 2026-08-17

This checkpoint records the first end-to-end YOLOv5s → YOLO11s migration. The
YOLOv5 implementation, model files, and validation directories remain intact;
the formal application target now uses YOLO11s while the old detector remains
available through a legacy test library.

## Source and model provenance

- Reference repository: `airockchip/rknn_model_zoo`.
- Reference release: `v2.3.2`, commit
  `bad6c7334531becaf90a561988519b7bec34d0ab`.
- Reference code license: Apache License 2.0, as stated by the upstream
  `LICENSE` file.
- YOLO11 model source: upstream optimized `examples/yolo11/model/yolo11s.onnx`
  artifact, whose README identifies the upstream `airockchip/ultralytics_yolo11`
  project. The optimized artifact is not the unchanged original Ultralytics
  export; its redistribution terms need a separate product/legal check.
- ONNX SHA-256:
  `691f87bd7eccb272ba0e8c884ab960964b10bd192c3d46bfb8a4d63ff91e2cc0`.
- ONNX metadata: IR version 7, opset 12, input `images` `[1,3,640,640]`
  `float32`, nine `float32` outputs.
- RKNN conversion: RKNN-Toolkit2 `2.3.2`, target `rk3568`, quantization `i8`.
- RKNN SHA-256:
  `caf30c2c21333ebbbbc2369dcab0af0aa672c3bdd03ddcd348954f6fd470ee5a`.

The conversion used the upstream YOLO11 conversion route and COCO subset
calibration data. Toolkit warnings were retained: two weight outliers were
reported, and the input plus all nine outputs were converted from float32 to
int8. No board runtime upgrade was performed.

## Tensor contract observed on RK3568

The board queried one NHWC RGB input and nine NCHW INT8 outputs under the
existing RKNN runtime 1.6.0:

```text
input  [1,640,640,3]  UINT8/NHWC  zp=-128 scale=0.00392157

stride 8:  [1,64,80,80], [1,80,80,80], [1,1,80,80]
stride 16: [1,64,40,40], [1,80,40,40], [1,1,40,40]
stride 32: [1,64,20,20], [1,80,20,20], [1,1,20,20]
```

The generic `RknnModel` now owns only lifecycle, metadata, input preparation,
and output lifetime. `Yolo11Detector` owns the nine-output grouping, INT8/UINT8
dequantization, DFL, anchor-free decode, class-wise NMS, and letterbox restore.
No raw RKNN output pointer escapes the `RknnOutputBatch` lifetime.

## Code and test result

- Formal `edge_vision` target uses `Yolo11Detector`.
- YOLOv5 detector source is retained in `edgevision_yolov5_legacy` for existing
  golden postprocess tests; it is not linked into the formal application.
- Added synthetic YOLO11 FP32/NCHW decode and class-wise NMS tests.
- ARM64 native build on the RK3568 with video enabled: passed.
- ARM64 CTest on the RK3568: 1/1 test passed.
- Ubuntu VM AArch64 cross-build with video disabled: passed with
  `aarch64-linux-gnu-g++ 7.5.0`. The VM's ARM64 OpenCV package lacks
  `videoio/highgui`; video validation therefore uses the board's existing
  OpenCV 4.2.0 native build.

## Board validation

Board baseline: RK3568 AArch64, Ubuntu 20.04.6, kernel 4.19.232,
`librknnrt.so` 1.6.0. The project-local runtime copy used by the executable
matches `/lib/librknnrt.so`, SHA-256
`9c5e43179e83f82b3ec1f2c2f9b520559af23ab76327aca63500db30d63d6910`, and is
loaded through `$ORIGIN/lib`.

On the same `bus.jpg` frame, YOLO11s produced 5 detections: 1 bus and 4
persons. The annotated image was visually checked and the boxes were
reasonable. One static run measured approximately:

```text
preprocess 25.0 ms, inference 152.9 ms, postprocess 4.5 ms, e2e 206.0 ms
```

The earlier static run measured 110.7 ms inference; the board-side variance is
kept as an observation rather than a fixed benchmark claim. The old YOLOv5s
binary on the same frame also produced 5 detections and measured approximately
67.0 ms inference, so the migration currently costs latency and needs a later
performance pass.

Headless camera validation opened `/dev/video0` at 1280x720 NV12/GStreamer,
wrote a 35-frame MP4, and reported 30 warmup plus 5 measured frames:

```text
inference 105.8 ms, postprocess 4.2 ms, full_loop 207.7 ms, actual_fps 4.81
display 0.0 ms
```

No GUI or display command was used during this validation.

## Remaining work

- Run longer camera/tracker/ROI/reopen/shutdown validation after user approval
  of the current YOLO11 board baseline.
- Compare a larger same-frame/image set against the YOLOv5 golden data and
  record recall/precision deltas.
- Decide whether the current official optimized YOLO11s i8 model meets the
  latency target; do not change the board runtime without a compatibility and
  rollback record.
- Keep YOLOv5 source/history and the `backup/yolov5s-stable-final` branch until
  the migration is explicitly accepted.
