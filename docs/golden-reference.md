# RKNN YOLOv5 Golden Reference — v1.6.0

This project references the Rockchip `rknn_model_zoo` checkout at tag `v1.6.0`.
Only the behavior needed by EdgeVision is recorded here; the Model Zoo is not
copied into this project.

## Exact reference files

- `examples/yolov5/cpp/rknpu2/yolov5.cc` — RKNN lifecycle, tensor queries, RGB UINT8 NHWC input, and output lifetime.
- `examples/yolov5/cpp/postprocess.cc` — quantized YOLOv5 decode, confidence filtering, sorting, NMS, and bbox restoration.
- `examples/yolov5/cpp/postprocess.h` — thresholds and result limits.
- `examples/yolov5/model/anchors_yolov5.txt` — anchor values.
- `utils/image_utils.c` / `utils/image_utils.h` — letterbox scale, alignment, padding, and metadata.
- `examples/yolov5/cpp/CMakeLists.txt` — ARM64 demo RPATH convention and runtime linkage.

## Locked model and tensor facts

- Input image buffer supplied by the application: RGB, UINT8, NHWC.
- The verified RKNN input descriptor is `RKNN_TENSOR_UINT8` and `RKNN_TENSOR_NHWC`.
- The model's queried input tensor may report INT8; RGB pixels are not manually quantized to INT8.
- Runtime output order is three YOLOv5 heads: 80x80, 40x40, and 20x20 grids.
- The verified RKNN attributes are NCHW tensors `[1, 255, 80, 80]`, `[1, 255, 40, 40]`, and `[1, 255, 20, 20]`, with INT8 affine quantization, `zp=-128`, and `scale=0.003922`.
- The semantic strides are 8, 16, and 32 respectively. Implementation maps heads from actual queried metadata instead of assuming ONNX ordering.

## Anchors

The three anchor groups, in stride order 8/16/32, are:

```text
stride 8:  10, 13, 16, 30, 33, 23
stride 16: 30, 61, 62, 45, 59, 119
stride 32: 116, 90, 156, 198, 373, 326
```

Each group is interpreted as three `(width, height)` pairs.

## Quantized decode

For the INT8 path, the reference dequantizes with:

```text
real = (int8_value - zero_point) * scale
```

The confidence threshold is also affine-quantized for the early objectness
and class checks. The reference uses the raw RKNN INT8 output; its C++ path
does not apply an additional sigmoid during `process_i8` or `process_fp32`.

For a head cell `(i, j)` and anchor `a`, with 85 values per anchor:

```text
x = (2 * dequant(tx) - 0.5 + j) * stride
y = (2 * dequant(ty) - 0.5 + i) * stride
w = (2 * dequant(tw))^2 * anchor_w
h = (2 * dequant(th))^2 * anchor_h
x -= w / 2
y -= h / 2
```

The best class is selected from channels 5 through 84. The detection score is
objectness multiplied by the best class probability. Defaults are confidence
`0.25` and NMS `0.45`.

## NMS and bbox restoration

The reference sorts scores descending and invokes NMS for each class in the
class set. Its literal implementation suppresses a later sorted candidate when
IoU is greater than `0.45`, using inclusive `+1` pixel overlap arithmetic.
EdgeVision preserves this v1.6.0 behavior rather than substituting a newer or
generic NMS implementation.

Letterbox restoration is:

```text
x1 = clamp(x - pad_x, 0, model_width) / scale
y1 = clamp(y - pad_y, 0, model_height) / scale
x2 = clamp(x + width - pad_x, 0, model_width) / scale
y2 = clamp(y + height - pad_y, 0, model_height) / scale
```

The reference letterbox chooses `min(dst_w/src_w, dst_h/src_h)`, floors the
non-limited resized dimension, aligns resized width down to a multiple of four
and height down to a multiple of two, centers padding, and aligns the active
left/top padding down to an even value. Padding color is `114`.

## Golden Regression baseline

The official RK3568 run on `bus.jpg` returned approximately:

```text
person @ (211 240 283 518) 0.836
person @ (475 231 560 520) 0.798
person @ (114 235 207 543) 0.796
bus    @ (90 133 553 461) 0.783
person @ (77 336 122 515) 0.399
```

Exact floating-point equality is not required. The own implementation must
preserve the classes, produce reasonably equivalent confidence values, and
match bbox geometry closely enough to identify the same objects.

Rockchip's implementation and related constants are third-party reference
material and must be attributed in `NOTICE` and project documentation.
