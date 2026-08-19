# Phase 10 Semantic Stabilizer Validation

Date: 2026-08-19

## Board build and correctness

The board is an RK3568 ARM64 system running Linux 4.19.232 with four logical
CPUs and OpenCV 4.2.0.

Two independent Release builds were completed in the board-only staging tree
`/root/edgevision_phase10_build`:

- `build-native-phase10`: `EDGEVISION_WITH_VIDEO=ON`, RTSP disabled;
- `build-native-phase10-rtsp`: `EDGEVISION_WITH_VIDEO=ON`,
  `EDGEVISION_WITH_RTSP=ON`.

Both builds produced `edge_vision` and `edgevision_tests`. Both CTest runs
passed: 1/1 tests passed.

The RTSP-enabled build also completed a 30-frame network smoke test using the
board's `ffmpeg` as an MJPEG sender:

```text
MJPEG sender -> TCP 127.0.0.1:5600 -> POSIX socket/JPEG decode -> BGR
              -> YOLO11 -> tracker -> SemanticStabilizer -> RTSP output
```

Observed board log:

```text
displayed_frames=30
completed_inferences=8
display_fps=18.699760
detection_fps=4.986603
captured_fps=16.061252
RTSP detection stream=rtsp://192.168.77.2:8554/live
```

The sender exits with Broken pipe after the application reaches
`--max-frames 30`; this is expected because the board closes the input after
the requested frame count.

## Offline performance comparison

The comparison uses the same YOLO11 RKNN model, the same 80-frame
`bus_regression_80.avi` input, and the same output path behavior. BEFORE is the
existing phase5 semantic-check ARM64 binary; AFTER is the current local commit
`09f1f1e`.

| Metric | BEFORE | AFTER | Change |
| --- | ---: | ---: | ---: |
| total wall time | 28.2806 s | 28.0175 s | -0.93% |
| throughput | 2.9669 FPS | 3.0009 FPS | +1.15% |
| preprocess | 7.1945 ms | 7.1870 ms | -0.10% |
| inference | 131.5375 ms | 128.3349 ms | -2.43% |
| postprocess | 4.9539 ms | 4.7983 ms | -3.14% |
| visualization | 11.7445 ms | 11.5005 ms | -2.08% |
| end-to-end | 161.6077 ms | 157.5136 ms | -2.53% |

A separate sampled run reported peak RSS of 128208 KB BEFORE and 128400 KB
AFTER (+192 KB, about +0.15%). Instantaneous sampled peak CPU was 101% BEFORE
and 117% AFTER; this is a coarse peak sample rather than a CPU average and is
not used as a regression verdict.

The result is an indicative board regression pass: measured throughput and
latency did not degrade, and the metadata-only stabilizer added negligible
memory. A longer repeated-run CPU average should be collected before making a
formal power/thermal claim.
