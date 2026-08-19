# Final validation record

## Current launcher smoke

Measured on 2026-08-19 after the one-command launcher was completed:

| Check | Result |
| --- | --- |
| SSH reachability | PASS |
| Board EdgeVision start | PASS |
| PC raw preview process | PASS |
| MJPEG/TCP sender | PASS |
| YOLO11 RKNN status | PASS |
| RK local display path | PASS |
| RTSP preview process | PASS |
| TCP event subscriber | PASS |
| Real business event | `DWELL,cup` received |
| Ctrl+C and normal stop | PASS |
| Cold restart | PASS |

The board `GET_STATUS` response during the smoke reported:

```text
camera_fps=13.954
display_fps=15.178
detection_fps=5.074
```

The short smoke did not measure a new YOLO-only duration, stabilizer-only
duration, CPU average, or memory average. Those values are therefore not
claimed as current final measurements.

## Earlier engineering evidence

The previous board validation used the same YOLO11 RKNN model and a 30-frame
MJPEG/TCP loopback. It reported approximately 16.06 capture FPS, 18.70 display
FPS, and 4.99 detection FPS. This is historical evidence, not a replacement
for the current launcher measurement above.

The live detection report found no near-identical duplicate boxes in the
sampled stabilized frames and no frame-level logical-ID explosion for the
tested cup. It did not constitute a labeled precision/recall benchmark.

## Acceptance boundary

The project is suitable for a short undergraduate project demonstration. It
does not claim formal detector precision, recall, cross-lifecycle identity, or
long-duration thermal/power stability. Persistent identity across a complete
exit and re-entry would require ReID and is not implemented.
