# SemanticStabilizer

The runtime path is:

```text
YOLO11 detections -> IouTracker(raw track_id) -> SemanticStabilizer(logical_id)
                 -> RegionMonitor -> display / TCP events / RTSP output
```

`SemanticStabilizer` is intentionally metadata-only. It does not add a model,
worker thread, image copy, crop, resize, ReID, or a second tracking system.

It provides four protections for the event layer:

- short raw-track gaps are reassociated by raw id first, then class-agnostic
  IoU/center distance;
- class labels are fused over time and a class switch requires consecutive
  observations;
- presence is accumulated for a three-observation cold start and decayed while
  an object is missing;
- only confirmed, currently observed objects are exposed to `RegionMonitor`,
  so cold-start detections cannot emit an early `ENTER` event.

The old `track_id` field remains on `Detection` and `RegionEvent` for source
compatibility. Business code should use `logical_id`; TCP includes both fields
with the same stable value for compatibility with existing clients.
