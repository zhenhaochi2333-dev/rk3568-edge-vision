# Engineering decisions

## Why the input is MJPEG/TCP

The earlier PC-to-board H.264/RTP/MPP input path was investigated with software
and live hardware decoding. The live Rockchip MPP path showed incomplete-frame,
reset, and timestamp-related corruption while the same general stream could be
decoded in a software path. The exact integration was not stable enough for a
small final demo.

The input was therefore reduced to:

```text
PC FFmpeg MJPEG -> TCP :5600 -> JPEG framing -> OpenCV imdecode -> BGR
```

This keeps the application-facing camera contract unchanged. It also avoids a
second GStreamer camera owner and avoids carrying a decoder diagnostic stack
into the formal path.

H.264 was not removed from the project completely. It remains on the output
side only:

```text
annotated BGR -> NV12 -> mpph264enc -> h264parse -> rtph264pay -> RTSP
```

## Identity boundary

The raw IoU tracker is not the business identity authority. The semantic layer
uses short detector gaps, geometry, IoU, confidence-weighted class evidence,
and time-based presence to maintain a `logical_id`. `RegionMonitor` converts
that lifecycle into one ENTER, optional one DWELL, and one EXIT.

The current identity is continuous presence identity. ReID or appearance
embeddings are future work, not an active dependency.
