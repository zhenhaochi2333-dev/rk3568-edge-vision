# SemanticStabilizer

The runtime path is:

```text
YOLO11 detections -> IouTracker(raw track_id) -> SemanticStabilizer(logical_id)
                 -> RegionMonitor -> display / TCP events / RTSP output
```

`SemanticStabilizer` is a metadata-only layer. It does not add a model,
thread, image copy, crop, resize, ReID feature, or second tracker.

## Lifecycle contract

Each logical object has exactly one of these states:

- `CANDIDATE`: presence and ENTER hysteresis are still accumulating.
- `ACTIVE`: the object is stable and is exposed to the business layer.
- `LOST_PENDING`: the raw observation is missing; the object may recover
  without another ENTER.
- `EXITED`: the real-time lost window expired. The object is never reused for
  reassociation and is retained only briefly for bounded cleanup.

The default time parameters are:

| Parameter | Default |
| --- | ---: |
| reassociation window | 1.2 s |
| maximum lost time | 2.0 s |
| presence detection gain | 2.0 / s |
| presence missing decay | 0.5 / s |
| ENTER threshold | 0.60 |
| EXIT threshold | 0.20 |
| ENTER stability | 0.40 s |
| bootstrap mute | 3.0 s |
| exited retention | 5.0 s |

Presence is updated from elapsed time, not detector frame counts. Reassociation
uses raw id first, then normalized center distance, bbox-size similarity, IoU,
and the real-time gap. Class mismatch is not a hard rejection. The geometric
fallback is greedy and intentionally avoids ReID, Hungarian matching,
DeepSORT, and ByteTrack.

Class semantics use confidence-weighted sparse evidence. Missing observations do
not rapidly erase the previous class. A new class must exceed the old evidence
by the configured ratio and lead for the configured hold time before a stable
class switch.

## Business-layer behavior

Only currently observed `ACTIVE` objects are returned from the stabilizer.
`RegionMonitor` keeps an ACTIVE ROI lifecycle during short
`LOST_PENDING` gaps, pauses DWELL during the gap, and emits one delayed EXIT
after the lost window. A recovery with the same logical id does not emit a
second ENTER. `DWELL` is emitted once per lifecycle; a true EXIT ends that
lifecycle.

The first three seconds run normal association, presence, and class fusion but
mute ENTER. Objects that become stable during this window are marked as
baseline objects, so no ENTER is backfilled later. Objects first seen after the
window follow the normal ENTER path.

The default ROI is the complete frame (`NormalizedRoi{}`); `--roi` still
supports a narrower normalized ROI. Display overlays show only stable class,
logical id, and bbox. TCP events use `logical_id` as the business identity and
retain `track_id` only as a compatibility alias carrying the same value.

## Validation checklist

The host unit suite covers:

1. same raw id through a short miss;
2. raw id change reassociation;
3. low-confidence class flicker;
4. sustained class switch;
5. far-object non-merge;
6. new id after the reassociation window;
7. ENTER hysteresis;
8. EXIT hysteresis;
9. one DWELL per lifecycle;
10. bootstrap mute without backfilled ENTER.

Board performance must be recorded separately from correctness tests using the
same input scenario before and after the stabilizer. The stabilizer itself is
metadata-only, so its expected CPU and memory cost is negligible; the board
report must still record the measured detection/display FPS, end-to-end
latency, CPU, and memory and distinguish historical baselines from current
measurements.
