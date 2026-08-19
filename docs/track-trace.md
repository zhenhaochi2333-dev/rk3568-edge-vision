# Detection pipeline trace

Use `--track-log /tmp/edgevision-tracks.csv` on the RK3568 process. The CSV is
diagnostic output only; it does not change the inference pipeline.

Each inference frame writes rows for:

- `raw`: YOLO11 decoded detections after confidence filtering and NMS, before tracking.
- `tracked`: detections after raw IoU tracking.
- `stabilized`: logical detections that are eligible for drawing and events.
- `rtsp_publish`: the annotated frame was handed to the RTSP streamer.
- `local_display`: the annotated frame reached the board display loop.

Empty stages are written as rows too. `decoder_candidates` and
`nms_suppressed_count` distinguish “YOLO produced no candidate” from “candidates
were removed by NMS.” The timing columns use the first captured frame as the
zero point. `detector_ms`, `tracker_ms`, and `stabilizer_ms` are stage durations;
`output_latency_ms` is from stabilizer completion to the corresponding output
stage.

Example network-camera launch:

```text
./edge_vision --model /root/edgevision_minimal/models/yolo11s_rk3568_i8.rknn \
  --labels /root/edgevision_minimal/assets/coco_80_labels_list.txt \
  --input network --show --smooth-preview --tcp --tcp-port 9000 \
  --track-log /tmp/edgevision-tracks.csv
```

Copy the trace back to the PC after stopping the process:

```text
scp root@192.168.77.2:/tmp/edgevision-tracks.csv D:/rk3568/edgevision-tracks.csv
```
