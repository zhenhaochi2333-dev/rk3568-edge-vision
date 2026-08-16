# Phase 5 Golden Regression Checkpoint

Validation was performed on the RK3568 in the project-specific temporary
directory `/root/edgevision_phase5_validation` using the cross-built ARM64
executable and a project-local `lib/librknnrt.so`.

## Command

```text
./edge_vision --model models/yolov5s_rk3568_i8.rknn \
  --labels assets/coco_80_labels_list.txt \
  --input assets/bus.jpg \
  --output output/bus_edgevision.png \
  --conf 0.25 --nms 0.45
```

## Result

```text
person @ (211 240 283 518) 0.836
person @ (475 231 560 520) 0.798
person @ (114 235 207 543) 0.796
bus    @ (90 133 553 461) 0.783
person @ (77 336 122 515) 0.399
```

Return code was `0`. The output was a readable 640x640 RGB PNG. The same
executable rejected an existing output without `--force` and returned `1`.

The executable was verified as ARM64 and `ldd` resolved `librknnrt.so` from
`/root/edgevision_phase5_validation/lib`. SHA-256 hashes of the project-local,
`/lib`, and `/usr/lib` RKNN runtime files were equal before/after the run; no
system runtime was overwritten.
