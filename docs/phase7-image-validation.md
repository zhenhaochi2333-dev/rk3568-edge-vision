# Phase 7 image validation

The polished image path was rebuilt for ARM64 and validated in the new board
directory `/root/edgevision_phase7_validation`.

Checks passed:

- `file` identifies an AArch64 ELF executable.
- `readelf -d` reports RUNPATH `$ORIGIN/lib`.
- `ldd` resolves the project-local `lib/librknnrt.so` with no missing library.
- `edgevision_tests` reports `PASS`.
- `bus.jpg` output is readable PNG, 640 x 640, RGB.
- Return code: 0.

Detections:

```text
person 0.836  (211,240)-(283,518)
person 0.798  (475,231)-(560,520)
person 0.796  (114,235)-(207,543)
bus    0.783  (90,133)-(553,461)
person 0.399  (77,336)-(122,515)
```

The result remains reasonably equivalent to the locked Rockchip v1.6.0 Golden
Reference. Generated output and logs remain outside normal Git history.

