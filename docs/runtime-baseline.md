# Runtime baseline

This record is for the Phase 5.5/6 engineering checkpoint. It describes the
runtime that was observed; it does not authorize changing the board or its
system packages.

## RK3568

- Board: AArch64 RK3568, Ubuntu 20.04.6, kernel `4.19.232`.
- Phase 5 executable: ARM64 ELF, dynamically linked.
- Executable RUNPATH: `$ORIGIN/lib`.
- `ldd` resolves `librknnrt.so` to the project-local deployment copy under
  `/root/edgevision_phase5_validation/lib/librknnrt.so`.
- No OpenCV shared objects are required by the Phase 5 executable because the
  image path was linked against the verified ARM64 static OpenCV package.
- Project-local RKNN runtime SHA-256:
  `9c5e43179e83f82b3ec1f2c2f9b520559af23ab76327aca63500db30d63d6910`.
- The board's `/lib/librknnrt.so` and `/usr/lib/librknnrt.so` were not modified.

## Board OpenCV

The board reports OpenCV `4.2.0`. The installed package provides `videoio` and
`highgui` shared libraries. `opencv_version --verbose` reports FFmpeg and
GStreamer support, plus GTK support for display. The video implementation will
use this existing runtime only; it will not install or upgrade multimedia
packages.

## Ubuntu cross-build OpenCV

The verified Model Zoo ARM64 package is OpenCV `3.4.5` and provides the static
image modules used by Phase 5/6 (`core`, `imgproc`, and `imgcodecs`). A
`videoio` library was not present in that package during inspection. The
cross-build remains the preferred path for image validation; video validation
may use a native RK3568 build if the cross package cannot provide videoio.

## Verification commands

For every deployed executable, inspect:

```text
file edge_vision
readelf -d edge_vision
ldd edge_vision
```

The expected RKNN deployment remains project-local, for example:

```text
edgevision_v1/
  edge_vision
  lib/librknnrt.so
  models/
  assets/
  output/
```

