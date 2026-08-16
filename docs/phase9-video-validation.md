# Phase 9 video validation

## Build decision

The verified Ubuntu ARM64 OpenCV 3.4.5 package does not provide `videoio` or
`highgui`, so the bounded Plan A video configuration stopped at CMake package
discovery. No toolkit, runtime, driver, kernel, or system package was changed.

Plan B compiled natively on RK3568 with the existing system OpenCV 4.2.0 and
the project-local RKNN 1.6.0 header/runtime staging. The executable is AArch64
and uses `$ORIGIN/lib` for `librknnrt.so`.

## Input provenance

`assets/bus_regression.avi` and the longer `bus_regression_80.avi` were generated
by `scripts/make_regression_video.cpp` from the approved `assets/bus.jpg` using
OpenCV's MJPG writer. They are engineering regression inputs, ignored by Git,
and are not final showcase media.

## RK3568 results

Validation directory:

```text
/root/edgevision_phase9_validation
```

MP4 run:

- Input: 640 x 640, FFmpeg backend, 5 FPS, 8 frames.
- Output: `output/demo.mp4`, requested/actual codec `mp4v`/MPEG-4.
- ffprobe: 640 x 640, 5/1 FPS, 8 frames.
- Application output verification reopened the file and read a frame.
- Return code: 0.

Fallback and shutdown:

- `edgevision_tests` passed the MJPG/AVI fallback writer path by forcing the
  MP4 request through the fallback policy and reopening the AVI output.
- An 80-frame input was interrupted with SIGINT during processing. The program
  wrote a readable partial MP4, released resources through RAII, logged the
  clean-stop warning, and returned 0.

The post-video bus.jpg image regression again produced the locked five
detections and a readable 640 x 640 PNG with return code 0.

