# Adaptive Sweet Spot (ASS)

Real-time spatial audio system that keeps the 5.1 surround "sweet spot" centered
on the listener. A webcam tracks the listener's head with MediaPipe BlazeFace, and
a DSP engine continuously adjusts per-channel delays and levels so the sound
stays centered as the listener moves.

## Components

| Module        | Role |
|---------------|------|
| `vision/`     | Webcam capture, BlazeFace detection, pixel → real-world offset conversion |
| `interface/`  | Bridge passing live (vision) or manual offsets to the DSP engine |
| `dsp/`        | 6-channel playback, per-channel delay/gain steering, artifact-free updates |
| `config/`     | Speaker layout, camera pose, system settings |
| `tools/`      | Per-milestone test scripts and calibration utilities |
| `assets/`     | Test audio (`assets/audio/`) and model weights (`assets/models/`) |
| `tests/`      | Unit tests |

## Hardware

- Compute: dev host → Qualcomm QIDK (edge MVP)
- Vision: USB webcam
- Audio: UAC 2.0 6-channel DAC → Logitech Z906 (5.1, discrete analog inputs)
- Constraint: motion-to-sound latency < 25 ms

## Conventions

- Units: meters, seconds, degrees.
- Room frame: origin at the designated center (baseline sweet spot),
  `+x` right, `+y` forward (toward the front speakers), `+z` up.
- Azimuth: clockwise from `+y` (front), so FR = +30°, FL = −30°.
- Channel order (WAV 5.1): `FL, FR, FC, LFE, SL, SR` → output channels 0–5.

## Milestones

| Phase | Milestone | Description | Status |
|-------|-----------|-------------|--------|
| 1 | 1.1 | Static audio centering (6-ch playback, fixed speaker layout) | ☐ |
| 1 | 1.2 | BlazeFace validation on webcam feed | ◐ tooling done, live validation pending |
| 2 | 2.1 | DSP sweet-spot shifting via delays + levels | ☐ |
| 2 | 2.2 | BlazeFace detections → real-world offsets | ◐ tooling done, live validation pending |
| 3 | 3.1 | Manual re-centering interface | ☐ |
| 3 | 3.2 | Vision → interface live pipeline | ☐ |
| 4 | 4.1 | QIDK deployment, end-to-end latency < 25 ms | ☐ |
| 5 | 5.0 | Arbitrary layouts and head rotation | ☐ |
| 6 | 6.0 | Automated speaker discovery / calibration | ☐ |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # core + tests
pip install -e ".[audio,vision]"   # hardware dependencies
pytest
python tools/fetch_model.py        # BlazeFace weights -> assets/models/
```

## Milestone 1.2 — BlazeFace validation

```bash
python tools/m1_2_vision_check.py              # test window
python tools/m1_2_vision_check.py --log m12.csv
python tools/m1_2_vision_check.py --list-cameras   # show connected cameras
python tools/m1_2_vision_check.py --camera 1       # start on a specific camera
```

All connected cameras are detected at startup; press `c` in the window to switch
between them (index 0 is usually the built-in camera).

Press a key to run a guided test (3 s countdown, then recording); results show
in the window and print to the terminal on exit:

| Key | Test | What to do | Pass |
|-----|------|------------|------|
| 1 | Still | Sit still, look at the camera (10 s) | detect ≥ 95%, jitter < 10 mm |
| 2 | Yaw | Slowly turn head fully left, then right (10 s) | reaches ±30°, detect ≥ 90% |
| 3 | Move | Move and lean at normal speed (15 s) | detect ≥ 95% |
| 4 | Distance | Slowly move back to your listening position (15 s) | info: does detection hold? |

`latency` is frame read → detection ready and excludes sensor/USB delay.

Head angles are geometric estimates from the keypoints (`vision/head_pose.py`):
yaw from the nose offset relative to the ear midpoint, roll from the eye line.
Image-space signs: yaw > 0 when the nose moves toward image right.

Distance: BlazeFace only sees a 128/192 px copy of its input, so faces far away
are too small in a full frame. `vision/tracker.py` handles this with two simple steps:

- **track** — face found last frame → detect only in a crop `roi_scale` × the face
  width around it (one inference per frame).
- **search** — no face → full frame, then 2×2, then 3×3 overlapping tiles.

Synthetic check (1280×720, 70° HFOV, Mac CPU):

| Model | Full frame only | With tracker (acquire) | Tracking cost |
|-------|-----------------|------------------------|---------------|
| `blaze_face_full_range` (default) | ~2 m | ~4 m | ~4.6 ms |
| `blaze_face_short_range` | ~0.9 m | ~1.5 m | ~1.6 ms |

Detection rate (`vision/rate.py`): the camera runs at 30 fps but the detector
only runs as often as head motion needs — `fps = head speed / (tolerance / 2)`,
with `tolerance_mm = 30` (~0.1 ms L/R arrival error). Still head → 5 fps,
0.3 m/s → 20 fps, ≥ 0.5 m/s → every frame. Skipped frames get a cheap motion
check (32×32 thumbnail of the face region) so a sudden move is caught at once.

Compute (counted from the model graphs):

| Model | Input | MACs | GFLOPs / inference | at 30 fps |
|-------|-------|------|--------------------|-----------|
| `blaze_face_short_range` | 128×128 | 31 M | 0.06 | 1.9 GFLOPS |
| `blaze_face_full_range` | 192×192 | 106 M | 0.21 | 6.3 GFLOPS |

Known limits:
- Ear keypoints become unreliable beyond roughly ±50° yaw.
- Searching (no face yet) can cost several inferences per frame.
- Default capture is 30 fps; the frame interval adds up to 33 ms of delay.

## Milestone 2.2 — Listener offset from the center

Assumes the camera sits **on top of the front center (FC) speaker**, level, facing
the listener. The designated center is then straight ahead of the camera at the
FC distance `D` (from `config/default.toml`). Output is the top-down offset
`(x, y)` in meters (`vision/spatial.py`). The ruler is the BlazeFace **face-box
width** `bw` (real width `Wf`):

```
distance  Z = scale · f · Wf / bw
lateral   x = −(u − cx) · Wf / bw − x_bias   (face width as a ruler; image right = room left)
front     y = D − Z                         (+ = toward the front speakers)
```

Why not the eye spacing: turning the head shrinks it by cos(yaw) (40° → distance
30% too far). The face-box width stays within ~2% up to ~50° yaw (measured in the
m1.2 log) and doesn't change when nodding.

```bash
python tools/m2_2_spatial_check.py
```

| Key | Action |
|-----|--------|
| `k` | Calibrate: sit at the intended center, face the camera, press `k` (1.5 s capture) |
| space | Record your averaged position, to compare with a tape measure |
| `c` / `q` | Switch camera / quit |

Calibration learns your face width `Wf` (from the eye spacing, while facing the
camera), `scale` (real camera FOV, so distance is exact at the center) and `x_bias`
(camera not exactly on the center line); it is saved to `config/calibration.json`.
Lateral accuracy then depends only on the IPD assumption — set `vision.ipd_m` to
your own value (the 63 mm default can be ~8% off).

Known limits: beyond ~50° yaw the box narrows (~10%); height and the eye→ear
offset are ignored.
