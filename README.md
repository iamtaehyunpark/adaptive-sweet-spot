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
| 2 | 2.2 | BlazeFace detections → real-world offsets | ☐ |
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
| 1 | Still | Sit still, look at the camera (10 s) | detect ≥ 95%, jitter < 2 px |
| 2 | Yaw | Slowly turn head fully left, then right (10 s) | reaches ±30°, detect ≥ 90% |
| 3 | Move | Move and lean at normal speed (15 s) | detect ≥ 95% |
| 4 | Distance | Slowly move back to your listening position (15 s) | info: does detection hold? |

`latency` is frame read → detection ready and excludes sensor/USB delay.

Head angles are geometric estimates from the keypoints (`vision/head_pose.py`):
yaw from the nose offset relative to the ear midpoint, roll from the eye line.
Image-space signs: yaw > 0 when the nose moves toward image right.

Known limits:
- `blaze_face_short_range` is tuned for faces within ~2 m of the camera.
- Ear keypoints become unreliable beyond roughly ±50° yaw.
- A 30 fps camera alone adds up to 33 ms between frames; the 25 ms budget needs
  a ≥60 fps camera and/or prediction (Milestone 3.2).
