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
python tools/m1_2_vision_check.py              # live overlay: box, 6 keypoints, fps, latency, yaw/roll
python tools/m1_2_vision_check.py --log m12.csv --duration 60
python tools/m1_2_vision_check.py --video clip.mp4
```

The window shows the face box, the six keypoints (eyes, nose, mouth, ear tragions)
and live stats. Hold still to read `jitter` (RMS eye-midpoint wobble in pixels).
`latency` is frame-grabbed → detection-ready and excludes sensor/USB delay.

Head angles are geometric estimates from the keypoints (`vision/head_pose.py`):
yaw from the nose offset relative to the ear midpoint, roll from the eye line.
Image-space signs: yaw > 0 when the nose moves toward image right.

Known limits:
- `blaze_face_short_range` is tuned for faces within ~2 m of the camera.
- Ear keypoints become unreliable beyond roughly ±50° yaw.
- A 30 fps camera alone adds up to 33 ms between frames; the 25 ms budget needs
  a ≥60 fps camera and/or prediction (Milestone 3.2).
