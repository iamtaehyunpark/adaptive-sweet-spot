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
| 3 | 3.2 | Vision → interface live pipeline | ◐ integrated app + UDP stream done; smoothing/prediction next |
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

## Sweet Spot Live (the integrated app)

```bash
python tools/app.py            # camera from the config; opens http://127.0.0.1:8765/
python tools/app.py --camera 1
python tools/app.py --no-camera   # room / speakers only
```

One page with everything:

- **Live** — camera → BlazeFace (tracker + adaptive rate) → listener position in the room
  frame, shown as a moving head (with yaw and a trail) in the **3D scan** and in a **top
  view**, next to the camera preview. Live x/y/z, distance, yaw, fps, latency, calibration
  and DSP-stream status.
- **Setup** — *Run setup* initializes everything from one capture: 5 s countdown (sit at
  your seat facing the camera), 2 s 1080p capture, MoGe-2 depth, calibration and the 3D
  scan; the page reloads the room when done. Tracking keeps running, and the tracking
  frames from the end of the countdown calibrate your face width exactly as tracking
  measures it.
- **Suggest** — suggested 5.1 positions on the scan (below), applied with one click.
- **Camera** — pick the camera from snapshots of every connected one (handy for a phone
  camera), choose its mode / fps, probe what the driver can adjust (zoom, focus, exposure)
  and adjust it, or press **Widest view** (minimum zoom + the full-sensor 4:3 mode where
  available). Saved in `config/camera.json`. Phone cameras choose their lens (e.g.
  ultra-wide) in the phone app. Changing the camera or its view invalidates the room setup;
  the page warns until Setup is run again, and Setup captures in the tracking aspect.
- **Speakers** — place speakers in 3D (drag arrows, Shift+click the scan, or type x/y/z);
  Save writes `config/room.json`, which `load_config()` returns to the DSP.
- **DSP stream** — every detection is sent over UDP (below).

### Speaker placement suggestion (Suggest tab, `interface/placement.py`)

Suggests where each speaker should go on the 3D scan for the best sweet spot, then lets
you apply the positions (and Save them for the DSP). FC is fixed: the camera sits on it.

| Guideline | Source | Used here |
|-----------|--------|-----------|
| All speakers on one arc around the listener (equal distance) | ITU-R BS.775 | arc radius = seat → FC; deviation penalized (the DSP compensates delay/level) |
| C 0°, L/R ±30° | ITU-R BS.775; Dolby allows 22–30° | ideal ±30°, allowed 22–30° |
| Surrounds 100–120° (110° reference) | ITU-R BS.775; Dolby 90–110° | ideal ±110°, allowed 90–120° |
| Tweeters at seated ear height; surrounds ear height (ITU) to 1–3 ft above (Dolby-style guides) | ITU / home-theater guides | fronts at ear height, surrounds +0.3 m |
| Mains 2–4 ft off the front wall, 2–3 ft off side walls; corners build up bass | speaker-placement guides | ≥ 0.6 m preferred for mains, surrounds may be wall-mounted |
| Aim (toe-in) at the listener | speaker-placement guides | every suggestion aims at the seat (arrow + tilt) |
| Subwoofer at the front blends best; corners boom; do a subwoofer crawl | SVS, Audioholics | front, off-center, off walls, on the floor |
| Mirror left/right pairs | — | symmetric angle per pair, distance mismatch penalized |

On the scan: walls are tall surfaces (> 1.4 m above the floor), so a speaker may stand
beyond low furniture but not behind a wall, inside furniture, or with something blocking
its line of sight to the listener's head. Directions the camera did not see (mostly
beside it) are marked *unverified*. The tab also shows the centering at the seat
(distance spread / max delay; 0 = centered without DSP correction) for the suggested
vs. the current layout, plus remarks on the seat (off-center between side walls,
too close to the back wall).

References: [ITU-R BS.775 / Dolby angles](https://theatercalc.com/guides/surround-sound-speaker-placement),
[speaker angle calculator](https://boomspeaker.com/calculators/surround-sound-speaker-angle-calculator/),
[Dolby 22–30° / 90–110°](https://www.the-home-cinema-guide.com/surround-sound-speaker-placement.html),
[surround setup](https://arqen.com/acoustics-101/surround-sound-speaker-placement/),
[subwoofer placement](https://www.svsound.com/blogs/subwoofer-setup-and-tuning/75365187-the-art-of-subwoofer-placement),
[subwoofer walls/corners](https://www.audioholics.com/easy-subwoofer-placement),
[wall distance, tweeter height, toe-in](https://www.soundandvision.com/content/setup-speakers-page-3-0).

### Vision → DSP bridge (UDP, message v1)

One JSON object per datagram to `[bridge] udp_host:udp_port` (default `127.0.0.1:9750`),
sent on every detection (`interface/bridge.py`):

```json
{"v": 1, "seq": 1234, "t": 1730000000.123, "tracking": true,
 "x": 0.12, "y": -0.05, "z": 0.02, "yaw_deg": -8.0, "confidence": 0.91}
```

| Field | Meaning |
|-------|---------|
| `t` | sender wall-clock time of the camera frame (s, Unix) |
| `tracking` | `false` = face lost; x/y/z repeat the last known position |
| `x`, `y`, `z` | listener offset from the seat, room frame, meters (x right, y front, z up) |
| `yaw_deg` | head yaw, room frame: 0 = facing front, + = turned right |
| `confidence` | detector score |

```python
from interface.bridge import BridgeReceiver
rx = BridgeReceiver()          # binds 127.0.0.1:9750
msg = rx.latest()              # newest ListenerMessage or None
```

Positions are raw per-detection (±2–4 cm jitter); smoothing/prediction is the next step.

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

## Milestone 3.2b — Depth model selection (room map / initial calibration)

A one-time metric depth run will build the room map and initialize the camera →
seat distance. Accuracy comes first, so candidates are compared on your own room
against tape measurements (`tools/depth_bench.py`, `tools/depth_infer.py`).

| Candidate | Weights | License | Notes |
|-----------|---------|---------|-------|
| MoGe-2 ViT-L | `Ruicheng/moge-2-vitl-normal` | MIT | metric point map + own focal estimate; official ONNX export |
| Depth Anything 3 Metric-L | `depth-anything/DA3METRIC-LARGE` | Apache 2.0 | needs the focal length (taken from MoGe-2's estimate) |

MoGe-3 is excluded: its refinement depends on FlexGEMM/Triton (no macOS, no Adreno/Hexagon).

Each model has its own environment (their dependencies conflict):

```bash
# MoGe-2 (v2 code does not need FlexGEMM, so install the package without deps)
uv venv .venvs/moge2 --python 3.12
VIRTUAL_ENV=.venvs/moge2 uv pip install "torch>=2.4" "torchvision>=0.19" "numpy>=2" opencv-python-headless \
  scipy pillow trimesh huggingface-hub requests tqdm click \
  "utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37" \
  "pipeline @ git+https://github.com/EasternJournalist/pipeline.git@1c511390d90226c00c101f34b84df26a0f8789b4"
VIRTUAL_ENV=.venvs/moge2 uv pip install --no-deps "moge @ git+https://github.com/microsoft/MoGe.git"

# Depth Anything 3 (no xformers on macOS; pycolmap left out - its libomp clashes with torch)
uv venv .venvs/da3 --python 3.12
VIRTUAL_ENV=.venvs/da3 uv pip install "torch>=2" torchvision einops huggingface_hub imageio "numpy<2" \
  opencv-python-headless pillow omegaconf safetensors trimesh evo plyfile "moviepy==1.0.3" e3nn typer requests addict
VIRTUAL_ENV=.venvs/da3 uv pip install --no-deps "depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"
```

Benchmark:

```bash
python tools/depth_bench.py capture --name room1          # space, then sit at the seat (5 s countdown, 2 s capture)
python tools/depth_bench.py annotate assets/captures/room1  # click points, type tape distances (first = your face)
.venvs/moge2/bin/python tools/depth_infer.py moge2 assets/captures/room1
.venvs/da3/bin/python   tools/depth_infer.py da3   assets/captures/room1
python tools/depth_bench.py evaluate assets/captures/room1
```

`evaluate` reports each model's error raw, after correcting the scale with the
seat distance, and after correcting it with the tape-measured camera height
(floor must be in view; `vision/room.py` fits the floor plane).

Status: **MoGe-2 is the provisional default.** On the first real capture (room2)
both models agreed with each other within a few cm but not with the tape points,
and the seat distance and camera height implied opposite scale corrections, so the
reference measurements were not reliable enough to rank the models. Planned
re-check: floor markers measured along the floor (distance computed from the
camera height), clicked directly in `annotate`.

## Milestone 3.2b — Room setup: calibration, 3D room scan, speaker placement

One-time flow (redo when the room or camera changes), `tools/room_setup.py`:

1. **Capture** from the seat, facing the camera on the FC speaker (2 s at 1080p). Tilt
   the camera so some floor is visible: it levels the scan and gives the camera height.
2. **Depth**: MoGe-2 (`.venvs/moge2`) gives a metric 3D point map. Flying pixels (points
   smeared across depth edges) and points beyond 8 m are removed.
3. **Calibrate**: your face box + its depth give the camera → seat distance, the camera
   FOV and your real face width → `config/calibration.json`. This replaces the IPD
   assumption and the `k` key of the 2.2 tool (`k` still re-centers).
4. **Scan**: colored 3D point cloud in the room frame → `config/room_cloud_{xyz,rgb}.bin`,
   plus a top-down map for the 2.2 tool → `config/room_map.png`.
5. **Place speakers** in a 3D viewer in the browser (`interface/room_viewer.py`,
   `interface/web/`) → `config/room.json`. FC is the camera; the others start at
   ITU-style defaults.

```bash
python tools/room_setup.py                                  # capture + build + place
python tools/room_setup.py --capture assets/captures/room2  # reuse a capture
python tools/room_setup.py --place                          # edit speakers only
python tools/room_setup.py --seat-distance 1.6              # optional tape: lens → face (m)
python tools/room_setup.py --camera-height 1.05             # optional tape: lens → floor (m)
```

The live app (`tools/app.py`) runs the same setup from its Setup tab. 3D viewer (opens at
`http://127.0.0.1:8765/`; loads three.js from unpkg, so it needs internet):

- orbit = drag, pan = right-drag, zoom = scroll; views: Camera / Top / From seat / Free
- select a speaker by clicking it, its row, or keys `1`–`6`
- move it by dragging its arrows, **Shift+click** the scan to put it on that surface,
  or type exact x / y / z
- Save, or Save & finish (ends the setup)

Room frame (shared with tracking and DSP): origin at the seat, `+x` listener's right,
`+y` toward the front (camera/FC), `+z` up relative to the listener's face height.
`load_config()` returns the placed positions as `speaker["position"]`;
`config.speaker_position(spk)` gives `(x, y, z)` for placed or configured speakers.
The 2.2 tool draws the room map and placed speakers behind the live listener.
