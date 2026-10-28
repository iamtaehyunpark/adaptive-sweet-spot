"""Room setup pipeline (Milestone 3.2b): depth -> calibration -> 3D scan -> room.json.

Shared by tools/room_setup.py (CLI) and the live app (tools/app.py).
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np

from config import CONFIG_DIR, ROOM_FILE, speaker_position

ROOT = CONFIG_DIR.parent
MAP_FILE = CONFIG_DIR / "room_map.png"
CLOUD_XYZ = CONFIG_DIR / "room_cloud_xyz.bin"  # float32 (N, 3) room frame
CLOUD_RGB = CONFIG_DIR / "room_cloud_rgb.bin"  # uint8 (N, 3)
CAL_FILE = CONFIG_DIR / "calibration.json"
CAPTURES = ROOT / "assets" / "captures"
MOGE_PYTHON = ROOT / ".venvs" / "moge2" / "bin" / "python"


class SetupError(RuntimeError):
    pass


def run_depth(capture: Path, log: Callable[[str], None] = print) -> Path:
    npz = capture / "moge2.npz"
    if npz.exists():
        log(f"depth: reusing {npz}")
        return npz
    if not MOGE_PYTHON.exists():
        raise SetupError(f"{MOGE_PYTHON} missing - set up the MoGe-2 env (README, Milestone 3.2b)")
    log("depth: running MoGe-2 (first run downloads ~1.3 GB)...")
    result = subprocess.run([str(MOGE_PYTHON), "-W", "ignore", str(ROOT / "tools" / "depth_infer.py"), "moge2",
                             str(capture)], capture_output=True, text=True)
    if result.returncode != 0:
        raise SetupError("depth model failed:\n" + (result.stderr or result.stdout)[-2000:])
    return npz


def detect_face(capture: Path, vcfg: dict):
    """Face in the middle capture frame: (face at the tracking resolution, face at capture
    resolution, capture frame).

    Detection runs on the frame resized to the tracking resolution, so the face box used
    for calibration behaves exactly like the boxes seen while tracking (BlazeFace box
    widths are not perfectly resolution-invariant).
    """
    import cv2

    from vision.detector import BlazeFaceDetector, FaceObservation
    from vision.tracker import FaceTracker

    frames = sorted((capture / "frames").glob("*.png"))
    if not frames:
        raise SetupError(f"no frames in {capture / 'frames'}")
    frame = cv2.imread(str(frames[len(frames) // 2]))
    small = cv2.resize(frame, (vcfg["width"], vcfg["height"]), interpolation=cv2.INTER_AREA)
    with BlazeFaceDetector(ROOT / vcfg["model_path"], vcfg["min_confidence"]) as det:
        tracker = FaceTracker(det, vcfg["roi_scale"], tuple(vcfg["search_levels"]))
        face = tracker.update(small, 0.0)
        # Second pass detects in the face crop ("track" mode), like nearly all live frames;
        # box widths from the full-frame search differ by a few percent.
        if face is not None:
            face = tracker.update(small, 0.0) or face
    if face is None:
        raise SetupError("no face found in the capture - sit at the seat facing the camera and retry")
    k = frame.shape[1] / small.shape[1]
    full = FaceObservation(bbox=face.bbox * k, keypoints=face.keypoints * k, score=face.score,
                           frame_size=(frame.shape[1], frame.shape[0]))
    return face, full, frame


def build(capture: Path, vcfg: dict, seat_distance: float | None = None, camera_height: float | None = None,
          log: Callable[[str], None] = print, live_faces: list | None = None) -> dict:
    """Run the setup on a capture; writes calibration, scan and map, returns room.json content.

    ``live_faces``: tracking detections from the seat just before the capture (the live
    app passes them). They calibrate the face width exactly as tracking measures it.
    """
    import cv2

    from vision.room import build_room, render_map
    from vision.spatial import calibration_from_depth

    npz = np.load(run_depth(capture, log))
    face, face_full, frame = detect_face(capture, vcfg)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    room = build_room(npz["points"], face_full.bbox, seat_distance, camera_height, colors=rgb)
    width = npz["points"].shape[1]
    hfov = math.degrees(2 * math.atan((width / 2) / float(npz["fx"])))

    if live_faces and len(live_faces) >= 3:
        cal = calibration_from_depth(live_faces, room.face_depth_m, hfov)
        log(f"calibration: {len(live_faces)} live tracking frames")
    else:
        cal = calibration_from_depth(face, room.face_depth_m, hfov)
        log("calibration: capture frame (no live frames)")
    cal.save(CAL_FILE)
    img, meta = render_map(room)
    cv2.imwrite(str(MAP_FILE), img)
    room.points.astype(np.float32).tofile(CLOUD_XYZ)
    room.colors.astype(np.uint8).tofile(CLOUD_RGB)

    log(f"scale: x{room.scale:.3f} ({room.scale_source})")
    log(f"camera -> seat: {room.center_distance_m:.2f} m horizontal ({room.face_depth_m:.2f} m along the camera axis)")
    log(f"camera FOV: {hfov:.1f} deg, your face width: {cal.face_width_m * 100:.1f} cm, "
        f"seat offset from camera axis: {cal.x_bias * 100:+.1f} cm")
    if room.has_floor:
        log(f"floor: camera {room.camera_height_m:.2f} m up, pitch {room.pitch_deg:+.1f}, "
            f"roll {room.roll_deg:+.1f}; your face {room.face_height_m:.2f} m above the floor")
    else:
        log("floor: not visible - level camera assumed, no heights")
    log(f"scan: {len(room.points):,} points")

    previous = json.loads(ROOM_FILE.read_text()) if ROOM_FILE.exists() else {}
    data = {
        "version": 1,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "capture": str(capture),
        "scale": room.scale,
        "scale_source": room.scale_source,
        "center_distance_m": room.center_distance_m,
        "hfov_deg": hfov,
        "camera_xy": [float(v) for v in room.camera_xy],
        "camera_height_m": room.camera_height_m,
        "camera_pitch_deg": room.pitch_deg,
        "camera_roll_deg": room.roll_deg,
        "face_height_m": room.face_height_m,
        "floor_z": room.floor_z,
        "map": {"image": MAP_FILE.name, **meta},
        "speakers": previous.get("speakers", []),
    }
    ROOM_FILE.write_text(json.dumps(data, indent=2))
    return data


def default_speakers(room: dict | None, cfg_speakers: list[dict]) -> list[dict]:
    """ITU-style starting positions around the seat, FC at the camera.

    Without a room setup, the configured positions (config/default.toml) are used.
    """
    if not room:
        return [{"name": s["name"], "channel": s["channel"], "lfe": s.get("lfe", False),
                 "position": [round(v, 3) for v in speaker_position(s)], "placed": False}
                for s in cfg_speakers]
    d = room["center_distance_m"]
    floor_z = -room["face_height_m"] if room.get("face_height_m") else -1.0
    cam = room["camera_xy"]
    cam_z = 0.0  # camera height relative to the face (FC sits under/at the camera)
    if room.get("floor_z") is not None and room.get("camera_height_m") is not None:
        cam_z = room["floor_z"] + room["camera_height_m"]
    defaults = {
        "FC": (cam[0], cam[1], cam_z),
        "FL": (d * math.sin(math.radians(-30)), d * math.cos(math.radians(-30)), 0.0),
        "FR": (d * math.sin(math.radians(30)), d * math.cos(math.radians(30)), 0.0),
        "SL": (0.9 * d * math.sin(math.radians(-110)), 0.9 * d * math.cos(math.radians(-110)), 0.0),
        "SR": (0.9 * d * math.sin(math.radians(110)), 0.9 * d * math.cos(math.radians(110)), 0.0),
        "LFE": (cam[0] + 0.5, cam[1], floor_z),
    }
    out = []
    for spk in cfg_speakers:
        x, y, z = defaults.get(spk["name"], (0.0, d, 0.0))
        out.append({"name": spk["name"], "channel": spk["channel"], "lfe": spk.get("lfe", False),
                    "position": [round(float(x), 3), round(float(y), 3), round(float(z), 3)],
                    "placed": spk["name"] == "FC"})
    return out


def load_room() -> dict | None:
    return json.loads(ROOM_FILE.read_text()) if ROOM_FILE.exists() else None
