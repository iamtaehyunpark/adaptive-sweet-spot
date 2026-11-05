"""Room setup pipeline (Milestone 3.2b): depth -> calibration -> 3D scan -> room.json.

Two captures from the same fixed camera:

* calibration (build)            listener at the seat facing the camera - required.
  Gives the scale, the room frame (origin at the face), the face width and a first scan.
* room scan (apply_room_scan)    the empty room, nobody in view - optional, after the
  calibration. Its depth is aligned to the calibration capture and replaces the scan,
  floor and map (no listener hiding the floor and what is behind them). The calibration
  itself (face width, seat distance, FOV) is unchanged.

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

from config import CONFIG_DIR, ROOM_FILE, camera_position, speaker_position

ROOT = CONFIG_DIR.parent
MAP_FILE = CONFIG_DIR / "room_map.png"
CLOUD_XYZ = CONFIG_DIR / "room_cloud_xyz.bin"  # float32 (N, 3) room frame
CLOUD_RGB = CONFIG_DIR / "room_cloud_rgb.bin"  # uint8 (N, 3)
CAL_FILE = CONFIG_DIR / "calibration.json"
CAPTURES = ROOT / "assets" / "captures"
MOGE_PYTHON = ROOT / ".venvs" / "moge2" / "bin" / "python"


SETUP_FILES = (ROOM_FILE, CAL_FILE, CLOUD_XYZ, CLOUD_RGB, MAP_FILE)  # what "Clear setup" removes


class SetupError(RuntimeError):
    pass


def clear_setup() -> list[Path]:
    """Remove the whole room setup: calibration, scan, room scan, map, speaker placement.

    Camera (camera.json) and audio (audio.json) choices and the raw captures stay.
    Returns the files removed.
    """
    removed = [p for p in SETUP_FILES if p.exists()]
    for p in removed:
        p.unlink()
    return removed


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


def detect_face(capture: Path, vcfg: dict, tracking_size: tuple[int, int] | None = None):
    """Face in the middle capture frame: (face at the tracking resolution, face at capture
    resolution, capture frame).

    ``tracking_size`` is the live tracking frame size; without it the capture is scaled
    to the configured tracking width, keeping its aspect.

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
    fh, fw = frame.shape[:2]
    if tracking_size is None or abs(tracking_size[0] / tracking_size[1] - fw / fh) > 0.01:
        tracking_size = (vcfg["width"], round(vcfg["width"] * fh / fw))  # never distort the image
    small = cv2.resize(frame, tracking_size, interpolation=cv2.INTER_AREA)
    with BlazeFaceDetector(ROOT / vcfg["model_path"], vcfg["min_confidence"]) as det:
        tracker = FaceTracker(det, vcfg["roi_scale"], tuple(vcfg["search_levels"]))
        face = tracker.update(small, 0.0)
        # Second pass detects in the face crop ("track" mode), like nearly all live frames;
        # box widths from the full-frame search differ by a few percent.
        if face is not None:
            face = tracker.update(small, 0.0) or face
    if face is None:
        raise SetupError("no face found in the capture - sit at the seat facing the camera and retry")
    kx, ky = fw / small.shape[1], fh / small.shape[0]
    full = FaceObservation(bbox=face.bbox * np.array([kx, ky, kx, ky]), keypoints=face.keypoints * np.array([kx, ky]),
                           score=face.score, frame_size=(fw, fh))
    return face, full, frame


def build(capture: Path, vcfg: dict, seat_distance: float | None = None, camera_height: float | None = None,
          log: Callable[[str], None] = print, live_faces: list | None = None,
          tracking_size: tuple[int, int] | None = None) -> dict:
    """Run the setup on a capture; writes calibration, scan and map, returns room.json content.

    ``live_faces``: tracking detections from the seat just before the capture (the live
    app passes them). They calibrate the face width exactly as tracking measures it.
    """
    import cv2

    from vision.room import build_room, render_map
    from vision.spatial import calibration_from_depth

    npz = np.load(run_depth(capture, log))
    face, face_full, frame = detect_face(capture, vcfg, tracking_size)
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
        "face_bbox_px": [float(v) for v in face_full.bbox],  # in the capture, for the room scan
        "speakers": [],
    }
    data["speakers"] = _speakers(data, previous)
    ROOM_FILE.write_text(json.dumps(data, indent=2))

    # a room scan from before still fits if the camera has not moved: re-apply it
    old_scan = (previous.get("room_scan") or {}).get("capture")
    if old_scan and Path(old_scan).exists():
        try:
            data = apply_room_scan(Path(old_scan), vcfg, log, tracking_size)
        except SetupError as e:
            log(f"previous room scan dropped: {str(e).splitlines()[0]}")
    return data


def _speakers(data: dict, previous: dict) -> list[dict]:
    """FC goes to the (possibly moved) camera; ones the user placed by hand are kept; the
    rest get this room's defaults (old defaults belong to the old room frame)."""
    from config import load_config

    placed = {s["name"]: s for s in previous.get("speakers", []) if s.get("placed") and s["name"] != "FC"}
    cfg_speakers = load_config(room=False)["speakers"]
    return [placed.get(s["name"], s) for s in default_speakers(data, cfg_speakers)]


def apply_room_scan(capture: Path, vcfg: dict, log: Callable[[str], None] = print,
                    tracking_size: tuple[int, int] | None = None) -> dict:
    """Rebuild the 3D scan, floor and map from an empty-room capture; returns room.json.

    Needs the calibration (build) first, from the same camera pose. The scan depth is
    scaled to the calibration capture's depth (median ratio over static pixels, the
    listener masked out) and re-projected with its intrinsics, so it lands in the same
    room frame with the same scale; the calibration's face point stays the origin.
    """
    import cv2

    from vision.room import (SCAN_MIN_AGREE, align_scan, body_mask, build_room, face_point,
                             points_from_depth, render_map)

    room = load_room()
    if room is None or not room.get("capture"):
        raise SetupError("run the calibration (Run setup) first - the room scan adds on to it")
    cal_capture = Path(room["capture"])
    if not (cal_capture / "moge2.npz").exists():
        raise SetupError(f"calibration capture {cal_capture} is missing - run Setup again")

    # nobody may be in view (checked before the slow depth model)
    try:
        detect_face(capture, vcfg, tracking_size)
    except SetupError as e:
        if "no face" not in str(e):
            raise
    else:
        raise SetupError("a face is in the room scan - step out of the camera view and retry")

    cal = np.load(cal_capture / "moge2.npz")
    scan = np.load(run_depth(capture, log))
    if "face_bbox_px" in room:
        bbox = np.array(room["face_bbox_px"])
    else:  # room.json from before the room scan existed
        bbox = detect_face(cal_capture, vcfg, tracking_size)[1].bbox
    try:
        ratio, agree = align_scan(cal["depth"], scan["depth"], body_mask(cal["depth"].shape, bbox))
    except ValueError as e:
        raise SetupError(f"room scan does not match the calibration: {e}") from None
    log(f"room scan: depth x{ratio:.3f} to the calibration, {agree * 100:.0f}% of the view agrees")
    if agree < SCAN_MIN_AGREE:
        raise SetupError(f"room scan does not match the calibration ({agree * 100:.0f}% agree) - "
                         "the camera or its view moved: run Setup (calibration) again")

    points = points_from_depth(scan["depth"] * ratio, float(cal["fx"]), float(cal["fy"]))
    face = face_point(cal["points"], bbox)
    frames = sorted((capture / "frames").glob("*.png"))
    rgb = cv2.cvtColor(cv2.imread(str(frames[len(frames) // 2])), cv2.COLOR_BGR2RGB)
    model = build_room(points, bbox, colors=rgb, face=face, scale=(room["scale"], room["scale_source"]))

    img, meta = render_map(model)
    cv2.imwrite(str(MAP_FILE), img)
    model.points.astype(np.float32).tofile(CLOUD_XYZ)
    model.colors.astype(np.uint8).tofile(CLOUD_RGB)
    log(f"room scan: {len(model.points):,} points")
    if model.has_floor:
        log(f"floor: camera {model.camera_height_m:.2f} m up, pitch {model.pitch_deg:+.1f}, "
            f"roll {model.roll_deg:+.1f}; your face {model.face_height_m:.2f} m above the floor")
    else:
        log("floor: not visible in the room scan either - level camera assumed")

    previous = dict(room)
    room.update({
        "camera_xy": [float(v) for v in model.camera_xy],
        "center_distance_m": model.center_distance_m,
        "camera_height_m": model.camera_height_m,
        "camera_pitch_deg": model.pitch_deg,
        "camera_roll_deg": model.roll_deg,
        "face_height_m": model.face_height_m,
        "floor_z": model.floor_z,
        "map": {"image": MAP_FILE.name, **meta},
        "room_scan": {"capture": str(capture), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "depth_ratio": ratio, "agreement": agree},
    })
    room["speakers"] = _speakers(room, previous)
    ROOM_FILE.write_text(json.dumps(room, indent=2))
    return room


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
    defaults = {
        "FC": tuple(camera_position(room)),
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
    """room.json, with FC always at the camera (it is never stored stale)."""
    if not ROOM_FILE.exists():
        return None
    room = json.loads(ROOM_FILE.read_text())
    cam = camera_position(room)
    for s in room.get("speakers", []):
        if s["name"] == "FC" and cam is not None:
            s["position"], s["placed"] = cam, True
    return room
