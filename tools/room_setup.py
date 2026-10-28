#!/usr/bin/env python3
"""Milestone 3.2b: room setup - initial calibration, 3D room scan, speaker placement.

One-time flow (redo when the room or the camera changes):

1. capture   sit at your seat facing the camera (on the FC speaker); 2 s at 1080p.
             Tilt the camera so some floor is visible: it levels the scan and gives
             the camera height.
2. depth     MoGe-2 (in .venvs/moge2) turns the capture into a metric 3D point map;
             flying pixels at depth edges are removed.
3. calibrate your face in the capture + its depth give the camera -> seat distance,
             the camera FOV and your real face width -> config/calibration.json
             (replaces the IPD assumption and the k key in m2_2_spatial_check).
4. scan      colored 3D point cloud in the room frame -> config/room_cloud_{xyz,rgb}.bin,
             plus a top-down map for the 2.2 tool -> config/room_map.png
5. place     3D viewer in the browser: orbit the scan and place each speaker
             (drag its arrows, Shift+click a surface, or type x/y/z) -> config/room.json

    python tools/room_setup.py                                  # full flow
    python tools/room_setup.py --capture assets/captures/room2  # reuse a capture
    python tools/room_setup.py --place                          # only edit speakers
    python tools/room_setup.py --seat-distance 1.6              # tape: lens -> face (m)
    python tools/room_setup.py --camera-height 1.05             # tape: lens -> floor (m)
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CONFIG_DIR, ROOM_FILE, load_config  # noqa: E402

MAP_FILE = CONFIG_DIR / "room_map.png"
CLOUD_XYZ = CONFIG_DIR / "room_cloud_xyz.bin"  # float32 (N, 3) room frame
CLOUD_RGB = CONFIG_DIR / "room_cloud_rgb.bin"  # uint8 (N, 3)
CAL_FILE = CONFIG_DIR / "calibration.json"
CAPTURES = ROOT / "assets" / "captures"
MOGE_PYTHON = ROOT / ".venvs" / "moge2" / "bin" / "python"


# ---------------------------------------------------------------- setup steps

def run_depth(capture: Path) -> Path:
    npz = capture / "moge2.npz"
    if npz.exists():
        print(f"depth: reusing {npz}")
        return npz
    if not MOGE_PYTHON.exists():
        raise SystemExit(f"{MOGE_PYTHON} missing - set up the MoGe-2 env (README, Milestone 3.2b)")
    print("depth: running MoGe-2 (first run downloads ~1.3 GB)...")
    subprocess.run([str(MOGE_PYTHON), "-W", "ignore", str(ROOT / "tools" / "depth_infer.py"), "moge2",
                    str(capture)], check=True)
    return npz


def detect_face(capture: Path, vcfg: dict):
    import cv2

    from vision.detector import BlazeFaceDetector
    from vision.tracker import FaceTracker

    frames = sorted((capture / "frames").glob("*.png"))
    frame = cv2.imread(str(frames[len(frames) // 2]))
    with BlazeFaceDetector(ROOT / vcfg["model_path"], vcfg["min_confidence"]) as det:
        face = FaceTracker(det, vcfg["roi_scale"], tuple(vcfg["search_levels"])).update(frame, 0.0)
    if face is None:
        raise SystemExit("no face found in the capture - sit at the seat facing the camera and recapture")
    return face, frame


def build(capture: Path, vcfg: dict, seat_distance: float | None, camera_height: float | None) -> dict:
    import cv2

    from vision.room import build_room, render_map
    from vision.spatial import calibration_from_depth

    npz = np.load(run_depth(capture))
    face, frame = detect_face(capture, vcfg)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    room = build_room(npz["points"], face.bbox, seat_distance, camera_height, colors=rgb)
    width = npz["points"].shape[1]
    hfov = math.degrees(2 * math.atan((width / 2) / float(npz["fx"])))

    cal = calibration_from_depth(face, room.face_depth_m, hfov)
    cal.save(CAL_FILE)
    img, meta = render_map(room)
    cv2.imwrite(str(MAP_FILE), img)
    room.points.astype(np.float32).tofile(CLOUD_XYZ)
    room.colors.astype(np.uint8).tofile(CLOUD_RGB)

    print(f"scale: x{room.scale:.3f} ({room.scale_source})")
    print(f"camera -> seat: {room.center_distance_m:.2f} m horizontal ({room.face_depth_m:.2f} m along the camera axis)")
    print(f"camera FOV: {hfov:.1f} deg, your face width: {cal.face_width_m * 100:.1f} cm, "
          f"seat offset from camera axis: {cal.x_bias * 100:+.1f} cm")
    if room.has_floor:
        print(f"floor: camera {room.camera_height_m:.2f} m up, pitch {room.pitch_deg:+.1f}, "
              f"roll {room.roll_deg:+.1f}; your face {room.face_height_m:.2f} m above the floor")
    else:
        print("floor: not visible - map assumes a level camera, no heights")
    print(f"scan: {len(room.points):,} points")
    print(f"saved {CAL_FILE.relative_to(ROOT)}, {MAP_FILE.relative_to(ROOT)}, {CLOUD_XYZ.relative_to(ROOT)}")

    previous = json.loads(ROOM_FILE.read_text()) if ROOM_FILE.exists() else {}
    return {
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


# ---------------------------------------------------------------- placement

def default_speakers(room: dict, cfg_speakers: list[dict]) -> list[dict]:
    """ITU-style starting positions around the seat, FC at the camera."""
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
                    "position": [round(x, 3), round(y, 3), round(z, 3)], "placed": spk["name"] == "FC"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", type=Path, help="reuse an existing capture folder")
    ap.add_argument("--name", help="name for a new capture (default: timestamp)")
    ap.add_argument("--camera", type=int)
    ap.add_argument("--place", action="store_true", help="only edit speaker placement")
    ap.add_argument("--seat-distance", type=float, help="tape: camera lens -> your face (m), fixes scale")
    ap.add_argument("--camera-height", type=float, help="tape: camera lens -> floor (m), fixes scale")
    ap.add_argument("--port", type=int, default=8765, help="local port for the 3D viewer")
    args = ap.parse_args()

    cfg = load_config(room=False)
    vcfg = cfg["vision"]

    if args.place:
        if not ROOM_FILE.exists():
            raise SystemExit("no config/room.json yet - run the full setup first")
        room = json.loads(ROOM_FILE.read_text())
    else:
        capture = args.capture
        if capture is None:
            from vision.capture import capture_frames

            capture = CAPTURES / (args.name or "setup-" + time.strftime("%Y%m%d-%H%M%S"))
            index = args.camera if args.camera is not None else vcfg["camera_index"]
            if not capture_frames(capture / "frames", index, title="room setup - capture"):
                return 1
        room = build(capture, vcfg, args.seat_distance, args.camera_height)
        ROOM_FILE.write_text(json.dumps(room, indent=2))

    from interface.room_viewer import serve

    if not CLOUD_XYZ.exists():
        raise SystemExit("no 3D scan yet - run the full setup first")
    room = serve(ROOM_FILE, CLOUD_XYZ, CLOUD_RGB, default_speakers(room, cfg["speakers"]), port=args.port)
    print(f"saved {ROOM_FILE.relative_to(ROOT)}:")
    for s in room["speakers"]:
        x, y, z = s["position"]
        print(f"  {s['name']:3s} ch{s['channel']}  x {x:+.2f}  y {y:+.2f}  z {z:+.2f} m"
              f"{'' if s['placed'] else '  (default, not placed)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
