#!/usr/bin/env python3
"""Milestone 3.2b: room setup - initial calibration, room map, speaker placement.

One-time flow (redo when the room or the camera changes):

1. capture   sit at your seat facing the camera (on the FC speaker); 2 s at 1080p.
             Tilt the camera so some floor is visible: it levels the map and gives
             the camera height.
2. depth     MoGe-2 (in .venvs/moge2) turns the capture into a metric 3D point map.
3. calibrate your face in the capture + its depth give the camera -> seat distance,
             the camera FOV and your real face width -> config/calibration.json
             (replaces the IPD assumption and the k key in m2_2_spatial_check).
4. map       top-down room map around the seat -> config/room_map.png
5. place     click on the map to place each speaker -> config/room.json

    python tools/room_setup.py                                  # full flow
    python tools/room_setup.py --capture assets/captures/room2  # reuse a capture
    python tools/room_setup.py --place                          # only edit speakers
    python tools/room_setup.py --seat-distance 1.6              # tape: lens -> face (m)
    python tools/room_setup.py --camera-height 1.05             # tape: lens -> floor (m)

Placement keys: 1-5 select FL FR SL SR LFE (FC is the camera), click/drag to place,
[ ] lower/raise the selected speaker by 5 cm, r reset defaults, q save and quit.
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
CAL_FILE = CONFIG_DIR / "calibration.json"
CAPTURES = ROOT / "assets" / "captures"
MOGE_PYTHON = ROOT / ".venvs" / "moge2" / "bin" / "python"
WINDOW = "room setup - speaker placement"
VIEW_MAX = (1100, 800)  # px, map view size limit


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
    return face


def build(capture: Path, vcfg: dict, seat_distance: float | None, camera_height: float | None) -> dict:
    import cv2

    from vision.room import build_room, render_map
    from vision.spatial import calibration_from_depth

    npz = np.load(run_depth(capture))
    face = detect_face(capture, vcfg)
    room = build_room(npz["points"], face.bbox, seat_distance, camera_height)
    width = npz["points"].shape[1]
    hfov = math.degrees(2 * math.atan((width / 2) / float(npz["fx"])))

    cal = calibration_from_depth(face, room.face_depth_m, hfov)
    cal.save(CAL_FILE)
    img, meta = render_map(room)
    cv2.imwrite(str(MAP_FILE), img)

    print(f"scale: x{room.scale:.3f} ({room.scale_source})")
    print(f"camera -> seat: {room.center_distance_m:.2f} m horizontal ({room.face_depth_m:.2f} m along the camera axis)")
    print(f"camera FOV: {hfov:.1f} deg, your face width: {cal.face_width_m * 100:.1f} cm, "
          f"seat offset from camera axis: {cal.x_bias * 100:+.1f} cm")
    if room.has_floor:
        print(f"floor: camera {room.camera_height_m:.2f} m up, pitch {room.pitch_deg:+.1f}, "
              f"roll {room.roll_deg:+.1f}; your face {room.face_height_m:.2f} m above the floor")
    else:
        print("floor: not visible - map assumes a level camera, no heights")
    print(f"saved {CAL_FILE.relative_to(ROOT)}, {MAP_FILE.relative_to(ROOT)}")

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
        "map": {"image": MAP_FILE.name, **meta},
        "speakers": previous.get("speakers", []),
    }


# ---------------------------------------------------------------- placement UI

def default_speakers(room: dict, cfg_speakers: list[dict]) -> list[dict]:
    """ITU-style starting positions around the seat, FC at the camera."""
    d = room["center_distance_m"]
    floor_z = -room["face_height_m"] if room.get("face_height_m") else -1.0
    cam = room["camera_xy"]
    defaults = {
        "FC": (cam[0], cam[1], 0.0),
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


class Placement:
    def __init__(self, room: dict, cfg_speakers: list[dict]):
        import cv2

        self.room = room
        base = cv2.imread(str(MAP_FILE)) if MAP_FILE.exists() else None
        if base is None:
            raise SystemExit("no room map - run the full setup first")
        self.base = base
        h, w = base.shape[:2]
        self.zoom = min(VIEW_MAX[0] / w, VIEW_MAX[1] / h, 3.0)
        self.mpp = room["map"]["m_per_px"]
        self.origin = room["map"]["origin_px"]
        self.speakers = room["speakers"] or default_speakers(room, cfg_speakers)
        self.movable = [s for s in self.speakers if s["name"] != "FC"]
        self.active = 0
        self.dragging = False

    # room <-> view pixels (front = up)
    def to_px(self, x, y):
        return (int((self.origin[0] + x / self.mpp) * self.zoom), int((self.origin[1] - y / self.mpp) * self.zoom))

    def to_room(self, u, v):
        return (u / self.zoom - self.origin[0]) * self.mpp, (self.origin[1] - v / self.zoom) * self.mpp

    def on_mouse(self, event, u, v, *_):
        import cv2

        if event == cv2.EVENT_LBUTTONDOWN:
            # grab a speaker near the cursor, else move the active one here
            for i, s in enumerate(self.movable):
                su, sv = self.to_px(*s["position"][:2])
                if (su - u) ** 2 + (sv - v) ** 2 < 15 ** 2:
                    self.active = i
                    break
            self.dragging = True
            self.move(u, v)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.move(u, v)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            spk = self.movable[self.active]
            # after placing, move on to the next unplaced speaker
            nxt = [i for i, s in enumerate(self.movable) if not s["placed"]]
            if spk["placed"] and nxt:
                self.active = nxt[0]

    def move(self, u, v):
        spk = self.movable[self.active]
        x, y = self.to_room(u, v)
        spk["position"][:2] = [round(x, 3), round(y, 3)]
        spk["placed"] = True

    def draw(self):
        import cv2

        from vision.overlay import draw_hud

        img = cv2.resize(self.base, None, fx=self.zoom, fy=self.zoom, interpolation=cv2.INTER_NEAREST)
        h, w = img.shape[:2]
        for m in range(-10, 11):  # 1 m grid around the seat
            u, _ = self.to_px(m, 0)
            _, v = self.to_px(0, m)
            if 0 <= u < w:
                cv2.line(img, (u, 0), (u, h), (235, 235, 235), 1)
            if 0 <= v < h:
                cv2.line(img, (0, v), (w, v), (235, 235, 235), 1)
        seat = self.to_px(0, 0)
        cv2.drawMarker(img, seat, (0, 0, 220), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(img, "seat", (seat[0] + 8, seat[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1)

        lines = ["Click/drag to place the selected speaker   1-5 select   [ ] height   r reset   q save+quit"]
        for i, s in enumerate(self.speakers):
            x, y, z = s["position"]
            c = self.to_px(x, y)
            selected = s["name"] != "FC" and self.movable[self.active] is s
            color = (0, 140, 255) if selected else (40, 40, 40)
            if s["placed"]:
                cv2.rectangle(img, (c[0] - 8, c[1] - 8), (c[0] + 8, c[1] + 8), color, -1)
            else:
                cv2.rectangle(img, (c[0] - 8, c[1] - 8), (c[0] + 8, c[1] + 8), color, 2)
            cv2.putText(img, s["name"], (c[0] + 11, c[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            az = math.degrees(math.atan2(x, y))
            key = f"{self.movable.index(s) + 1}" if s in self.movable else "-"
            lines.append(f"{'>' if selected else ' '} {key} {s['name']:3s}  az {az:+6.1f}  dist {math.hypot(x, y):4.2f} m  "
                         f"height {z:+.2f} m{'' if s['placed'] else '  (not placed)'}")
        draw_hud(img, lines)
        return img

    def run(self) -> list[dict]:
        import cv2

        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        while True:
            cv2.imshow(WINDOW, self.draw())
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if ord("1") <= key < ord("1") + len(self.movable):
                self.active = key - ord("1")
            elif key in (ord("["), ord("]")):
                spk = self.movable[self.active]
                spk["position"][2] = round(spk["position"][2] + (0.05 if key == ord("]") else -0.05), 3)
            elif key == ord("r"):
                self.speakers = default_speakers(self.room, [{"name": s["name"], "channel": s["channel"],
                                                               "lfe": s["lfe"]} for s in self.speakers])
                self.movable = [s for s in self.speakers if s["name"] != "FC"]
                self.active = 0
        cv2.destroyAllWindows()
        return self.speakers


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", type=Path, help="reuse an existing capture folder")
    ap.add_argument("--name", help="name for a new capture (default: timestamp)")
    ap.add_argument("--camera", type=int)
    ap.add_argument("--place", action="store_true", help="only edit speaker placement")
    ap.add_argument("--seat-distance", type=float, help="tape: camera lens -> your face (m), fixes scale")
    ap.add_argument("--camera-height", type=float, help="tape: camera lens -> floor (m), fixes scale")
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

    room["speakers"] = Placement(room, cfg["speakers"]).run()
    ROOM_FILE.write_text(json.dumps(room, indent=2))
    print(f"saved {ROOM_FILE.relative_to(ROOT)}:")
    for s in room["speakers"]:
        x, y, z = s["position"]
        print(f"  {s['name']:3s} ch{s['channel']}  x {x:+.2f}  y {y:+.2f}  z {z:+.2f} m"
              f"{'' if s['placed'] else '  (default, not placed)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
