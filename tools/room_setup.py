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
4b. room scan (optional, --room-scan) the empty room from the same camera, nobody in
             view: aligned to the calibration capture's depth, it replaces the scan,
             floor and map (no listener hiding the floor); the calibration is kept.
             A later calibration re-applies it while the camera has not moved.
5. place     3D viewer in the browser: orbit the scan and place each speaker
             (drag its arrows, Shift+click a surface, or type x/y/z) -> config/room.json

The live app (tools/app.py) runs the same setup from its Setup tab; this CLI is for
scripted runs and reusing captures.

    python tools/room_setup.py                                  # full flow
    python tools/room_setup.py --capture assets/captures/room2  # reuse a capture
    python tools/room_setup.py --place                          # only edit speakers
    python tools/room_setup.py --seat-distance 1.6              # tape: lens -> face (m)
    python tools/room_setup.py --camera-height 1.05             # tape: lens -> floor (m)
    python tools/room_setup.py --room-scan                      # after the setup: empty room
    python tools/room_setup.py --clear                          # remove the whole setup
    python tools/room_setup.py --room-scan --capture assets/captures/roomscan1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import ROOM_FILE, load_config  # noqa: E402
from interface import setup as room_setup  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", type=Path, help="reuse an existing capture folder")
    ap.add_argument("--name", help="name for a new capture (default: timestamp)")
    ap.add_argument("--camera", type=int)
    ap.add_argument("--place", action="store_true", help="only edit speaker placement (opens the app)")
    ap.add_argument("--seat-distance", type=float, help="tape: camera lens -> your face (m), fixes scale")
    ap.add_argument("--camera-height", type=float, help="tape: camera lens -> floor (m), fixes scale")
    ap.add_argument("--room-scan", action="store_true",
                    help="capture (or --capture) the empty room and add it to the existing setup")
    ap.add_argument("--clear", action="store_true",
                    help="remove the whole setup (calibration, scans, room scan, speaker placement)")
    ap.add_argument("--port", type=int, default=8765, help="local port for the 3D viewer")
    args = ap.parse_args()

    if args.clear:
        removed = room_setup.clear_setup()
        for p in removed:
            print(f"removed {p.relative_to(ROOT)}")
        print("setup cleared" if removed else "nothing to clear")
        return 0

    cfg = load_config(room=False)
    if not args.place:
        capture = args.capture
        if capture is None:
            from vision.capture import capture_frames

            prefix = "roomscan-" if args.room_scan else "setup-"
            capture = room_setup.CAPTURES / (args.name or prefix + time.strftime("%Y%m%d-%H%M%S"))
            index = args.camera if args.camera is not None else cfg["vision"]["camera_index"]
            title = "room scan - step out of view after space" if args.room_scan else "room setup - capture"
            countdown = {"countdown_s": 10.0, "prompt": "Step out of the camera view"} if args.room_scan else {}
            if not capture_frames(capture / "frames", index, title=title, **countdown):
                return 1
        try:
            if args.room_scan:
                room_setup.apply_room_scan(capture, cfg["vision"])
            else:
                room_setup.build(capture, cfg["vision"], args.seat_distance, args.camera_height)
        except room_setup.SetupError as e:
            raise SystemExit(str(e))
        print(f"saved {ROOM_FILE.relative_to(ROOT)} and the 3D scan")

    from interface.app import serve

    print("placing speakers in the app (no camera); Save in the Speakers tab, Ctrl-C here when done")
    serve(load_config(), None, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
