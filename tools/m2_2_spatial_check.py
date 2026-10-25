#!/usr/bin/env python3
"""Milestone 2.2: listener offset from the designated center.

Assumes the camera sits on top of the front center speaker, facing the
listener (see vision/spatial.py). Shows the camera view next to a top-down
map of the speakers, the center and the listener.

    python tools/m2_2_spatial_check.py

Keys:
    k      calibrate - sit at the intended center, face the camera, press k
    space  record your averaged position (compare with a tape measure)
    c      switch camera
    q      quit

Calibration is saved to config/calibration.json and loaded next time.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CONFIG_DIR, load_config  # noqa: E402
from vision.pipeline import VisionPipeline  # noqa: E402
from vision.source import open_source  # noqa: E402
from vision.spatial import Calibration, CenterMapper  # noqa: E402

WINDOW = "m2.2 listener offset"
CAL_PATH = CONFIG_DIR / "calibration.json"
CAPTURE_S = 1.5   # how long calibration / recording average over
MAP_SIZE = 480    # px, top-down map panel
MAP_RANGE_M = 3.0  # map shows +-3 m around the center


def speaker_xy(spk: dict) -> tuple[float, float]:
    az = math.radians(spk["azimuth_deg"])
    return spk["distance_m"] * math.sin(az), spk["distance_m"] * math.cos(az)


def draw_map(speakers, center_distance, offset, records, height):
    import cv2

    panel = np.full((max(height, MAP_SIZE), MAP_SIZE, 3), 255, np.uint8)
    ppm = MAP_SIZE / (2 * MAP_RANGE_M)
    cx, cy = MAP_SIZE // 2, MAP_SIZE // 2

    def px(x, y):  # room (m) -> panel px; +y (front) is up
        return int(cx + x * ppm), int(cy - y * ppm)

    for m in range(-int(MAP_RANGE_M), int(MAP_RANGE_M) + 1):  # 1 m grid
        cv2.line(panel, px(m, -MAP_RANGE_M), px(m, MAP_RANGE_M), (230, 230, 230), 1)
        cv2.line(panel, px(-MAP_RANGE_M, m), px(MAP_RANGE_M, m), (230, 230, 230), 1)

    for spk in speakers:
        x, y = speaker_xy(spk)
        cv2.rectangle(panel, (px(x, y)[0] - 7, px(x, y)[1] - 7), (px(x, y)[0] + 7, px(x, y)[1] + 7),
                      (60, 60, 60), -1)
        cv2.putText(panel, spk["name"], (px(x, y)[0] + 10, px(x, y)[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cam = px(0, center_distance + 0.15)
    cv2.putText(panel, "camera", (cam[0] - 25, cam[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    c = px(0, 0)
    cv2.line(panel, (c[0] - 10, c[1]), (c[0] + 10, c[1]), (0, 0, 0), 1)
    cv2.line(panel, (c[0], c[1] - 10), (c[0], c[1] + 10), (0, 0, 0), 1)

    for i, (x, y) in enumerate(records, 1):
        cv2.circle(panel, px(x, y), 5, (255, 140, 0), 2)
        cv2.putText(panel, str(i), (px(x, y)[0] + 7, px(x, y)[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 140, 0), 1)
    if offset is not None:
        cv2.circle(panel, px(offset.x, offset.y), 9, (0, 0, 220), -1)
    return panel


def main() -> int:
    import cv2

    from vision.overlay import draw_face, draw_hud

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--camera", type=int)
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--fps", type=int)
    ap.add_argument("--video", help="use a video file instead of the webcam")
    args = ap.parse_args()

    cfg = load_config(args.config)
    vcfg = cfg["vision"]
    speakers = cfg["speakers"]
    center_distance = next(s["distance_m"] for s in speakers if s["name"] == "FC")

    source = open_source(args, vcfg)
    pipe = VisionPipeline(vcfg, ROOT, max_fps=args.fps)
    mapper: CenterMapper | None = None
    records: list[tuple[float, float]] = []
    capture: dict | None = None  # {"kind": "calibrate"|"record", "until": t, "faces": [...]}
    message = f"calibration: {'loaded' if CAL_PATH.exists() else 'none - sit at center and press k'}"

    while True:
        item = source.read()
        if item is None:
            if source.is_video:
                break
            blank = np.zeros((360, 640, 3), np.uint8)
            draw_hud(blank, [source.label, "", "No signal - press c to switch camera, q to quit"])
            cv2.imshow(WINDOW, blank)
            key = cv2.waitKey(100) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                source.switch()
                pipe.reset()
                mapper = None
            continue

        ts, frame = item
        h, w = frame.shape[:2]
        if mapper is None or mapper.cx != w / 2:
            mapper = CenterMapper(w, h, vcfg["hfov_deg"], center_distance, Calibration.load(CAL_PATH),
                                  ipd_m=vcfg["ipd_m"])

        face, processed = pipe.process(frame, ts)
        offset = mapper.map(face) if face is not None else None

        # Calibration / recording: collect faces for CAPTURE_S, then use the median.
        if capture is not None:
            if processed and face is not None:
                capture["faces"].append(face)
            if ts >= capture["until"]:
                faces = capture["faces"]
                if len(faces) < 3:
                    message = f"{capture['kind']} failed: face not detected"
                elif capture["kind"] == "calibrate":
                    cal = mapper.calibrate(faces)
                    cal.save(CAL_PATH)
                    message = (f"calibrated: face width {cal.face_width_m * 100:.1f} cm, scale {cal.scale:.3f}, "
                               f"x_bias {cal.x_bias * 100:+.1f} cm (saved)")
                else:
                    offs = [mapper.map(f) for f in faces]
                    x = float(np.median([o.x for o in offs]))
                    y = float(np.median([o.y for o in offs]))
                    records.append((x, y))
                    message = f"record {len(records)}: x {x * 100:+.1f} cm  y {y * 100:+.1f} cm"
                print(message)
                capture = None

        hud = [source.label]
        if offset is not None:
            hud.append(f"x {offset.x * 100:+6.1f} cm   y {offset.y * 100:+6.1f} cm   "
                       f"distance {offset.distance:.2f} m   yaw {offset.yaw_deg:+5.1f}")
        else:
            hud.append("no face")
        hud.append(f"face width {mapper.cal.face_width_m * 100:.1f} cm  scale {mapper.cal.scale:.3f}  "
                   f"x_bias {mapper.cal.x_bias * 100:+.1f} cm  (FC distance {center_distance:.2f} m)")
        hud += ["", message]
        if capture is not None:
            hud.append(f"{capture['kind']}: hold still... {capture['until'] - ts:3.1f}s")
        hud += ["", "k calibrate at center   space record   c camera   q quit"]

        if face is not None:
            draw_face(frame, face)
        draw_hud(frame, hud)
        panel = draw_map(speakers, center_distance, offset, records, h)
        if panel.shape[0] != h:
            panel = cv2.resize(panel, (int(MAP_SIZE * h / panel.shape[0]), h))
        cv2.imshow(WINDOW, np.hstack([frame, panel]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            source.switch()
            pipe.reset()
            mapper, capture = None, None
        elif key in (ord("k"), ord(" ")) and capture is None:
            capture = {"kind": "calibrate" if key == ord("k") else "record",
                       "until": ts + CAPTURE_S, "faces": []}

    source.close()
    pipe.close()
    cv2.destroyAllWindows()
    if records:
        print("\n=== recorded positions (cm) ===")
        for i, (x, y) in enumerate(records, 1):
            print(f"{i:2d}  x {x * 100:+6.1f}  y {y * 100:+6.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
