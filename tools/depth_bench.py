#!/usr/bin/env python3
"""Compare metric depth models on your own room against tape measurements.

1. capture   live preview; press space, sit at the seat during the 5 s countdown,
             then 2 s of 1080p frames are saved to assets/captures/<name>/frames/
2. annotate  click points on the captured frame and type their tape-measured
             distance from the camera lens in the window, Enter to save.
             First point = your face (the seat).
3. run the models (each in its own env, see tools/depth_infer.py)
4. evaluate  error of every <model>.npz vs the tape points: raw, after
             correcting the scale with the seat distance, and after correcting
             it with the tape-measured camera height (floor must be in view)

    python tools/depth_bench.py capture --name room1
    python tools/depth_bench.py annotate assets/captures/room1
    .venvs/moge2/bin/python tools/depth_infer.py moge2 assets/captures/room1
    .venvs/da3/bin/python   tools/depth_infer.py da3   assets/captures/room1
    python tools/depth_bench.py evaluate assets/captures/room1

Tape tips: measure straight-line from the camera lens to a flat, matte spot
(wall, door, sofa back) - not edges, glass, mirrors or screens.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402

CAPTURES = ROOT / "assets" / "captures"
COUNTDOWN_S = 5.0
CAPTURE_S = 2.0
PATCH = 7  # px, median window around each tape point


def cmd_capture(args) -> int:
    import cv2

    from vision.camera import Camera
    from vision.overlay import draw_hud

    vcfg = load_config(args.config)["vision"]
    out = CAPTURES / (args.name or time.strftime("%Y%m%d-%H%M%S"))
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    index = args.camera if args.camera is not None else vcfg["camera_index"]
    with Camera(index, args.width, args.height, 30) as cam:
        print(f"camera {index}: {cam.width}x{cam.height}")
        start = None
        saved = 0
        while True:
            item = cam.read()
            if item is None:
                print("camera read failed", file=sys.stderr)
                return 1
            ts, frame = item
            if start is None:
                status = "Press space, then sit at your seat facing the camera. q to quit."
            elif ts - start < COUNTDOWN_S:
                status = f"Sit at your seat, hold still... {COUNTDOWN_S - (ts - start):.0f}"
            elif ts - start < COUNTDOWN_S + CAPTURE_S:
                cv2.imwrite(str(frames_dir / f"{saved:03d}.png"), frame)
                saved += 1
                status = f"CAPTURING ({saved})"
            else:
                break
            preview = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            draw_hud(preview, [f"{cam.width}x{cam.height}", status])
            cv2.imshow("depth capture", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return 1
            if key == ord(" ") and start is None:
                start = ts
    cv2.destroyAllWindows()
    print(f"saved {saved} frames to {frames_dir.relative_to(ROOT)}")
    print(f"next: python tools/depth_bench.py annotate {out.relative_to(ROOT)}")
    return 0


def first_frame(capture: Path):
    import cv2

    paths = sorted((capture / "frames").glob("*.png"))
    if not paths:
        raise SystemExit(f"no frames in {capture / 'frames'}")
    return cv2.imread(str(paths[len(paths) // 2]))


def cmd_annotate(args) -> int:
    """Click a point, type its distance in the window, Enter to save (no terminal input needed)."""
    import cv2

    from vision.overlay import draw_hud

    capture = args.capture
    img = first_frame(capture)
    scale = 0.5 if img.shape[1] > 1400 else 1.0
    gt_path = capture / "gt.json"
    meta_path = capture / "meta.json"
    points = json.loads(gt_path.read_text()) if gt_path.exists() else []
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    pending: list[tuple[int, int]] = []  # clicked point waiting for its distance
    typed = ""
    entering_height = False

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending[:] = [(int(x / scale), int(y / scale))]

    def save():
        gt_path.write_text(json.dumps(points, indent=2))
        meta_path.write_text(json.dumps(meta, indent=2))

    cv2.namedWindow("annotate")
    cv2.setMouseCallback("annotate", on_mouse)
    while True:
        view = cv2.resize(img, None, fx=scale, fy=scale)
        for i, p in enumerate(points):
            c = (int(p["u"] * scale), int(p["v"] * scale))
            cv2.circle(view, c, 6, (0, 0, 255), 2)
            cv2.putText(view, f"{i + 1} {p['label']} {p['distance_m']:.2f}m", (c[0] + 8, c[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        if entering_height:
            prompt = f"camera height above the floor = {typed or '_'} m   [Enter save, Esc cancel]"
        elif pending:
            u, v = pending[0]
            cv2.circle(view, (int(u * scale), int(v * scale)), 8, (0, 200, 255), 2)
            label = "seat (your face)" if not points else f"point {len(points) + 1}"
            prompt = f"{label}: tape distance from camera = {typed or '_'} m   [Enter save, Esc cancel]"
        elif not points:
            prompt = "Click your face (the seat) first"
        else:
            prompt = "Click a flat, matte spot (wall, door, sofa back)"
        height = f"{meta['camera_height_m']:.2f} m" if "camera_height_m" in meta else "not set"
        draw_hud(view, [prompt, f"{len(points)} points saved   camera height {height}   "
                                "h: set camera height   u: undo last   q: quit"])
        cv2.imshow("annotate", view)

        key = cv2.waitKey(30) & 0xFF
        if key == 255:
            continue
        typing = pending or entering_height
        if typing and (chr(key).isdigit() or chr(key) == "."):
            typed += chr(key)
        elif typing and key in (8, 127):  # backspace / delete
            typed = typed[:-1]
        elif typing and key == 27:  # esc
            pending.clear()
            typed, entering_height = "", False
        elif entering_height and key in (13, 10):
            try:
                meta["camera_height_m"] = float(typed)
            except ValueError:
                continue
            typed, entering_height = "", False
            save()
        elif pending and key in (13, 10):  # enter
            try:
                dist = float(typed)
            except ValueError:
                continue
            u, v = pending.pop()
            points.append({"u": u, "v": v, "distance_m": dist,
                           "label": "seat" if not points else f"p{len(points) + 1}"})
            typed = ""
            save()
        elif key == ord("h") and not pending:
            pending.clear()
            typed, entering_height = "", True
        elif key == ord("u") and points and not pending:
            points.pop()
            save()
        elif key == ord("q") and not pending:
            break
    cv2.destroyAllWindows()
    print(f"{len(points)} points saved to {gt_path}; meta {meta}")
    return 0


def model_distance(npz, u: int, v: int) -> float:
    """Straight-line distance camera -> surface at (u, v), median over a small patch."""
    pts = npz["points"]
    h, w = pts.shape[:2]
    r = PATCH // 2
    patch = pts[max(v - r, 0):min(v + r + 1, h), max(u - r, 0):min(u + r + 1, w)].reshape(-1, 3)
    return float(np.nanmedian(np.linalg.norm(patch, axis=1)))


def cmd_evaluate(args) -> int:
    from vision.room import fit_floor

    capture = args.capture
    gt = json.loads((capture / "gt.json").read_text())
    meta_path = capture / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    tape_height = meta.get("camera_height_m")
    if len(gt) < 2:
        raise SystemExit("need at least 2 tape points (seat + one more)")
    seat = gt[0]
    results = sorted(capture.glob("*.npz"))
    if not results:
        raise SystemExit("no <model>.npz yet - run tools/depth_infer.py")

    print(f"{len(gt)} tape points; seat correction uses '{seat['label']}' ({seat['distance_m']:.2f} m); "
          f"floor correction uses camera height {tape_height if tape_height else 'not set (press h in annotate)'}\n")
    summary = []
    for path in results:
        npz = np.load(path)
        name = path.stem
        pred = [model_distance(npz, p["u"], p["v"]) for p in gt]
        corrections = {"seat": seat["distance_m"] / pred[0]}
        floor = fit_floor(npz["points"])
        floor_note = "no floor found"
        if floor is not None:
            floor_note = (f"floor {floor.inlier_frac:.0%} of image, camera {floor.height_m:.2f} m up, "
                          f"pitch {floor.pitch_deg:+.1f}, roll {floor.roll_deg:+.1f}")
            if tape_height:
                corrections["floor"] = tape_height / floor.height_m
        hfov = np.degrees(2 * np.arctan((npz["points"].shape[1] / 2) / float(npz["fx"])))
        print(f"== {name}  ({npz['model']}, focal from {npz['focal_source']}, HFOV {hfov:.1f} deg, "
              f"{float(npz['seconds']):.1f}s)")
        print(f"   {floor_note}")
        print("   scale corrections: " + ", ".join(f"{k} x{v:.3f}" for k, v in corrections.items()))
        header = f"   {'point':8s} {'tape':>6s} {'raw':>7s}" + "".join(f" {k + ' corr':>11s}" for k in corrections)
        print(header)
        errors = {"raw": []} | {k: [] for k in corrections}
        for i, p in enumerate(gt):
            row = f"   {p['label']:8s} {p['distance_m']:6.2f} {(pred[i] - p['distance_m']) * 100:+6.0f}cm"
            errors["raw"].append(abs(pred[i] - p["distance_m"]) / p["distance_m"])
            for k, s in corrections.items():
                e = pred[i] * s - p["distance_m"]
                used = k == "seat" and i == 0  # the seat point defines that correction
                row += f" {'(ref)' if used else f'{e * 100:+6.0f}cm':>11s}"
                if not used:
                    errors[k].append(abs(e) / p["distance_m"])
            print(row)
        if tape_height and floor is not None:
            print(f"   camera height: tape {tape_height:.2f} m, model {floor.height_m:.2f} m "
                  f"({(floor.height_m - tape_height) * 100:+.0f} cm)")
        summary.append((name, {k: float(np.mean(v)) for k, v in errors.items() if v}))
        print()

    print("== summary: mean |error| / distance (a correction's own reference point is excluded)")
    for name, errs in summary:
        print(f"   {name:8s} " + "   ".join(f"{k} {v * 100:5.1f}%" for k, v in errs.items()))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--name")
    c.add_argument("--config")
    c.add_argument("--camera", type=int)
    c.add_argument("--width", type=int, default=1920)
    c.add_argument("--height", type=int, default=1080)
    a = sub.add_parser("annotate")
    a.add_argument("capture", type=Path)
    e = sub.add_parser("evaluate")
    e.add_argument("capture", type=Path)
    args = ap.parse_args()
    return {"capture": cmd_capture, "annotate": cmd_annotate, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
