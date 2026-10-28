#!/usr/bin/env python3
"""Compare metric depth models on your own room against tape measurements.

1. capture   live preview; press space, sit at the seat during the 5 s countdown,
             then 2 s of 1080p frames are saved to assets/captures/<name>/frames/
2. annotate  click points on the captured frame and type their tape-measured
             distance from the camera lens. First point = your face (the seat).
3. run the models (each in its own env, see tools/depth_infer.py)
4. evaluate  error of every <model>.npz vs the tape points, raw and after
             correcting the scale with the seat distance alone

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
    import cv2

    capture = args.capture
    img = first_frame(capture)
    scale = 0.5 if img.shape[1] > 1400 else 1.0
    gt_path = capture / "gt.json"
    points = json.loads(gt_path.read_text()) if gt_path.exists() else []
    clicked: list[tuple[int, int]] = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((int(x / scale), int(y / scale)))

    cv2.namedWindow("annotate")
    cv2.setMouseCallback("annotate", on_mouse)
    print("Click a point, then type its tape distance (m) here. "
          "First point: your face (seat). u = undo last, q = save and quit.")
    while True:
        view = cv2.resize(img, None, fx=scale, fy=scale)
        for i, p in enumerate(points):
            c = (int(p["u"] * scale), int(p["v"] * scale))
            cv2.circle(view, c, 6, (0, 0, 255), 2)
            cv2.putText(view, f"{i + 1}:{p['distance_m']:.2f}m {p['label']}", (c[0] + 8, c[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("annotate", view)
        key = cv2.waitKey(30) & 0xFF
        if clicked:
            u, v = clicked.pop(0)
            default = "seat" if not points else f"p{len(points) + 1}"
            try:
                dist = float(input(f"point {len(points) + 1} at ({u},{v}) - tape distance in m: "))
            except ValueError:
                print("skipped (not a number)")
                continue
            label = input(f"  label [{default}]: ").strip() or default
            points.append({"u": u, "v": v, "distance_m": dist, "label": label})
            gt_path.write_text(json.dumps(points, indent=2))
        if key == ord("u") and points:
            print(f"removed {points.pop()}")
            gt_path.write_text(json.dumps(points, indent=2))
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    print(f"{len(points)} points saved to {gt_path.relative_to(ROOT)}")
    return 0


def model_distance(npz, u: int, v: int) -> float:
    """Straight-line distance camera -> surface at (u, v), median over a small patch."""
    pts = npz["points"]
    h, w = pts.shape[:2]
    r = PATCH // 2
    patch = pts[max(v - r, 0):min(v + r + 1, h), max(u - r, 0):min(u + r + 1, w)].reshape(-1, 3)
    return float(np.nanmedian(np.linalg.norm(patch, axis=1)))


def cmd_evaluate(args) -> int:
    capture = args.capture
    gt = json.loads((capture / "gt.json").read_text())
    if len(gt) < 2:
        raise SystemExit("need at least 2 tape points (seat + one more)")
    seat, others = gt[0], gt[1:]
    results = sorted(capture.glob("*.npz"))
    if not results:
        raise SystemExit("no <model>.npz yet - run tools/depth_infer.py")

    print(f"{len(gt)} tape points; scale correction uses '{seat['label']}' ({seat['distance_m']:.2f} m)\n")
    summary = []
    for path in results:
        npz = np.load(path)
        name = path.stem
        pred = {i: model_distance(npz, p["u"], p["v"]) for i, p in enumerate(gt)}
        s = seat["distance_m"] / pred[0]
        hfov = np.degrees(2 * np.arctan((npz["points"].shape[1] / 2) / float(npz["fx"])))
        print(f"== {name}  ({npz['model']}, focal from {npz['focal_source']}, HFOV {hfov:.1f} deg, "
              f"{float(npz['seconds']):.1f}s, scale correction x{s:.3f})")
        print(f"   {'point':12s} {'tape':>6s} {'raw':>6s} {'err':>7s} {'scaled':>7s} {'err':>7s}")
        raw_err, cor_err = [], []
        for i, p in enumerate(gt):
            d_raw, d_cor = pred[i], pred[i] * s
            e_raw, e_cor = d_raw - p["distance_m"], d_cor - p["distance_m"]
            raw_err.append(abs(e_raw) / p["distance_m"])
            if i > 0:
                cor_err.append(abs(e_cor) / p["distance_m"])
            print(f"   {p['label']:12s} {p['distance_m']:6.2f} {d_raw:6.2f} {e_raw * 100:+6.0f}cm "
                  f"{d_cor:7.2f} {(e_cor * 100 if i else 0):+6.0f}cm")
        summary.append((name, np.mean(raw_err), np.mean(cor_err)))
        print()

    print("== summary (mean |error| / distance; scaled excludes the seat point)")
    for name, r, c in sorted(summary, key=lambda x: x[2]):
        print(f"   {name:8s} raw {r * 100:5.1f}%   scale-corrected {c * 100:5.1f}%")
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
