#!/usr/bin/env python3
"""Milestone 1.2: validate BlazeFace face position and head angle in real time.

Shows the webcam feed with the face box, 6 keypoints and live stats, and
optionally logs every frame to CSV for offline analysis.

    python tools/m1_2_vision_check.py                    # live window
    python tools/m1_2_vision_check.py --log run.csv      # + per-frame log
    python tools/m1_2_vision_check.py --video clip.mp4   # recorded input
    python tools/m1_2_vision_check.py --no-window --duration 10

Keys: q quit, r reset stats.

Note: "latency" here is frame-grabbed -> detection-ready. It excludes sensor
exposure and USB transfer, which must be measured externally (e.g. LED test).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402
from vision.detector import KEYPOINT_NAMES, BlazeFaceDetector, primary_face  # noqa: E402
from vision.head_pose import estimate_roll_deg, estimate_yaw_deg  # noqa: E402


class Stats:
    def __init__(self, window: int = 120):
        self.window = window
        self.reset()

    def reset(self) -> None:
        w = self.window
        self.frame_ts = deque(maxlen=w)
        self.infer_ms = deque(maxlen=w)
        self.latency_ms = deque(maxlen=w)
        self.detected = deque(maxlen=w)
        self.eye_mid = deque(maxlen=30)
        self.total = 0
        self.total_detected = 0
        self.all_infer = []
        self.all_latency = []

    def add(self, ts: float, infer: float, latency: float, face) -> None:
        self.frame_ts.append(ts)
        self.infer_ms.append(infer)
        self.latency_ms.append(latency)
        self.detected.append(face is not None)
        self.total += 1
        self.all_infer.append(infer)
        self.all_latency.append(latency)
        if face is not None:
            self.total_detected += 1
            self.eye_mid.append(face.eye_mid)

    @property
    def fps(self) -> float:
        if len(self.frame_ts) < 2:
            return 0.0
        return (len(self.frame_ts) - 1) / (self.frame_ts[-1] - self.frame_ts[0])

    @property
    def jitter_px(self) -> float:
        """RMS deviation of the eye midpoint over the last 30 detections (hold still to read)."""
        if len(self.eye_mid) < 5:
            return float("nan")
        pts = np.array(self.eye_mid)
        return float(np.sqrt(((pts - pts.mean(0)) ** 2).sum(1).mean()))

    def summary(self) -> str:
        if not self.total:
            return "no frames processed"
        inf, lat = np.array(self.all_infer), np.array(self.all_latency)
        return (
            f"frames={self.total}  detection_rate={self.total_detected / self.total:.1%}\n"
            f"infer_ms    p50={np.percentile(inf, 50):.2f}  p95={np.percentile(inf, 95):.2f}  max={inf.max():.2f}\n"
            f"latency_ms  p50={np.percentile(lat, 50):.2f}  p95={np.percentile(lat, 95):.2f}  max={lat.max():.2f}"
        )


def frames_from_camera(args, vcfg):
    from vision.camera import FrameGrabber

    grabber = FrameGrabber(args.camera if args.camera is not None else vcfg["camera_index"],
                           args.width or vcfg["width"], args.height or vcfg["height"],
                           args.fps or vcfg["fps"])
    print(f"camera: {grabber.width}x{grabber.height} @ {grabber.reported_fps:.1f} fps (reported)")
    try:
        seq = 0
        while True:
            item = grabber.read(seq)
            if item is None:
                print("camera timeout", file=sys.stderr)
                return
            seq, ts, frame = item
            yield seq, ts, frame
    finally:
        grabber.close()


def frames_from_video(path: str):
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"video: {path} @ {fps:.1f} fps")
    seq = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        seq += 1
        # Synthetic capture time so latency reflects processing only.
        yield seq, time.perf_counter(), frame
    cap.release()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--camera", type=int)
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--fps", type=int)
    ap.add_argument("--video", help="process a video file instead of the webcam")
    ap.add_argument("--log", help="write per-frame CSV")
    ap.add_argument("--duration", type=float, help="stop after N seconds")
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args()

    vcfg = load_config(args.config)["vision"]
    model = ROOT / vcfg["model_path"]
    show = not args.no_window
    if show:
        import cv2
        from vision.overlay import draw_face, draw_hud

    stats = Stats()
    log_file = writer = None
    if args.log:
        log_file = open(args.log, "w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(["t", "seq", "infer_ms", "latency_ms", "n_faces", "score",
                         "bx", "by", "bw", "bh",
                         *[f"{n}_{a}" for n in KEYPOINT_NAMES for a in "xy"],
                         "ipd_px", "yaw_deg", "roll_deg"])

    source = frames_from_video(args.video) if args.video else frames_from_camera(args, vcfg)
    t_start = time.perf_counter()
    last_print = t_start
    try:
        with BlazeFaceDetector(model, vcfg["min_confidence"]) as det:
            for seq, ts, frame in source:
                faces = det.detect(frame, ts)
                latency = (time.perf_counter() - ts) * 1000.0
                face = primary_face(faces)
                stats.add(ts, det.last_infer_ms, latency, face)

                yaw = roll = float("nan")
                if face is not None:
                    yaw, roll = estimate_yaw_deg(face.keypoints), estimate_roll_deg(face.keypoints)

                if writer:
                    row = [f"{ts - t_start:.4f}", seq, f"{det.last_infer_ms:.3f}", f"{latency:.3f}", len(faces)]
                    if face is not None:
                        row += [f"{face.score:.3f}", *np.round(face.bbox, 1),
                                *np.round(face.keypoints.ravel(), 1),
                                f"{face.ipd_px:.1f}", f"{yaw:.1f}", f"{roll:.1f}"]
                    writer.writerow(row)

                hud = [
                    f"fps {stats.fps:5.1f}   infer {det.last_infer_ms:5.2f} ms   latency {latency:5.2f} ms",
                    f"detect {np.mean(stats.detected):.0%}   faces {len(faces)}",
                ]
                if face is not None:
                    hud.append(f"yaw {yaw:+6.1f}  roll {roll:+6.1f}  ipd {face.ipd_px:5.1f}px  "
                               f"jitter {stats.jitter_px:4.2f}px")

                now = time.perf_counter()
                if now - last_print >= 2.0:
                    print(" | ".join(hud))
                    last_print = now

                if show:
                    for f in faces:
                        draw_face(frame, f, primary=f is face)
                    draw_hud(frame, hud)
                    cv2.imshow("m1.2 BlazeFace check", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("r"):
                        stats.reset()

                if args.duration and now - t_start >= args.duration:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if log_file:
            log_file.close()
        if show:
            cv2.destroyAllWindows()

    print("\n" + stats.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
