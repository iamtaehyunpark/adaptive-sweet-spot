#!/usr/bin/env python3
"""Milestone 1.2: BlazeFace test interface.

Opens a window with the live webcam feed, face box, 6 keypoints and stats.
Press a number key to run a guided test; results appear in the window and
are printed to the terminal.

    python tools/m1_2_vision_check.py
    python tools/m1_2_vision_check.py --log m12.csv      # also log every frame
    python tools/m1_2_vision_check.py --video clip.mp4   # recorded input
    python tools/m1_2_vision_check.py --camera 1         # start on camera index 1
    python tools/m1_2_vision_check.py --list-cameras

Keys:
    1  Still     sit still and look at the camera
    2  Yaw       slowly turn your head fully left, then right
    3  Move      move and lean around at normal speed
    4  Distance  slowly lean/walk back to your listening position
    c  switch to the next camera
    q  quit

"latency" = frame read -> detection ready (excludes sensor/USB delay).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402
from vision.detector import KEYPOINT_NAMES, BlazeFaceDetector  # noqa: E402
from vision.head_pose import estimate_roll_deg, estimate_yaw_deg  # noqa: E402
from vision.rate import AdaptiveRate  # noqa: E402
from vision.tracker import FaceTracker, roi_around  # noqa: E402

COUNTDOWN_S = 3.0
IPD_MM = 63.0


@dataclass
class Sample:
    ts: float
    found: bool
    eye_mid: np.ndarray | None = None
    yaw: float = float("nan")
    roll: float = float("nan")
    ipd: float = float("nan")


def detection_rate(samples):
    return sum(s.found for s in samples) / max(len(samples), 1)


def longest_dropout_ms(samples):
    longest, start = 0.0, None
    for s in samples:
        if not s.found and start is None:
            start = s.ts
        elif s.found and start is not None:
            longest, start = max(longest, s.ts - start), None
    if start is not None:
        longest = max(longest, samples[-1].ts - start)
    return longest * 1000.0


def found(samples):
    return [s for s in samples if s.found]


def result_still(samples):
    hits = found(samples)
    if len(hits) < 5:
        return False, "face not detected"
    pts = np.array([s.eye_mid for s in hits])
    jitter_px = float(np.sqrt(((pts - pts.mean(0)) ** 2).sum(1).mean()))
    # Scale pixels to mm using the average interpupillary distance (~63 mm).
    jitter_mm = jitter_px / np.mean([s.ipd for s in hits]) * IPD_MM
    yaw_sd = float(np.std([s.yaw for s in hits]))
    rate = detection_rate(samples)
    ok = rate >= 0.95 and jitter_mm < 10.0
    return ok, f"detect {rate:.0%}  jitter {jitter_mm:.1f}mm ({jitter_px:.1f}px)  yaw sd {yaw_sd:.1f}deg"


def result_yaw(samples):
    hits = found(samples)
    if len(hits) < 5:
        return False, "face not detected"
    yaws = [s.yaw for s in hits]
    rate = detection_rate(samples)
    ok = rate >= 0.90 and min(yaws) <= -30 and max(yaws) >= 30
    return ok, f"yaw {min(yaws):+.0f}..{max(yaws):+.0f}deg  detect {rate:.0%}"


def result_move(samples):
    rate = detection_rate(samples)
    drop = longest_dropout_ms(samples)
    return rate >= 0.95, f"detect {rate:.0%}  longest dropout {drop:.0f}ms"


def result_distance(samples):
    hits = found(samples)
    if not hits:
        return False, "face not detected"
    ipds = [s.ipd for s in hits]
    # Pass/fail is judged by you: did detection hold at your listening position?
    return None, (f"detect {detection_rate(samples):.0%}  ipd {max(ipds):.0f}->{min(ipds):.0f}px  "
                  f"last face ipd {hits[-1].ipd:.0f}px")


@dataclass
class Test:
    name: str
    instruction: str
    duration_s: float
    evaluate: callable
    result: str = ""
    passed: bool | None = None
    samples: list = field(default_factory=list)


TESTS = {
    ord("1"): Test("Still", "Sit still, look at the camera", 10, result_still),
    ord("2"): Test("Yaw", "Slowly turn head fully left, then right", 10, result_yaw),
    ord("3"): Test("Move", "Move and lean around at normal speed", 15, result_move),
    ord("4"): Test("Distance", "Slowly move back to your listening position", 15, result_distance),
}


WINDOW = "m1.2 BlazeFace test"


class CameraSource:
    """Webcam input; ``switch()`` cycles through all detected cameras."""

    is_video = False

    def __init__(self, cameras, start_index: int, width: int, height: int, fps: int):
        self.cameras = cameras
        self.settings = (width, height, fps)
        self.pos = next((i for i, c in enumerate(cameras) if c.index == start_index), 0)
        self.cam = None
        self._open()

    def _open(self) -> None:
        from vision.camera import Camera

        if self.cam is not None:
            self.cam.close()
        info = self.cameras[self.pos]
        self.cam = Camera(info.index, *self.settings)
        print(f"using camera {info.index} ({info.name}): "
              f"{self.cam.width}x{self.cam.height} @ {self.cam.reported_fps:.1f} fps (reported)")

    def switch(self) -> None:
        self.pos = (self.pos + 1) % len(self.cameras)
        self._open()

    @property
    def label(self) -> str:
        info = self.cameras[self.pos]
        return f"camera {self.pos + 1}/{len(self.cameras)}: {info.name} (index {info.index})  [c: switch]"

    def read(self):
        return self.cam.read()

    def close(self) -> None:
        self.cam.close()


class VideoSource:
    is_video = True

    def __init__(self, path: str):
        import cv2

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        self.label = f"video: {path}"

    def switch(self) -> None:
        pass

    def read(self):
        ok, frame = self.cap.read()
        return (time.perf_counter(), frame) if ok else None

    def close(self) -> None:
        self.cap.release()


def open_source(args, vcfg):
    if args.video:
        return VideoSource(args.video)

    from vision.camera import CameraInfo, list_cameras

    print("looking for cameras...")
    cameras = list_cameras()
    for c in cameras:
        print(f"  index {c.index}: {c.name} ({c.width}x{c.height})")
    start = args.camera if args.camera is not None else vcfg["camera_index"]
    if args.camera is not None and all(c.index != start for c in cameras):
        cameras.insert(0, CameraInfo(start, f"camera {start}", 0, 0))
    if not cameras:
        raise SystemExit("no cameras found")
    return CameraSource(cameras, start, args.width or vcfg["width"], args.height or vcfg["height"],
                        args.fps or vcfg["fps"])


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
    ap.add_argument("--log", help="write per-frame CSV")
    ap.add_argument("--list-cameras", action="store_true", help="list cameras and exit")
    args = ap.parse_args()

    if args.list_cameras:
        from vision.camera import list_cameras

        for c in list_cameras():
            print(f"index {c.index}: {c.name} ({c.width}x{c.height})")
        return 0

    vcfg = load_config(args.config)["vision"]
    source = open_source(args, vcfg)

    writer = log_file = None
    if args.log:
        log_file = open(args.log, "w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(["t", "test", "infer_ms", "latency_ms", "n_faces", "score",
                         "bx", "by", "bw", "bh",
                         *[f"{n}_{a}" for n in KEYPOINT_NAMES for a in "xy"],
                         "ipd_px", "yaw_deg", "roll_deg"])

    frame_times: list[float] = []
    latencies: list[float] = []
    active: Test | None = None
    active_start = 0.0

    with BlazeFaceDetector(ROOT / vcfg["model_path"], vcfg["min_confidence"]) as det:
        tracker = FaceTracker(det, vcfg["roi_scale"], tuple(vcfg["search_levels"]))
        rate = None
        if vcfg["adaptive_rate"]:
            rate = AdaptiveRate(vcfg["tolerance_mm"], vcfg["min_fps"], max_fps=args.fps or vcfg["fps"])
        det_times: list[float] = []
        latency = 0.0
        while True:
            item = source.read()
            if item is None:
                if source.is_video:
                    break
                # Camera stopped delivering frames: keep the window up so the user can switch.
                blank = np.zeros((360, 640, 3), np.uint8)
                draw_hud(blank, [source.label, "", "No signal - press c to switch camera, q to quit"])
                cv2.imshow(WINDOW, blank)
                key = cv2.waitKey(100) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    source.switch()
                    tracker.reset()
                    det_times = []
                    if rate is not None:
                        rate.reset()
                    frame_times, active = [], None
                continue

            ts, frame = item
            frame_times = (frame_times + [ts])[-60:]
            # Adaptive rate: skip detection when the head is moving slowly enough.
            processed = rate is None or rate.should_process(ts, frame)
            if processed:
                face = tracker.update(frame, ts)
                if rate is not None:
                    rate.update(face, ts, frame, roi_around(face, tracker.roi_scale) if face else None)
                latency = (time.perf_counter() - ts) * 1000.0
                latencies.append(latency)
                det_times = (det_times + [ts])[-30:]
            else:
                face = tracker.last
            faces = [face] if face is not None else []

            sample = Sample(ts, face is not None)
            if face is not None:
                sample.eye_mid = face.eye_mid
                sample.yaw = estimate_yaw_deg(face.keypoints)
                sample.roll = estimate_roll_deg(face.keypoints)
                sample.ipd = face.ipd_px

            # Run the active test: countdown, then record for its duration.
            status = "Press 1-4 to run a test, c to switch camera, q to quit"
            if active is not None:
                elapsed = ts - active_start
                if elapsed < COUNTDOWN_S:
                    status = f"[{active.name}] {active.instruction} - starting in {COUNTDOWN_S - elapsed:.0f}"
                elif elapsed < COUNTDOWN_S + active.duration_s:
                    if processed:
                        active.samples.append(sample)
                    left = COUNTDOWN_S + active.duration_s - elapsed
                    status = f"[{active.name}] RECORDING {left:4.1f}s - {active.instruction}"
                else:
                    active.passed, active.result = active.evaluate(active.samples)
                    print(f"{active.name:9s} {verdict(active.passed):5s} {active.result}")
                    active = None

            if writer and processed:
                row = [f"{ts:.4f}", active.name if active else "", f"{tracker.infer_ms:.3f}",
                       f"{latency:.3f}", len(faces)]
                if face is not None:
                    row += [f"{face.score:.3f}", *np.round(face.bbox, 1),
                            *np.round(face.keypoints.ravel(), 1),
                            f"{face.ipd_px:.1f}", f"{sample.yaw:.1f}", f"{sample.roll:.1f}"]
                writer.writerow(row)

            fps = rate_of(frame_times)
            det_line = f"detect {rate_of(det_times):4.1f} fps"
            if rate is not None:
                det_line += (f" (target {rate.target_fps:4.1f}, head {rate.speed_mm_s:4.0f} mm/s, "
                             f"motion {rate.motion:4.1f})")
            hud = [source.label,
                   f"camera {fps:4.1f} fps  infer {tracker.infer_ms:4.1f}ms  latency {latency:4.1f}ms  "
                   f"{tracker.mode} ({tracker.inferences} run{'s' if tracker.inferences != 1 else ''})", det_line]
            if face is not None:
                hud.append(f"yaw {sample.yaw:+5.1f}  roll {sample.roll:+5.1f}  ipd {face.ipd_px:4.0f}px  "
                           f"score {face.score:.2f}")
            else:
                hud.append("no face")
            hud += ["", status, ""]
            hud += [f"{k - ord('0')} {t.name:9s} {verdict(t.passed) if t.result else '-':5s} {t.result}"
                    for k, t in TESTS.items()]

            for x0, y0, x1, y1 in tracker.regions:
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 200, 255), 1)
            if face is not None:
                draw_face(frame, face)
            draw_hud(frame, hud)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                source.switch()
                tracker.reset()
                det_times = []
                if rate is not None:
                    rate.reset()
                frame_times, active = [], None  # a test can't span two cameras
                continue
            if key in TESTS and active is None:
                active = TESTS[key]
                active.samples, active.result, active.passed = [], "", None
                active_start = ts

    source.close()
    cv2.destroyAllWindows()
    if log_file:
        log_file.close()

    print("\n=== Milestone 1.2 summary ===")
    for t in TESTS.values():
        print(f"{t.name:9s} {verdict(t.passed) if t.result else 'not run':7s} {t.result}")
    if latencies:
        lat = np.array(latencies)
        print(f"latency   p50 {np.percentile(lat, 50):.1f}ms  p95 {np.percentile(lat, 95):.1f}ms")
    return 0


def rate_of(times: list[float]) -> float:
    return (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0


def verdict(passed: bool | None) -> str:
    return {True: "PASS", False: "FAIL", None: "INFO"}[passed]


if __name__ == "__main__":
    sys.exit(main())
