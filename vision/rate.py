"""Adaptive detection rate: run the detector only as often as head motion needs.

Between two detections the head moves ``speed * interval``. Keeping that
below half the position tolerance (the other half is left for pipeline
latency) gives the detection rate needed:

    fps = speed / (0.5 * tolerance), clamped to [min_fps, max_fps]

The tolerance comes from the audio side: ~0.1 ms of L/R arrival-time error
noticeably shifts the phantom image, and a lateral offset dx changes the L/R
path difference by ~dx (speakers at +-30 deg), so 0.1 ms ~= 34 mm.

Speed is measured in the image plane (eye midpoint), scaled to mm with the
interpupillary distance. Depth motion is ignored: its estimate is too noisy
at this stage.

At a low rate the start of a movement would go unseen until the next
detection, so skipped frames get a cheap motion check: a 32x32 grayscale
thumbnail of the face region is compared with the one from the last detection,
and a large enough change triggers a detection immediately.
"""

from __future__ import annotations

import numpy as np

from .detector import FaceObservation

IPD_MM = 63.0
THUMB = 32


def thumbnail(frame: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    import cv2

    x0, y0, x1, y1 = region
    gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (THUMB, THUMB), interpolation=cv2.INTER_AREA).astype(np.int16)


class AdaptiveRate:
    def __init__(self, tolerance_mm: float = 30.0, min_fps: float = 5.0, max_fps: float = 30.0,
                 smoothing: float = 0.5, motion_threshold: float = 6.0):
        self.tolerance_mm = tolerance_mm
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.smoothing = smoothing  # EMA weight of the newest speed sample
        self.motion_threshold = motion_threshold  # mean abs gray-level change (0-255)
        self.motion = 0.0
        self._thumb: np.ndarray | None = None
        self._region: tuple[int, int, int, int] | None = None
        self.speed_mm_s = 0.0
        self.target_fps = max_fps
        self._last_ts: float | None = None
        self._last: FaceObservation | None = None

    def reset(self) -> None:
        self.speed_mm_s, self.target_fps, self.motion = 0.0, self.max_fps, 0.0
        self._last_ts = self._last = self._thumb = self._region = None

    def should_process(self, ts: float, frame: np.ndarray | None = None) -> bool:
        """True if enough time has passed since the last detection, or the face region moved."""
        if self._last_ts is None:
            return True
        # Small slack so a 30 fps camera isn't skipped at a 30 fps target by jitter.
        if ts - self._last_ts >= 0.9 / self.target_fps:
            return True
        if frame is not None and self._thumb is not None:
            self.motion = float(np.abs(thumbnail(frame, self._region) - self._thumb).mean())
            return self.motion > self.motion_threshold
        return False

    def update(self, face: FaceObservation | None, ts: float, frame: np.ndarray | None = None,
               region: tuple[int, int, int, int] | None = None) -> None:
        """Call after every detection with its result (and the frame + face region for motion checks)."""
        if frame is not None and region is not None and face is not None:
            self._region, self._thumb = region, thumbnail(frame, region)
        else:
            self._region = self._thumb = None
        if face is None:
            # Lost or searching: run at full rate until the face is back.
            self.speed_mm_s, self.target_fps, self._last = 0.0, self.max_fps, None
        else:
            if self._last is not None and ts > self._last_ts:
                mm_per_px = IPD_MM / max(face.ipd_px, 1.0)
                dist_mm = float(np.linalg.norm(face.eye_mid - self._last.eye_mid)) * mm_per_px
                speed = dist_mm / (ts - self._last_ts)
                self.speed_mm_s += self.smoothing * (speed - self.speed_mm_s)
            fps = self.speed_mm_s / (0.5 * self.tolerance_mm)
            self.target_fps = float(np.clip(fps, self.min_fps, self.max_fps))
            self._last = face
        self._last_ts = ts
