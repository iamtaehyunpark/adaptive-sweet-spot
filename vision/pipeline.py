"""Detector + tracker + adaptive rate, as one per-frame call."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .detector import BlazeFaceDetector, FaceObservation
from .rate import AdaptiveRate
from .tracker import FaceTracker, roi_around


class VisionPipeline:
    def __init__(self, vcfg: dict, root: Path, max_fps: float | None = None):
        self.detector = BlazeFaceDetector(root / vcfg["model_path"], vcfg["min_confidence"])
        self.tracker = FaceTracker(self.detector, vcfg["roi_scale"], tuple(vcfg["search_levels"]))
        self.rate = None
        if vcfg["adaptive_rate"]:
            self.rate = AdaptiveRate(vcfg["tolerance_mm"], vcfg["min_fps"], max_fps=max_fps or vcfg["fps"])

    def process(self, frame: np.ndarray, ts: float) -> tuple[FaceObservation | None, bool]:
        """Returns (face, processed). When the rate skips a frame, the last face is returned."""
        if self.rate is not None and not self.rate.should_process(ts, frame):
            return self.tracker.last, False
        face = self.tracker.update(frame, ts)
        if self.rate is not None:
            self.rate.update(face, ts, frame, roi_around(face, self.tracker.roi_scale) if face else None)
        return face, True

    def reset(self) -> None:
        self.tracker.reset()
        if self.rate is not None:
            self.rate.reset()

    def close(self) -> None:
        self.detector.close()
