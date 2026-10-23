"""Distance-robust face tracking on top of BlazeFace.

BlazeFace sees a downscaled copy of its input (128 or 192 px), so a distant
face in a full frame is too small to detect. Two simple steps fix that:

* track:  a face was found last frame -> detect only in a square crop
          ``roi_scale`` x the face width around it, so the face fills the
          model input regardless of distance.
* search: no face yet -> try the full frame, then n x n overlapping tiles
          for each n in ``search_levels``, stopping at the first hit.
"""

from __future__ import annotations

import numpy as np

from .detector import BlazeFaceDetector, FaceObservation


def tile_regions(width: int, height: int, n: int, overlap: float = 0.25) -> list[tuple[int, int, int, int]]:
    """n x n tiles covering the frame, each overlapping its neighbours by ``overlap``."""
    if n <= 1:
        return [(0, 0, width, height)]
    tw = int(width / n * (1 + overlap))
    th = int(height / n * (1 + overlap))
    xs = np.linspace(0, width - tw, n).astype(int)
    ys = np.linspace(0, height - th, n).astype(int)
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def roi_around(face: FaceObservation, scale: float, min_size: int = 96) -> tuple[int, int, int, int]:
    """Square region ``scale`` x the face width, centered on the face, clipped to the frame."""
    fw, fh = face.frame_size
    x, y, w, h = face.bbox
    size = int(max(scale * max(w, h), min_size))
    size = min(size, fw, fh)
    cx, cy = x + w / 2, y + h / 2
    x0 = int(np.clip(cx - size / 2, 0, fw - size))
    y0 = int(np.clip(cy - size / 2, 0, fh - size))
    return x0, y0, x0 + size, y0 + size


class FaceTracker:
    def __init__(self, detector: BlazeFaceDetector, roi_scale: float = 4.0,
                 search_levels: tuple[int, ...] = (1, 2, 3)):
        self.detector = detector
        self.roi_scale = roi_scale
        self.search_levels = search_levels
        self.last: FaceObservation | None = None
        # Debug info about the last call
        self.mode = "search"
        self.regions: list[tuple[int, int, int, int]] = []
        self.infer_ms = 0.0
        self.inferences = 0

    def update(self, frame: np.ndarray, timestamp_s: float) -> FaceObservation | None:
        self.infer_ms, self.inferences, self.regions = 0.0, 0, []

        if self.last is not None:
            self.mode = "track"
            face = self._best(frame, timestamp_s, [roi_around(self.last, self.roi_scale)])
            if face is not None:
                self.last = face
                return face

        self.mode = "search"
        h, w = frame.shape[:2]
        for n in self.search_levels:
            face = self._best(frame, timestamp_s, tile_regions(w, h, n))
            if face is not None:
                self.last = face
                return face
        self.last = None
        return None

    def reset(self) -> None:
        self.last = None

    def _best(self, frame, timestamp_s, regions) -> FaceObservation | None:
        faces = []
        for region in regions:
            faces += self.detector.detect(frame, timestamp_s, region)
            self.infer_ms += self.detector.last_infer_ms
            self.inferences += 1
            self.regions.append(region)
        return max(faces, key=lambda f: f.score, default=None)
