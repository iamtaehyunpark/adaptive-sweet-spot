"""Milestone 2.2: BlazeFace detection -> listener offset from the designated center.

Assumed setup: the camera sits on top of the front center (FC) speaker, level,
facing the listener. The designated center is then straight ahead of the
camera at the FC distance ``D``.

Top-down 2D, room frame (+x right, +y toward the front speakers), meters:

    Z = scale * f * IPD / ipd_px          distance from the camera (eye spacing)
    x = -(u - cx) * IPD / ipd_px - x_bias  eye spacing as a ruler at the face's depth;
                                          camera faces the listener: image right = room left
    y = D - Z                             + = listener moved toward the front

Calibration: sit at the intended center and capture; that sets ``scale``
(absorbs the camera's real FOV and the listener's real IPD, so distance is
exact) and ``x_bias`` (camera not exactly on the center line).

x does not depend on the camera FOV, only on the listener's real IPD: set
``vision.ipd_m`` to the listener's own value for best lateral accuracy
(the 63 mm default can be ~8% off).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .detector import FaceObservation
from .head_pose import estimate_yaw_deg

IPD_M = 0.063


@dataclass
class ListenerOffset:
    x: float           # m, + = listener's right of center
    y: float           # m, + = toward the front speakers
    distance: float    # m from the camera
    yaw_deg: float     # image-space head yaw (not used for steering yet)
    timestamp_s: float
    score: float


@dataclass
class Calibration:
    scale: float = 1.0
    x_bias: float = 0.0

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> Calibration:
        return cls(**json.loads(path.read_text())) if path.exists() else cls()


class CenterMapper:
    def __init__(self, width: int, height: int, hfov_deg: float, center_distance_m: float,
                 calibration: Calibration | None = None, ipd_m: float = IPD_M):
        self.cx = width / 2
        self.focal_px = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
        self.center_distance_m = center_distance_m
        self.ipd_m = ipd_m
        self.cal = calibration or Calibration()

    def _raw(self, face: FaceObservation) -> tuple[float, float]:
        """Uncalibrated (distance, lateral offset)."""
        z = self.focal_px * self.ipd_m / face.ipd_px
        x = -(face.eye_mid[0] - self.cx) * self.ipd_m / face.ipd_px
        return z, x

    def map(self, face: FaceObservation) -> ListenerOffset:
        z, x = self._raw(face)
        z *= self.cal.scale
        x -= self.cal.x_bias
        return ListenerOffset(x, self.center_distance_m - z, z, estimate_yaw_deg(face.keypoints),
                              face.timestamp_s, face.score)

    def calibrate(self, faces: list[FaceObservation]) -> Calibration:
        """The listener is at the center for all ``faces``: solve scale and x_bias (medians)."""
        raw = [self._raw(f) for f in faces]
        z_raw = float(np.median([z for z, _ in raw]))
        x_raw = float(np.median([x for _, x in raw]))
        self.cal = Calibration(self.center_distance_m / z_raw, x_raw)
        return self.cal
