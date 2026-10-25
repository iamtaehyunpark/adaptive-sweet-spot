"""Milestone 2.2: BlazeFace detection -> listener offset from the designated center.

Assumed setup: the camera sits on top of the front center (FC) speaker, level,
facing the listener. The designated center is then straight ahead of the
camera at the FC distance ``D``.

Top-down 2D, room frame (+x right, +y toward the front speakers), meters.
The ruler is the BlazeFace face-box width ``bw`` (px), real width ``Wf`` (m):

    Z = scale * f * Wf / bw            distance from the camera
    x = -(u - cx) * Wf / bw - x_bias   face width as a ruler at the face's depth;
                                       camera faces the listener: image right = room left
    y = D - Z                          + = listener moved toward the front

Why the face box and not the eye spacing: turning the head by yaw shrinks the
apparent eye spacing by cos(yaw) (40 deg -> distance 30% too far), while the
box width stays within ~2% up to ~50 deg yaw (measured, m1.2 log) and is
unaffected by nodding.

Calibration: sit at the intended center, facing the camera, and capture. That
learns ``Wf`` from the eye spacing (``Wf = IPD * bw / ipd_px`` while facing the
camera), ``scale`` (camera FOV, so distance is exact at the center) and
``x_bias`` (camera not exactly on the center line). Lateral accuracy then
depends only on the IPD assumption: set ``vision.ipd_m`` to the listener's own.
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
# Typical face-box width: BlazeFace box ~2.44 x eye spacing when facing the camera (m1.2 log).
DEFAULT_FACE_WIDTH_M = 2.44 * IPD_M
FACING_YAW_DEG = 10.0  # frames turned more than this are not used to learn the face width


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
    face_width_m: float = DEFAULT_FACE_WIDTH_M

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> Calibration:
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        if "face_width_m" not in data:  # older eye-spacing calibration: no longer valid
            return cls()
        return cls(**data)


class CenterMapper:
    def __init__(self, width: int, height: int, hfov_deg: float, center_distance_m: float,
                 calibration: Calibration | None = None, ipd_m: float = IPD_M):
        self.cx = width / 2
        self.focal_px = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
        self.center_distance_m = center_distance_m
        self.ipd_m = ipd_m
        self.cal = calibration or Calibration()

    def _raw(self, face: FaceObservation) -> tuple[float, float]:
        """Uncalibrated (distance, lateral offset), using the face-box width as the ruler."""
        bw = face.bbox[2]
        z = self.focal_px * self.cal.face_width_m / bw
        x = -(face.eye_mid[0] - self.cx) * self.cal.face_width_m / bw
        return z, x

    def map(self, face: FaceObservation) -> ListenerOffset:
        z, x = self._raw(face)
        z *= self.cal.scale
        x -= self.cal.x_bias
        return ListenerOffset(x, self.center_distance_m - z, z, estimate_yaw_deg(face.keypoints),
                              face.timestamp_s, face.score)

    def calibrate(self, faces: list[FaceObservation]) -> Calibration:
        """The listener is at the center, facing the camera, for all ``faces``."""
        facing = [f for f in faces if abs(estimate_yaw_deg(f.keypoints)) < FACING_YAW_DEG] or faces
        face_width = self.ipd_m * float(np.median([f.bbox[2] / f.ipd_px for f in facing]))
        self.cal = Calibration(face_width_m=face_width)
        raw = [self._raw(f) for f in faces]
        z_raw = float(np.median([z for z, _ in raw]))
        x_raw = float(np.median([x for _, x in raw]))
        self.cal = Calibration(self.center_distance_m / z_raw, x_raw, face_width)
        return self.cal
