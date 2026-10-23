"""Head angle estimates from the 6 BlazeFace keypoints.

These are geometric approximations (BlazeFace has no 3D landmarks):

* yaw: the nose tip sits in front of the ear axis, so as the head turns its
  projection shifts relative to the ear midpoint while the ear span shrinks.
  With nose depth ``Rn`` and ear half-span ``Re``:
      (nose_x - ear_mid_x) / ear_half_span = (Rn / Re) * tan(yaw)
* roll: angle of the eye line.

Sign conventions (image space, camera facing the subject):
    yaw  > 0  nose moves toward image right (subject turns to *their* left)
    roll > 0  eye line rotates clockwise in the image
Conversion to the room frame happens in Milestone 2.2.
"""

from __future__ import annotations

import math

import numpy as np

from .detector import LEFT_EAR, LEFT_EYE, NOSE, RIGHT_EAR, RIGHT_EYE

# Nose-tip depth in front of the tragion axis / half the tragion span (~9.5 cm / 7.5 cm).
DEFAULT_NOSE_DEPTH_RATIO = 1.25


def estimate_yaw_deg(keypoints: np.ndarray, nose_depth_ratio: float = DEFAULT_NOSE_DEPTH_RATIO) -> float:
    ear_r, ear_l, nose = keypoints[RIGHT_EAR], keypoints[LEFT_EAR], keypoints[NOSE]
    half_span = 0.5 * (ear_l[0] - ear_r[0])
    if half_span <= 1e-6:
        # Ears collapsed or crossed: head turned ~90° or keypoints unreliable.
        return math.copysign(90.0, nose[0] - 0.5 * (ear_l[0] + ear_r[0]))
    k = (nose[0] - 0.5 * (ear_l[0] + ear_r[0])) / half_span
    return math.degrees(math.atan(k / nose_depth_ratio))


def estimate_roll_deg(keypoints: np.ndarray) -> float:
    d = keypoints[LEFT_EYE] - keypoints[RIGHT_EYE]
    return math.degrees(math.atan2(d[1], d[0]))
