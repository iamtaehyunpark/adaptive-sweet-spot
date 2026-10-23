import math

import numpy as np
import pytest

from vision.head_pose import DEFAULT_NOSE_DEPTH_RATIO, estimate_roll_deg, estimate_yaw_deg


def synthetic_keypoints(yaw_deg: float, roll_deg: float = 0.0, scale: float = 100.0) -> np.ndarray:
    """Project a simple head model (camera facing subject) to image space.

    Head frame: x toward image right, z toward camera. Turning the subject to
    their left moves the nose toward image right (positive yaw).
    """
    re, rn = 1.0, DEFAULT_NOSE_DEPTH_RATIO
    pts3 = np.array([
        [-0.45, -0.3, 0.8],   # right eye (image left)
        [0.45, -0.3, 0.8],    # left eye
        [0.0, 0.0, rn],       # nose
        [0.0, 0.5, 0.8],      # mouth
        [-re, 0.0, 0.0],      # right ear
        [re, 0.0, 0.0],       # left ear
    ])
    y = math.radians(yaw_deg)
    rot_yaw = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    r = math.radians(roll_deg)
    rot_roll = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    xy = (pts3 @ rot_yaw.T)[:, :2] @ rot_roll.T  # orthographic
    return xy * scale + np.array([640.0, 360.0])


@pytest.mark.parametrize("yaw", [-40, -15, 0, 10, 30])
def test_yaw_recovers_model_angle(yaw):
    assert estimate_yaw_deg(synthetic_keypoints(yaw)) == pytest.approx(yaw, abs=0.5)


@pytest.mark.parametrize("roll", [-20, 0, 15])
def test_roll_from_eye_line(roll):
    assert estimate_roll_deg(synthetic_keypoints(0, roll)) == pytest.approx(roll, abs=0.5)


def test_yaw_sign_nose_toward_image_right_is_positive():
    kps = synthetic_keypoints(0)
    kps[2, 0] += 20
    assert estimate_yaw_deg(kps) > 0
