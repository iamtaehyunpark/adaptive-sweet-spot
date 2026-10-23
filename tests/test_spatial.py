import math

import numpy as np
import pytest

from vision.detector import FaceObservation
from vision.spatial import CenterMapper

W, H = 1280, 720
D = 2.0  # FC speaker -> center


def face_at(x_m, dist_m, hfov_deg=70.0, ipd_m=0.063):
    """Face of a listener at room offset x (m) and distance (m) from the camera."""
    f = (W / 2) / math.tan(math.radians(hfov_deg) / 2)
    u = W / 2 - x_m * f / dist_m           # room +x appears toward image left
    half = 0.5 * f * ipd_m / dist_m
    kps = np.zeros((6, 2))
    kps[0] = [u - half, H / 2]   # right eye
    kps[1] = [u + half, H / 2]   # left eye
    kps[2] = [u, H / 2 + 10]     # nose (frontal)
    kps[4] = [u - 2.5 * half, H / 2]
    kps[5] = [u + 2.5 * half, H / 2]
    return FaceObservation(bbox=np.zeros(4), keypoints=kps, score=1.0, frame_size=(W, H))


def test_listener_at_center_maps_to_zero():
    off = CenterMapper(W, H, 70.0, D).map(face_at(0.0, D))
    assert off.x == pytest.approx(0, abs=1e-6)
    assert off.y == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("x, dist", [(0.3, 2.0), (-0.5, 1.5), (0.2, 2.6)])
def test_offset_recovered(x, dist):
    off = CenterMapper(W, H, 70.0, D).map(face_at(x, dist))
    assert off.x == pytest.approx(x, abs=1e-3)
    assert off.y == pytest.approx(D - dist, abs=1e-3)  # + = toward the front


def test_calibration_fixes_wrong_fov_and_camera_offset():
    # Real camera is 60 deg and the "center" seat is 5 cm right of the camera
    # axis; the mapper assumes 70 deg and on-axis.
    def real(x, dist):
        return face_at(x + 0.05, dist, hfov_deg=60.0)

    m = CenterMapper(W, H, 70.0, D)
    assert abs(m.map(real(0, D)).y) > 0.1  # uncalibrated distance is clearly wrong

    m.calibrate([real(0, D)] * 5)
    at_center = m.map(real(0, D))
    assert at_center.x == pytest.approx(0, abs=1e-3)
    assert at_center.y == pytest.approx(0, abs=1e-3)

    moved = m.map(real(0.4, 1.6))
    assert moved.x == pytest.approx(0.4, abs=1e-3)  # lateral is FOV-independent
    assert moved.y == pytest.approx(0.4, abs=1e-3)


def test_wrong_ipd_only_scales_lateral_offset():
    # Real IPD 58 mm vs assumed 63 mm: distance is fixed by calibration,
    # lateral offset is off by 63/58 (~8.6%).
    def real(x, dist):
        return face_at(x, dist, ipd_m=0.058)

    m = CenterMapper(W, H, 70.0, D)
    m.calibrate([real(0, D)] * 5)
    moved = m.map(real(0.4, 1.6))
    assert moved.y == pytest.approx(0.4, abs=1e-3)
    assert moved.x == pytest.approx(0.4 * 63 / 58, abs=1e-3)
