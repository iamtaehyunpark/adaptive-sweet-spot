import numpy as np
import pytest

from vision.detector import FaceObservation
from vision.rate import AdaptiveRate


def face(x, ipd_px=63.0):
    """Face whose eyes are ipd_px apart, so 1 px == 1 mm when ipd_px == 63."""
    kps = np.zeros((6, 2))
    kps[0] = [x - ipd_px / 2, 0]
    kps[1] = [x + ipd_px / 2, 0]
    return FaceObservation(bbox=np.zeros(4), keypoints=kps, score=1.0, frame_size=(1280, 720))


def run(rate, speed_mm_s, seconds=2.0, cam_fps=30.0):
    """Feed a head moving at a constant speed; return the fraction of frames processed."""
    processed = 0
    n = int(seconds * cam_fps)
    for i in range(n):
        ts = i / cam_fps
        if rate.should_process(ts):
            rate.update(face(speed_mm_s * ts), ts)
            processed += 1
    return processed / n


def test_still_head_runs_at_min_fps():
    r = AdaptiveRate(tolerance_mm=30, min_fps=5, max_fps=30)
    frac = run(r, speed_mm_s=0)
    assert r.target_fps == 5
    assert frac == pytest.approx(5 / 30, abs=0.05)


def test_fps_follows_speed():
    r = AdaptiveRate(tolerance_mm=30, min_fps=5, max_fps=30)
    run(r, speed_mm_s=300)  # 300 / 15 = 20 fps
    assert r.target_fps == pytest.approx(20, rel=0.15)


def test_fast_head_uses_every_frame():
    r = AdaptiveRate(tolerance_mm=30, min_fps=5, max_fps=30)
    assert run(r, speed_mm_s=1000) > 0.9


def test_lost_face_goes_full_rate():
    r = AdaptiveRate(tolerance_mm=30, min_fps=5, max_fps=30)
    run(r, speed_mm_s=0)
    r.update(None, 10.0)
    assert r.target_fps == 30 and r.should_process(10.0 + 1 / 30)


def test_motion_in_face_region_triggers_detection():
    r = AdaptiveRate(tolerance_mm=30, min_fps=5, max_fps=30)
    still = np.full((720, 1280, 3), 100, np.uint8)
    region = (500, 200, 700, 400)
    r.update(face(600), 0.0, still, region)
    assert not r.should_process(1 / 30, still)      # nothing changed, below min interval
    moved = still.copy()
    moved[200:400, 500:600] = 200                    # half the region changes
    assert r.should_process(1 / 30, moved)
