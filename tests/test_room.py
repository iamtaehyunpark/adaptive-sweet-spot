import math

import numpy as np
import pytest

from vision.room import fit_floor


def synthetic_room(height=0.8, pitch_deg=-20.0, w=320, h=240, f=250.0):
    """Camera ``height`` above a floor, tilted by ``pitch_deg``; a wall 4 m ahead."""
    p = math.radians(pitch_deg)
    # Camera axes in world coords (world: x right, y forward, z up).
    fwd = np.array([0, math.cos(p), math.sin(p)])
    down = np.array([0, math.sin(p), -math.cos(p)])
    right = np.array([1.0, 0, 0])
    u, v = np.meshgrid(np.arange(w) - w / 2, np.arange(h) - h / 2)
    rays = u[..., None] / f * right + v[..., None] / f * down + fwd  # world directions
    cam = np.array([0, 0, height])
    t_floor = np.where(rays[..., 2] < -1e-6, -height / np.minimum(rays[..., 2], -1e-6), np.inf)
    t_wall = np.where(rays[..., 1] > 1e-6, 4.0 / np.maximum(rays[..., 1], 1e-6), np.inf)
    t = np.minimum(t_floor, t_wall)
    world = cam + rays * t[..., None]
    rel = world - cam  # back to camera coords
    return np.stack([rel @ right, rel @ down, rel @ fwd], axis=-1)


@pytest.mark.parametrize("height, pitch", [(0.8, -20.0), (1.2, -35.0), (0.5, -10.0)])
def test_floor_height_and_tilt(height, pitch):
    floor = fit_floor(synthetic_room(height, pitch))
    assert floor is not None
    assert floor.height_m == pytest.approx(height, rel=0.01)
    assert floor.pitch_deg == pytest.approx(pitch, abs=0.5)
    assert floor.roll_deg == pytest.approx(0.0, abs=0.5)


def test_no_floor_in_view():
    assert fit_floor(synthetic_room(0.8, pitch_deg=25.0)) is None or \
        fit_floor(synthetic_room(0.8, pitch_deg=25.0)).inlier_frac < 0.02
