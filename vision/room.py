"""Room geometry from a metric point map (camera frame: x right, y down, z forward)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class FloorPlane:
    up: np.ndarray        # unit normal pointing from the floor toward the camera (camera frame)
    height_m: float       # camera height above the floor
    inlier_frac: float    # fraction of valid pixels on the floor
    inliers: np.ndarray   # (H, W) bool

    @property
    def pitch_deg(self) -> float:
        """Camera tilt: + = looking up, - = looking down."""
        return math.degrees(math.asin(float(np.clip(self.up[2], -1, 1))))

    @property
    def roll_deg(self) -> float:
        """+ = image right side lower."""
        return math.degrees(math.asin(float(np.clip(self.up[0], -1, 1))))


def fit_floor(points: np.ndarray, max_tilt_deg: float = 60.0, iterations: int = 500,
              tolerance: float = 0.015, seed: int = 0) -> FloorPlane | None:
    """RANSAC plane fit to the floor, searched in the lower half of the image.

    ``tolerance`` is relative to the median depth so it works at any (uncorrected) scale.
    Planes whose normal is more than ``max_tilt_deg`` away from the camera's down axis are
    rejected (walls, screens).
    """
    h, w = points.shape[:2]
    valid = np.isfinite(points).all(axis=2)
    rows = np.arange(h)[:, None].repeat(w, axis=1)
    lower = points[valid & (rows > h // 2)]
    if len(lower) < 100:
        return None
    sample = lower[np.random.default_rng(seed).choice(len(lower), min(len(lower), 20000), replace=False)]
    thresh = tolerance * float(np.median(sample[:, 2]))
    min_up = math.cos(math.radians(max_tilt_deg))
    rng = np.random.default_rng(seed)

    best_n, best_p, best_count = None, None, 0
    for _ in range(iterations):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if abs(n[1]) < min_up:  # camera y axis points down; floor normal must be close to it
            continue
        count = int((np.abs((sample - a) @ n) < thresh).sum())
        if count > best_count:
            best_n, best_p, best_count = n, a, count
    if best_n is None:
        return None

    # Refine with least squares on the inliers.
    near = sample[np.abs((sample - best_p) @ best_n) < thresh]
    centroid = near.mean(axis=0)
    n = np.linalg.svd(near - centroid)[2][-1]
    d = -float(n @ centroid)
    if d < 0:  # orient the normal toward the camera (origin above the floor)
        n, d = -n, -d

    full = np.full((h, w), False)
    full[valid] = np.abs(points[valid] @ n + d) < thresh
    return FloorPlane(up=n, height_m=d, inlier_frac=float(full[valid].mean()), inliers=full)
