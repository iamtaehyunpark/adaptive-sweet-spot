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

    ``tolerance`` is relative to each point's distance (depth error grows with range),
    so it also works at any (uncorrected) scale.
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
    thresh = tolerance * np.linalg.norm(sample, axis=1)
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

    # Refine with least squares, tightening the band each pass so points where the floor
    # meets walls or furniture (inside the loose RANSAC band) don't tilt the plane.
    n, d = best_n, -float(best_n @ best_p)
    for shrink in (1.0, 0.5, 0.25):
        near = sample[np.abs(sample @ n + d) < shrink * thresh]
        if len(near) < 3:
            break
        centroid = near.mean(axis=0)
        n = np.linalg.svd(near - centroid)[2][-1]
        d = -float(n @ centroid)
    if d < 0:  # orient the normal toward the camera (origin above the floor)
        n, d = -n, -d

    full = np.full((h, w), False)
    pv = points[valid]
    full[valid] = np.abs(pv @ n + d) < tolerance * np.linalg.norm(pv, axis=1)
    return FloorPlane(up=n, height_m=d, inlier_frac=float(full[valid].mean()), inliers=full)


# ---------------------------------------------------------------------------
# Room frame and top-down map (Milestone 3.2b)
#
# Room frame: origin at the seat (horizontal position of the listener's face in
# the setup capture), +x = listener's right, +y = toward the front (the camera,
# which sits on the FC speaker), +z = up, relative to the listener's face height.
# The camera faces the listener, so room +x is camera-left and room +y is
# camera-backward. With a visible floor the frame is leveled using its normal;
# otherwise the camera is assumed level.
# ---------------------------------------------------------------------------

FLOOR_BAND_M = 0.05        # within this of the floor plane = floor
OBSTACLE_BAND_M = (0.10, 2.20)  # heights above the floor drawn as obstacles


def level_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera-frame unit vectors (right, forward, up) of a level frame at the camera."""
    up = up / np.linalg.norm(up)
    z = np.array([0.0, 0.0, 1.0])
    forward = z - (z @ up) * up
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    return right, forward, up


@dataclass
class RoomModel:
    points: np.ndarray            # (N, 3) room-frame points (subsampled)
    height_above_floor: np.ndarray | None  # (N,) per point, None without a floor
    camera_xy: np.ndarray         # camera (= FC speaker) position, room frame
    center_distance_m: float      # horizontal camera -> seat distance
    face_depth_m: float           # seat face distance along the camera axis (tracking uses this)
    face_height_m: float | None   # listener face height above the floor
    camera_height_m: float | None
    pitch_deg: float | None
    roll_deg: float | None
    scale: float                  # applied to the model output
    scale_source: str

    @property
    def has_floor(self) -> bool:
        return self.height_above_floor is not None


def face_point(points: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Median camera-frame point over the inner half of the face box."""
    x, y, w, h = bbox
    x0, x1 = int(x + w / 4), int(x + 3 * w / 4)
    y0, y1 = int(y + h / 4), int(y + 3 * h / 4)
    patch = points[max(y0, 0):y1, max(x0, 0):x1].reshape(-1, 3)
    patch = patch[np.isfinite(patch).all(axis=1)]
    if len(patch) == 0:
        raise ValueError("no depth at the face")
    return np.median(patch, axis=0)


def build_room(points: np.ndarray, face_bbox: np.ndarray, seat_distance_m: float | None = None,
               camera_height_m: float | None = None, stride: int = 2) -> RoomModel:
    """Room model from a camera-frame metric point map and the listener's face box.

    Optional tape measurements fix the model's scale: ``seat_distance_m`` (camera
    lens -> face, straight line) or ``camera_height_m`` (lens -> floor).
    """
    floor = fit_floor(points)
    face = face_point(points, face_bbox)
    scale, source = 1.0, "model"
    if seat_distance_m:
        scale, source = seat_distance_m / float(np.linalg.norm(face)), "seat distance tape"
    elif camera_height_m and floor is not None:
        scale, source = camera_height_m / floor.height_m, "camera height tape"

    pts = points[::stride, ::stride].reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)] * scale
    face = face * scale
    up = floor.up if floor is not None else np.array([0.0, -1.0, 0.0])
    right, forward, upv = level_basis(up)

    def to_room(p):
        # level coords at the camera, then flip to the listener-facing room frame
        r, f, u = p @ right, p @ forward, p @ upv
        fr, ff, fu = face @ right, face @ forward, face @ upv
        return np.stack([-(r - fr), -(f - ff), u - fu], axis=-1)

    room_pts = to_room(pts)
    camera_xy = to_room(np.zeros(3))[:2]
    height = None
    face_height = cam_height = pitch = roll = None
    if floor is not None:
        cam_height = floor.height_m * scale
        height = room_pts[:, 2] + (face @ upv) + cam_height  # height above the floor
        face_height = float(face @ upv + cam_height)
        pitch, roll = floor.pitch_deg, floor.roll_deg
    return RoomModel(
        points=room_pts, height_above_floor=height, camera_xy=camera_xy,
        center_distance_m=float(np.hypot(*camera_xy)), face_depth_m=float(face[2]),
        face_height_m=face_height, camera_height_m=cam_height, pitch_deg=pitch, roll_deg=roll,
        scale=scale, scale_source=source,
    )


def render_map(room: RoomModel, m_per_px: float = 0.02, half_extent_m: float = 5.0):
    """Top-down BGR map image (front = up) and metadata to convert room <-> pixels."""
    x, y = room.points[:, 0], room.points[:, 1]
    keep = (np.abs(x) < half_extent_m) & (np.abs(y) < half_extent_m)
    xmin, xmax = min(x[keep].min(), room.camera_xy[0], 0) - 0.3, max(x[keep].max(), room.camera_xy[0], 0) + 0.3
    ymin, ymax = min(y[keep].min(), 0) - 0.3, max(y[keep].max(), room.camera_xy[1]) + 0.3
    w = int((xmax - xmin) / m_per_px) + 1
    h = int((ymax - ymin) / m_per_px) + 1
    origin = (-xmin / m_per_px, ymax / m_per_px)  # pixel of room (0, 0)

    def px(xs, ys):
        return ((xs - xmin) / m_per_px).astype(int), ((ymax - ys) / m_per_px).astype(int)

    img = np.full((h, w, 3), 255, np.uint8)
    if room.has_floor:
        hgt = room.height_above_floor[keep]
        floor = np.abs(hgt) < FLOOR_BAND_M
        obstacle = (hgt > OBSTACLE_BAND_M[0]) & (hgt < OBSTACLE_BAND_M[1])
    else:
        z = room.points[keep, 2]
        floor = np.zeros(keep.sum(), bool)
        obstacle = (z > -1.5) & (z < 1.0)
    for mask, color in ((floor, (225, 215, 200)), (obstacle, (70, 70, 70))):
        u, v = px(x[keep][mask], y[keep][mask])
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        img[v[ok], u[ok]] = color
    meta = {"m_per_px": m_per_px, "origin_px": [float(origin[0]), float(origin[1])], "size_px": [w, h]}
    return img, meta
