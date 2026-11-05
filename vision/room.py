"""Room geometry from a metric point map (camera frame: x right, y down, z forward)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


MIN_FACE_ABOVE_FLOOR_M = 0.5  # a seated/standing face is ~1.1-1.7 m above the floor
MAX_CAMERA_TILT_DEG = 40.0    # the camera sits on the FC speaker facing the listener: near level


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


def fit_floor(points: np.ndarray, max_tilt_deg: float = MAX_CAMERA_TILT_DEG, iterations: int = 500,
              tolerance: float = 0.015, seed: int = 0, exclude: np.ndarray | None = None,
              above: np.ndarray | None = None, min_below_m: float = MIN_FACE_ABOVE_FLOOR_M) -> FloorPlane | None:
    """RANSAC plane fit to the floor, searched in the lower half of the image.

    ``tolerance`` is relative to each point's distance (depth error grows with range),
    so it also works at any (uncorrected) scale.
    Planes whose normal is more than ``max_tilt_deg`` away from the camera's down axis are
    rejected (walls, screens, a reclined torso, planes slicing across walls). ``exclude`` masks pixels that are not floor (the listener's
    body: a big, often reclined surface in the lower half). With ``above`` (the face point),
    only planes at least ``min_below_m`` below it count - not a lap, a desk or a chair seat.
    """
    h, w = points.shape[:2]
    valid = np.isfinite(points).all(axis=2)
    rows = np.arange(h)[:, None].repeat(w, axis=1)
    cand = valid & (rows > h // 2)
    if exclude is not None:
        cand &= ~exclude
    lower = points[cand]
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
        if above is not None and abs((above - a) @ n) < min_below_m:
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
    if above is not None and above @ n + d < min_below_m:
        return None  # refinement slid onto something that is not the floor

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
BODY_WIDTHS = 2.5          # listener's body: face center +- this many face widths, face down
EDGE_REL = 0.03            # depth jump to a neighbour (relative) that marks a flying pixel
MAX_RANGE_M = 8.0          # drop points farther than this from the camera
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
    colors: np.ndarray | None = None  # (N, 3) uint8 RGB per point
    floor_z: float | None = None      # floor height in the room frame (z relative to the face)

    @property
    def has_floor(self) -> bool:
        return self.height_above_floor is not None


def flying_pixel_mask(points: np.ndarray, rel: float = EDGE_REL) -> np.ndarray:
    """True where a pixel's depth jumps by more than ``rel`` to any 4-neighbour.

    Depth models smear points across depth discontinuities (between a foreground
    object and the wall behind it); those points float in mid-air and must go.
    """
    d = np.linalg.norm(points, axis=2)
    jump = np.zeros(d.shape, bool)
    for axis in (0, 1):
        diff = np.abs(np.diff(d, axis=axis)) / np.minimum(
            np.take(d, range(d.shape[axis] - 1), axis=axis), np.take(d, range(1, d.shape[axis]), axis=axis))
        big = ~(diff <= rel)  # NaN counts as a jump
        if axis == 0:
            jump[:-1] |= big
            jump[1:] |= big
        else:
            jump[:, :-1] |= big
            jump[:, 1:] |= big
    return jump


def body_mask(shape: tuple[int, int], bbox: np.ndarray) -> np.ndarray:
    """Pixels of the listener's body: a column under the face down to the image bottom."""
    x, y, w, _ = bbox
    cx = x + w / 2
    mask = np.zeros(shape, bool)
    x0, x1 = int(max(cx - BODY_WIDTHS * w, 0)), int(min(cx + BODY_WIDTHS * w, shape[1]))
    mask[int(max(y, 0)):, x0:x1] = True
    return mask


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
               camera_height_m: float | None = None, stride: int = 2,
               colors: np.ndarray | None = None, face: np.ndarray | None = None,
               scale: tuple[float, str] | None = None) -> RoomModel:
    """Room model from a camera-frame metric point map and the listener's face box.

    Optional tape measurements fix the model's scale: ``seat_distance_m`` (camera
    lens -> face, straight line) or ``camera_height_m`` (lens -> floor). ``colors``
    is the (H, W, 3) RGB image the points came from. Flying pixels and points
    beyond MAX_RANGE_M are removed.

    Room scan (empty room, see align_scan): pass the calibration's camera-frame
    ``face`` point and its ``scale`` (factor, source); ``points`` then hold no
    listener, so no body is masked out of the floor search.
    """
    empty_room = face is not None
    if face is None:
        face = face_point(points, face_bbox)
    floor = fit_floor(points, exclude=None if empty_room else body_mask(points.shape[:2], face_bbox), above=face)
    scale, source = scale or (1.0, "model")
    if seat_distance_m:
        scale, source = seat_distance_m / float(np.linalg.norm(face)), "seat distance tape"
    elif camera_height_m and floor is not None:
        scale, source = camera_height_m / floor.height_m, "camera height tape"

    keep = np.isfinite(points).all(axis=2) & ~flying_pixel_mask(points)
    keep &= np.linalg.norm(np.nan_to_num(points, nan=np.inf), axis=2) < MAX_RANGE_M / scale
    keep = keep[::stride, ::stride].reshape(-1)
    pts = points[::stride, ::stride].reshape(-1, 3)[keep] * scale
    cols = colors[::stride, ::stride].reshape(-1, 3)[keep] if colors is not None else None
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
    face_height = cam_height = pitch = roll = floor_z = None
    if floor is not None:
        cam_height = floor.height_m * scale
        height = room_pts[:, 2] + (face @ upv) + cam_height  # height above the floor
        face_height = float(face @ upv + cam_height)
        floor_z = -face_height
        pitch, roll = floor.pitch_deg, floor.roll_deg
    return RoomModel(
        points=room_pts, height_above_floor=height, camera_xy=camera_xy,
        center_distance_m=float(np.hypot(*camera_xy)), face_depth_m=float(face[2]),
        face_height_m=face_height, camera_height_m=cam_height, pitch_deg=pitch, roll_deg=roll,
        scale=scale, scale_source=source, colors=cols, floor_z=floor_z,
    )


# ---------------------------------------------------------------------------
# Room scan: a second capture of the empty room from the same (fixed) camera.
# The calibration capture has the listener in it, hiding the floor and whatever
# is behind them; the room scan sees all of it. Its depth is brought to the
# calibration's scale and intrinsics so it lands in the same room frame.
# ---------------------------------------------------------------------------

SCAN_AGREE_REL = 0.05     # static pixels within this (relative) of the calibration depth agree
SCAN_MIN_AGREE = 0.5      # below this fraction of agreeing pixels the camera moved


def points_from_depth(depth: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """(H, W, 3) camera-frame points from z-depth, principal point at the image center."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    return np.stack([(u - w / 2) * depth / fx, (v - h / 2) * depth / fy, depth], axis=-1)


def align_scan(ref_depth: np.ndarray, scan_depth: np.ndarray, exclude: np.ndarray | None = None
               ) -> tuple[float, float]:
    """Depth ratio that brings the room scan to the calibration capture, and agreement.

    Both are z-depth maps from the same camera pose. ``exclude`` masks the listener in
    the calibration capture (it is not in the scan). Returns (ratio, fraction of the
    compared pixels within SCAN_AGREE_REL after scaling); a low fraction means the
    camera (or its view) moved between the captures.
    """
    if ref_depth.shape != scan_depth.shape:
        raise ValueError(f"capture sizes differ: {ref_depth.shape[::-1]} vs {scan_depth.shape[::-1]}")
    ok = np.isfinite(ref_depth) & np.isfinite(scan_depth) & (ref_depth > 0) & (scan_depth > 0)
    if exclude is not None:
        ok &= ~exclude
    if ok.sum() < 100:
        raise ValueError("no depth to compare between the captures")
    ratio = ref_depth[ok] / scan_depth[ok]
    k = float(np.median(ratio))
    agree = float((np.abs(ratio / k - 1) < SCAN_AGREE_REL).mean())
    return k, agree


def render_map(room: RoomModel, m_per_px: float = 0.02, half_extent_m: float = 5.0):
    """Top-down BGR map image (front = up) and metadata to convert room <-> pixels."""
    x, y = room.points[:, 0], room.points[:, 1]
    keep = (np.abs(x) < half_extent_m) & (np.abs(y) < half_extent_m)
    # bounds always include the seat (0, 0) and the camera, even with no scan points nearby
    xs = np.concatenate([x[keep], [0.0, room.camera_xy[0]]])
    ys = np.concatenate([y[keep], [0.0, room.camera_xy[1]]])
    xmin, xmax = xs.min() - 0.3, xs.max() + 0.3
    ymin, ymax = ys.min() - 0.3, ys.max() + 0.3
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
