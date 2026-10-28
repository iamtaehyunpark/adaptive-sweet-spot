"""5.1 speaker placement suggestion on the 3D room scan (Milestone 3.2b).

Guidelines (references in the README):

* ITU-R BS.775: all speakers on an arc around the listener (equal distance),
  C 0 deg, L/R +-30 deg, surrounds 100-120 deg (110 reference).
* Dolby-style home guidance: L/R 22-30 deg, surrounds 90-110 deg, surrounds
  1-3 ft above ear height (ITU: ear height); tweeters at seated ear height.
* Keep mains ~2-4 ft off the front wall and ~2-3 ft off side walls; corners
  build up bass. Aim (toe-in) speakers at the listener.
* Subwoofer: front of the room blends best and is least localizable; corners
  give more output but boomier bass; confirm with a "subwoofer crawl".
* Left/right pairs mirrored around the listener.

The scan is one view from the camera on FC: directions it did not see (mostly
beside the camera) cannot be checked and are reported as "not verified".
FC itself is fixed: the camera sits on it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

SPEED_OF_SOUND = 343.0
BIN_DEG = 2.0
SELF_RADIUS_M = 0.5          # ignore the listener's own body / seat in the scan
WALL_MIN_HEIGHT_M = 1.4      # surfaces taller than this above the floor count as walls
MIN_BIN_POINTS = 12          # points needed to call a direction "observed"
MAIN_CLEARANCE_M = 0.6       # preferred distance of mains from walls (~2 ft)
SURROUND_CLEARANCE_M = 0.15  # surrounds may be wall-mounted
MIN_CLEARANCE_M = 0.12       # hard limit: speaker cabinet + gap
LOS_RADIUS_M = 0.12          # line-of-sight corridor radius
SURROUND_HEIGHT_M = 0.3      # above ear height: between ITU (0) and Dolby (0.3-0.9)

# name: (ideal azimuth deg, allowed |azimuth| range, height above ear, preferred wall clearance)
GUIDE = {
    "FL": (-30.0, (22.0, 30.0), 0.0, MAIN_CLEARANCE_M),
    "FR": (30.0, (22.0, 30.0), 0.0, MAIN_CLEARANCE_M),
    "SL": (-110.0, (90.0, 120.0), SURROUND_HEIGHT_M, SURROUND_CLEARANCE_M),
    "SR": (110.0, (90.0, 120.0), SURROUND_HEIGHT_M, SURROUND_CLEARANCE_M),
}
PAIRS = (("FL", "FR"), ("SL", "SR"))


@dataclass
class SpeakerSuggestion:
    name: str
    channel: int
    lfe: bool
    position: list[float]          # room frame (m)
    azimuth_deg: float
    distance_m: float
    aim_yaw_deg: float             # direction to point the speaker (room azimuth, 0 = front)
    aim_pitch_deg: float           # + = tilt up
    verified: bool                 # location covered by the scan
    score: float
    notes: list[str] = field(default_factory=list)


class RoomGeometry:
    """Walls and obstacles around the seat from the scan, in polar bins.

    A *wall* is a tall surface (more than WALL_MIN_HEIGHT_M above the floor): a speaker
    behind it is impossible. Lower points (sofa, table, the listener's own chair) are
    obstacles: a speaker may stand beyond them, but not inside them or with them in
    the line of sight (checked separately). ``extent`` is how far the scan saw in each
    direction; positions beyond it and without a wall are unverified.
    """

    def __init__(self, points: np.ndarray, floor_z: float | None):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        if floor_z is not None:
            band = (z > floor_z + 0.15) & (z < floor_z + 2.2)
            tall = z > floor_z + WALL_MIN_HEIGHT_M
        else:  # no floor: z is relative to the face; walls reach well above it
            band = (z > -1.0) & (z < 1.0)
            tall = z > 0.3
        r = np.hypot(x, y)
        keep = band & (r > SELF_RADIUS_M)
        self.obstacles = points[keep]
        n_bins = int(360 / BIN_DEG)
        idx_all = ((np.degrees(np.arctan2(x, y)) + 180) / BIN_DEG).astype(int) % n_bins
        self.wall = np.full(n_bins, np.inf)
        self.extent = np.zeros(n_bins)
        self.count = np.bincount(idx_all[keep], minlength=n_bins)
        wall_sel = tall & (r > SELF_RADIUS_M)
        for b in range(n_bins):
            in_bin = keep & (idx_all == b)
            if in_bin.sum() >= MIN_BIN_POINTS:
                self.extent[b] = np.percentile(r[in_bin], 95)
            w = wall_sel & (idx_all == b)
            if w.sum() >= MIN_BIN_POINTS:
                self.wall[b] = np.percentile(r[w], 10)  # near face of the wall

    def _bin(self, az_deg: float) -> int:
        return int(((az_deg + 180) / BIN_DEG)) % len(self.wall)

    def wall_distance(self, az_deg: float) -> float | None:
        """Nearest wall in this direction (min over +-1 bin), None if no wall was seen."""
        b = self._bin(az_deg)
        d = min(self.wall[(b + k) % len(self.wall)] for k in (-1, 0, 1))
        return None if math.isinf(d) else float(d)

    def seen_to(self, az_deg: float) -> float:
        """How far the scan saw in this direction (0 = not at all)."""
        b = self._bin(az_deg)
        return float(max(self.extent[(b + k) % len(self.extent)] for k in (-1, 0, 1)))

    def collides(self, p: np.ndarray, radius: float = 0.2) -> bool:
        if len(self.obstacles) == 0:
            return False
        d = np.hypot(self.obstacles[:, 0] - p[0], self.obstacles[:, 1] - p[1])
        return int(((d < radius) & (np.abs(self.obstacles[:, 2] - p[2]) < 0.4)).sum()) > 10

    def occluded(self, p: np.ndarray) -> bool:
        """Anything in the corridor between the speaker and the listener's head?"""
        if len(self.obstacles) == 0:
            return False
        v = -p  # head at the origin
        length2 = float(v @ v)
        t = ((self.obstacles - p) @ v) / length2
        inside = (t > 0.1) & (t < 0.85)
        closest = p + np.outer(t[inside], v)
        dist = np.linalg.norm(self.obstacles[inside] - closest, axis=1)
        return int((dist < LOS_RADIUS_M).sum()) > 8

    @property
    def observed_fraction(self) -> float:
        return float((self.count >= MIN_BIN_POINTS).mean())


def _aim(p: np.ndarray) -> tuple[float, float]:
    yaw = math.degrees(math.atan2(-p[0], -p[1]))
    pitch = math.degrees(math.atan2(-p[2], math.hypot(p[0], p[1])))
    return yaw, pitch


def _evaluate(geo: RoomGeometry, name: str, az: float, r: float, radius: float):
    """(score, position, notes, verified) for one candidate, or None if infeasible."""
    ideal, (lo, hi), z, pref_clear = GUIDE[name]
    a = math.radians(az)
    p = np.array([r * math.sin(a), r * math.cos(a), z])
    wall = geo.wall_distance(az)
    verified = bool(wall is not None or r <= geo.seen_to(az))
    notes = []
    score = ((abs(az) - abs(ideal)) / 8.0) ** 2 + ((r - radius) / (0.25 * radius)) ** 2
    if wall is not None:
        clearance = wall - r
        if clearance < MIN_CLEARANCE_M:
            return None  # behind or in the wall
        if clearance < pref_clear:
            score += 2.0 * ((pref_clear - clearance) / pref_clear) ** 2
            notes.append(f"{clearance:.2f} m from the wall behind it (prefer >= {pref_clear:.1f} m)")
    if verified and geo.collides(p):
        return None  # inside furniture
    if not verified:
        score += 0.5
        notes.append("not visible in the scan - check this spot is free")
    if geo.occluded(p):
        score += 5.0
        notes.append("something in the scan blocks the line of sight to your head")
    return score, p, notes, verified


def _radius_grid(radius: float) -> np.ndarray:
    return np.round(np.arange(0.6 * radius, 1.25 * radius + 1e-9, 0.05), 3)


def suggest_layout(room: dict, points: np.ndarray, cfg_speakers: list[dict]) -> dict:
    """Suggested 5.1 positions on the scan, plus a centering analysis."""
    geo = RoomGeometry(points, room.get("floor_z"))
    cam = np.array(room["camera_xy"], float)
    radius = float(np.hypot(*cam))  # FC (camera) sets the arc radius
    fc_z = 0.0
    if room.get("floor_z") is not None and room.get("camera_height_m") is not None:
        fc_z = room["floor_z"] + room["camera_height_m"]
    channel = {s["name"]: s for s in cfg_speakers}
    out: dict[str, SpeakerSuggestion] = {}

    for left, right in PAIRS:
        lo, hi = GUIDE[right][1]
        best = None
        for a in np.arange(lo, hi + 1e-9, 1.0):
            cand_l = [c for r in _radius_grid(radius) if (c := _evaluate(geo, left, -a, r, radius))]
            cand_r = [c for r in _radius_grid(radius) if (c := _evaluate(geo, right, a, r, radius))]
            for cl in cand_l:
                for cr in cand_r:
                    rl, rr = np.hypot(*cl[1][:2]), np.hypot(*cr[1][:2])
                    total = cl[0] + cr[0] + 4.0 * abs(rl - rr) / radius  # keep pairs symmetric
                    if best is None or total < best[0]:
                        best = (total, a, cl, cr)
        if best is None:
            continue
        _, a, cl, cr = best
        for name, (score, p, notes, verified), az in ((left, cl, -a), (right, cr, a)):
            notes = list(notes)
            ideal = GUIDE[name][0]
            if abs(abs(az) - abs(ideal)) >= 3:
                notes.insert(0, f"{abs(az):.0f} deg instead of the ideal {abs(ideal):.0f} deg (room constraint)")
            d = float(np.linalg.norm(p))
            if abs(np.hypot(*p[:2]) - radius) > 0.05:
                notes.append(f"{np.hypot(*p[:2]):.2f} m from the seat vs {radius:.2f} m for FC - "
                             "the DSP compensates the delay/level")
            yaw, pitch = _aim(p)
            out[name] = SpeakerSuggestion(name, channel[name]["channel"], False, [round(float(v), 3) for v in p],
                                          round(float(az), 1), round(d, 3), round(yaw, 1), round(pitch, 1),
                                          verified, round(float(score), 3), notes)

    # FC is fixed at the camera.
    p_fc = np.array([cam[0], cam[1], fc_z])
    yaw, pitch = _aim(p_fc)
    fc_notes = ["fixed: the camera sits on FC"]
    if abs(pitch) > 10:
        fc_notes.append(f"{abs(pitch):.0f} deg {'below' if fc_z < 0 else 'above'} ear height - tilt it toward the seat "
                        "and keep its tweeter close to the L/R tweeter height")
    out["FC"] = SpeakerSuggestion("FC", channel["FC"]["channel"], False, [round(float(v), 3) for v in p_fc],
                                  round(math.degrees(math.atan2(cam[0], cam[1])), 1),
                                  round(float(np.linalg.norm(p_fc)), 3), round(yaw, 1), round(pitch, 1),
                                  True, 0.0, fc_notes)

    # LFE: front of the room, off-center, off the walls, on the floor.
    if "LFE" in channel:
        out["LFE"] = _suggest_lfe(geo, room, cam, channel["LFE"]["channel"])

    ordered = [asdict(out[s["name"]]) for s in cfg_speakers if s["name"] in out]
    return {"speakers": ordered, "analysis": centering_analysis(ordered), "room": room_notes(geo),
            "arc_radius_m": round(radius, 3), "scan_coverage": round(geo.observed_fraction, 2)}


def _suggest_lfe(geo: RoomGeometry, room: dict, cam: np.ndarray, channel: int) -> SpeakerSuggestion:
    floor_z = room.get("floor_z") if room.get("floor_z") is not None else -1.0
    left, right = geo.wall_distance(-90), geo.wall_distance(90)
    target_dx = 0.25 * (left + right) if left and right else 0.8  # ~quarter of the room width
    best = None
    for dx in np.arange(0.4, 2.0, 0.1):
        for side in (-1, 1):
            p = np.array([cam[0] + side * dx, cam[1] - 0.2, floor_z + 0.2])
            az = math.degrees(math.atan2(p[0], p[1]))
            wall = geo.wall_distance(az)
            if wall is not None and wall - np.hypot(*p[:2]) < 0.1:
                continue
            side_wall = right if side > 0 else left
            if side_wall is not None and side_wall - abs(p[0]) < 0.3:
                continue  # avoid the corner
            score = ((dx - target_dx) / 0.5) ** 2 + (0.3 if wall is None else 0.0)
            if best is None or score < best[0]:
                best = (score, p, wall is not None)
    if best is None:
        best = (1.0, np.array([cam[0] + 0.6, cam[1] - 0.2, floor_z + 0.2]), False)
    score, p, verified = best
    yaw, pitch = _aim(p)
    notes = ["front of the room blends best with the mains; off-center and off the walls avoids boomy bass",
             "fine-tune with a subwoofer crawl: put it at the seat, crawl along the front, place it where the "
             "bass sounds smoothest"]
    if not verified:
        notes.append("not visible in the scan - check this spot is free")
    return SpeakerSuggestion("LFE", channel, True, [round(float(v), 3) for v in p],
                             round(math.degrees(math.atan2(p[0], p[1])), 1), round(float(np.linalg.norm(p)), 3),
                             round(yaw, 1), round(pitch, 1), bool(verified), round(float(score), 3), notes)


def centering_analysis(speakers: list[dict]) -> dict:
    """Delay/level differences at the seat (0 = perfectly centered without DSP correction)."""
    mains = [s for s in speakers if not s["lfe"]]
    d = np.array([np.linalg.norm(s["position"]) for s in mains])
    far = d.max()
    return {
        "distance_spread_m": round(float(d.max() - d.min()), 3),
        "max_delay_ms": round(float((far - d.min()) / SPEED_OF_SOUND * 1000), 3),
        "per_speaker": {s["name"]: {"distance_m": round(float(di), 3),
                                    "delay_ms": round(float((far - di) / SPEED_OF_SOUND * 1000), 3),
                                    "level_db": round(float(20 * math.log10(di / far)), 2)}
                        for s, di in zip(mains, d)},
    }


def room_notes(geo: RoomGeometry) -> list[str]:
    """General sweet-spot remarks about the seat in this room."""
    notes = []
    left, right = geo.wall_distance(-90), geo.wall_distance(90)
    if left is not None and right is not None:
        off = (right - left) / 2
        if abs(off) > 0.25:
            side = "left" if off < 0 else "right"
            notes.append(f"the seat is {abs(off):.2f} m off the middle between the side walls (closer to the {side}); "
                         "a centered seat makes left/right reflections symmetric")
    back = geo.wall_distance(180)
    if back is not None and back < 0.6:
        notes.append(f"the seat is {back:.2f} m from the wall behind it; sitting farther from it reduces "
                     "bass build-up and early reflections")
    if geo.observed_fraction < 0.4:
        notes.append("the scan covers little of the room; suggestions in unseen directions are unverified")
    return notes
