"""Speaker -> listener geometry: per-channel delay, level and off-axis correction.

Time alignment: every channel is delayed so that sound from all speakers reaches the
listener at the same moment. The arrival time is held constant (a fixed reference
distance ``ref_m``, the farthest any speaker can be from a listener within
``max_offset_m`` of the seat), so a moving listener hears no pitch change at all -
only each channel's own delay moves, by exactly its own distance change.

Level: inverse-distance law (6 dB per doubling), nearer speakers are attenuated to
the level of the farthest one (never boosted, so no clipping). The LFE gets delay
only: at bass wavelengths the room dominates its level.

Off-axis: speakers are assumed aimed (toed-in) at the seat. When the listener moves
off a speaker's axis, its treble drops; a gentle high shelf restores it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import speaker_position

SEAT = np.zeros(3)
MIN_GAIN_DB = -12.0  # level compensation limit (very close speaker)


@dataclass
class Layout:
    names: list[str]
    channels: np.ndarray      # output channel per speaker
    positions: np.ndarray     # (n, 3) room frame, meters
    lfe: np.ndarray           # bool per speaker
    trim_db: np.ndarray       # static per-speaker trim
    ref_m: float              # constant arrival distance (see module doc)

    @classmethod
    def from_config(cls, speakers: list[dict], max_offset_m: float = 1.5) -> Layout:
        pos = np.array([speaker_position(s) for s in speakers], dtype=float)
        far = float(np.linalg.norm(pos - SEAT, axis=1).max()) if len(pos) else 0.0
        return cls(names=[s["name"] for s in speakers],
                   channels=np.array([s["channel"] for s in speakers], dtype=int),
                   positions=pos,
                   lfe=np.array([bool(s.get("lfe", False)) for s in speakers]),
                   trim_db=np.array([float(s.get("trim_db", 0.0)) for s in speakers]),
                   ref_m=far + max_offset_m)


@dataclass
class Steering:
    delay_s: np.ndarray   # per speaker
    gain_db: np.ndarray   # per speaker (includes trim)
    shelf_db: np.ndarray  # high-shelf boost per speaker
    distance_m: np.ndarray


def clamp_listener(p, max_offset_m: float) -> np.ndarray:
    """Keep the listener within ``max_offset_m`` (horizontal) of the seat."""
    p = np.asarray(p, dtype=float).copy()
    r = float(np.hypot(p[0], p[1]))
    if r > max_offset_m:
        p[:2] *= max_offset_m / r
    p[2] = float(np.clip(p[2], -max_offset_m, max_offset_m))
    return p


def steering(layout: Layout, listener, speed_of_sound: float = 343.0, level: bool = True,
             offaxis_max_db: float = 3.0, offaxis_full_deg: float = 30.0) -> Steering:
    """Delays / gains / shelf boosts that center the sound on ``listener`` (room frame)."""
    p = np.asarray(listener, dtype=float)
    vec = p - layout.positions
    d = np.maximum(np.linalg.norm(vec, axis=1), 0.05)
    delay = np.clip((layout.ref_m - d) / speed_of_sound, 0.0, None)

    gain = layout.trim_db.copy()
    mains = ~layout.lfe
    if level and mains.any():
        g = 20 * np.log10(d / d[mains].max())
        gain[mains] += np.maximum(g[mains], MIN_GAIN_DB)

    shelf = np.zeros(len(d))
    if offaxis_max_db > 0:
        axis = SEAT - layout.positions  # aimed at the seat
        cos = np.einsum("ij,ij->i", axis, vec) / (np.linalg.norm(axis, axis=1).clip(1e-6) * d)
        angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        shelf[mains] = offaxis_max_db * np.clip(angle[mains] / offaxis_full_deg, 0, 1)
    return Steering(delay, gain, shelf, d)


def bypass(layout: Layout) -> Steering:
    n = len(layout.names)
    return Steering(np.zeros(n), np.zeros(n), np.zeros(n), np.linalg.norm(layout.positions, axis=1))
