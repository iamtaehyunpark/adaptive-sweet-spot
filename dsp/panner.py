"""Sound object panning: one mono source at a position in the room -> 5.1 channel gains.

2D VBAP (Pulkki 1997) over the five mains: the object's direction, seen from the
reference point (the listener in Follow mode, else the seat), lies between two
adjacent speakers; their gains solve p = g1*l1 + g2*l2 for the unit direction
vectors, power-normalized (g1^2 + g2^2 = 1). Directions are taken from where the
speakers actually stand as seen from the reference - so the object keeps its
direction when the listener moves (re-panning). Level follows the object's
distance (1/d, re-referenced to REF_M, clamped). The LFE gets nothing (mains only).
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import Layout

REF_M = 1.5                 # object distance rendered at unity gain
GAIN_RANGE = (0.25, 2.0)    # distance gain limits (-12 .. +6 dB)


def vbap_gains(layout: Layout, reference, obj) -> np.ndarray:
    """Per-speaker gains (layout order) placing a source at ``obj`` as heard from ``reference``."""
    ref = np.asarray(reference, dtype=float)
    mains = np.where(~layout.lfe)[0]
    az = np.array([math.atan2(*(layout.positions[i][:2] - ref[:2])) for i in mains])  # clockwise from +y
    order = np.argsort(az)
    mains, az = mains[order], az[order]
    v = np.asarray(obj, dtype=float)[:2] - ref[:2]
    dist = float(np.hypot(*v))
    target = math.atan2(v[0], v[1]) if dist > 1e-6 else 0.0

    gains = np.zeros(len(layout.names))
    n = len(mains)
    for k in range(n):  # adjacent pair (k, k+1) around the circle
        a, b = az[k], az[(k + 1) % n]
        span = (b - a) % (2 * math.pi) or 2 * math.pi
        off = (target - a) % (2 * math.pi)
        if off <= span + 1e-9:
            base = np.array([[math.sin(a), math.cos(a)], [math.sin(b), math.cos(b)]])  # rows: l1, l2
            p = np.array([math.sin(target), math.cos(target)])
            g = np.linalg.solve(base.T, p) if abs(np.linalg.det(base)) > 1e-6 else np.array([1.0, 0.0])
            g = np.clip(g, 0, None)
            g /= max(np.linalg.norm(g), 1e-9)
            gains[mains[k]], gains[mains[(k + 1) % n]] = g
            break
    distance_gain = float(np.clip(REF_M / max(dist, 0.05), *GAIN_RANGE))
    return gains * distance_gain
