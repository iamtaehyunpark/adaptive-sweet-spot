"""Listener position smoothing and short-term prediction (DSP side of the bridge).

Raw tracking positions jitter by +-2-4 cm and arrive at an irregular rate (the
adaptive detection rate runs 5-30 fps). A One Euro filter (Casiez et al., CHI 2012)
smooths hard while the head is still and follows quickly when it moves (its cutoff
rises with speed). The filtered velocity then extrapolates the position to the
moment the sound reaches the ears, hiding the camera -> speaker pipeline latency.
"""

from __future__ import annotations

import math

import numpy as np

MAX_SPEED = 2.0  # m/s, cap for prediction


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    """Vector One Euro filter with variable time step."""

    def __init__(self, min_cutoff_hz: float = 0.5, beta: float = 2.0, d_cutoff_hz: float = 1.0):
        self.min_cutoff = min_cutoff_hz
        self.beta = beta
        self.d_cutoff = d_cutoff_hz
        self.x: np.ndarray | None = None
        self.dx: np.ndarray | None = None
        self.t: float | None = None

    def reset(self) -> None:
        self.x = self.dx = self.t = None

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.x is None or self.t is None or t <= self.t:
            if self.x is None or self.t is None:
                self.x, self.dx, self.t = x, np.zeros_like(x), t
            return self.x
        dt = t - self.t
        dx = (x - self.x) / dt
        self.dx = self.dx + _alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        self.x = self.x + _alpha(cutoff, dt) * (x - self.x)
        self.t = t
        return self.x


class ListenerFilter:
    """Turns bridge messages into the listener position the DSP should use now."""

    def __init__(self, min_cutoff_hz: float = 0.5, beta: float = 2.0, predict: bool = True,
                 max_predict_s: float = 0.1):
        self.filter = OneEuro(min_cutoff_hz, beta)
        self.predict = predict
        self.max_predict_s = max_predict_s
        self.seq: int | None = None
        self.tracking = False
        self.msg_t = 0.0

    def update(self, msg) -> bool:
        """Feed the newest bridge message; returns True if it was new."""
        if msg is None or msg.seq == self.seq:
            return False
        if self.seq is not None and msg.seq < self.seq - 1000:
            self.filter.reset()  # sender restarted
        self.seq = msg.seq
        self.tracking = msg.tracking
        self.msg_t = msg.t
        if msg.tracking or self.filter.x is None:
            self.filter((msg.x, msg.y, msg.z), msg.t)
        if not msg.tracking and self.filter.dx is not None:
            self.filter.dx = np.zeros(3)  # face lost: hold still, don't extrapolate
        return True

    def position(self, at_t: float) -> np.ndarray | None:
        """Position at wall-clock time ``at_t`` (e.g. when the next audio block plays)."""
        f = self.filter
        if f.x is None:
            return None
        if not (self.predict and self.tracking):
            return f.x.copy()
        h = float(np.clip(at_t - f.t, 0.0, self.max_predict_s))
        v = f.dx
        speed = float(np.linalg.norm(v))
        if speed > MAX_SPEED:
            v = v * (MAX_SPEED / speed)
        return f.x + v * h
