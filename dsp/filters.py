"""Click-free per-channel high shelf (off-axis treble correction).

A first-order shelf written as ``x + (g - 1) * highpass(x)``: the high-pass filter is
fixed, only the gain g changes, ramped per sample - so changing the boost can never
cause zipper noise (unlike recomputing biquad coefficients on the fly).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter


class ShelfBank:
    def __init__(self, channels: int, corner_hz: float, sample_rate: int):
        # first-order high-pass (bilinear transform)
        k = math.tan(math.pi * corner_hz / sample_rate)
        a1 = (k - 1) / (k + 1)
        b0 = 1 / (1 + k)
        self.b = np.array([b0, -b0])
        self.a = np.array([1.0, a1])
        self.zi = np.zeros((1, channels))
        self.gain = np.ones(channels)  # linear high-frequency gain, current

    def process(self, x: np.ndarray, target_db: np.ndarray) -> np.ndarray:
        target = 10 ** (np.asarray(target_db) / 20)
        hp, self.zi = lfilter(self.b, self.a, x, axis=0, zi=self.zi)
        n = len(x)
        ramp = np.arange(1, n + 1)[:, None] / n
        g = self.gain + (target - self.gain) * ramp
        self.gain = target
        if np.allclose(g, 1.0):
            return x
        return (x + (g - 1) * hp).astype(np.float32)
