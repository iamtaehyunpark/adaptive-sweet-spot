"""Multichannel fractional delay lines with slew-rate-limited delay changes.

* Fractional (sub-sample) delays: 4-point cubic Lagrange interpolation - flat to
  within ~0.1 dB up to 10 kHz at 48 kHz, where linear interpolation loses ~2 dB.
* Slew limiting: the delay moves at most ``max_slew`` seconds per second, ramped per
  sample. A delay changing at rate r shifts the pitch by r (0.002 = 0.2 % = 3.5 cents,
  inaudible), so head movements never cause clicks or audible pitch wobble.
"""

from __future__ import annotations

import numpy as np

BASE = 2  # samples of fixed latency the interpolator needs (2 future samples)


class DelayBank:
    def __init__(self, channels: int, max_delay_s: float, sample_rate: int):
        self.fs = sample_rate
        self.max_delay = int(np.ceil(max_delay_s * sample_rate))
        self.hist_len = self.max_delay + BASE + 4
        self.hist = np.zeros((self.hist_len, channels), dtype=np.float32)
        self.current = np.zeros(channels)  # samples
        self.target = np.zeros(channels)
        self._cols = np.arange(channels)

    def set_target(self, delay_s: np.ndarray) -> None:
        self.target = np.clip(np.asarray(delay_s, dtype=float) * self.fs, 0, self.max_delay)

    def jump(self) -> None:
        """Move to the target immediately (before audio starts)."""
        self.current = self.target.copy()

    @property
    def current_s(self) -> np.ndarray:
        return self.current / self.fs

    def process(self, x: np.ndarray, max_slew: float) -> np.ndarray:
        """x: (n, channels) block -> delayed block; delays ramp toward the target."""
        n = len(x)
        buf = np.concatenate([self.hist, x.astype(np.float32, copy=False)])
        limit = max_slew * n
        step = np.clip(self.target - self.current, -limit, limit)
        ramp = np.arange(1, n + 1)[:, None] / n
        d = self.current + step * ramp + BASE                     # (n, ch) delay per sample
        pos = self.hist_len + np.arange(n)[:, None] - d
        i = np.floor(pos).astype(np.intp)
        f = pos - i
        c = self._cols
        xm1, x0, x1, x2 = buf[i - 1, c], buf[i, c], buf[i + 1, c], buf[i + 2, c]
        fm1, fp1, fm2 = f - 1, f + 1, f - 2
        y = (-f * fm1 * fm2 / 6 * xm1 + fp1 * fm1 * fm2 / 2 * x0
             - fp1 * f * fm2 / 2 * x1 + fp1 * f * fm1 / 6 * x2)
        self.current = self.current + step
        self.hist = buf[-self.hist_len:]
        return y.astype(np.float32)
