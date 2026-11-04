"""Sweet-spot steering engine: 6 channels in -> delayed, level- and treble-corrected out.

Modes:
* ``bypass``    no processing (delays 0, unity gain) - the reference for A/B tests
* ``static``    centered on the seat with the measured speaker positions (Milestone 1.1)
* ``tracking``  centered on the current listener position (Milestones 2.1 / 3.x)

Every parameter change is smoothed (delays slew-limited, gains and shelf ramped per
sample), so switching modes or moving the listener never clicks.
"""

from __future__ import annotations

import time

import numpy as np

from .delay import DelayBank
from .filters import ShelfBank
from .geometry import SEAT, Layout, bypass, clamp_listener, steering

MODES = ("bypass", "static", "tracking")


class SteeringEngine:
    def __init__(self, speakers: list[dict], sample_rate: int, dcfg: dict, speed_of_sound: float = 343.0,
                 max_offset_m: float = 1.5, channels: int = 6):
        self.fs = sample_rate
        self.cfg = dcfg
        self.c = speed_of_sound
        self.max_offset_m = max_offset_m
        self.channels = channels
        self.mode = dcfg.get("mode", "static")
        self.listener = SEAT.copy()
        self.delays = DelayBank(channels, dcfg.get("max_delay_s", 0.03), sample_rate)
        self.shelves = ShelfBank(channels, dcfg.get("shelf_hz", 3000.0), sample_rate)
        self.gain = np.ones(channels)  # linear, current (per channel)
        self.proc_ms = 0.0
        self.set_speakers(speakers)
        self._update_targets()
        self.delays.jump()
        self.gain = self._target_gain.copy()

    # ------------------------------------------------------------ control
    def set_speakers(self, speakers: list[dict]) -> None:
        self.layout = Layout.from_config(speakers, self.max_offset_m)
        need = self.layout.ref_m / self.c
        if need > self.delays.max_delay / self.fs:
            raise ValueError(f"speakers need {need * 1000:.1f} ms delay, max_delay_s is "
                             f"{self.delays.max_delay / self.fs * 1000:.1f} ms")

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode

    def set_listener(self, position) -> None:
        if position is not None:
            self.listener = clamp_listener(position, self.max_offset_m)

    def steering(self):
        if self.mode == "bypass":
            return bypass(self.layout)
        target = self.listener if self.mode == "tracking" else SEAT
        level = self.cfg.get("level_compensation", True)
        off_db = self.cfg.get("offaxis_max_db", 3.0) if self.cfg.get("offaxis_eq", True) else 0.0
        return steering(self.layout, target, self.c, level, off_db, self.cfg.get("offaxis_full_deg", 30.0))

    def _update_targets(self) -> None:
        st = self.steering()
        delay = np.zeros(self.channels)
        gain_db = np.zeros(self.channels)
        shelf = np.zeros(self.channels)
        ch = self.layout.channels
        delay[ch], gain_db[ch], shelf[ch] = st.delay_s, st.gain_db, st.shelf_db
        self.delays.set_target(delay)
        self._target_gain = 10 ** (gain_db / 20)
        self._target_shelf = shelf
        self.last = st

    # ------------------------------------------------------------ audio
    def process(self, x: np.ndarray) -> np.ndarray:
        """x: (n, channels) float32 block in channel order -> processed block."""
        t0 = time.perf_counter()
        self._update_targets()
        n = len(x)
        y = self.delays.process(x, self.cfg.get("slew", 0.002))
        y = self.shelves.process(y, self._target_shelf)
        # gain: one-pole toward the target per block, linear ramp inside the block
        a = 1 - np.exp(-n / (self.cfg.get("gain_tau_s", 0.05) * self.fs))
        new = self.gain + a * (self._target_gain - self.gain)
        ramp = np.arange(1, n + 1)[:, None] / n
        y = y * (self.gain + (new - self.gain) * ramp)
        self.gain = new
        self.proc_ms = (time.perf_counter() - t0) * 1000
        return y.astype(np.float32)

    def state(self) -> dict:
        """Per-speaker current/target values for the UI."""
        ch = self.layout.channels
        return {
            "mode": self.mode,
            "listener": [round(float(v), 3) for v in self.listener],
            "ref_m": self.layout.ref_m,
            "speakers": [{
                "name": name, "channel": int(c),
                "position": [round(float(v), 3) for v in self.layout.positions[i]],
                "lfe": bool(self.layout.lfe[i]),
                "distance_m": round(float(self.last.distance_m[i]), 3),
                "delay_ms": round(float(self.delays.current_s[c]) * 1000, 3),
                "target_delay_ms": round(float(self.delays.target[c] / self.fs) * 1000, 3),
                "gain_db": round(float(20 * np.log10(max(self.gain[c], 1e-6))), 2),
                "shelf_db": round(float(self._target_shelf[c]), 2),
            } for i, (name, c) in enumerate(zip(self.layout.names, ch))],
        }
