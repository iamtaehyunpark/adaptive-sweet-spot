"""Virtual surround: hear the 5.1 speakers over stereo (headphones / laptop) for testing.

Not a separate DSP: it sits after the steering engine and simulates what the real room
would do to its 6 outputs - the sound of each speaker travels to the listener's two
ears from where the speaker really stands (config / room setup):

* path delay to each ear: distance / c to the head (the arrival differences the engine
  has to compensate) plus the interaural time difference of a spherical head
  (Woodworth: up to ~0.66 ms for a speaker at the side)
* level falling with distance (1/d)
* head shadow: treble cut at the ear facing away from the speaker
* speaker directivity: treble falls off the speaker's axis (aimed at the seat), with the
  same model the engine's off-axis EQ assumes - so that EQ cancels it, as in the room
* head yaw (from the camera): the ears turn with your head while the speakers stay put
  in the room - turning your head is what tells front from back
* rear cue: sound from behind is duller (the pinna shades it) - a treble cut that grows
  toward the back

It uses the engine's own building blocks (fractional DelayBank, ShelfBank). The
simulated listener is the tracked (or manual) position, so in Bypass / Static an
off-center listener hears the image pulled toward the nearer speaker, and Follow
re-centers it - the same A/B as with the real speakers. Without measured HRTFs front
and back stay ambiguous when the head is still: turn your head a little to resolve it.
"""

from __future__ import annotations

import math

import numpy as np

from .delay import DelayBank
from .filters import ShelfBank
from .geometry import Layout

EAR_M = 0.0875        # half the ear spacing
SHADOW_DB = -10.0     # treble at the far ear for a speaker fully to the side
REF_M = 2.0           # distance rendered at unity gain
SLEW = 0.01           # physical movement: a real moving head hears Doppler too
REAR_DB = -4.0        # treble straight behind (pinna shading), fading to 0 at the sides


class VirtualSurround:
    def __init__(self, layout: Layout, sample_rate: int, max_offset_m: float = 1.5,
                 speed_of_sound: float = 343.0, channels: int = 6, offaxis_db: float = 3.0,
                 offaxis_full_deg: float = 30.0):
        self.fs = sample_rate
        self.offaxis_db = offaxis_db            # speaker treble loss off-axis (same model as the engine EQ)
        self.offaxis_full_deg = offaxis_full_deg
        self.c = speed_of_sound
        self.channels = channels
        self.set_layout(layout, max_offset_m)
        far = float(np.linalg.norm(layout.positions, axis=1).max()) + max_offset_m + EAR_M
        self.delays = DelayBank(2 * channels, far / speed_of_sound + 0.002, sample_rate)
        self.shelves = ShelfBank(2 * channels, 2000.0, sample_rate)
        self.gain = None

    def set_layout(self, layout: Layout, max_offset_m: float = 1.5) -> None:
        self.layout = layout
        # skip the common part of the flight time (keeps the latency low)
        near = float(np.linalg.norm(layout.positions, axis=1).min()) - max_offset_m - EAR_M
        self.offset_s = max(near, 0.0) / self.c

    def _targets(self, listener, yaw_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = np.asarray(listener, dtype=float)
        yaw = math.radians(yaw_deg)  # + = head turned right (clockwise from above)
        facing = np.array([math.sin(yaw), math.cos(yaw)])
        right = np.array([math.cos(yaw), -math.sin(yaw)])
        delay = np.zeros(2 * self.channels)
        gain = np.zeros(2 * self.channels)
        shelf = np.zeros(2 * self.channels)
        for i, ch in enumerate(self.layout.channels):
            spk = self.layout.positions[i]
            v = spk - p
            dist = max(float(np.linalg.norm(v)), 0.1)
            horiz = max(float(np.linalg.norm(v[:2])), 1e-6)
            side = float(v[:2] @ right) / horiz             # + = speaker to the right of the head
            front = float(v[:2] @ facing) / horiz           # - = behind the head
            lateral = math.asin(min(abs(side), 1.0))       # 0 = ahead/behind, pi/2 = at the side
            rear = REAR_DB * max(0.0, -front) if not self.layout.lfe[i] else 0.0
            # speaker directivity: angle between its axis (aimed at the seat) and the head
            aim = -spk
            cos = float(aim @ v) / (max(np.linalg.norm(aim), 1e-6) * dist) * -1.0
            off = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
            directivity = -self.offaxis_db * min(off / self.offaxis_full_deg, 1.0) if not self.layout.lfe[i] else 0.0
            for e, sign in enumerate((-1.0, 1.0)):          # left ear, right ear
                near = sign * side > 0
                # Woodworth spherical head: near ear a*sin(t) closer, far ear a*t farther (around the head)
                d = dist - EAR_M * math.sin(lateral) if near else dist + EAR_M * lateral
                k = 2 * ch + e
                delay[k] = d / self.c - self.offset_s
                gain[k] = min(REF_M / d, 4.0)
                shelf[k] = directivity + rear + (0.0 if near else SHADOW_DB * math.sin(lateral))
        return delay, gain, shelf

    def process(self, y: np.ndarray, listener, yaw_deg: float = 0.0) -> np.ndarray:
        """y: (n, 6) engine output in channel order -> (n, 2) binaural stereo."""
        delay, gain, shelf = self._targets(listener, yaw_deg)
        self.delays.set_target(delay)
        if self.gain is None:
            self.delays.jump()
            self.gain = gain
        x = np.repeat(y, 2, axis=1)  # column 2*ch + ear
        z = self.delays.process(x, SLEW)
        z = self.shelves.process(z, shelf)
        n = len(y)
        ramp = np.arange(1, n + 1)[:, None] / n
        z = z * (self.gain + (gain - self.gain) * ramp)
        self.gain = gain
        out = np.stack([z[:, 0::2].sum(axis=1), z[:, 1::2].sum(axis=1)], axis=1)
        return (out * 0.5).astype(np.float32)  # 6 sources summed: headroom
