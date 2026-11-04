#!/usr/bin/env python3
"""Validate the DSP end to end (Static / Follow steering, virtual surround), offline.

Drives the real Player audio callback (the code path used live: sources, engine,
virtual surround, volume) without a sound device, and measures the result:

1. Static at the seat, real speakers: does sound from every speaker arrive at the
   seat at the same time and level? (propagation simulated: delay d/c, level 1/d)
2. Off the seat: arrival spread and level spread for Bypass / Static / Follow.
3. Virtual surround, one speaker at a time (identify): the direction the binaural
   output encodes (interaural time difference -> azimuth) vs. the true direction of
   that speaker from the listener.
4. Virtual surround, phantom center (FL + FR): where the image lands for listeners at
   different positions, Static vs. Follow, vs. where it should land.
5. Moving listener: no clicks (largest sample step) and pitch stays put.

Uses config/default.toml + config/room.json speakers.

    python tools/validate_dsp.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402
from dsp.player import Player  # noqa: E402
from dsp.virtual import EAR_M  # noqa: E402

FS = 48000
BLOCK = 128
C = 343.0
OK, BAD = "ok", "FAIL"


def make_player(render: str) -> Player:
    cfg = load_config()
    cfg["audio"]["sample_rate"] = FS
    cfg["audio"]["block_size"] = BLOCK
    cfg["dsp"]["volume_db"] = 0.0
    p = Player(cfg, receiver=None)
    p.render = render
    p.device_info = {"channels": 2 if render == "virtual" else 6}
    return p


def run(p: Player, seconds: float, out_ch: int, listener_path=None) -> np.ndarray:
    """Call the audio callback block by block; listener_path(t) -> position (manual)."""
    status = SimpleNamespace(output_underflow=False, input_overflow=False)
    blocks = []
    n = int(seconds * FS / BLOCK)
    for b in range(n):
        if listener_path is not None:
            p.set(manual=list(listener_path(b * BLOCK / FS)))
        out = np.zeros((BLOCK, out_ch), np.float32)
        p._callback(out, BLOCK, None, status)
        blocks.append(out)
    return np.concatenate(blocks)


def settle(p: Player, mode: str, listener) -> None:
    p.set(mode=mode, manual=list(listener))
    pos = np.asarray(listener, dtype=float)
    p._s1, p._s2 = pos.copy(), pos.copy()  # skip the audio-rate position smoothing glide
    p.engine.set_listener(pos)
    p.engine._update_targets()
    p.engine.delays.jump()
    p.engine.gain = p.engine._target_gain.copy()
    p.virtual.gain = None  # virtual jumps to its targets on the next block


def lag(left: np.ndarray, right: np.ndarray, max_lag: int = 48) -> float:
    """Sub-sample interaural lag (+ = right ear leads), by cross-correlation."""
    a, b = left - left.mean(), right - right.mean()
    lags = np.arange(-max_lag, max_lag + 1)
    xc = np.array([np.dot(a[max_lag:-max_lag], np.roll(b, k)[max_lag:-max_lag]) for k in lags])
    i = int(np.argmax(xc))
    if 0 < i < len(xc) - 1:  # parabolic peak
        y0, y1, y2 = xc[i - 1], xc[i], xc[i + 1]
        return lags[i] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return float(lags[i])


def itd_to_azimuth(itd_s: float) -> float:
    """Woodworth inverse: lateral angle (deg) from the interaural time difference."""
    target = abs(itd_s) * C / EAR_M
    lo, hi = 0.0, math.pi / 2
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mid + math.sin(mid) < target else (lo, mid)
    return math.copysign(math.degrees(lo), itd_s)


def lateral_of(spk, listener) -> float:
    """True lateral angle (deg, + = right) of a speaker from the listener (front/back folded)."""
    v = np.asarray(spk[:2]) - np.asarray(listener[:2])
    return math.degrees(math.asin(np.clip(v[0] / np.linalg.norm(v), -1, 1)))


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def arrivals(p: Player, mode: str, listener) -> tuple[float, float]:
    """(arrival spread us, level spread dB) of the 5 mains at ``listener`` (real room)."""
    settle(p, mode, listener)
    t = np.arange(FS // 2)
    pulse = np.exp(-0.5 * ((t - 300) / 4.0) ** 2).astype(np.float32)
    y = np.concatenate([p.engine.process(np.repeat(pulse[i:i + BLOCK, None], 6, 1)) for i in range(0, len(t), BLOCK)])
    lay = p.engine.layout
    times, levels = [], []
    for i, ch in enumerate(lay.channels):
        if lay.lfe[i]:
            continue
        d = np.linalg.norm(np.asarray(listener) - lay.positions[i])
        sig = y[:, ch]
        times.append((t * sig).sum() / sig.sum() / FS + d / C)
        levels.append(20 * math.log10(sig.sum() / d))
    return (max(times) - min(times)) * 1e6, max(levels) - min(levels)


def main() -> int:
    fails = 0

    def check(cond: bool) -> str:
        nonlocal fails
        fails += not cond
        return OK if cond else BAD

    # ---------------------------------------------------------------- 1, 2 real speakers
    p = make_player("speakers")
    lay = p.engine.layout
    print("speakers (room frame, m):", ", ".join(f"{n} ({x:+.2f}, {y:+.2f}, {z:+.2f})"
                                               for n, (x, y, z) in zip(lay.names, lay.positions)))
    section("1. Static at the seat (real speakers): arrival + level at the seat")
    dt, dl = arrivals(p, "static", (0, 0, 0))
    print(f"arrival spread {dt:6.2f} us   level spread {dl:5.2f} dB   {check(dt < 5 and dl < 0.1)}")
    dtb, dlb = arrivals(p, "bypass", (0, 0, 0))
    print(f"(bypass, for comparison: {dtb:.0f} us, {dlb:.2f} dB - the layout's own mismatch)")

    section("2. Off the seat: arrival spread / level spread of the 5 mains (treble EQ off)")
    p.dcfg["offaxis_eq"] = False  # timing check: delays only (the shelf's phase is reported below)
    print(f"{'listener':>18s}  {'Bypass':>17s}  {'Static':>17s}  {'Follow':>17s}")
    for pos in [(0.3, 0, 0), (-0.5, 0, 0), (0, 0.4, 0), (0, -0.4, 0), (0.4, 0.3, 0.1)]:
        row = [arrivals(p, m, pos) for m in ("bypass", "static", "tracking")]
        good = row[2][0] < 10 and row[2][1] < 0.1
        print(f"({pos[0]:+.1f},{pos[1]:+.1f},{pos[2]:+.1f})".rjust(18) + "  "
              + "  ".join(f"{t:7.0f}us {l:5.2f}dB" for t, l in row) + f"   {check(good)}")
    p.dcfg["offaxis_eq"] = True
    eq = max(arrivals(p, "tracking", pos)[0] for pos in [(0.3, 0, 0), (-0.5, 0, 0), (0.4, 0.3, 0.1)])
    print(f"with the off-axis treble EQ on, its phase shifts arrivals by up to {eq:.0f} us "
          f"(people notice ~10-20 us; arrival spreads of ~1 ms)")

    # ---------------------------------------------------------------- 3 virtual: directions
    section("3. Virtual surround: direction encoded per speaker (ITD -> azimuth) vs. true")
    v = make_player("virtual")
    for pos in [(0, 0, 0), (0.4, 0, 0)]:
        print(f"listener at ({pos[0]:+.1f}, {pos[1]:+.1f}) - lateral angle, + = right (front/back folded)")
        for i, name in enumerate(v.engine.layout.names):
            if v.engine.layout.lfe[i]:
                continue
            settle(v, "static", pos)
            v.set_source("identify")
            # skip to this speaker's slot
            from dsp.sources import IDENTIFY_ORDER, NAMES

            slot = IDENTIFY_ORDER.index(NAMES.index(name))
            v.source.pos = slot * (v.source.step + v.source.gap) + FS // 10
            out = run(v, 0.8, 2)
            itd = lag(out[:, 0], out[:, 1]) / FS
            got = itd_to_azimuth(itd)
            want = lateral_of(v.engine.layout.positions[i], pos)
            ild = 20 * math.log10(np.std(out[:, 1]) / np.std(out[:, 0]))
            print(f"  {name:3s} true {want:+6.1f}  rendered {got:+6.1f}  (ITD {itd * 1e6:+5.0f} us, "
                  f"ILD R-L {ild:+5.1f} dB)  {check(abs(got - want) < 5 and ild * want >= 0)}")

    # ---------------------------------------------------------------- 4 virtual: phantom
    section("4. Virtual surround: phantom center (FL+FR) image, lateral angle")
    print(f"{'listener':>12s}  {'should be':>9s}  {'Static':>7s}  {'Follow':>7s}   (Static: strongest interaural")
    print(f"{'':37s}correlation of two arrivals - only its side is meaningful)")
    fl, fr = (v.engine.layout.positions[v.engine.layout.names.index(n)] for n in ("FL", "FR"))
    for pos in [(0, 0, 0), (0.2, 0, 0), (0.4, 0, 0), (-0.4, 0, 0), (0.3, 0.3, 0)]:
        want = (lateral_of(fl, pos) + lateral_of(fr, pos)) / 2  # aligned: midway between them
        got = {}
        for mode in ("static", "tracking"):
            settle(v, mode, pos)
            v.set_source("front")
            out = run(v, 0.6, 2)[FS // 10:]
            got[mode] = itd_to_azimuth(lag(out[:, 0], out[:, 1]) / FS)
        good = abs(got["tracking"] - want) < 4 and (pos[0] == 0 or got["static"] * pos[0] > 0)
        print(f"({pos[0]:+.1f},{pos[1]:+.1f})".rjust(12) + f"  {want:+9.1f}  {got['static']:+7.1f}  "
              f"{got['tracking']:+7.1f}   {check(good)}")
    print("Static off-center: pulled into the nearer speaker (precedence effect); Follow: midway")
    print("between FL and FR as seen from the listener (not straight ahead: that needs re-panning, 5.0).")

    # ---------------------------------------------------------------- 5 moving listener
    section("5. Moving listener (virtual, FL only, 1 kHz tone): clicks and pitch")
    tone = (np.sin(2 * math.pi * 1000 * np.arange(FS * 4) / FS) * 0.1).astype(np.float32)

    class Tone:
        name, pos = "tone", 0

        def read(self, n):
            x = np.zeros((n, 6), np.float32)
            x[:, 0] = tone[self.pos:self.pos + n]
            self.pos += n
            return x

        def info(self):
            return {}

    def pitch(mode: str, amp: float, hz: float) -> tuple[float, float, float]:
        """(min, max frequency at the left ear, largest sample step) while swaying sideways."""
        settle(v, mode, (0, 0, 0))
        v.source = Tone()
        out = run(v, 3.0, 2, listener_path=lambda t: (amp * math.sin(2 * math.pi * hz * t), 0, 0))[FS // 2:, 0]
        zc = np.where(np.diff(np.sign(out)) > 0)[0]
        frac = zc + out[zc] / (out[zc] - out[zc + 1])  # sub-sample zero crossings
        f = FS / np.diff(frac)
        return f.min(), f.max(), float(np.abs(np.diff(out)).max())

    still_step = 2 * math.pi * 1000 / FS * np.abs(tone).max() * 1.6  # generous bound
    for label, amp, hz in (("sway 0.3 m/s", 0.25, 0.2), ("fast 1.6 m/s", 0.5, 0.5)):
        speed = amp * 2 * math.pi * hz
        real = 1000 * speed * 0.5 / C  # FL is ~30 deg off the sway direction: cos(60) of the speed
        lo_b, hi_b, _ = pitch("bypass", amp, hz)
        lo_f, hi_f, step = pitch("tracking", amp, hz)
        ok = (hi_f - lo_f) < 1.0 if speed * 0.5 / C < 0.002 else True
        print(f"{label}: Bypass {lo_b:7.1f}-{hi_b:7.1f} Hz (real Doppler ~+-{real:.1f})   "
              f"Follow {lo_f:7.1f}-{hi_f:7.1f} Hz   max step {step:.4f}   {check(ok and step < still_step)}")
    print("Follow cancels the Doppler while the delay change stays under the slew limit")
    print("(0.2 %/s = 0.69 m/s toward a speaker); faster, it catches up with a slight pitch shift.")

    print(f"\n{'all checks passed' if not fails else f'{fails} check(s) failed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
