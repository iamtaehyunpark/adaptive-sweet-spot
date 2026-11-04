"""6-channel audio sources (channel order FL, FR, FC, LFE, SL, SR).

Test signals for the milestones:

* ``identify``  pink noise on one speaker at a time (FL, FC, FR, SR, SL, LFE) - wiring
                and level check; ``current`` names the active speaker.
* ``center``    the same (correlated) pink noise on all five mains: when the delays
                and levels are right, it fuses into one image at your head.
* ``front``     the same noise on FL + FR: a phantom center, the most sensitive test -
                moving 10 cm sideways pulls it into the nearer speaker without steering.
* ``surround``  the same on SL + SR.
* ``clicks``    an impulse on every main every 0.5 s - for measuring alignment with a
                microphone at the listening position.
* ``file``      a WAV/FLAC file (6-ch in WAV order; stereo -> FL/FR, mono -> FC), looped.
* ``object``    one mono sound placed anywhere in the room, panned over the speakers
                (dsp/panner.py): noise bursts (easiest to localize), steady noise, or a
                file mixed down to mono.
"""

from __future__ import annotations

import threading
from collections import deque
from pathlib import Path

import numpy as np

CHANNELS = 6
FL, FR, FC, LFE, SL, SR = range(6)
MAINS = [FL, FR, FC, SL, SR]
NAMES = ["FL", "FR", "FC", "LFE", "SL", "SR"]
IDENTIFY_ORDER = [FL, FC, FR, SR, SL, LFE]
TEST_KINDS = ["identify", "center", "front", "surround", "clicks"]
NOISE_RMS = 10 ** (-20 / 20)  # -20 dBFS


def pink_noise(n: int, fs: int, seed: int = 1, lowpass_hz: float | None = None) -> np.ndarray:
    """Seamlessly looping pink noise (1/f spectrum shaped in the frequency domain)."""
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    shape = np.zeros_like(f)
    band = f >= 20
    shape[band] = 1 / np.sqrt(f[band])
    if lowpass_hz:
        shape[f > lowpass_hz] = 0
    x = np.fft.irfft(spec * shape, n)
    return (x * NOISE_RMS / x.std()).astype(np.float32)


class Silence:
    name = "off"

    def read(self, n: int) -> np.ndarray:
        return np.zeros((n, CHANNELS), np.float32)

    def info(self) -> dict:
        return {}


class TestSignal:
    def __init__(self, kind: str, fs: int, step_s: float = 1.5, gap_s: float = 0.4):
        if kind not in TEST_KINDS:
            raise ValueError(f"unknown test signal {kind!r}")
        self.name = kind
        self.fs = fs
        self.noise = pink_noise(fs * 8, fs)
        self.bass = pink_noise(fs * 8, fs, seed=2, lowpass_hz=120) if kind == "identify" else None
        self.pos = 0
        self.step = int(step_s * fs)
        self.gap = int(gap_s * fs)
        self.current: str | None = None

    def _loop(self, src: np.ndarray, n: int) -> np.ndarray:
        idx = (self.pos + np.arange(n)) % len(src)
        return src[idx]

    def read(self, n: int) -> np.ndarray:
        out = np.zeros((n, CHANNELS), np.float32)
        kind = self.name
        if kind == "identify":
            period = self.step + self.gap
            slot = (self.pos // period) % len(IDENTIFY_ORDER)
            ch = IDENTIFY_ORDER[slot]
            t = (self.pos + np.arange(n)) % period
            on = (t < self.step).astype(np.float32)
            src = self.bass if ch == LFE else self.noise
            out[:, ch] = self._loop(src, n) * on
            self.current = NAMES[ch]
        elif kind == "clicks":
            t = (self.pos + np.arange(n)) % (self.fs // 2)
            out[t == 0, :] = 0.5
            out[:, LFE] = 0
        else:
            chans = {"center": MAINS, "front": [FL, FR], "surround": [SL, SR]}[kind]
            x = self._loop(self.noise, n)
            for c in chans:
                out[:, c] = x
        self.pos += n
        return out

    def info(self) -> dict:
        return {"current": self.current} if self.name == "identify" else {}


class FileSource:
    """Streams a sound file from a reader thread (no disk access in the audio callback)."""

    CHUNK = 4096

    def __init__(self, path: str | Path, loop: bool = True, start_s: float = 0.0):
        import soundfile as sf

        self.path = Path(path)
        self.name = "file"
        self._file = sf.SoundFile(str(self.path))
        self.fs = self._file.samplerate
        self.file_channels = self._file.channels
        self.start_s = max(0.0, min(float(start_s), self._file.frames / self.fs - 1))
        self._file.seek(int(self.start_s * self.fs))
        self.loop = loop
        self._chunks: deque = deque()
        self._buffered = 0
        self._lock = threading.Lock()
        self._need = threading.Event()
        self._need.set()
        self._running = True
        self.ended = False
        self._thread = threading.Thread(target=self._reader, daemon=True, name="FileSource")
        self._thread.start()

    def _to6(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros((len(x), CHANNELS), np.float32)
        c = x.shape[1]
        if c == 1:
            out[:, FC] = x[:, 0]
        elif c == 2:
            out[:, FL], out[:, FR] = x[:, 0], x[:, 1]
        else:
            out[:, :min(c, CHANNELS)] = x[:, :CHANNELS]
        return out

    def _reader(self) -> None:
        while self._running:
            self._need.wait(0.2)
            while self._running and self._buffered < self.fs:  # keep ~1 s buffered
                x = self._file.read(self.CHUNK, dtype="float32", always_2d=True)
                if len(x) == 0:
                    if not self.loop:
                        self.ended = True
                        break
                    self._file.seek(0)
                    continue
                with self._lock:
                    self._chunks.append(self._to6(x))
                    self._buffered += len(x)
            self._need.clear()

    def read(self, n: int) -> np.ndarray:
        out = np.zeros((n, CHANNELS), np.float32)
        filled = 0
        with self._lock:
            while filled < n and self._chunks:
                c = self._chunks[0]
                take = min(n - filled, len(c))
                out[filled:filled + take] = c[:take]
                filled += take
                if take == len(c):
                    self._chunks.popleft()
                else:
                    self._chunks[0] = c[take:]
            self._buffered -= filled
        if self._buffered < self.fs // 2:
            self._need.set()
        return out

    def info(self) -> dict:
        return {"file": self.path.name, "file_channels": self.file_channels, "file_rate": self.fs,
                "start_s": self.start_s}

    def close(self) -> None:
        self._running = False
        self._need.set()
        self._thread.join(timeout=1.0)
        self._file.close()


OBJECT_SIGNALS = ["bursts", "noise", "file"]


class ObjectSource:
    """A mono signal panned to 6 channels; the player sets the per-channel gains each
    block (ramped across the block, so moving the object never clicks)."""

    name = "object"

    def __init__(self, fs: int, signal: str = "bursts", file: str | Path | None = None, start_s: float = 0.0):
        if signal not in OBJECT_SIGNALS:
            raise ValueError(f"object signal must be one of {OBJECT_SIGNALS}")
        self.signal = signal
        self.fs = fs
        self.pos = 0
        self.noise = pink_noise(fs * 8, fs, seed=3) * 2.0  # -14 dBFS: a single source, not 5
        self.file = FileSource(file, start_s=start_s) if signal == "file" else None
        if self.file is not None:
            self.fs = self.file.fs
        self.gains = np.zeros(CHANNELS)
        self._target = np.zeros(CHANNELS)

    def set_gains(self, gains: np.ndarray) -> None:
        self._target = np.asarray(gains, dtype=float)

    def _mono(self, n: int) -> np.ndarray:
        if self.file is not None:
            x = self.file.read(n)
            return x.sum(axis=1) / max(1, self.file.file_channels) ** 0.5
        idx = (self.pos + np.arange(n)) % len(self.noise)
        x = self.noise[idx]
        if self.signal == "bursts":  # 250 ms bursts every 0.5 s, 10 ms ramps
            t = ((self.pos + np.arange(n)) % (self.fs // 2)) / self.fs
            env = np.clip(np.minimum(t, 0.25 - t) / 0.01, 0, 1)
            x = x * env
        return x

    def read(self, n: int) -> np.ndarray:
        x = self._mono(n)
        self.pos += n
        ramp = np.arange(1, n + 1)[:, None] / n
        g = self.gains + (self._target - self.gains) * ramp
        self.gains = self._target.copy()
        return (x[:, None] * g).astype(np.float32)

    def info(self) -> dict:
        info = {"signal": self.signal}
        if self.file is not None:
            info.update(self.file.info())
        return info

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
