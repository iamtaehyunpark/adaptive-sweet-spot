"""Real-time audio: source -> steering engine -> sound device, driven by the bridge.

The audio callback (PortAudio thread) does everything for one block:
  newest bridge message -> One Euro filter -> position predicted to when this block
  is heard -> engine targets -> process -> meters -> device channels.

Render (what reaches the device):
* ``speakers``  the 6 channels to a 6-channel DAC (the real system)
* ``virtual``   virtual surround over stereo (dsp/virtual.py): the room simulated for
                headphones / laptop speakers - tests the same steering without 5.1
* ``downmix``   plain ITU stereo downmix (checks that things play; no spatial cues)
``auto`` = speakers on a 6-channel device, else virtual.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import numpy as np

from config import CONFIG_DIR

from .engine import SteeringEngine
from .listener import ListenerFilter
from .geometry import SEAT
from .panner import vbap_gains
from .sources import CHANNELS, FileSource, ObjectSource, Silence, TestSignal
from .virtual import VirtualSurround

RENDERS = ("auto", "speakers", "virtual", "downmix")
YAW_TAU_S = 0.08

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "assets" / "audio"
AUDIO_FILE = CONFIG_DIR / "audio.json"  # chosen output device (per machine)
# ITU-R BS.775 downmix: L = FL + .707 FC + .707 SL (LFE dropped), normalized
_D = 1 / (1 + 2 * 0.7071)
DOWNMIX = np.array([[1, 0], [0, 1], [0.7071, 0.7071], [0, 0], [0.7071, 0], [0, 0.7071]], np.float32) * _D


def list_devices(refresh: bool = False) -> list[dict]:
    """Audio devices. ``refresh`` re-scans (PortAudio only sees devices present at start);
    no stream may be open while refreshing."""
    import sounddevice as sd

    if refresh:
        sd._terminate()
        sd._initialize()
    out = []
    for i, d in enumerate(sd.query_devices()):
        out.append({"index": i, "name": d["name"], "out": d["max_output_channels"], "in": d["max_input_channels"],
                    "rate": d["default_samplerate"], "hostapi": sd.query_hostapis(d["hostapi"])["name"]})
    return out


def find_device(name: str | int | None, kind: str) -> int | None:
    """Device index by index or name substring (case-insensitive); None = system default."""
    if name is None or name == "":
        return None
    if isinstance(name, int) or str(name).isdigit():
        return int(name)
    key = "out" if kind == "output" else "in"
    devices = [d for d in list_devices() if d[key] > 0]
    for d in devices:  # exact name first, then substring
        if d["name"] == name:
            return d["index"]
    for d in devices:
        if str(name).lower() in d["name"].lower():
            return d["index"]
    raise ValueError(f"no {kind} device matching {name!r}")



def head_to_room(offset, yaw_deg: float) -> np.ndarray:
    """Turn an offset in the listener's head frame (x right, y facing) into the room frame.

    ``yaw_deg``: + = head turned right (clockwise from above), as on the bridge.
    """
    yaw = math.radians(yaw_deg)
    facing = np.array([math.sin(yaw), math.cos(yaw)])
    right = np.array([math.cos(yaw), -math.sin(yaw)])
    o = np.asarray(offset, dtype=float)
    xy = o[0] * right + o[1] * facing
    return np.array([xy[0], xy[1], o[2]])

class Player:
    def __init__(self, cfg: dict, receiver=None):
        self.cfg = cfg
        self.acfg = cfg["audio"]
        self.dcfg = cfg.get("dsp", {})
        self.receiver = receiver
        self.lock = threading.Lock()
        d = self.dcfg
        self.filter = ListenerFilter(d.get("min_cutoff_hz", 1.5), d.get("beta", 6.0), d.get("predict", False),
                                     d.get("max_predict_s", 0.1))
        self.volume_db = float(d.get("volume_db", -6.0))
        self.source = Silence()
        self.stream = None
        self.fs = int(self.acfg["sample_rate"])
        self.block = int(self.acfg["block_size"])
        self.output = self.acfg.get("device") or None
        if self.output is None and AUDIO_FILE.exists():  # last choice from the Sound tab
            self.output = json.loads(AUDIO_FILE.read_text()).get("output")
        self.input = None          # input device for pass-through ("input" source)
        self.input_channels = 0
        self.device_info: dict = {}
        self.render = d.get("render", "auto")
        self.engine = self._make_engine(self.fs)
        self.peaks = np.zeros(CHANNELS)
        self.xruns = 0
        self.blocks = 0
        self.error = ""
        self.msg_age_ms: float | None = None   # camera frame -> applied in the DSP
        self.msg_rate = 0.0
        self._msg_times: list[float] = []
        self.out_latency_s = 0.0
        self.manual = False
        self.manual_pos: np.ndarray | None = None
        # sound object (source "object"): placed relative to the seat, facing front (room
        # frame). In Follow the whole scene moves and turns with the listener's head.
        self.object_pos = np.array([0.0, 1.5, 0.0])
        self.object_signal = "bursts"
        # audio-rate position smoothing (two cascaded one-pole stages): positions arrive
        # in steps (5-30 per second, manual drags too); stepping the delays makes a tiny
        # pitch chirp per step - heard as ticking. This turns them into a smooth path.
        self.position_tau_s = float(d.get("position_tau_s", 0.025))
        self._s1: np.ndarray | None = None
        self._s2: np.ndarray | None = None
        self.yaw_deg = 0.0        # smoothed head yaw (virtual surround only; the steering ignores it)
        self.head_yaw_deg = 0.0   # smoothed head yaw that turns the sound scene in Follow (0 = front)
        self._yaw_target = 0.0

    def _make_engine(self, fs: int) -> SteeringEngine:
        old = getattr(self, "engine", None)
        eng = SteeringEngine(self.cfg["speakers"], fs, self.dcfg, self.acfg.get("speed_of_sound", 343.0),
                             self.cfg.get("listener", {}).get("max_offset_m", 1.5))
        if old is not None:
            eng.mode, eng.listener = old.mode, old.listener
        self.virtual = VirtualSurround(eng.layout, fs, eng.max_offset_m, eng.c,
                                       offaxis_db=self.dcfg.get("offaxis_max_db", 3.0),
                                       offaxis_full_deg=self.dcfg.get("offaxis_full_deg", 30.0))
        return eng

    @property
    def active_render(self) -> str:
        out_ch = self.device_info.get("channels", CHANNELS)
        if self.render == "auto":
            return "speakers" if out_ch >= CHANNELS else "virtual"
        if self.render == "speakers" and out_ch < CHANNELS:
            return "downmix"
        return self.render

    # ------------------------------------------------------------ stream
    def start(self) -> None:
        import sounddevice as sd

        self.stop()
        out_dev = find_device(self.output, "output")
        info = sd.query_devices(out_dev if out_dev is not None else sd.default.device[1])
        out_ch = min(CHANNELS, int(info["max_output_channels"]))
        out_ch = CHANNELS if out_ch >= CHANNELS else 2 if out_ch >= 2 else 1
        fs = getattr(self.source, "fs", None) or self.fs
        if fs != self.engine.fs:
            self.engine = self._make_engine(fs)
        kw = dict(samplerate=fs, blocksize=self.block, dtype="float32", latency="low", callback=self._callback)
        in_name = None
        if self.input is not None:
            in_dev = find_device(self.input, "input")
            in_info = sd.query_devices(in_dev if in_dev is not None else sd.default.device[0])
            in_name = in_info["name"]
            self.input_channels = min(CHANNELS, int(in_info["max_input_channels"]))
            self.stream = sd.Stream(device=(in_dev, out_dev), channels=(self.input_channels, out_ch), **kw)
        else:
            self.input_channels = 0
            self.stream = sd.OutputStream(device=out_dev, channels=out_ch, **kw)
        self.stream.start()
        lat = self.stream.latency
        self.out_latency_s = float(lat[1] if isinstance(lat, (tuple, list)) else lat)
        self.device_info = {"name": info["name"], "channels": out_ch, "rate": fs, "block": self.block,
                            "downmix": out_ch < CHANNELS,
                            "input": in_name}
        self.error = ""

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def restart(self) -> None:
        try:
            self.start()
        except Exception as e:  # report to the UI, keep the process alive
            self.stream = None
            self.error = str(e)

    # ------------------------------------------------------------ control
    def set_source(self, name: str, file: str | None = None, start_s: float = 0.0) -> None:
        old = self.source
        if name == "off":
            src = Silence()
        elif name == "file":
            path = Path(file) if file else None
            if path and not path.is_absolute():
                path = AUDIO_DIR / path
            if not path or not path.exists():
                raise ValueError(f"file not found: {file}")
            src = FileSource(path, start_s=start_s)
        elif name == "input":
            src = Silence()
            src.name = "input"
        elif name == "object":
            if self.object_signal == "file":
                path = Path(file) if file else None
                if path and not path.is_absolute():
                    path = AUDIO_DIR / path
                if not path or not path.exists():
                    raise ValueError(f"file not found: {file}")
                src = ObjectSource(self.engine.fs, "file", path, start_s)
            else:
                src = ObjectSource(self.engine.fs, self.object_signal)
        else:
            src = TestSignal(name, self.engine.fs)
        need_restart = (getattr(src, "fs", self.engine.fs) != self.engine.fs) or \
                       ((name == "input") != (self.input is not None))
        if name == "input" and self.input is None:
            self.input = self.dcfg.get("input_device") or ""
        if name != "input":
            self.input = None
        with self.lock:
            self.source = src
        if hasattr(old, "close"):
            old.close()
        if need_restart and self.stream is not None:
            self.restart()

    def set(self, **kw) -> None:
        """Apply settings from the control port / CLI."""
        if "mode" in kw:
            self.engine.set_mode(kw["mode"])
        if "render" in kw:
            if kw["render"] not in RENDERS:
                raise ValueError(f"render must be one of {RENDERS}")
            self.render = kw["render"]
        if "volume_db" in kw:
            self.volume_db = float(np.clip(kw["volume_db"], -60, 0))
        for key in ("offaxis_eq", "level_compensation", "predict", "virtual_yaw"):
            if key in kw:
                self.dcfg[key] = bool(kw[key])
                if key == "predict":
                    self.filter.predict = bool(kw[key])
        if "slew" in kw:
            self.dcfg["slew"] = float(np.clip(kw["slew"], 0.0001, 0.05))
        if "speakers" in kw:  # live positions from the Speakers tab (before Save)
            pos = {sp["name"]: sp["position"] for sp in kw["speakers"]}
            speakers = [{**sp, "position": [float(v) for v in pos[sp["name"]]]} if sp["name"] in pos else sp
                        for sp in self.cfg["speakers"]]
            self.reload_speakers(speakers)
        if "manual" in kw:  # manual listener position (Milestone 3.1) instead of the bridge
            m = kw["manual"]
            self.manual = m is not None
            self.manual_pos = None if m is None else np.asarray(m, dtype=float)
        restart = False
        if "output" in kw and kw["output"] != self.output:
            self.select_output(kw["output"] or None)
        if "input_device" in kw:
            self.dcfg["input_device"] = kw["input_device"]
            if self.input is not None:
                self.input, restart = kw["input_device"], True
        if "object" in kw:  # sound object position, room frame
            self.object_pos = np.asarray(kw["object"], dtype=float)
        if "object_signal" in kw:
            self.object_signal = kw["object_signal"]
            if self.source.name == "object" and "source" not in kw:
                kw["source"] = "object"
        if "source" in kw:
            self.set_source(kw["source"], kw.get("file"), float(kw.get("start_s", 0.0)))
        if restart:
            self.restart()

    def select_output(self, name: str | None) -> None:
        """Switch the output device (name, or None = system default); remembered."""
        previous = self.output
        self.output = name
        self.restart()
        if self.error:  # keep playing on the previous device
            error = self.error
            self.output = previous
            self.restart()
            raise ValueError(f"cannot open {name}: {error}")
        AUDIO_FILE.write_text(json.dumps({"output": name}, indent=2))

    def devices(self, refresh: bool = False) -> list[dict]:
        if refresh:
            running = self.stream is not None
            self.stop()
            try:
                return list_devices(refresh=True)
            finally:
                if running:
                    self.restart()
        return list_devices()

    def reload_speakers(self, speakers: list[dict]) -> None:
        self.engine.set_speakers(speakers)  # raises if out of the delay range: keep the old layout
        self.cfg["speakers"] = speakers
        self.virtual.set_layout(self.engine.layout, self.engine.max_offset_m)

    # ------------------------------------------------------------ audio thread
    def _callback(self, *args) -> None:
        if len(args) == 5:
            indata, outdata, frames, _t, status = args
        else:
            outdata, frames, _t, status = args
            indata = None
        if status.output_underflow or status.input_overflow:
            self.xruns += 1
        self._track()
        with self.lock:
            src = self.source
        if src.name == "object":
            src.set_gains(self.object_gains())
        if indata is not None and src.name == "input":
            x = np.zeros((frames, CHANNELS), np.float32)
            x[:, :indata.shape[1]] = indata[:, :CHANNELS]
        else:
            x = src.read(frames)
        y = self.engine.process(x) * (10 ** (self.volume_db / 20))
        peak = np.abs(y).max(axis=0)
        self.peaks = np.maximum(peak, self.peaks * 0.9)
        render = self.active_render
        out_ch = outdata.shape[1]
        if render == "virtual" and out_ch >= 2:
            v = self.virtual.process(y, self.engine.listener, self.yaw_deg)  # the room, simulated at the listener
            y = np.zeros((frames, out_ch), np.float32)
            y[:, :2] = v
        elif out_ch < CHANNELS or render == "downmix":
            mix = y @ DOWNMIX if out_ch >= 2 else y.mean(axis=1, keepdims=True)
            y = np.zeros((frames, out_ch), np.float32)
            y[:, :mix.shape[1]] = mix
        np.clip(y, -1.0, 1.0, out=outdata)
        self.blocks += 1

    def _track(self) -> None:
        """Listener target (manual, or newest bridge message -> filter -> prediction),
        smoothed at audio rate, -> the engine's listener."""
        target = None
        if self.manual:
            target = self.manual_pos
        elif self.receiver is not None:
            msg = self.receiver.latest()
            now = time.time()
            if self.filter.update(msg):
                if msg.tracking:
                    self._yaw_target = float(msg.yaw_deg)
                self.msg_age_ms = (now - msg.t) * 1000
                self._msg_times = [t for t in self._msg_times if now - t < 2.0] + [now]
                self.msg_rate = len(self._msg_times) / 2.0
            # the smoothing delays the position by ~2 tau: predict that far further ahead
            heard_at = now + self.out_latency_s + self.block / self.engine.fs + 2 * self.position_tau_s
            target = self.filter.position(heard_at)
        if target is None:
            return
        if self._s1 is None:
            self._s1, self._s2 = target.copy(), target.copy()
        a = 1.0 - np.exp(-self.block / (max(self.position_tau_s, 1e-4) * self.engine.fs))
        self._s1 = self._s1 + a * (target - self._s1)
        self._s2 = self._s2 + a * (self._s1 - self._s2)
        self.engine.set_listener(self._s2)
        yaw_target = 0.0 if self.manual or not self.dcfg.get("virtual_yaw", True) else self._yaw_target
        a_yaw = 1.0 - np.exp(-self.block / (YAW_TAU_S * self.engine.fs))  # yaw is noisier: smooth more
        self.yaw_deg += a_yaw * (yaw_target - self.yaw_deg)
        head_target = 0.0 if self.manual else self._yaw_target
        self.head_yaw_deg += a_yaw * (head_target - self.head_yaw_deg)

    def object_heard(self) -> np.ndarray:
        """Where the sound object is rendered, room frame.

        Its placement is relative to the seat (the sweet-spot center), facing front. Follow
        carries the sweet spot - and so the whole sound scene - with the listener: the object
        keeps its offset from you and turns with your head (placed 1.5 m ahead = 1.5 m in
        front of your face, wherever you sit and wherever you look). Static / Bypass keep it
        where it was placed.
        """
        eng = self.engine
        if eng.mode != "tracking":
            return self.object_pos.copy()
        return eng.listener + head_to_room(self.object_pos - SEAT, self.head_yaw_deg)

    def object_gains(self) -> np.ndarray:
        """Channel gains for the sound object, panned as heard from the listener (Follow)
        or the seat (Static / Bypass)."""
        eng = self.engine
        ref = eng.listener if eng.mode == "tracking" else SEAT
        g = vbap_gains(eng.layout, ref, self.object_heard())
        out = np.zeros(CHANNELS)
        out[eng.layout.channels] = g
        return out

    # ------------------------------------------------------------ status
    def status(self) -> dict:
        st = self.engine.state()
        levels = 20 * np.log10(np.maximum(self.peaks, 1e-5))
        for s in st["speakers"]:
            s["peak_db"] = round(float(levels[s["channel"]]), 1)
        block_ms = self.block / self.engine.fs * 1000
        out_ms = self.out_latency_s * 1000 + block_ms
        smooth_ms = 2 * self.position_tau_s * 1000  # two cascaded one-pole stages
        recent = self.msg_age_ms is not None and time.time() - self.filter.msg_t < 1.0
        st.update({
            "running": self.stream is not None and self.stream.active,
            "error": self.error,
            "device": self.device_info, "output": self.output,
            "render": self.render, "active_render": self.active_render,
            "source": self.source.name, "source_info": self.source.info(),
            "volume_db": self.volume_db,
            "offaxis_eq": self.dcfg.get("offaxis_eq", True),
            "level_compensation": self.dcfg.get("level_compensation", True),
            "predict": self.filter.predict,
            "virtual_yaw": self.dcfg.get("virtual_yaw", True), "yaw_deg": round(self.yaw_deg, 1),
            "slew": self.dcfg.get("slew", 0.002),
            "manual": self.manual,
            "object": [round(float(v), 3) for v in self.object_pos], "object_signal": self.object_signal,
            "object_heard": [round(float(v), 3) for v in self.object_heard()],
            "head_yaw_deg": round(self.head_yaw_deg, 1),
            "bridge": {"tracking": self.filter.tracking, "rate": self.msg_rate,
                       "age_ms": self.msg_age_ms if recent else None},
            "latency": {"output_ms": out_ms, "block_ms": block_ms,
                        "vision_ms": self.msg_age_ms if recent else None,
                        "smoothing_ms": smooth_ms,
                        "motion_to_sound_ms": (self.msg_age_ms + smooth_ms + out_ms) if recent else None,
                        "proc_ms": self.engine.proc_ms,
                        "cpu": float(self.stream.cpu_load) if self.stream is not None else 0.0},
            "xruns": self.xruns,
            "files": sorted(p.name for p in AUDIO_DIR.glob("*") if p.suffix.lower() in (".wav", ".flac", ".aiff", ".ogg")),
        })
        return st
