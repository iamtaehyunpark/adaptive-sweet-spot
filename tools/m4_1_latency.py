#!/usr/bin/env python3
"""Milestone 4.1: latency budget (motion -> sound < 25 ms), on the dev host or the QIDK.

    python tools/m4_1_latency.py bench        # DSP cost per audio block (no audio device)
    python tools/m4_1_latency.py live [s]     # measure a running app + DSP for s seconds

``live`` reads the DSP's own measurements while you move in front of the camera:
camera frame -> DSP (vision + bridge), DSP -> speaker (device latency + one block), and
their sum. Not included: camera exposure/readout before the frame timestamp
(~1 frame, measure with a blinking LED if needed).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402


def bench() -> None:
    from dsp.engine import SteeringEngine
    from dsp.virtual import VirtualSurround

    cfg = load_config()
    fs = cfg["audio"]["sample_rate"]
    for block in (64, 128, 256):
        eng = SteeringEngine(cfg["speakers"], fs, cfg["dsp"])
        eng.set_mode("tracking")
        virt = VirtualSurround(eng.layout, fs)
        x = (np.random.default_rng(0).standard_normal((block, 6)) * 0.1).astype(np.float32)
        for name, fn in (("engine", lambda p: eng.process(x)),
                         ("engine + virtual", lambda p: virt.process(eng.process(x), p))):
            n = 3000
            t = time.perf_counter()
            for i in range(n):
                p = (0.3 * np.sin(i / 200), 0.1, 0.0)
                eng.set_listener(p)
                fn(p)
            ms = (time.perf_counter() - t) / n * 1000
            budget = block / fs * 1000
            print(f"block {block:3d} ({budget:.2f} ms): {name:17s} {ms:.3f} ms  = {ms / budget * 100:4.1f}% of real time")


def live(seconds: float) -> None:
    from dsp.control import DspClient

    client = DspClient(port=load_config()["dsp"]["control_port"])
    vis, out, tot = [], [], []
    end = time.time() + seconds
    print(f"sampling the DSP for {seconds:.0f} s - move your head in front of the camera...")
    while time.time() < end:
        st = client.request("status")
        if st is None:
            sys.exit("DSP not running (start python tools/app.py)")
        lat = st["latency"]
        out.append(lat["output_ms"])
        if lat["vision_ms"] is not None:
            vis.append(lat["vision_ms"])
            tot.append(lat["motion_to_sound_ms"])
        time.sleep(0.05)
    if not tot:
        sys.exit("no tracking messages reached the DSP (is the camera tracking you?)")

    def row(name, v):
        v = np.array(v)
        print(f"{name:18s} median {np.median(v):6.1f} ms   p95 {np.percentile(v, 95):6.1f} ms   max {v.max():6.1f} ms")

    row("camera -> DSP", vis)
    row("DSP -> speaker", out)
    row("motion -> sound", tot)
    p95 = np.percentile(tot, 95)
    print(f"target < 25 ms: {'PASS' if p95 < 25 else 'FAIL'} (p95, excluding camera exposure)")
    print(f"device: {st['device'].get('name')}, xruns {st['xruns']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "bench"
    if cmd == "bench":
        bench()
    elif cmd == "live":
        live(float(sys.argv[2]) if len(sys.argv) > 2 else 20.0)
    else:
        sys.exit(__doc__)
