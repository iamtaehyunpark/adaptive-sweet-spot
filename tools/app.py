#!/usr/bin/env python3
"""Sweet Spot Live: everything in one browser page.

* live face tracking -> listener position in the room frame, shown in the 3D scan
  and a top view, with the camera preview and live stats
* room setup from the page (Setup tab): countdown, 1080p capture, MoGe-2 depth,
  calibration and 3D scan - tracking keeps running
* speaker placement in 3D (Speakers tab) -> config/room.json (read by the DSP config)
* every detection streamed to the DSP over UDP (interface/bridge.py, message v1)
* the sweet-spot DSP (tools/dsp.py) started as its own process (audio never shares a
  process with vision) and driven from the Sound tab

    python tools/app.py                 # camera from config, opens the browser
    python tools/app.py --camera 1
    python tools/app.py --no-camera     # only the 3D room / speaker placement (+ manual DSP)
    python tools/app.py --no-stream     # don't send UDP bridge messages
    python tools/app.py --no-dsp        # don't start the DSP (e.g. it runs elsewhere)
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402
from interface.app import DspLink, LiveTracker, serve  # noqa: E402


def _interrupt(*_):
    raise KeyboardInterrupt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, help="camera index (default: config)")
    ap.add_argument("--port", type=int, default=8765, help="local web port")
    ap.add_argument("--no-camera", action="store_true", help="no tracking, only the room / speakers")
    ap.add_argument("--no-stream", action="store_true", help="don't publish on the UDP bridge")
    ap.add_argument("--no-browser", action="store_true", help="don't open the browser")
    ap.add_argument("--no-dsp", action="store_true", help="don't start the DSP process")
    ap.add_argument("--audio-device", help="DSP output device (index or name substring)")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _interrupt)  # `kill` shuts down like Ctrl-C (stops the DSP too)
    cfg = load_config()
    dsp_proc = None
    if not args.no_dsp:
        cmd = [sys.executable, str(ROOT / "tools" / "dsp.py"), "--quiet", "--parent", str(os.getpid())]
        if args.audio_device:
            cmd += ["--device", args.audio_device]
        dsp_proc = subprocess.Popen(cmd)
    dsp = DspLink(cfg)
    dsp.start()
    tracker = None
    if not args.no_camera:
        tracker = LiveTracker(cfg, args.camera, publish=not args.no_stream)
        tracker.start()
    try:
        serve(cfg, tracker, args.port, open_browser=not args.no_browser, dsp=dsp)
    finally:
        dsp.running = False
        if tracker:
            tracker.stop()
            tracker.join(timeout=2.0)
        if dsp_proc:
            dsp_proc.terminate()
            dsp_proc.wait(timeout=3.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
