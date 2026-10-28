#!/usr/bin/env python3
"""Sweet Spot Live: everything in one browser page.

* live face tracking -> listener position in the room frame, shown in the 3D scan
  and a top view, with the camera preview and live stats
* room setup from the page (Setup tab): countdown, 1080p capture, MoGe-2 depth,
  calibration and 3D scan - tracking keeps running
* speaker placement in 3D (Speakers tab) -> config/room.json (read by the DSP config)
* every detection streamed to the DSP over UDP (interface/bridge.py, message v1)

    python tools/app.py                 # camera from config, opens the browser
    python tools/app.py --camera 1
    python tools/app.py --no-camera     # only the 3D room / speaker placement
    python tools/app.py --no-stream     # don't send UDP bridge messages
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402
from interface.app import LiveTracker, serve  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, help="camera index (default: config)")
    ap.add_argument("--port", type=int, default=8765, help="local web port")
    ap.add_argument("--no-camera", action="store_true", help="no tracking, only the room / speakers")
    ap.add_argument("--no-stream", action="store_true", help="don't publish on the UDP bridge")
    ap.add_argument("--no-browser", action="store_true", help="don't open the browser")
    args = ap.parse_args()

    cfg = load_config()
    tracker = None
    if not args.no_camera:
        tracker = LiveTracker(cfg, args.camera, publish=not args.no_stream)
        tracker.start()
    try:
        serve(cfg, tracker, args.port, open_browser=not args.no_browser)
    finally:
        if tracker:
            tracker.stop()
            tracker.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
