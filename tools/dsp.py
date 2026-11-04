#!/usr/bin/env python3
"""Sweet-spot DSP process: 6-channel audio steered to the listener.

Receives listener positions on the UDP bridge (from tools/app.py), takes commands on
the control port (the app's Sound tab), plays to the 6-channel DAC.

    python tools/dsp.py --list-devices
    python tools/dsp.py --device "USB Audio" --source center --mode tracking
    python tools/dsp.py --source file --file movie_51.wav
    python tools/dsp.py --source input --input BlackHole      # live pass-through

Normally the app starts it (python tools/app.py); run it alone for a headless setup
(e.g. on the QIDK, with the vision side elsewhere).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--device", help="output device (index or name substring; default: config / system)")
    ap.add_argument("--input", help="input device for --source input")
    ap.add_argument("--source", default="off", help="off|identify|center|front|surround|clicks|file|input")
    ap.add_argument("--file", help="sound file (path or name in assets/audio/)")
    ap.add_argument("--mode", choices=["bypass", "static", "tracking"])
    ap.add_argument("--volume", type=float, help="master volume dB (<= 0)")
    ap.add_argument("--no-bridge", action="store_true", help="don't listen for listener positions")
    ap.add_argument("--quiet", action="store_true", help="no status line")
    ap.add_argument("--parent", type=int, help="exit when this process (the app) exits")
    args = ap.parse_args()

    from dsp.player import list_devices

    if args.list_devices:
        for d in list_devices():
            print(f"{d['index']:3d}  out {d['out']:2d}  in {d['in']:2d}  {d['rate']:.0f} Hz  {d['name']} ({d['hostapi']})")
        return 0

    from dsp.control import ControlServer
    from dsp.player import Player
    from interface.bridge import BridgeReceiver

    cfg = load_config()
    dcfg = cfg.setdefault("dsp", {})
    if args.mode:
        dcfg["mode"] = args.mode
    if args.device:
        cfg["audio"]["device"] = args.device
    if args.input:
        dcfg["input_device"] = args.input
    receiver = None
    if not args.no_bridge:
        b = cfg["bridge"]
        try:
            receiver = BridgeReceiver(b["udp_host"], b["udp_port"])
        except OSError as e:
            print(f"bridge port {b['udp_port']} busy ({e}) - is another DSP running?")
            return 1
    player = Player(cfg, receiver)
    if args.volume is not None:
        player.set(volume_db=args.volume)
    player.set_source(args.source, args.file)
    player.restart()
    if player.error:
        print(f"audio: {player.error}")
    else:
        d = player.device_info
        print(f"dsp: {d['name']} {d['channels']} ch @ {d['rate']} Hz, block {d['block']}"
              + (" (stereo downmix: device has < 6 outputs)" if d["downmix"] else ""))
    control = ControlServer(player, port=dcfg.get("control_port", 9751),
                            reload=lambda: load_config()["speakers"])
    print(f"dsp: mode {player.engine.mode}, source {player.source.name}, control udp {dcfg.get('control_port', 9751)}"
          + ("" if receiver is None else f", listening for the listener on udp {cfg['bridge']['udp_port']}"))
    try:
        while True:
            time.sleep(0.5)
            if args.parent and os.getppid() != args.parent:
                break  # the app is gone
            if args.quiet:
                continue
            st = player.status()
            lat = st["latency"]
            delays = " ".join(f"{s['name']} {s['delay_ms']:.2f}" for s in st["speakers"])
            lst = st["listener"]
            m2s = lat["motion_to_sound_ms"]
            print(f"\r{st['mode']:8s} {st['source']:8s} listener {lst[0]:+.2f} {lst[1]:+.2f} {lst[2]:+.2f}  "
                  f"delay ms: {delays}  motion->sound {'-' if m2s is None else f'{m2s:.0f} ms'}  "
                  f"xruns {st['xruns']}   ", end="", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        control.close()
        player.stop()
        if receiver:
            receiver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
