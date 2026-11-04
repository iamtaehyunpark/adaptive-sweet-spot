"""DSP control port: JSON commands over local UDP, one reply per request.

    {"cmd": "status"}                                -> DSP state (delays, gains, levels, latency)
    {"cmd": "set", "mode": "tracking", ...}          -> apply settings, reply with the state
    {"cmd": "reload"}                                -> re-read speaker positions (after Save)
    {"cmd": "devices", "refresh": true}              -> audio devices (refresh: re-scan, e.g. a DAC plugged in)

``set`` keys: mode (bypass|static|tracking), source (off|identify|center|front|surround|
clicks|file|input|object), file, start_s (file start time), volume_db, offaxis_eq, level_compensation, predict, virtual_yaw, slew,
manual ([x, y, z] or null = follow the bridge), output, input_device, render,
speakers ([{"name", "position"}] - live placement from the page, before it is saved),
object ([x, y, z] of the sound object), object_signal (bursts|noise|file).

Listener positions themselves arrive on the bridge (interface/bridge.py), not here.
"""

from __future__ import annotations

import json
import socket
import threading

DEFAULT_PORT = 9751
MAX_DATAGRAM = 65000


class ControlServer:
    def __init__(self, player, host: str = "127.0.0.1", port: int = DEFAULT_PORT, reload=None):
        self.player = player
        self.reload = reload  # () -> speakers list
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(0.5)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="DspControl")
        self._thread.start()

    def handle(self, req: dict) -> dict:
        cmd = req.pop("cmd", None)
        req.pop("id", None)
        if cmd == "status":
            return self.player.status()
        if cmd == "set":
            self.player.set(**req)
            return self.player.status()
        if cmd == "reload":
            if self.reload is None:
                raise ValueError("reload not available")
            self.player.reload_speakers(self.reload())
            return self.player.status()
        if cmd == "devices":
            return {"devices": self.player.devices(bool(req.get("refresh")))}
        raise ValueError(f"unknown command {cmd!r}")

    def _loop(self) -> None:
        while self._running:
            try:
                data, addr = self.sock.recvfrom(MAX_DATAGRAM)
            except (TimeoutError, socket.timeout):
                continue
            except OSError:
                break
            req_id = None
            try:
                req = json.loads(data)
                req_id = req.get("id")
                reply = {"ok": True, **self.handle(req)}
            except Exception as e:  # any bad command: report, keep serving
                reply = {"ok": False, "error": str(e)}
            reply["id"] = req_id
            try:
                self.sock.sendto(json.dumps(reply).encode(), addr)
            except OSError:
                pass

    def close(self) -> None:
        self._running = False
        self.sock.close()
        self._thread.join(timeout=1.0)


class DspClient:
    """Talks to a running DSP process (used by the app)."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, timeout: float = 0.3):
        self.addr = (host, port)
        self.timeout = timeout
        self._lock = threading.Lock()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)
        self._id = 0

    def request(self, cmd: str, **kw) -> dict | None:
        """Reply dict, or None if the DSP is not running."""
        with self._lock:
            self._id += 1
            try:
                self.sock.sendto(json.dumps({"cmd": cmd, "id": self._id, **kw}).encode(), self.addr)
                while True:  # skip late replies to earlier (timed-out) requests
                    reply = json.loads(self.sock.recvfrom(MAX_DATAGRAM)[0])
                    if reply.get("id") == self._id:
                        return reply
            except (TimeoutError, socket.timeout, ConnectionRefusedError, OSError):
                return None

    def close(self) -> None:
        self.sock.close()
