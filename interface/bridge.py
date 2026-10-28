"""Vision -> DSP bridge (Milestone 3.2): listener position messages over UDP.

Message v1: one UTF-8 JSON object per datagram, sent on every detection.

    {
      "v": 1,              # message version
      "seq": 1234,         # increments per message
      "t": 1730000000.123, # sender wall-clock time of the camera frame (s, Unix)
      "tracking": true,    # false = face lost; x/y/z hold the last known position
      "x": 0.12,           # m, + = listener's right of the seat (room frame)
      "y": -0.05,          # m, + = toward the front (FC / camera)
      "z": 0.02,           # m, + = up, relative to the face height at setup
      "yaw_deg": -8.0,     # head yaw, room frame: 0 = facing front, + = turned right
      "confidence": 0.91   # detector score
    }

Room frame and speaker positions: see config.load_config() / config.speaker_position().
Defaults: 127.0.0.1:9750 ([bridge] in config/default.toml).

Receiving (DSP side):

    rx = BridgeReceiver()            # binds udp_host:udp_port from the config
    msg = rx.latest()                # newest ListenerMessage or None (non-blocking)
"""

from __future__ import annotations

import json
import socket
import threading
from dataclasses import asdict, dataclass

VERSION = 1


@dataclass
class ListenerMessage:
    seq: int
    t: float
    tracking: bool
    x: float
    y: float
    z: float
    yaw_deg: float
    confidence: float
    v: int = VERSION

    def to_bytes(self) -> bytes:
        d = asdict(self)
        for k in ("x", "y", "z", "yaw_deg", "confidence"):
            d[k] = round(d[k], 4)
        return json.dumps(d, separators=(",", ":")).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> ListenerMessage:
        d = json.loads(data)
        if d.get("v") != VERSION:
            raise ValueError(f"unsupported message version {d.get('v')}")
        return cls(**d)


class BridgePublisher:
    def __init__(self, host: str = "127.0.0.1", port: int = 9750):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0
        self.sent = 0

    def send(self, t: float, tracking: bool, x: float, y: float, z: float, yaw_deg: float,
             confidence: float) -> ListenerMessage:
        self.seq += 1
        msg = ListenerMessage(self.seq, t, tracking, x, y, z, yaw_deg, confidence)
        try:
            self.sock.sendto(msg.to_bytes(), self.addr)
            self.sent += 1
        except OSError:
            pass  # nobody listening / network down: tracking must not stop
        return msg

    def close(self) -> None:
        self.sock.close()


class BridgeReceiver:
    """Background UDP listener keeping the newest message."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9750):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(0.5)
        self._latest: ListenerMessage | None = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        return self.sock.getsockname()[1]

    def _loop(self) -> None:
        while self._running:
            try:
                data, _ = self.sock.recvfrom(4096)
                msg = ListenerMessage.from_bytes(data)
            except (TimeoutError, socket.timeout):
                continue
            except (OSError, ValueError, TypeError):
                if not self._running:
                    break
                continue
            with self._lock:
                if self._latest is None or msg.seq > self._latest.seq or msg.seq < self._latest.seq - 1000:
                    self._latest = msg  # ignore late, out-of-order datagrams (seq reset = sender restart)

    def latest(self) -> ListenerMessage | None:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._running = False
        self.sock.close()
        self._thread.join(timeout=1.0)
