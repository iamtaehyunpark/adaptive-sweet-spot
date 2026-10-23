"""Low-latency webcam capture.

A background thread reads frames continuously and keeps only the newest one,
so a slow consumer never processes stale, buffered frames.
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np


class FrameGrabber:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: int = 60):
        import cv2

        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(index, backend)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera {index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.reported_fps = float(self._cap.get(cv2.CAP_PROP_FPS))

        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._ts = 0.0
        self._seq = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="FrameGrabber", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            ts = time.perf_counter()
            if not ok:
                time.sleep(0.005)
                continue
            with self._cond:
                self._frame, self._ts = frame, ts
                self._seq += 1
                self._cond.notify_all()

    def read(self, last_seq: int = 0, timeout: float = 1.0) -> tuple[int, float, np.ndarray] | None:
        """Block until a frame newer than ``last_seq`` arrives; returns (seq, timestamp, frame)."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > last_seq or not self._running, timeout):
                return None
            if self._frame is None:
                return None
            return self._seq, self._ts, self._frame

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
