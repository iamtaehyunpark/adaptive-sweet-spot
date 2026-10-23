"""Webcam capture."""

from __future__ import annotations

import time

import numpy as np


class Camera:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: int = 60):
        import cv2

        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera {index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.reported_fps = float(self._cap.get(cv2.CAP_PROP_FPS))

    def read(self) -> tuple[float, np.ndarray] | None:
        """Returns (timestamp, frame) or None if the read failed."""
        ok, frame = self._cap.read()
        if not ok:
            return None
        return time.perf_counter(), frame

    def close(self) -> None:
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
