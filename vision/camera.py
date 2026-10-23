"""Webcam capture."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CameraInfo:
    index: int
    name: str
    width: int
    height: int


def _device_name(index: int) -> str:
    # Linux exposes V4L2 device names in sysfs; elsewhere fall back to the index.
    sysfs = Path(f"/sys/class/video4linux/video{index}/name")
    if sysfs.exists():
        return sysfs.read_text().strip()
    return f"camera {index}"


def list_cameras(max_index: int = 6) -> list[CameraInfo]:
    """Probe camera indices 0..max_index-1 and return the ones that deliver frames."""
    import cv2

    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                found.append(CameraInfo(i, _device_name(i), frame.shape[1], frame.shape[0]))
        cap.release()
    return found


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
