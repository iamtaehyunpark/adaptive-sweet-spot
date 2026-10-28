"""Webcam capture, camera discovery and (where the driver allows) camera controls.

What can be changed depends on the OS backend and the camera: Linux/V4L2 webcams
(e.g. on the QIDK) usually expose zoom/focus/exposure; macOS (AVFoundation) mostly
only resolution and fps. Phone cameras (Continuity Camera, DroidCam, Camo, ...)
pick their lens (e.g. ultra-wide) in the phone app, not through the driver.
Every capability is therefore probed at runtime.

A camera is an index (local device) or a URL (network camera, e.g. a phone app
serving MJPEG/RTSP over Wi-Fi) - the portable way to use a phone on any OS.
"""

from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NETWORK_OPEN_TIMEOUT_MS = 4000

# Modes tried when probing (the driver snaps to what it supports).
PROBE_MODES = [(640, 480), (1280, 720), (1280, 960), (1600, 1200), (1920, 1080), (1920, 1440),
               (2560, 1440), (2592, 1944), (3840, 2160), (4032, 3024)]
CONTROL_PROPS = {  # name -> cv2 property name
    "zoom": "CAP_PROP_ZOOM",
    "focus": "CAP_PROP_FOCUS",
    "autofocus": "CAP_PROP_AUTOFOCUS",
    "exposure": "CAP_PROP_EXPOSURE",
    "auto_exposure": "CAP_PROP_AUTO_EXPOSURE",
}


@dataclass
class CameraInfo:
    index: int | str  # device index or network URL
    name: str
    width: int
    height: int


def mac_devices() -> list[dict] | None:
    """macOS video devices in OpenCV's index order (AVFoundation), or None elsewhere.

    Needs pyobjc-framework-AVFoundation (optional). OpenCV's AVFoundation backend
    indexes devicesWithMediaType:video, so the order matches its camera indices.
    """
    if sys.platform != "darwin":
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from AVFoundation import AVCaptureDevice, AVMediaTypeVideo

            devices = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)
    except Exception:
        return None
    return [{"name": str(d.localizedName()), "id": str(d.uniqueID()), "model": str(d.modelID()),
             "type": str(d.deviceType())} for d in devices]


def _device_name(index: int, mac: list[dict] | None = None) -> str:
    # Linux exposes V4L2 device names in sysfs; macOS via AVFoundation; else the index.
    sysfs = Path(f"/sys/class/video4linux/video{index}/name")
    if sysfs.exists():
        return sysfs.read_text().strip()
    mac = mac if mac is not None else mac_devices()
    if mac and 0 <= index < len(mac):
        return mac[index]["name"]
    return f"camera {index}"


def is_builtin(device: dict) -> bool:
    return "BuiltIn" in device.get("type", "")


def resolve_camera_index(saved: dict, devices: list[dict] | None) -> int | str:
    """Camera index to open for the saved choice.

    Indices shift when devices come and go (e.g. an iPhone appearing as Continuity
    Camera takes index 0), so the choice is matched by name. With no saved name,
    or if that camera is gone, a built-in camera is preferred so a phone is never
    picked up unasked.
    """
    index = saved.get("index", 0)
    if isinstance(index, str) or not devices:
        return index
    name = saved.get("name")
    if name:
        for i, d in enumerate(devices):
            if d["name"] == name:
                return i
    for i, d in enumerate(devices):
        if is_builtin(d):
            return i
    return index if 0 <= index < len(devices) else 0


def open_capture(source: int | str):
    """cv2.VideoCapture for a device index or a network URL (with an open timeout)."""
    import cv2

    if isinstance(source, str):
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG,
                                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, NETWORK_OPEN_TIMEOUT_MS,
                                 cv2.CAP_PROP_READ_TIMEOUT_MSEC, NETWORK_OPEN_TIMEOUT_MS])
    return cv2.VideoCapture(source)


def snapshot(source: int | str):
    """(CameraInfo, frame) for one source, or None if it delivers no frames."""
    cap = open_capture(source)
    try:
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        for _ in range(5):  # some cameras deliver black frames right after opening
            if ok and frame.mean() > 8:
                break
            ok, frame = cap.read()
        if not ok:
            return None
        name = source if isinstance(source, str) else _device_name(source)
        return CameraInfo(source, name, frame.shape[1], frame.shape[0]), frame
    finally:
        cap.release()


def list_cameras(max_index: int = 6, skip: set[int] | None = None, snapshots: bool = False):
    """Probe camera indices 0..max_index-1 and return the ones that deliver frames.

    With ``snapshots`` returns (CameraInfo, frame) pairs, to show the user what each
    camera sees (macOS gives no device names through OpenCV). Indices in ``skip``
    (e.g. the camera already in use) are not opened.
    """
    import cv2

    mac = mac_devices()
    found = []
    for i in range(max_index):
        if skip and i in skip:
            continue
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, frame = cap.read()
            for _ in range(5):  # some cameras deliver black frames right after opening
                if ok and frame.mean() > 8:
                    break
                ok, frame = cap.read()
            if ok:
                info = CameraInfo(i, _device_name(i, mac), frame.shape[1], frame.shape[0])
                found.append((info, frame) if snapshots else info)
        cap.release()
    return found


class Camera:
    def __init__(self, index: int | str = 0, width: int = 1280, height: int = 720, fps: int = 60,
                 controls: dict[str, float] | None = None):
        import cv2

        self.index = index
        self._cap = open_capture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera {index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        for name, value in (controls or {}).items():
            self.set_control(name, value)

        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.reported_fps = float(self._cap.get(cv2.CAP_PROP_FPS))

    def read(self) -> tuple[float, np.ndarray] | None:
        """Returns (timestamp, frame) or None if the read failed."""
        ok, frame = self._cap.read()
        if not ok:
            return None
        return time.perf_counter(), frame

    # ------------------------------------------------------------ controls
    def _prop(self, name: str):
        import cv2

        return getattr(cv2, CONTROL_PROPS[name])

    def get_control(self, name: str) -> float:
        return float(self._cap.get(self._prop(name)))

    def set_control(self, name: str, value: float) -> float:
        """Set a control; returns the value the driver reports afterwards."""
        self._cap.set(self._prop(name), float(value))
        return self.get_control(name)

    def probe_controls(self) -> dict[str, dict]:
        """Controls the driver actually changes, with their range (min/max by clamping)."""
        out = {}
        for name in CONTROL_PROPS:
            prop = self._prop(name)
            orig = self._cap.get(prop)
            if not self._cap.set(prop, orig):
                continue
            lo = self.set_control(name, -1e6)
            hi = self.set_control(name, 1e6)
            self._cap.set(prop, orig)
            if hi > lo:
                out[name] = {"min": lo, "max": hi, "value": float(orig)}
        return out

    def probe_modes(self) -> list[tuple[int, int]]:
        """Resolutions the camera delivers (read from real frames), largest first.

        Changes the capture mode: reopen the camera with the wanted mode afterwards.
        """
        import cv2

        modes = set()
        for w, h in PROBE_MODES:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            frame = None
            for _ in range(3):
                ok, f = self._cap.read()
                if ok:
                    frame = f
            if frame is not None:
                modes.add((frame.shape[1], frame.shape[0]))
        return sorted(modes, key=lambda m: m[0] * m[1], reverse=True)

    def close(self) -> None:
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def aspect(mode: tuple[int, int]) -> float:
    return mode[0] / mode[1]


def widest_mode(modes: list[tuple[int, int]], max_width: int = 1920) -> tuple[int, int] | None:
    """Mode most likely to show the widest view: full-sensor 4:3 beats a 16:9 crop.

    Heuristic (most webcam and phone sensors are 4:3 and crop for 16:9): prefer the
    tallest aspect, then the largest mode up to ``max_width`` (tracking cost).
    """
    usable = [m for m in modes if m[0] <= max_width] or modes
    if not usable:
        return None
    tallest = min(aspect(m) for m in usable)
    candidates = [m for m in usable if abs(aspect(m) - tallest) < 0.02]
    return max(candidates, key=lambda m: m[0] * m[1])


def setup_mode(modes: list[tuple[int, int]] | None, tracking: tuple[int, int],
               max_width: int = 1920) -> tuple[int, int]:
    """Largest mode (up to ``max_width``) with the tracking aspect: the setup capture
    must show the same view as tracking. The depth model gains little beyond ~1080p."""
    same = [m for m in (modes or []) if abs(aspect(m) - aspect(tracking)) < 0.01 and m[0] <= max_width]
    if same:
        return max(same, key=lambda m: m[0] * m[1])
    if abs(aspect(tracking) - 16 / 9) < 0.01 and tracking[0] < 1920:
        return (1920, 1080)
    return tracking
