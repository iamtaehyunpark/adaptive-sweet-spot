"""Frame sources for the tools: webcam (switchable) or video file."""

from __future__ import annotations

import time

from .camera import Camera, CameraInfo, list_cameras


class CameraSource:
    """Webcam input; ``switch()`` cycles through all detected cameras."""

    is_video = False

    def __init__(self, cameras: list[CameraInfo], start_index: int, width: int, height: int, fps: int):
        self.cameras = cameras
        self.settings = (width, height, fps)
        self.pos = next((i for i, c in enumerate(cameras) if c.index == start_index), 0)
        self.cam = None
        self._open()

    def _open(self) -> None:
        if self.cam is not None:
            self.cam.close()
        info = self.cameras[self.pos]
        self.cam = Camera(info.index, *self.settings)
        print(f"using camera {info.index} ({info.name}): "
              f"{self.cam.width}x{self.cam.height} @ {self.cam.reported_fps:.1f} fps (reported)")

    def switch(self) -> None:
        self.pos = (self.pos + 1) % len(self.cameras)
        self._open()

    @property
    def label(self) -> str:
        info = self.cameras[self.pos]
        return f"camera {self.pos + 1}/{len(self.cameras)}: {info.name} (index {info.index})  [c: switch]"

    def read(self):
        return self.cam.read()

    def close(self) -> None:
        self.cam.close()


class VideoSource:
    is_video = True

    def __init__(self, path: str):
        import cv2

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        self.label = f"video: {path}"

    def switch(self) -> None:
        pass

    def read(self):
        ok, frame = self.cap.read()
        return (time.perf_counter(), frame) if ok else None

    def close(self) -> None:
        self.cap.release()


def open_source(args, vcfg: dict):
    """Build a source from tool args (--video / --camera / --width / --height / --fps)."""
    if args.video:
        return VideoSource(args.video)

    print("looking for cameras...")
    cameras = list_cameras()
    for c in cameras:
        print(f"  index {c.index}: {c.name} ({c.width}x{c.height})")
    start = args.camera if args.camera is not None else vcfg["camera_index"]
    if args.camera is not None and all(c.index != start for c in cameras):
        cameras.insert(0, CameraInfo(start, f"camera {start}", 0, 0))
    if not cameras:
        raise SystemExit("no cameras found")
    return CameraSource(cameras, start, args.width or vcfg["width"], args.height or vcfg["height"],
                        args.fps or vcfg["fps"])
