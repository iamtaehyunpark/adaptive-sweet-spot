"""Timed still capture from the seat (used by the room setup and the depth bench)."""

from __future__ import annotations

import sys
from pathlib import Path

from .camera import Camera

COUNTDOWN_S = 5.0
CAPTURE_S = 2.0


def capture_frames(out_dir: Path, camera_index: int, width: int = 1920, height: int = 1080,
                   countdown_s: float = COUNTDOWN_S, duration_s: float = CAPTURE_S,
                   title: str = "capture") -> int:
    """Live preview; space starts a countdown, then frames are saved as out_dir/NNN.png.

    Returns the number of frames saved (0 if the user quit).
    """
    import cv2

    from .overlay import draw_hud

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    with Camera(camera_index, width, height, 30) as cam:
        print(f"camera {camera_index}: {cam.width}x{cam.height}")
        start = None
        while True:
            item = cam.read()
            if item is None:
                print("camera read failed", file=sys.stderr)
                break
            ts, frame = item
            if start is None:
                status = "Press space, then sit at your seat facing the camera. q to quit."
            elif ts - start < countdown_s:
                status = f"Sit at your seat, hold still... {countdown_s - (ts - start):.0f}"
            elif ts - start < countdown_s + duration_s:
                cv2.imwrite(str(out_dir / f"{saved:03d}.png"), frame)
                saved += 1
                status = f"CAPTURING ({saved})"
            else:
                break
            preview = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            draw_hud(preview, [f"{cam.width}x{cam.height}", status])
            cv2.imshow(title, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                saved = 0
                break
            if key == ord(" ") and start is None:
                start = ts
    cv2.destroyAllWindows()
    return saved
