"""Debug drawing for detections."""

from __future__ import annotations

import numpy as np

from .detector import FaceObservation

_KP_COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]


def draw_face(frame: np.ndarray, face: FaceObservation, primary: bool = True) -> None:
    import cv2

    x, y, w, h = face.bbox.astype(int)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0) if primary else (128, 128, 128), 2)
    for (kx, ky), color in zip(face.keypoints.astype(int), _KP_COLORS):
        cv2.circle(frame, (kx, ky), 4, color, -1)
    cv2.putText(frame, f"{face.score:.2f}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def draw_hud(frame: np.ndarray, lines: list[str]) -> None:
    import cv2

    font, scale, line_h, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 20, 8
    width = max((cv2.getTextSize(t, font, scale, 1)[0][0] for t in lines), default=0)
    cv2.rectangle(frame, (10 - pad, 10 - pad), (10 + width + pad, 10 + line_h * len(lines) + pad // 2),
                  (255, 255, 255), -1)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 25 + line_h * i), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
