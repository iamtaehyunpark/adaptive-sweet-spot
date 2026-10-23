"""MediaPipe BlazeFace wrapper (Tasks API).

Keypoint order (MediaPipe face detector, *subject's* left/right):
    0 right eye, 1 left eye, 2 nose tip, 3 mouth center,
    4 right ear tragion, 5 left ear tragion
For a subject facing the camera, their right eye/ear appear on the image left.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RIGHT_EYE, LEFT_EYE, NOSE, MOUTH, RIGHT_EAR, LEFT_EAR = range(6)
KEYPOINT_NAMES = ("right_eye", "left_eye", "nose", "mouth", "right_ear", "left_ear")


@dataclass
class FaceObservation:
    """One detected face, in pixel coordinates of the source frame."""

    bbox: np.ndarray        # (x, y, w, h) px, top-left origin
    keypoints: np.ndarray   # (6, 2) px
    score: float
    frame_size: tuple[int, int]  # (width, height)
    timestamp_s: float = 0.0     # capture time (perf_counter clock)
    extra: dict = field(default_factory=dict)

    @property
    def eye_mid(self) -> np.ndarray:
        return 0.5 * (self.keypoints[RIGHT_EYE] + self.keypoints[LEFT_EYE])

    @property
    def ear_mid(self) -> np.ndarray:
        return 0.5 * (self.keypoints[RIGHT_EAR] + self.keypoints[LEFT_EAR])

    @property
    def ipd_px(self) -> float:
        return float(np.linalg.norm(self.keypoints[LEFT_EYE] - self.keypoints[RIGHT_EYE]))

    @property
    def area(self) -> float:
        return float(self.bbox[2] * self.bbox[3])


class BlazeFaceDetector:
    """Runs BlazeFace on BGR frames (or a region of one).

    Works with both blaze_face_short_range (faces within ~1 m at 720p/70 deg)
    and blaze_face_full_range (~2 m). Use vision.tracker for longer distances.
    """

    def __init__(self, model_path: str | Path, min_confidence: float = 0.5,
                 min_suppression: float = 0.3):
        from mediapipe.tasks.python import BaseOptions, vision

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} not found; run tools/fetch_model.py")
        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=min_confidence,
            min_suppression_threshold=min_suppression,
        )
        self._detector = vision.FaceDetector.create_from_options(options)
        self.last_infer_ms = 0.0

    def detect(self, frame_bgr: np.ndarray, timestamp_s: float,
               region: tuple[int, int, int, int] | None = None) -> list[FaceObservation]:
        """Detect faces in the frame, or only in ``region`` = (x0, y0, x1, y1).

        Results are always in full-frame pixel coordinates.
        """
        import cv2
        import mediapipe as mp

        fh, fw = frame_bgr.shape[:2]
        x0, y0, x1, y1 = region if region is not None else (0, 0, fw, fh)
        crop = frame_bgr[y0:y1, x0:x1]
        h, w = crop.shape[:2]

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        t0 = time.perf_counter()
        result = self._detector.detect(image)
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0

        faces = []
        for det in result.detections:
            bb = det.bounding_box
            kps = np.array([[kp.x * w + x0, kp.y * h + y0] for kp in (det.keypoints or [])])
            if kps.shape != (6, 2):
                continue
            score = det.categories[0].score if det.categories else 0.0
            faces.append(FaceObservation(
                bbox=np.array([bb.origin_x + x0, bb.origin_y + y0, bb.width, bb.height], float),
                keypoints=kps, score=float(score), frame_size=(fw, fh),
                timestamp_s=timestamp_s,
            ))
        return faces

    def close(self) -> None:
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def primary_face(faces: list[FaceObservation]) -> FaceObservation | None:
    """The listener is assumed to be the largest (closest) face."""
    return max(faces, key=lambda f: f.area, default=None)
