#!/usr/bin/env python3
"""Download BlazeFace model weights into assets/models/."""

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "blaze_face_short_range.tflite": (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
    ),
}


def main() -> int:
    out_dir = ROOT / "assets" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        dest = out_dir / name
        if dest.exists():
            print(f"exists  {dest.relative_to(ROOT)}")
            continue
        print(f"fetch   {url}")
        urllib.request.urlretrieve(url, dest)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        print(f"saved   {dest.relative_to(ROOT)}  sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
