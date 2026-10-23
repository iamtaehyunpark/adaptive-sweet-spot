import numpy as np

from vision.detector import FaceObservation
from vision.tracker import roi_around, tile_regions


def test_single_tile_is_full_frame():
    assert tile_regions(1280, 720, 1) == [(0, 0, 1280, 720)]


def test_tiles_cover_frame_and_overlap():
    tiles = tile_regions(1280, 720, 3)
    assert len(tiles) == 9
    assert min(t[0] for t in tiles) == 0 and max(t[2] for t in tiles) == 1280
    assert min(t[1] for t in tiles) == 0 and max(t[3] for t in tiles) == 720
    (a, b) = tiles[0], tiles[1]
    assert b[0] < a[2]  # horizontal neighbours overlap


def face_at(x, y, w, frame=(1280, 720)):
    return FaceObservation(bbox=np.array([x, y, w, w], float), keypoints=np.zeros((6, 2)),
                           score=1.0, frame_size=frame)


def test_roi_is_centered_square_of_scaled_face():
    x0, y0, x1, y1 = roi_around(face_at(600, 300, 40), scale=4.0)
    assert x1 - x0 == y1 - y0 == 160
    assert (x0 + x1) / 2 == 620 and (y0 + y1) / 2 == 320


def test_roi_clipped_to_frame():
    x0, y0, x1, y1 = roi_around(face_at(0, 0, 100), scale=4.0)
    assert (x0, y0) == (0, 0) and x1 - x0 == 400
    x0, y0, x1, y1 = roi_around(face_at(1200, 650, 300), scale=4.0)
    assert y1 - y0 == 720 and x1 <= 1280
