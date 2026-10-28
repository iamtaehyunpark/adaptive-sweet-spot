import math

import numpy as np
import pytest

from vision.room import fit_floor


def synthetic_room(height=0.8, pitch_deg=-20.0, w=320, h=240, f=250.0):
    """Camera ``height`` above a floor, tilted by ``pitch_deg``; a wall 4 m ahead."""
    p = math.radians(pitch_deg)
    # Camera axes in world coords (world: x right, y forward, z up).
    fwd = np.array([0, math.cos(p), math.sin(p)])
    down = np.array([0, math.sin(p), -math.cos(p)])
    right = np.array([1.0, 0, 0])
    u, v = np.meshgrid(np.arange(w) - w / 2, np.arange(h) - h / 2)
    rays = u[..., None] / f * right + v[..., None] / f * down + fwd  # world directions
    cam = np.array([0, 0, height])
    t_floor = np.where(rays[..., 2] < -1e-6, -height / np.minimum(rays[..., 2], -1e-6), np.inf)
    t_wall = np.where(rays[..., 1] > 1e-6, 4.0 / np.maximum(rays[..., 1], 1e-6), np.inf)
    t = np.minimum(t_floor, t_wall)
    world = cam + rays * t[..., None]
    rel = world - cam  # back to camera coords
    return np.stack([rel @ right, rel @ down, rel @ fwd], axis=-1)


@pytest.mark.parametrize("height, pitch", [(0.8, -20.0), (1.2, -35.0), (0.5, -10.0)])
def test_floor_height_and_tilt(height, pitch):
    floor = fit_floor(synthetic_room(height, pitch))
    assert floor is not None
    assert floor.height_m == pytest.approx(height, rel=0.01)
    assert floor.pitch_deg == pytest.approx(pitch, abs=0.5)
    assert floor.roll_deg == pytest.approx(0.0, abs=0.5)


def test_no_floor_in_view():
    assert fit_floor(synthetic_room(0.8, pitch_deg=25.0)) is None or \
        fit_floor(synthetic_room(0.8, pitch_deg=25.0)).inlier_frac < 0.02


def room_with_face(height=1.0, pitch_deg=-15.0, face_cam=(0.2, -0.1, 2.0), w=320, h=240):
    pts = synthetic_room(height, pitch_deg, w, h)
    # paint a face patch (camera frame point) into the middle of the image
    box = np.array([w / 2 - 20, h / 2 - 20, 40, 40])
    pts[int(box[1]):int(box[1] + box[3]), int(box[0]):int(box[0] + box[2])] = face_cam
    return pts, box


def test_room_frame_puts_seat_at_origin_and_camera_in_front():
    from vision.room import build_room

    pts, box = room_with_face(height=1.0, pitch_deg=-15.0, face_cam=(0.0, 0.0, 2.0))
    room = build_room(pts, box, stride=1)
    p = math.radians(-15.0)
    # face on the optical axis 2 m out; camera pitched down 15 deg
    assert room.center_distance_m == pytest.approx(2.0 * math.cos(p), rel=0.01)
    assert room.camera_xy[0] == pytest.approx(0.0, abs=0.01)
    assert room.camera_xy[1] == pytest.approx(2.0 * math.cos(p), rel=0.01)  # camera is toward +y (front)
    assert room.camera_height_m == pytest.approx(1.0, rel=0.01)
    assert room.face_height_m == pytest.approx(1.0 + 2.0 * math.sin(p), abs=0.02)
    assert room.pitch_deg == pytest.approx(-15.0, abs=0.5)


def test_room_x_is_listener_right():
    from vision.room import build_room

    # face 0.3 m to the camera's right = listener's left -> camera appears at listener's right (+x)
    pts, box = room_with_face(height=1.0, pitch_deg=0.0, face_cam=(0.3, 0.0, 2.0))
    room = build_room(pts, box, stride=1)
    assert room.camera_xy[0] == pytest.approx(0.3, abs=0.01)
    assert room.camera_xy[1] == pytest.approx(2.0, abs=0.01)


def test_tape_seat_distance_rescales():
    from vision.room import build_room

    pts, box = room_with_face(height=1.0, pitch_deg=0.0, face_cam=(0.0, 0.0, 2.0))
    room = build_room(pts, box, seat_distance_m=2.5, stride=1)
    assert room.scale == pytest.approx(1.25)
    assert room.center_distance_m == pytest.approx(2.5, rel=0.01)
    assert room.camera_height_m == pytest.approx(1.25, rel=0.01)


def test_render_map_origin_is_seat():
    from vision.room import build_room, render_map

    pts, box = room_with_face(height=1.0, pitch_deg=-10.0, face_cam=(0.0, 0.0, 2.0))
    room = build_room(pts, box, stride=1)
    img, meta = render_map(room, m_per_px=0.05)
    assert img.shape[:2] == (meta["size_px"][1], meta["size_px"][0])
    ou, ov = meta["origin_px"]
    cu = ou + room.camera_xy[0] / 0.05
    cv = ov - room.camera_xy[1] / 0.05
    assert cv < ov  # camera (front) is drawn above the seat
    assert 0 <= cu < img.shape[1] and 0 <= cv < img.shape[0]


def test_flying_pixels_at_depth_edges_are_removed():
    from vision.room import flying_pixel_mask

    # wall at 4 m with a 1 m-away box in the middle, and one smeared pixel in between
    pts = np.zeros((40, 40, 3))
    pts[..., 2] = 4.0
    pts[10:30, 10:30, 2] = 1.0
    pts[20, 30, 2] = 2.5  # flying pixel next to the box edge
    mask = flying_pixel_mask(pts)
    assert mask[20, 30]
    assert not mask[0, 0] and not mask[20, 20]  # flat wall and box interior kept




def test_render_map_with_no_points_near_the_seat():
    from vision.room import RoomModel, render_map

    room = RoomModel(points=np.array([[0.0, 30.0, 0.0]]), height_above_floor=None, camera_xy=np.array([0.0, 2.0]),
                     center_distance_m=2.0, face_depth_m=2.0, face_height_m=None, camera_height_m=None,
                     pitch_deg=None, roll_deg=None, scale=1.0, scale_source="model")
    img, meta = render_map(room)
    assert img.size > 0
