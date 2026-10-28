import numpy as np
import pytest

from interface.placement import GUIDE, centering_analysis, suggest_layout

CFG = [{"name": n, "channel": i, **({"lfe": True} if n == "LFE" else {})}
       for i, n in enumerate(["FL", "FR", "FC", "LFE", "SL", "SR"])]


def box_room(x0, x1, y0, y1, floor_z=-1.2, height=2.5, step=0.05):
    """Wall points of a rectangular room (room frame: seat at the origin)."""
    zs = np.arange(floor_z, floor_z + height, step)
    walls = []
    for x in np.arange(x0, x1, step):
        walls += [(x, y0, z) for z in zs] + [(x, y1, z) for z in zs]
    for y in np.arange(y0, y1, step):
        walls += [(x0, y, z) for z in zs] + [(x1, y, z) for z in zs]
    return np.array(walls)


def room_meta(cam_y=2.0, floor_z=-1.2):
    return {"camera_xy": [0.0, cam_y], "floor_z": floor_z, "camera_height_m": 1.0}


def by_name(result):
    return {s["name"]: s for s in result["speakers"]}


def test_big_room_gets_ideal_itu_layout():
    pts = box_room(-4, 4, -4, 2.6)
    res = by_name(suggest_layout(room_meta(), pts, CFG))
    for name in ("FL", "FR", "SL", "SR"):
        assert res[name]["azimuth_deg"] == pytest.approx(GUIDE[name][0], abs=1.0)
        assert res[name]["distance_m"] == pytest.approx(np.hypot(2.0, GUIDE[name][2]), abs=0.06)
    assert res["FC"]["position"][:2] == [0.0, 2.0]
    assert res["LFE"]["position"][2] == pytest.approx(-1.0)  # on the floor


def test_close_side_walls_pull_surrounds_in_but_stay_symmetric():
    pts = box_room(-1.4, 1.4, -2.5, 2.6)  # narrow room: walls 1.4 m to each side
    res = by_name(suggest_layout(room_meta(), pts, CFG))
    sl, sr = res["SL"], res["SR"]
    assert sl["azimuth_deg"] == pytest.approx(-sr["azimuth_deg"])
    for s in (sl, sr):
        assert 90 <= abs(s["azimuth_deg"]) <= 120
        assert abs(s["position"][0]) <= 1.4 - 0.12  # inside the side wall with clearance
    assert sl["distance_m"] == pytest.approx(sr["distance_m"], abs=0.06)


def test_unobserved_directions_are_flagged():
    pts = box_room(-4, 4, -4, 2.6)
    pts = pts[pts[:, 1] < 0.5]  # remove everything in front (like the area beside the camera)
    res = by_name(suggest_layout(room_meta(), pts, CFG))
    assert not res["FL"]["verified"] and not res["FR"]["verified"]
    assert res["SL"]["verified"] and res["SR"]["verified"]


def test_blocked_line_of_sight_is_avoided_or_reported():
    pts = box_room(-4, 4, -4, 2.6)
    # a tall cabinet between the seat and the ideal SR spot (110 deg, 2 m)
    a = np.radians(110)
    cx, cy = 1.1 * np.sin(a), 1.1 * np.cos(a)
    g = np.mgrid[-0.15:0.15:0.03, -0.15:0.15:0.03, -1.2:0.8:0.05].reshape(3, -1).T
    pts = np.vstack([pts, g + [cx, cy, 0]])
    sr = by_name(suggest_layout(room_meta(), pts, CFG))["SR"]
    blocked = any("line of sight" in n for n in sr["notes"])
    assert blocked or abs(sr["azimuth_deg"] - 110) >= 3


def test_centering_analysis_equal_distances_need_no_delay():
    spk = [{"name": n, "lfe": False, "position": [2 * np.sin(np.radians(a)), 2 * np.cos(np.radians(a)), 0]}
           for n, a in (("FL", -30), ("FR", 30), ("FC", 0))]
    an = centering_analysis(spk)
    assert an["max_delay_ms"] == pytest.approx(0, abs=1e-6)
    spk[0]["position"] = [-0.5, 0.866, 0]  # FL at 1 m
    an = centering_analysis(spk)
    assert an["per_speaker"]["FL"]["delay_ms"] == pytest.approx(1000 / 343, rel=1e-2)
    assert an["per_speaker"]["FL"]["level_db"] == pytest.approx(-6.02, abs=0.05)


def test_low_furniture_next_to_the_seat_does_not_block_placement():
    pts = box_room(-4, 4, -4, 2.6)
    # sofa armrests / chair 0.55 m to both sides, 0.6 m tall
    arm = np.mgrid[-0.1:0.1:0.03, -0.6:0.3:0.03, -1.2:-0.6:0.05].reshape(3, -1).T
    pts = np.vstack([pts, arm + [0.6, 0, 0], arm + [-0.6, 0, 0]])
    res = by_name(suggest_layout(room_meta(), pts, CFG))
    for name in ("FL", "FR", "SL", "SR"):
        assert name in res
        assert res[name]["distance_m"] > 1.5


def test_result_is_json_serializable():
    import json

    json.dumps(suggest_layout(room_meta(), box_room(-4, 4, -4, 2.6), CFG))
