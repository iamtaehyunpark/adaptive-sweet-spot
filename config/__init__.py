"""Configuration loading."""

import json
import math
import tomllib
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CONFIG_DIR / "default.toml"
ROOM_FILE = CONFIG_DIR / "room.json"  # written by tools/room_setup.py


def load_config(path: str | Path | None = None, room: bool | None = None) -> dict:
    """Load a TOML config; defaults to config/local.toml if present, else default.toml.

    Speaker positions placed in the room setup (config/room.json) replace the
    configured ones. ``room`` defaults to True only when ``path`` is not given.
    """
    if path is None:
        local = CONFIG_DIR / "local.toml"
        path = local if local.exists() else DEFAULT_CONFIG
        room = True if room is None else room
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    if room and ROOM_FILE.exists():
        data = json.loads(ROOM_FILE.read_text())
        placed = {s["name"]: s for s in data.get("speakers", [])}
        cam = camera_position(data)
        for spk in cfg["speakers"]:
            if spk["name"] == "FC" and cam is not None:
                spk["position"] = cam  # the camera sits on FC: always where the setup found the camera
            elif spk["name"] in placed:
                spk["position"] = placed[spk["name"]]["position"]
    return cfg


def camera_position(room: dict) -> list[float] | None:
    """Room-frame (x, y, z) of the camera (= the FC speaker) from a room setup, or None."""
    if not room or "camera_xy" not in room:
        return None
    z = 0.0  # level with the face when the floor was not seen
    if room.get("floor_z") is not None and room.get("camera_height_m") is not None:
        z = room["floor_z"] + room["camera_height_m"]
    x, y = room["camera_xy"]
    return [round(float(x), 3), round(float(y), 3), round(float(z), 3)]


def speaker_position(spk: dict) -> tuple[float, float, float]:
    """Room-frame (x, y, z) in meters: placed ``position`` or azimuth/distance/height."""
    if "position" in spk:
        x, y, z = spk["position"]
        return float(x), float(y), float(z)
    az = math.radians(spk["azimuth_deg"])
    return (spk["distance_m"] * math.sin(az), spk["distance_m"] * math.cos(az), spk.get("height_m", 0.0))
