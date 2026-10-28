from config import DEFAULT_CONFIG, load_config


def test_default_config_has_six_channels():
    cfg = load_config(DEFAULT_CONFIG)
    channels = sorted(s["channel"] for s in cfg["speakers"])
    assert channels == list(range(cfg["audio"]["channels"]))


def test_room_speakers_override_positions(tmp_path, monkeypatch):
    import json

    import config

    room = tmp_path / "room.json"
    room.write_text(json.dumps({"speakers": [{"name": "FL", "position": [-1.0, 2.0, 0.1]}]}))
    monkeypatch.setattr(config, "ROOM_FILE", room)
    cfg = config.load_config(config.DEFAULT_CONFIG, room=True)
    fl = next(s for s in cfg["speakers"] if s["name"] == "FL")
    fr = next(s for s in cfg["speakers"] if s["name"] == "FR")
    assert config.speaker_position(fl) == (-1.0, 2.0, 0.1)
    assert config.speaker_position(fr)[0] > 0  # untouched: from azimuth/distance
