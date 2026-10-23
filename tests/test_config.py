from config import DEFAULT_CONFIG, load_config


def test_default_config_has_six_channels():
    cfg = load_config(DEFAULT_CONFIG)
    channels = sorted(s["channel"] for s in cfg["speakers"])
    assert channels == list(range(cfg["audio"]["channels"]))
