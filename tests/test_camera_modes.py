from vision.camera import setup_mode, widest_mode


def test_widest_prefers_full_sensor_4x3():
    modes = [(3840, 2160), (1920, 1440), (1920, 1080), (1280, 720), (640, 480)]
    assert widest_mode(modes) == (1920, 1440)


def test_widest_falls_back_to_largest_when_only_16x9():
    assert widest_mode([(1920, 1080), (1280, 720)]) == (1920, 1080)


def test_widest_none_without_modes():
    assert widest_mode([]) is None


def test_setup_mode_keeps_tracking_aspect():
    modes = [(3840, 2160), (1920, 1440), (1920, 1080), (1280, 960), (1280, 720)]
    assert setup_mode(modes, (1280, 960)) == (1920, 1440)
    assert setup_mode(modes, (1280, 720)) == (1920, 1080)  # capped at 1920 wide


def test_setup_mode_without_probe():
    assert setup_mode(None, (1280, 720)) == (1920, 1080)
    assert setup_mode(None, (1280, 960)) == (1280, 960)
