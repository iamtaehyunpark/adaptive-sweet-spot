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


from vision.camera import resolve_camera_index  # noqa: E402

BUILTIN = {"name": "FaceTime HD Camera", "type": "AVCaptureDeviceTypeBuiltInWideAngleCamera"}
IPHONE = {"name": "T's iPhone Camera", "type": "AVCaptureDeviceTypeContinuityCamera"}


def test_saved_camera_found_by_name_when_indices_shift():
    saved = {"index": 0, "name": "FaceTime HD Camera"}
    assert resolve_camera_index(saved, [IPHONE, BUILTIN]) == 1  # iPhone took index 0
    assert resolve_camera_index({"index": 1, "name": IPHONE["name"]}, [IPHONE, BUILTIN]) == 0


def test_no_saved_choice_prefers_builtin_over_phone():
    assert resolve_camera_index({"index": 0}, [IPHONE, BUILTIN]) == 1


def test_missing_saved_camera_falls_back_to_builtin():
    assert resolve_camera_index({"index": 0, "name": IPHONE["name"]}, [BUILTIN]) == 0


def test_network_url_and_unknown_platform_unchanged():
    assert resolve_camera_index({"index": "http://phone:8080/video"}, [BUILTIN]) == "http://phone:8080/video"
    assert resolve_camera_index({"index": 2}, None) == 2
