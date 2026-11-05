import math

import numpy as np
import pytest

from dsp.delay import BASE, DelayBank
from dsp.engine import SteeringEngine
from dsp.filters import ShelfBank
from dsp.geometry import Layout, clamp_listener, steering
from dsp.listener import ListenerFilter, OneEuro

FS = 48000
C = 343.0
SPEAKERS = [
    {"name": "FL", "channel": 0, "azimuth_deg": -30, "distance_m": 2.0, "height_m": 0.0},
    {"name": "FR", "channel": 1, "azimuth_deg": 30, "distance_m": 2.0, "height_m": 0.0},
    {"name": "FC", "channel": 2, "azimuth_deg": 0, "distance_m": 2.2, "height_m": 0.0},
    {"name": "LFE", "channel": 3, "azimuth_deg": 20, "distance_m": 2.2, "height_m": -0.8, "lfe": True},
    {"name": "SL", "channel": 4, "azimuth_deg": -110, "distance_m": 1.5, "height_m": 0.0},
    {"name": "SR", "channel": 5, "azimuth_deg": 110, "distance_m": 1.8, "height_m": 0.0},
]
DCFG = {"slew": 0.002, "offaxis_eq": False, "max_delay_s": 0.04}


def arrivals(engine, listener, n=FS // 2):
    """Play one (smooth, band-limited) pulse on every channel, propagate it through the
    air to ``listener``: returns (arrival time in samples, amplitude) per speaker."""
    t = np.arange(n)
    pulse = np.exp(-0.5 * ((t - 200) / 4.0) ** 2).astype(np.float32)
    x = np.repeat(pulse[:, None], 6, axis=1)
    y = np.concatenate([engine.process(x[i:i + 128]) for i in range(0, n, 128)])
    out = []
    for i, ch in enumerate(engine.layout.channels):
        d = np.linalg.norm(np.asarray(listener) - engine.layout.positions[i])
        sig = y[:, ch]
        centroid = float((t * sig).sum() / sig.sum())  # sub-sample pulse position
        out.append((centroid + d / C * FS, float(sig.sum()) / d))
    return np.array(out)


def settled_engine(listener, mode="tracking"):
    eng = SteeringEngine(SPEAKERS, FS, DCFG, C)
    eng.set_mode(mode)
    eng.set_listener(listener)
    eng._update_targets()
    eng.delays.jump()
    eng.gain = eng._target_gain.copy()
    return eng


@pytest.mark.parametrize("listener", [(0, 0, 0), (0.3, 0, 0), (-0.4, 0.3, 0.1), (0.2, -0.5, 0)])
def test_sound_from_all_speakers_arrives_together_and_equally_loud(listener):
    eng = settled_engine(listener)
    arr = arrivals(eng, listener)
    t, amp = arr[:, 0], arr[:, 1]
    assert t.max() - t.min() < 0.05  # samples (1 us): all channels aligned at the listener: all channels aligned at the listener
    mains = ~eng.layout.lfe
    assert 20 * np.log10(amp[mains].max() / amp[mains].min()) < 0.1  # dB


def test_without_steering_an_offset_listener_is_misaligned():
    listener = (0.3, 0, 0)
    t = arrivals(settled_engine(listener, "bypass"), listener)[:, 0]
    assert t.max() - t.min() > 10  # samples


def test_arrival_time_is_constant_while_the_listener_moves():
    # fixed reference distance: no Doppler for the listener
    a = arrivals(settled_engine((0, 0, 0)), (0, 0, 0))[:, 0].mean()
    b = arrivals(settled_engine((0.5, 0.4, 0)), (0.5, 0.4, 0))[:, 0].mean()
    assert abs(a - b) < 0.05


def test_fractional_delay_is_accurate():
    bank = DelayBank(1, 0.01, FS)
    bank.set_target(np.array([10.37 / FS]))
    bank.jump()
    f = 1000.0
    t = np.arange(4096)
    y = bank.process(np.sin(2 * math.pi * f * t / FS)[:, None].astype(np.float32), 0.002)[:, 0]
    expected = np.sin(2 * math.pi * f * (t - 10.37 - BASE) / FS)
    assert np.abs(y[200:] - expected[200:]).max() < 1e-3


def test_delay_change_is_slew_limited_and_click_free():
    bank = DelayBank(1, 0.02, FS)
    bank.set_target(np.array([0.010]))  # 10 ms jump requested
    x = np.sin(2 * math.pi * 440 * np.arange(FS) / FS).astype(np.float32)[:, None]
    y = np.concatenate([bank.process(x[i:i + 128], 0.002) for i in range(0, FS, 128)])[:, 0]
    assert bank.current_s[0] == pytest.approx(0.002, rel=0.01)  # 1 s at 0.002 s/s
    # no discontinuities: sample-to-sample change stays within a 440 Hz sine's slope
    assert np.abs(np.diff(y)).max() < 2 * math.pi * 440 / FS * 1.01


def test_level_compensation_never_boosts():
    st = steering(Layout.from_config(SPEAKERS), (0.4, 0.2, 0))
    assert (st.gain_db <= 1e-9).all()


def test_offaxis_shelf_grows_with_angle():
    lay = Layout.from_config(SPEAKERS)
    on = steering(lay, (0, 0, 0), offaxis_max_db=3.0)
    off = steering(lay, (0.6, 0, 0), offaxis_max_db=3.0)
    assert np.allclose(on.shelf_db, 0, atol=1e-6)
    assert off.shelf_db.max() > 1.0 and off.shelf_db.max() <= 3.0
    assert off.shelf_db[lay.lfe].max() == 0


def test_shelf_boosts_treble_only():
    bank = ShelfBank(1, 3000, FS)
    t = np.arange(FS // 4)

    def gain(f):
        x = np.sin(2 * math.pi * f * t / FS).astype(np.float32)[:, None]
        bank.process(x[:512], np.array([6.0]))  # settle the gain ramp
        y = bank.process(x, np.array([6.0]))
        return 20 * np.log10(y[2000:].std() / x[2000:].std())

    assert abs(gain(100)) < 0.3
    assert gain(15000) > 5.0


def test_listener_clamped_to_max_offset():
    p = clamp_listener((3.0, 4.0, 0.0), 1.5)
    assert np.hypot(p[0], p[1]) == pytest.approx(1.5)


def test_one_euro_smooths_jitter_and_follows_moves():
    rng = np.random.default_rng(0)
    f = OneEuro(0.5, 2.0)
    still = [f(np.array([0.0, 0.0, 0.0]) + rng.normal(0, 0.02, 3), i / 10) for i in range(100)]
    assert np.std([p[0] for p in still[20:]]) < 0.01  # 2 cm jitter -> < 1 cm
    for i in range(100, 130):  # move 1 m/s for 0.3 s
        p = f(np.array([(i - 100) / 10 * 1.0, 0, 0]), i / 10)
    assert p[0] > 2.0  # follows (2.9 m), lag < 1 m


def test_listener_filter_predicts_and_holds_when_lost():
    from interface.bridge import ListenerMessage

    lf = ListenerFilter(0.5, 2.0, predict=True, max_predict_s=0.1)
    for i in range(30):
        lf.update(ListenerMessage(i + 1, i / 30, True, i / 30 * 0.5, 0, 0, 0, 0.9))  # 0.5 m/s to the right
    t_last = 29 / 30
    assert lf.position(t_last + 0.05)[0] > lf.position(t_last)[0]
    lf.update(ListenerMessage(31, t_last + 0.03, False, 0.5, 0, 0, 0, 0.0))
    assert lf.position(t_last + 0.1)[0] == pytest.approx(lf.filter.x[0])


def test_control_roundtrip():
    from dsp.control import ControlServer, DspClient

    class FakePlayer:
        def __init__(self):
            self.kw = {}

        def status(self):
            return {"mode": self.kw.get("mode", "static")}

        def set(self, **kw):
            self.kw.update(kw)

    server = ControlServer(FakePlayer(), port=0)
    client = DspClient(port=server.sock.getsockname()[1], timeout=1.0)
    try:
        assert client.request("status")["mode"] == "static"
        assert client.request("set", mode="tracking")["mode"] == "tracking"
        assert client.request("bogus")["ok"] is False
    finally:
        client.close()
        server.close()
    assert DspClient(port=1, timeout=0.1).request("status") is None


def _phantom_lag(mode, listener):
    """Interaural lag (samples, + = right ear first) of the FL+FR phantom center,
    virtual surround after the engine."""
    from dsp.sources import TestSignal
    from dsp.virtual import VirtualSurround

    eng = settled_engine(listener, mode)
    virt = VirtualSurround(eng.layout, FS)
    src = TestSignal("front", FS)
    out = np.concatenate([virt.process(eng.process(src.read(128)), listener) for _ in range(FS // 128)])
    left, right = out[FS // 4:, 0], out[FS // 4:, 1]
    lags = np.arange(-40, 41)
    xc = [np.dot(left[50:-50], np.roll(right, k)[50:-50]) for k in lags]
    return int(lags[int(np.argmax(xc))])  # roll(right, k) matches left: right leads by k


def test_virtual_surround_off_center_pulls_image_and_follow_recenters():
    from dsp.virtual import EAR_M

    listener = (0.4, 0.0, 0.0)  # 40 cm right: FR is nearer
    assert abs(_phantom_lag("static", (0, 0, 0))) <= 1       # at the seat: centered
    pulled = _phantom_lag("static", listener)
    assert pulled > 5                                         # off-center: right ear first (image pulled into FR)
    # Follow: aligned at the head, the phantom sits midway between FL and FR as seen from
    # the new spot (they are no longer symmetric: -39 / +19 deg -> about -10 deg)
    lay = Layout.from_config(SPEAKERS)
    az = [math.atan2(*(lay.positions[i][:2] - np.array(listener[:2]))) for i in (0, 1)]
    t = np.mean(az)
    expected = math.copysign(EAR_M * (abs(t) + math.sin(abs(t))), t) / C * FS  # Woodworth ITD
    follow = _phantom_lag("tracking", listener)
    assert abs(follow - expected) <= 1.5
    assert abs(pulled - expected) > 8


def test_virtual_itd_matches_a_real_head():
    from dsp.virtual import VirtualSurround

    side = Layout.from_config([{"name": "X", "channel": 0, "azimuth_deg": 90, "distance_m": 2.0, "height_m": 0}])
    d, _, shelf = VirtualSurround(side, FS)._targets((0, 0, 0))
    assert (d[0] - d[1]) * 1000 == pytest.approx(0.66, abs=0.03)  # ms, right ear first
    assert shelf[0] < -9 and shelf[1] == 0  # shadow at the far (left) ear only


def test_virtual_speaker_directivity_cancels_the_engine_offaxis_eq():
    from dsp.virtual import REAR_DB, VirtualSurround

    lay = Layout.from_config(SPEAKERS)
    listener = (0.5, 0.2, 0.0)
    eq = steering(lay, listener, offaxis_max_db=3.0).shelf_db
    _, _, shelf = VirtualSurround(lay, FS, offaxis_db=3.0)._targets(listener)
    head_avg = (shelf[0::2] + shelf[1::2]) / 2  # remove the head shadow (it differs per ear)
    shadow_avg = []
    for i in range(len(lay.names)):
        v = lay.positions[i] - np.array(listener)
        side = v[0] / np.linalg.norm(v[:2])
        front = v[1] / np.linalg.norm(v[:2])
        rear = REAR_DB * max(0.0, -front) if not lay.lfe[i] else 0.0
        shadow_avg.append(-10.0 * math.sin(math.asin(min(abs(side), 1))) / 2 + rear)
    directivity = head_avg[lay.channels] - np.array(shadow_avg)
    assert np.allclose(directivity + eq, 0, atol=0.01)  # EQ boost == simulated treble loss


def test_vbap_puts_the_object_on_the_right_speakers():
    from dsp.panner import vbap_gains

    lay = Layout.from_config(SPEAKERS)
    names = lay.names
    g = vbap_gains(lay, (0, 0, 0), lay.positions[names.index("FR")] * 0.75)  # at FR, 1.5 m
    assert g[names.index("FR")] == pytest.approx(1.0, abs=1e-6)
    assert np.count_nonzero(g) == 1
    g = vbap_gains(lay, (0, 0, 0), (-2.0, 0.0, 0.0))  # hard left: between FL and SL
    assert g[names.index("FL")] > 0 and g[names.index("SL")] > 0
    assert g[names.index("FR")] == g[names.index("SR")] == g[names.index("LFE")] == 0
    g = vbap_gains(lay, (0, 0, 0), (0.0, -1.5, 0.0))  # behind: SL + SR equally
    assert g[names.index("SL")] == pytest.approx(g[names.index("SR")], rel=0.05)


def test_object_keeps_its_direction_when_the_listener_moves():
    from dsp.panner import vbap_gains

    lay = Layout.from_config(SPEAKERS)
    obj = lay.positions[lay.names.index("FL")] * 0.75  # straight at FL from the seat
    moved = vbap_gains(lay, (0.5, 0, 0), obj)          # 0.5 m right: FL is no longer exactly there
    # the energy vector of the pan points at the object as seen from the new spot
    dirs = np.array([math.atan2(*(p[:2] - np.array([0.5, 0]))) for p in lay.positions])
    e = np.array([(moved ** 2 * np.sin(dirs)).sum(), (moved ** 2 * np.cos(dirs)).sum()])
    want = math.atan2(*(obj[:2] - np.array([0.5, 0])))
    assert math.atan2(*e) == pytest.approx(want, abs=math.radians(8))


def test_virtual_head_turn_tells_front_from_back():
    from dsp.virtual import VirtualSurround

    lay = Layout.from_config([
        {"name": "F", "channel": 0, "azimuth_deg": 0, "distance_m": 2.0, "height_m": 0},
        {"name": "B", "channel": 1, "azimuth_deg": 180, "distance_m": 2.0, "height_m": 0}])
    v = VirtualSurround(lay, FS, channels=2, offaxis_db=0.0)

    def itd(yaw):  # + = right ear first, per speaker
        d, _, _ = v._targets((0, 0, 0), yaw)
        return [(d[2 * c] - d[2 * c + 1]) * 1e6 for c in (0, 1)]

    front, back = itd(0)
    assert abs(front) < 1 and abs(back) < 1  # head still: identical cues (cone of confusion)
    front, back = itd(20)                    # turn right: front moves left, back moves right
    assert front < -50 and back > 50


def test_virtual_rear_sources_are_duller():
    from dsp.virtual import REAR_DB, VirtualSurround

    lay = Layout.from_config([
        {"name": "F", "channel": 0, "azimuth_deg": 0, "distance_m": 2.0, "height_m": 0},
        {"name": "B", "channel": 1, "azimuth_deg": 180, "distance_m": 2.0, "height_m": 0}])
    _, _, shelf = VirtualSurround(lay, FS, channels=2, offaxis_db=0.0)._targets((0, 0, 0))
    assert shelf[0] == shelf[1] == 0                       # front: no cut
    assert shelf[2] == pytest.approx(REAR_DB) and shelf[3] == pytest.approx(REAR_DB)  # behind: both ears


def test_follow_carries_the_object_with_the_listener():
    from types import SimpleNamespace

    from dsp.panner import vbap_gains
    from dsp.player import Player

    lay = Layout.from_config(SPEAKERS)
    placed = np.array([0.0, 1.5, 0.0])  # 1.5 m ahead of the seat
    you = np.array([0.6, -0.3, 0.0])

    def heard(mode, yaw=0.0, obj=placed):
        stub = SimpleNamespace(engine=SimpleNamespace(mode=mode, listener=you, layout=lay), object_pos=obj,
                               head_yaw_deg=yaw)
        return Player.object_heard(stub)

    # Follow: same offset from you as from the seat -> straight ahead of you, same distance
    assert np.allclose(heard("tracking"), you + placed)
    g = vbap_gains(lay, you, heard("tracking"))
    dirs = np.array([math.atan2(*(p[:2] - you[:2])) for p in lay.positions])
    e = np.array([(g ** 2 * np.sin(dirs)).sum(), (g ** 2 * np.cos(dirs)).sum()])
    assert math.atan2(*e) == pytest.approx(0.0, abs=math.radians(8))
    # Static / Bypass: the object stays where it was placed
    assert np.allclose(heard("static"), placed) and np.allclose(heard("bypass"), placed)

    # ... and turns with the head: placed ahead stays in front of the face
    turned = heard("tracking", yaw=90.0)          # turned right: "ahead" is now room +x
    assert np.allclose(turned, you + [1.5, 0, 0], atol=1e-9)
    right = heard("tracking", yaw=90.0, obj=np.array([1.0, 0.0, 0.2]))  # placed to the right
    assert np.allclose(right, you + [0, -1.0, 0.2], atol=1e-9)       # -> behind in the room
    assert np.allclose(heard("static", yaw=90.0), placed)            # Static ignores the head
