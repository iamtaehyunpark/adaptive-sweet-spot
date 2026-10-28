import time

import pytest

from interface.bridge import BridgePublisher, BridgeReceiver, ListenerMessage


def test_message_roundtrip():
    msg = ListenerMessage(7, 123.5, True, 0.12345678, -0.05, 0.0, -8.0, 0.9)
    back = ListenerMessage.from_bytes(msg.to_bytes())
    assert back.seq == 7 and back.tracking and back.x == pytest.approx(0.1235, abs=1e-4)


def test_rejects_other_versions():
    with pytest.raises(ValueError):
        ListenerMessage.from_bytes(b'{"v": 2, "seq": 1}')


def test_udp_publish_receive_keeps_newest():
    rx = BridgeReceiver("127.0.0.1", 0)
    tx = BridgePublisher("127.0.0.1", rx.port)
    try:
        for i in range(5):
            tx.send(time.time(), True, 0.1 * i, 0.0, 0.0, 0.0, 0.9)
        deadline = time.time() + 2
        while time.time() < deadline and (rx.latest() is None or rx.latest().seq < 5):
            time.sleep(0.01)
        assert rx.latest().seq == 5
        assert rx.latest().x == pytest.approx(0.4)
    finally:
        tx.close()
        rx.close()
