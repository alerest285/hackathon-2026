import msgpack
import numpy as np
import pytest

from so101_client import protocol
from so101_client.fakes import FakeNet
from so101_client.history import Window


def window(image_format="jpeg"):
    state = np.arange(48, dtype=np.float32).reshape(8, 6)
    if image_format == "raw":
        images = {cam: [np.full((256, 256, 3), i, np.uint8) for i in (0, 7)] for cam in ("scene", "wrist")}
    else:
        images = {"scene": [b"s0", b"s7"], "wrist": [b"w0", b"w7"]}
    return Window(state=state, cmd_hist=state + 100, images=images, image_format=image_format,
                  frame_t={"scene": 1.0, "wrist": 1.0})


def test_array_map_layout_matches_the_architecture_table():
    a = np.arange(12, dtype=np.float32).reshape(2, 6)
    assert protocol.encode_array(a) == {"__nd__": True, "dtype": "<f4", "shape": [2, 6], "data": a.tobytes()}
    assert protocol.encode_array(np.zeros((4, 5, 3), np.uint8))["dtype"] == "|u1"
    plain = msgpack.unpackb(protocol.pack(protocol.infer_message(3, 120, "t", window())), raw=False)
    assert plain["type"] == "infer"
    assert set(plain["state"]) == {"__nd__", "dtype", "shape", "data"}
    assert plain["state"]["dtype"] == "<f4" and plain["state"]["shape"] == [8, 6]
    assert isinstance(plain["images"]["scene"][0], bytes)


@pytest.mark.parametrize("image_format", ["jpeg", "png", "raw"])
def test_infer_round_trip(image_format):
    w = window(image_format)
    msg = protocol.unpack(protocol.pack(protocol.infer_message(3, 120, "pick up the disk", w)))
    assert (msg["type"], msg["id"], msg["tick"], msg["task"]) == ("infer", 3, 120, "pick up the disk")
    assert msg["image_format"] == image_format
    assert msg["state"].dtype == np.float32
    np.testing.assert_array_equal(msg["state"], w.state)
    np.testing.assert_array_equal(msg["cmd_hist"], w.cmd_hist)
    for cam in ("scene", "wrist"):
        for got, sent in zip(msg["images"][cam], w.images[cam]):
            if image_format == "raw":
                assert got.dtype == np.uint8 and got.shape == (256, 256, 3)
                np.testing.assert_array_equal(got, sent)
            else:
                assert got == sent


def test_infer_message_rejects_bad_windows():
    w = window()
    w.state = np.zeros((7, 6), np.float32)
    with pytest.raises(ValueError):
        protocol.infer_message(1, 0, "t", w)
    w = window()
    w.images["scene"] = [b"only one"]
    with pytest.raises(ValueError):
        protocol.infer_message(1, 0, "t", w)


def test_message_type():
    assert protocol.message_type({"type": "chunk"}) == "chunk"
    assert protocol.message_type({"protocol": "so101.v1", "engine": "fake"}) == "hello"
    assert protocol.message_type({"id": 1, "tick": 2, "actions": None}) == "chunk"
    assert protocol.message_type({"id": 1, "error": "x"}) == "error"
    assert protocol.message_type({"t": 1.0, "server_t": 2.0}) == "pong"
    assert protocol.message_type({"t": 1.0}) == "ping"


def test_parse_actions():
    a = np.ones((42, 6), np.float32)
    out = protocol.parse_actions(protocol.unpack(protocol.pack({"type": "chunk", "actions": a})))
    assert out.flags.writeable and out.dtype == np.float32
    assert protocol.parse_actions({"actions": a[None]}).shape == (42, 6)
    a[3, 2] = np.nan
    with pytest.raises(ValueError):
        protocol.parse_actions({"actions": a})
    with pytest.raises(ValueError):
        protocol.parse_actions({"actions": np.ones((42, 7), np.float32)})


def test_check_hello():
    assert protocol.check_hello(FakeNet().connect()) == []
    assert protocol.check_hello({"protocol": "so101.v0"})
    assert protocol.check_hello({"protocol": "so101.v1", "cameras": ["wrist", "scene"]})
