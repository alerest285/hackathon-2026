import numpy as np
import pytest

from so101_client.cameras import Frame
from so101_client.history import Ring

CAMS = ("scene", "wrist")


def frames(tick):
    return {cam: Frame(bgr=np.zeros((1, 1, 3), np.uint8), t=100.0 + tick, seq=tick, encoded=f"{cam}{tick}".encode())
            for cam in CAMS}


def state(tick):
    return np.full(6, float(tick), np.float32)


def command(tick):
    """The command sent at tick `tick`."""
    return np.full(6, 1000.0 + tick, np.float32)


def test_reset_pads_with_the_first_observation():
    ring = Ring(8)
    with pytest.raises(RuntimeError):
        ring.window()
    ring.reset(state(0), frames(0))
    w = ring.window()
    assert w.state.shape == (8, 6) and w.cmd_hist.shape == (8, 6)
    assert w.state.dtype == np.float32 and w.cmd_hist.dtype == np.float32
    np.testing.assert_array_equal(w.state, np.tile(state(0), (8, 1)))
    np.testing.assert_array_equal(w.cmd_hist, w.state)  # command = measured state at the start
    assert w.images == {"scene": [b"scene0", b"scene0"], "wrist": [b"wrist0", b"wrist0"]}
    assert w.frame_t == {"scene": 100.0, "wrist": 100.0}
    assert w.image_format == "jpeg"


def test_entry_j_pairs_tick_t_minus_7_plus_j_with_the_command_before_it():
    ring = Ring(8)
    ring.reset(state(0), frames(0))
    for k in range(1, 13):
        ring.append(state(k), command(k - 1), frames(k))
        w = ring.window()
        for j in range(8):
            tick = k - 7 + j
            if tick <= 0:  # reset padding
                np.testing.assert_array_equal(w.state[j], state(0))
                np.testing.assert_array_equal(w.cmd_hist[j], state(0))
            else:
                np.testing.assert_array_equal(w.state[j], state(tick))
                np.testing.assert_array_equal(w.cmd_hist[j], command(tick - 1))
        first = max(k - 7, 0)
        assert w.images["scene"] == [f"scene{first}".encode(), f"scene{k}".encode()]
        assert w.images["wrist"] == [f"wrist{first}".encode(), f"wrist{k}".encode()]
        assert w.frame_t["scene"] == 100.0 + k


def test_images_are_taken_at_indices_0_and_7_only():
    ring = Ring(8, image_format="png")
    ring.reset(state(0), frames(0))
    for k in range(1, 20):
        ring.append(state(k), command(k - 1), frames(k))
    w = ring.window()
    assert w.images["scene"] == [b"scene12", b"scene19"]
    assert w.image_format == "png"


def test_append_to_an_empty_ring_pads_and_copies():
    ring = Ring(8)
    s = state(3)
    ring.append(s, command(2), frames(3))
    s[:] = -1  # later changes to the caller's array must not leak into the history
    w = ring.window()
    np.testing.assert_array_equal(w.state, np.tile(state(3), (8, 1)))
    np.testing.assert_array_equal(w.cmd_hist, np.tile(command(2), (8, 1)))


def test_missing_camera_reuses_its_previous_frame():
    ring = Ring(8)
    ring.reset(state(0), frames(0))
    only_scene = {"scene": frames(1)["scene"]}
    ring.append(state(1), command(0), only_scene)
    w = ring.window()
    assert w.images["wrist"] == [b"wrist0", b"wrist0"]
    assert w.images["scene"] == [b"scene0", b"scene1"]
