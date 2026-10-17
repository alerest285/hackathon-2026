"""Cameras with video files as sources (no webcam is opened), plus the encode/decode helpers."""

import threading
import time

import cv2
import numpy as np
import pytest

from so101_client.cameras import Cameras, compose, decode_image, encode_image
from so101_client.fakes import FakeCameras


def write_video(path, color, n=90, size=(640, 480)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, size)
    assert writer.isOpened()
    for i in range(n):
        frame = np.full((size[1], size[0], 3), color, np.uint8)
        cv2.putText(frame, str(i), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        writer.write(frame)
    writer.release()


@pytest.mark.parametrize("image_format", ["jpeg", "png", "raw"])
def test_encode_decode_round_trip(image_format):
    bgr = FakeCameras().pattern(0, 0.3)
    encoded = encode_image(bgr, 256, image_format, 95)
    rgb = decode_image(encoded, image_format)
    assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
    expected = cv2.cvtColor(cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    tolerance = 12 if image_format == "jpeg" else 0
    assert np.abs(rgb.astype(int) - expected.astype(int)).mean() <= tolerance
    assert compose(rgb, rgb).shape == (256, 512, 3)


def test_cameras_from_video_files(tmp_path):
    write_video(tmp_path / "scene.avi", (0, 0, 200))  # red in BGR
    write_video(tmp_path / "wrist.avi", (200, 0, 0))  # blue
    cams = Cameras({"scene": str(tmp_path / "scene.avi"), "wrist": str(tmp_path / "wrist.avi")})
    seen = []
    lock = threading.Lock()

    def tap(cam, bgr, t):
        with lock:
            seen.append((cam, bgr.shape))

    cams.add_tap(tap)
    cams.start()
    try:
        assert cams.running
        latest = cams.latest()
        time.sleep(0.2)
    finally:
        cams.stop()
    assert not cams.running
    assert set(latest) == {"scene", "wrist"}
    for frame in latest.values():
        assert frame.bgr.shape == (480, 640, 3) and frame.seq >= 1 and isinstance(frame.encoded, bytes)
    scene = decode_image(latest["scene"].encoded, "jpeg")
    wrist = decode_image(latest["wrist"].encoded, "jpeg")
    assert scene.shape == (256, 256, 3)
    assert scene[..., 0].mean() > 150 and wrist[..., 2].mean() > 150  # RGB: scene is red, wrist is blue
    with lock:
        assert {cam for cam, _ in seen} == {"scene", "wrist"} and seen[0][1] == (480, 640, 3)


def test_a_camera_that_does_not_open_raises(tmp_path):
    with pytest.raises(RuntimeError):
        Cameras({"scene": str(tmp_path / "missing.avi"), "wrist": str(tmp_path / "missing2.avi")}).start()
