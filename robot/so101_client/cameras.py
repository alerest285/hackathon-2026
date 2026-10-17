"""Scene and wrist cameras: one capture thread per camera, the newest frame kept with its capture time.

Each new frame is kept at full resolution (for taps, e.g. video) and as the squashed 256x256 image the
model reads, encoded on the camera thread so the control loop never waits for it. On macOS the captures
are opened on the calling thread (AVFoundation asks for camera permission there) and read on their own.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

from .protocol import IMAGE_FORMATS

log = logging.getLogger(__name__)


@dataclass
class Frame:
    bgr: np.ndarray  # full-resolution BGR (H, W, 3) uint8
    t: float  # perf_counter at capture
    seq: int
    encoded: bytes | np.ndarray  # image_size x image_size: jpeg/png bytes, or RGB uint8 array for raw


def encode_image(bgr: np.ndarray, image_size: int = 256, image_format: str = "jpeg",
                 jpeg_quality: int = 95) -> bytes | np.ndarray:
    """Squash a BGR frame to image_size x image_size (INTER_AREA) and encode it."""
    import cv2

    small = cv2.resize(bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    if image_format == "jpeg":
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    elif image_format == "png":
        ok, buf = cv2.imencode(".png", small)
    elif image_format == "raw":
        return np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError(f"image_format must be one of {IMAGE_FORMATS}")
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for {image_format}")
    return buf.tobytes()


def decode_image(data, image_format: str) -> np.ndarray:
    """Encoded image -> RGB uint8 (H, W, 3), as the server sees it."""
    import cv2

    if image_format == "raw":
        return np.asarray(data, dtype=np.uint8)
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not decode a {image_format} image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def compose(scene_rgb: np.ndarray, wrist_rgb: np.ndarray) -> np.ndarray:
    """scene | wrist side by side, the side_by_side layout of the SO-101 checkpoint."""
    return np.concatenate([scene_rgb, wrist_rgb], axis=1)


def capture_backend(source) -> int:
    """AVFoundation for camera indices on macOS; files and URLs go through OpenCV's default (FFMPEG)."""
    import cv2

    return cv2.CAP_AVFOUNDATION if sys.platform == "darwin" and isinstance(source, int) else cv2.CAP_ANY


def open_capture(source, width: int = 640, height: int = 480, fps: int = 30):
    """cv2.VideoCapture with the requested mode and a one-frame buffer. Raises when it does not open."""
    import cv2

    src = int(source) if isinstance(source, str) and source.strip().isdigit() else source
    cap = cv2.VideoCapture(src, capture_backend(src))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"camera {source!r} did not open. List cameras with camera_check.py; on macOS the terminal needs "
            "camera permission (System Settings > Privacy & Security > Camera)."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def capture_mode(cap) -> tuple[int, int, float]:
    import cv2

    return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(cap.get(cv2.CAP_PROP_FPS)))


class Cameras:
    """sources: {"scene": 0, "wrist": 1}; values are OpenCV indices (int or digit string) or paths."""

    def __init__(self, sources: dict, width: int = 640, height: int = 480, fps: int = 30, image_size: int = 256,
                 image_format: str = "jpeg", jpeg_quality: int = 95):
        if image_format not in IMAGE_FORMATS:
            raise ValueError(f"image_format must be one of {IMAGE_FORMATS}")
        self.sources = dict(sources)
        self.names = tuple(self.sources)
        self.width, self.height, self.fps = width, height, fps
        self.image_size = image_size
        self.image_format = image_format
        self.jpeg_quality = jpeg_quality
        self.modes: dict[str, tuple[int, int, float]] = {}
        self.errors: dict[str, int] = {name: 0 for name in self.names}
        self.running = False
        self._latest: dict[str, Frame] = {}
        self._lock = threading.Lock()
        self._taps: list = []
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    def start(self, timeout_s: float = 5.0) -> None:
        """Open every camera, start the capture threads and wait for a first frame from each."""
        if self.running:
            return
        caps = {}
        try:
            for name, source in self.sources.items():
                caps[name] = open_capture(source, self.width, self.height, self.fps)
                self.modes[name] = capture_mode(caps[name])
                w, h, f = self.modes[name]
                if (w, h) != (self.width, self.height):
                    log.warning("camera %s delivers %dx%d, asked for %dx%d", name, w, h, self.width, self.height)
        except Exception:
            for cap in caps.values():
                cap.release()
            raise
        self._stop.clear()
        self._threads = [threading.Thread(target=self._reader, args=(name, cap), name=f"camera-{name}", daemon=True)
                         for name, cap in caps.items()]
        for thread in self._threads:
            thread.start()
        self.running = True
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            with self._lock:
                missing = [name for name in self.names if name not in self._latest]
            if not missing:
                return
            time.sleep(0.01)
        self.stop()
        raise RuntimeError(f"no frames from camera(s) {missing} within {timeout_s:.0f} s")

    def latest(self) -> dict[str, Frame]:
        with self._lock:
            return dict(self._latest)

    def add_tap(self, fn) -> None:
        """fn(camera, frame_bgr, t), called on the camera thread for every new frame. Keep it fast."""
        self._taps = [*self._taps, fn]

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads = []
        self.running = False

    def _reader(self, name: str, cap) -> None:
        seq = 0
        failures = 0
        tap_errors = 0
        try:
            while not self._stop.is_set():
                if not cap.grab():
                    failures += 1
                    self.errors[name] += 1
                    if failures in (1, 30) or failures % 300 == 0:
                        log.warning("camera %s: grab failed (%d in a row)", name, failures)
                    time.sleep(0.005)
                    continue
                t = time.perf_counter()
                ok, bgr = cap.retrieve()
                if not ok or bgr is None:
                    failures += 1
                    self.errors[name] += 1
                    continue
                failures = 0
                seq += 1
                try:
                    encoded = encode_image(bgr, self.image_size, self.image_format, self.jpeg_quality)
                except Exception:
                    log.exception("camera %s: encoding failed", name)
                    continue
                frame = Frame(bgr=bgr, t=t, seq=seq, encoded=encoded)
                with self._lock:
                    self._latest[name] = frame
                for fn in self._taps:
                    try:
                        fn(name, bgr, t)
                    except Exception:
                        tap_errors += 1
                        if tap_errors in (1, 100):
                            log.exception("camera %s: frame tap failed (%d so far)", name, tap_errors)
        finally:
            cap.release()
