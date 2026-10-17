"""Full-resolution camera video per episode, written from the loop's frame taps on a writer thread per camera."""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


class VideoRecorder:
    """One camera's frames -> `<name>.mp4` (fourcc, nominal fps) plus `<name>.times.csv` (frame, perf_counter t).

    push() is called on the camera thread and never blocks: when the writer falls behind, frames are dropped
    and counted. The video opens at the size of its first frame; later frames of another size are resized.
    """

    def __init__(self, path: str | Path, fps: float = 30.0, codec: str = "mp4v", max_queue: int = 90):
        self.path = Path(path)
        self.times_path = self.path.with_suffix(".times.csv")
        self.fps = float(fps)
        self.codec = codec
        self.frames = 0
        self.dropped = 0
        self.size: tuple[int, int] | None = None
        self.error: str | None = None
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._closed = False
        self._thread = threading.Thread(target=self._run, name=f"video-{self.path.stem}", daemon=True)
        self._thread.start()

    def push(self, frame_bgr: np.ndarray, t: float) -> None:
        if self._closed or self.error is not None:
            return
        try:
            self._q.put_nowait((np.array(frame_bgr, copy=True), float(t)))
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        writer = None
        times = None
        try:
            while True:
                item = self._q.get()
                if item is None:
                    break
                frame, t = item
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                if writer is None:
                    h, w = frame.shape[:2]
                    self.size = (w, h)
                    writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*self.codec), self.fps, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"cv2.VideoWriter could not open {self.path} ({self.codec})")
                    times = open(self.times_path, "w")
                    times.write("frame,t\n")
                if (frame.shape[1], frame.shape[0]) != self.size:
                    frame = cv2.resize(frame, self.size)
                writer.write(np.ascontiguousarray(frame, dtype=np.uint8))
                times.write(f"{self.frames},{t:.6f}\n")
                self.frames += 1
        except Exception as e:  # noqa: BLE001 - reported in episode.json, never stops the episode
            self.error = repr(e)
            log.error("video %s: %s", self.path.name, e)
            while self._q.get() is not None:
                pass
        finally:
            if writer is not None:
                writer.release()
            if times is not None:
                times.close()

    def close(self, timeout: float = 30.0) -> dict:
        """Flush the queue, finish the file and return its stats."""
        if not self._closed:
            self._closed = True
            try:
                self._q.put(None, timeout=timeout)
            except queue.Full:
                self.error = self.error or "writer thread did not drain its queue"
            self._thread.join(timeout=timeout)
        return {
            "file": self.path.name if self.frames else None,
            "frames": self.frames,
            "dropped": self.dropped,
            "size": list(self.size) if self.size else None,
            "fps": self.fps,
            "error": self.error,
        }


class VideoTaps:
    """The one frame tap the runner registers for the whole run; it feeds the current episode's recorders.

    The loop API has no way to remove a tap, so episodes switch recorders here instead.
    """

    def __init__(self, cameras=("scene", "wrist"), fps: float = 30.0, codec: str = "mp4v"):
        self.cameras = tuple(cameras)
        self.fps = fps
        self.codec = codec
        self._recorders: dict[str, VideoRecorder] = {}

    def __call__(self, camera: str, frame_bgr: np.ndarray, t: float) -> None:
        recorder = self._recorders.get(camera)
        if recorder is not None:
            recorder.push(frame_bgr, t)

    def open(self, directory: Path) -> None:
        self.close()
        self._recorders = {
            cam: VideoRecorder(Path(directory) / f"{cam}.mp4", fps=self.fps, codec=self.codec) for cam in self.cameras
        }

    def close(self) -> dict:
        recorders, self._recorders = self._recorders, {}
        return {cam: rec.close() for cam, rec in recorders.items()}
