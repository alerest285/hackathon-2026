"""The 8-tick observation ring and the window sent with every request.

Entry j of a window pairs the measured state and frames of tick t-7+j with the command sent at the tick
before (flux-action ObservationHistory.append). At an episode start the ring holds 8 copies of the first
observation with command = measured state. Only the frames at indices 0 and 7 are encoded by the model,
so the window carries just those two per camera.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from . import N_OBS


@dataclass
class Window:
    state: np.ndarray  # (8, 6) float32, oldest first
    cmd_hist: np.ndarray  # (8, 6) float32, entry j = the command sent at the tick before state j
    images: dict  # {"scene": [i0, i7], "wrist": [i0, i7]}; bytes (jpeg/png) or uint8 (H, W, 3) RGB for raw
    image_format: str  # "jpeg" | "png" | "raw"
    frame_t: dict  # {"scene": capture time of i7, ...} in perf_counter seconds


@dataclass(frozen=True)
class _Entry:
    state: np.ndarray
    cmd: np.ndarray
    images: dict
    t: dict


class Ring:
    """Last n ticks of (state, command sent before, frames). Frames are Frame-like: .encoded and .t."""

    def __init__(self, n: int = N_OBS, image_format: str = "jpeg"):
        self.n = n
        self.image_format = image_format
        self._entries: deque[_Entry] = deque(maxlen=n)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def ready(self) -> bool:
        return len(self._entries) == self.n

    def append(self, state, cmd_before, frames: dict) -> None:
        """One tick. An empty ring is padded with copies of this entry."""
        entry = self._entry(state, cmd_before, frames)
        if not self._entries:
            self._entries.extend([entry] * self.n)
        else:
            self._entries.append(entry)

    def reset(self, state, frames: dict) -> None:
        """Episode start or intervention: n copies of this observation, each with command = state."""
        self._entries.clear()
        self._entries.extend([self._entry(state, state, frames)] * self.n)

    def window(self) -> Window:
        if not self.ready:
            raise RuntimeError("the ring is empty: reset it with the first observation")
        first, last = self._entries[0], self._entries[-1]
        return Window(
            state=np.stack([e.state for e in self._entries]),
            cmd_hist=np.stack([e.cmd for e in self._entries]),
            images={cam: [first.images.get(cam, img), img] for cam, img in last.images.items()},
            image_format=self.image_format,
            frame_t=dict(last.t),
        )

    def _entry(self, state, cmd, frames: dict) -> _Entry:
        state = np.array(state, dtype=np.float32).reshape(-1)
        cmd = np.array(cmd, dtype=np.float32).reshape(-1)
        if state.shape != cmd.shape:
            raise ValueError(f"state {state.shape} and command {cmd.shape} differ")
        images, times = {}, {}
        previous = self._entries[-1] if self._entries else None
        for cam, frame in frames.items():
            images[cam] = frame.encoded
            times[cam] = frame.t
        if previous is not None:
            for cam in previous.images:
                if cam not in images:
                    images[cam] = previous.images[cam]
                    times[cam] = previous.t[cam]
        return _Entry(state, cmd, images, times)
