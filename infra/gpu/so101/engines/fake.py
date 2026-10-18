"""Engine fake: no torch, no GPU. Deterministic motion for tests and for driving the Mac client.

Modes (all integrate joint motion from cmd_hist[-1], like the model; the gripper is absolute):
- hold: every action equals cmd_hist[-1].
- sine: joints 0-3 oscillate a few degrees around the anchor (4 s period); the gripper stays put.
- pickplace: a 9 s cycle that restarts whenever the task string changes: reach (gripper open), close,
  transport (closed), open, return, then 2.5 s of rest. A step-end detector (grasp, release, 1 s rest)
  fires about 7.5 s after a task starts. period_s scales every phase in proportion (tests use 2-3 s).

guidance and steps, when given, are only reported in hello (the fake does not sample), so the server's
--guidance and --steps can be checked without a GPU.
"""

import math
import time

import numpy as np

from .base import Engine

PERIOD_S = 9.0
HOME = np.zeros(5)
PICK = np.array([8.0, 6.0, -6.0, 4.0, 0.0])  # degrees, relative to where the cycle started
PLACE = np.array([-8.0, 6.0, -6.0, 4.0, 0.0])
SINE_AMP = np.array([5.0, 3.0, 3.0, 4.0, 0.0])
SINE_PERIOD_S = 4.0
MODES = ("hold", "sine", "pickplace")


def _ease(u: float) -> float:
    return (1 - math.cos(math.pi * min(max(u, 0.0), 1.0))) / 2


def pickplace_pose(
    t: float, gripper_open: float, gripper_closed: float, period_s: float = PERIOD_S
) -> tuple[np.ndarray, float]:
    """Joint offset (5,) in degrees and absolute gripper at time t into the cycle of length period_s."""
    s = (t % period_s) * (PERIOD_S / period_s)
    if s < 1.5:
        return HOME + (PICK - HOME) * _ease(s / 1.5), gripper_open
    if s < 2.5:
        return PICK.copy(), gripper_open + (gripper_closed - gripper_open) * (s - 1.5)
    if s < 4.0:
        return PICK + (PLACE - PICK) * _ease((s - 2.5) / 1.5), gripper_closed
    if s < 5.0:
        return PLACE.copy(), gripper_closed + (gripper_open - gripper_closed) * (s - 4.0)
    if s < 6.5:
        return PLACE + (HOME - PLACE) * _ease((s - 5.0) / 1.5), gripper_open
    return HOME.copy(), gripper_open


class FakeEngine(Engine):
    name = "fake"

    def __init__(
        self,
        mode: str = "pickplace",
        latency_ms: float = 0.0,
        jitter_ms: float = 0.0,
        gripper_open: float = 60.0,
        gripper_closed: float = 10.0,
        seed: int = 0,
        clock=time.monotonic,
        fps: float = 30.0,
        chunk: int = 42,
        period_s: float = PERIOD_S,
        guidance: tuple[float, float] | None = None,
        steps: int | None = None,
        capture_video: bool = False,
    ):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"fake mode must be one of {', '.join(MODES)}, got {mode!r}")
        if not period_s > 0:
            raise ValueError(f"period_s must be > 0, got {period_s!r}")
        self.period_s = float(period_s)
        self.mode, self.latency_ms, self.jitter_ms = mode, float(latency_ms), float(jitter_ms)
        self.gripper_open, self.gripper_closed = float(gripper_open), float(gripper_closed)
        self.fps, self.chunk = float(fps), int(chunk)
        self.clock = clock
        self.rng = np.random.default_rng(seed)
        self._t_start = clock()
        self._task, self._t_task = None, self._t_start
        self.capture_video, self.last_video = bool(capture_video), None
        self.meta = {
            "checkpoint": "fake",
            "latent_hw": [2, 2],
            "revision": None,
            "fake": {
                "mode": mode,
                "latency_ms": self.latency_ms,
                "jitter_ms": self.jitter_ms,
                "gripper_open": self.gripper_open,
                "gripper_closed": self.gripper_closed,
                "period_s": self.period_s,
            },
        }
        if guidance is not None:
            self.meta["guidance"] = [float(g) for g in guidance]
        if steps is not None:
            self.meta["steps"] = int(steps)
        print(
            f"fake: mode {mode}, latency {self.latency_ms:.0f} ms, jitter {self.jitter_ms:.0f} ms, "
            f"pickplace period {self.period_s:g} s",
            flush=True,
        )

    def _offsets(self, t: float) -> tuple[np.ndarray, float | None]:
        if self.mode == "sine":
            return SINE_AMP * math.sin(2 * math.pi * t / SINE_PERIOD_S), None
        return pickplace_pose(t, self.gripper_open, self.gripper_closed, self.period_s)

    def infer(self, state, cmd_hist, frames, task):
        now = self.clock()
        anchor = np.asarray(cmd_hist[-1], dtype=np.float32)
        actions = np.tile(anchor, (self.chunk, 1))
        if self.mode != "hold":
            if self.mode == "pickplace" and task != self._task:
                self._task, self._t_task = task, now
            t = now - (self._t_task if self.mode == "pickplace" else self._t_start)
            before, _ = self._offsets(t - 1 / self.fps)
            for i in range(self.chunk):
                offset, gripper = self._offsets(t + i / self.fps)
                actions[i, :5] = anchor[:5] + offset - before
                if gripper is not None:
                    actions[i, 5] = gripper
        delay_ms = self.latency_ms
        if self.jitter_ms > 0:
            delay_ms = max(0.0, float(self.rng.normal(self.latency_ms, self.jitter_ms)))
        if delay_ms > 0:
            time.sleep(delay_ms / 1000)
        # --save-latents plumbing: a placeholder "video" of 2 latent frames x latent_hw 2x2, 96 channels
        self.last_video = np.full((8, 96), len(task), dtype=np.float16) if self.capture_video else None
        return actions.astype(np.float32)
