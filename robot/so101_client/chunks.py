"""Chunk arithmetic, stitching and request scheduling. Pure: no I/O, no clock, unit-tested.

A chunk holds 42 absolute commands for ticks r..r+41: joints integrated from the window's last command
(cmd_hist[-1], the anchor), gripper absolute. The client turns it back into per-tick joint deltas
(D0 = a0 - anchor, Di = ai - a(i-1)), joins it at index d = k - r and adds each delta to the command it
last sent, which is what BFL's select_action does tick by tick. A position jump at a seam is therefore
impossible; the K-tick crossfade smooths the change of velocity.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from . import CHUNK, EXEC

JOINTS = 5  # channels 0-4 are joints in degrees
GRIPPER = 5  # channel 5 is the gripper, RANGE_0_100, absolute
MODES = ("prefetch", "sync")


def to_deltas(actions, anchor) -> tuple[np.ndarray, np.ndarray]:
    """(n, 6) absolute chunk and its anchor (6,) -> joint deltas (n, 5) and gripper (n,)."""
    a = np.asarray(actions, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != JOINTS + 1:
        raise ValueError(f"actions must be (n, 6), got {a.shape}")
    anchor = np.asarray(anchor, dtype=np.float64).reshape(JOINTS + 1)
    previous = np.vstack([anchor[None, :JOINTS], a[:-1, :JOINTS]])
    return a[:, :JOINTS] - previous, a[:, GRIPPER].copy()


def integrate(joint_deltas, gripper, start) -> np.ndarray:
    """Inverse of to_deltas: absolute (n, 6) commands from a start command (6,)."""
    joint_deltas = np.asarray(joint_deltas, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64).reshape(JOINTS + 1)
    out = np.empty((len(joint_deltas), JOINTS + 1))
    out[:, :JOINTS] = start[:JOINTS] + np.cumsum(joint_deltas, axis=0)
    out[:, GRIPPER] = np.asarray(gripper, dtype=np.float64)
    return out


def crossfade_weights(k: int) -> np.ndarray:
    """Weight of the new chunk on the j-th tick after a seam: (j + 1) / (K + 1)."""
    return (np.arange(k) + 1.0) / (k + 1.0)


def clamp(cmd, last_cmd, joint_limits=None, step_max: float | None = 6.0) -> np.ndarray:
    """Joint limits first, then |cmd - last_cmd| <= step_max per joint, gripper into 0..100.

    In this order a command never jumps by more than step_max, and it never leaves the limits once the
    last command is inside them. joint_limits: 5 x (lo, hi) degrees, or None.
    """
    out = np.array(cmd, dtype=np.float64).reshape(JOINTS + 1)
    last = np.asarray(last_cmd, dtype=np.float64).reshape(JOINTS + 1)
    if joint_limits is not None:
        lim = np.asarray(joint_limits, dtype=np.float64).reshape(JOINTS, 2)
        out[:JOINTS] = np.clip(out[:JOINTS], lim[:, 0], lim[:, 1])
    if step_max is not None and step_max > 0:
        out[:JOINTS] = np.clip(out[:JOINTS], last[:JOINTS] - step_max, last[:JOINTS] + step_max)
    out[GRIPPER] = min(100.0, max(0.0, out[GRIPPER]))
    return out


class Stitcher:
    """The active chunk in joint-delta space.

    accept() joins a reply at index d with a K-tick crossfade of the joint deltas, or at index 0 when
    nothing is active (after a hold). A reply that would join a running chunk with d > chunk - 1 - K is
    dropped. The gripper is taken from the newest chunk as it is.
    """

    def __init__(self, chunk: int = CHUNK, crossfade: int = 6):
        if not 0 <= crossfade < chunk:
            raise ValueError("crossfade must be in [0, chunk)")
        self.chunk = chunk
        self.crossfade = crossfade
        self.reset()

    def reset(self) -> None:
        self._jd: np.ndarray | None = None
        self._g: np.ndarray | None = None
        self._idx = 0
        self._old: tuple[np.ndarray, int] | None = None
        self._fade_j = 0

    @property
    def active(self) -> bool:
        return self._g is not None and self._idx < len(self._g)

    @property
    def remaining(self) -> int:
        return len(self._g) - self._idx if self._g is not None else 0

    @property
    def index(self) -> int:
        """Index of the next action of the active chunk."""
        return self._idx

    def accept(self, actions, anchor, d: int, after_hold: bool | None = None) -> int | None:
        """Join a chunk planned for request tick r, d = k - r ticks later. Returns the start index, or
        None when the chunk is dropped. after_hold defaults to "nothing is active"."""
        jd, g = to_deltas(actions, anchor)
        if after_hold is None:
            after_hold = not self.active
        if after_hold:
            start, old = 0, None
        else:
            if d > len(g) - 1 - self.crossfade:
                return None
            start = max(0, int(d))
            old = (self._jd, self._idx) if self.crossfade > 0 and self.active else None
        self._jd, self._g, self._idx, self._old, self._fade_j = jd, g, start, old, 0
        return start

    def next(self) -> tuple[np.ndarray, float] | None:
        """(joint delta (5,), gripper) for this tick, or None when nothing is left."""
        if not self.active:
            return None
        delta = self._jd[self._idx]
        if self._old is not None:
            old_jd, old_idx = self._old
            if self._fade_j < self.crossfade and old_idx < len(old_jd):
                w = (self._fade_j + 1.0) / (self.crossfade + 1.0)
                delta = (1.0 - w) * old_jd[old_idx] + w * delta
                self._old = (old_jd, old_idx + 1)
                self._fade_j += 1
            else:
                self._old = None
        gripper = float(self._g[self._idx])
        self._idx += 1
        return delta, gripper


class Scheduler:
    """When to send the next request. One request is in flight at a time.

    prefetch: request when k - last_request >= P, or when the remaining actions <= d_max + K + 2, where
    d_max is the worst turnaround in ticks over the last 100 plans; immediately while holding.
    sync (BFL's cadence): execute exec_horizon actions, hold, request from the current window; the ticks
    spent waiting for the reply do not enter the history, and the reply starts at index 0.
    """

    def __init__(self, mode: str = "prefetch", period: int = 16, crossfade: int = 6, exec_horizon: int = EXEC,
                 history: int = 100):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.period = period
        self.crossfade = crossfade
        self.exec_horizon = exec_horizon
        self.inflight: int | None = None
        self.last_request: int | None = None
        self._d: deque[int] = deque(maxlen=history)
        self.executed = 0
        self.holding = True
        self._hold_requested = False

    @property
    def d_max(self) -> int:
        return max(self._d) if self._d else 0

    def reset(self) -> None:
        """Task change or pause: back to a hold. A request in flight keeps its slot until its reply."""
        self.hold()

    def hold(self) -> None:
        self.holding = True
        self.executed = 0
        self._hold_requested = False

    def started(self) -> None:
        """A chunk was accepted."""
        self.holding = False
        self.executed = 0
        self._hold_requested = False

    def stepped(self) -> bool:
        """One action executed. True when sync mode has executed exec_horizon actions: the caller drops the
        rest of the chunk. The hold, and its request, start at the next tick, so the request's window holds
        this last command (BFL's select_action predicts on the tick after the queue ran out)."""
        self.executed += 1
        return self.mode == "sync" and self.executed >= self.exec_horizon

    def remaining(self, chunk_left: int) -> int:
        """Actions left before a hold: the chunk's tail in prefetch, up to exec_horizon in sync."""
        if self.holding:
            return 0
        if self.mode == "sync":
            return max(0, min(chunk_left, self.exec_horizon - self.executed))
        return chunk_left

    def appends(self) -> bool:
        """Whether this tick enters the history: always in prefetch, not while sync waits for its reply."""
        return not (self.mode == "sync" and self.holding and self._hold_requested)

    def due(self, k: int, remaining: int) -> bool:
        if self.inflight is not None:
            return False
        if self.mode == "sync":
            return self.holding and not self._hold_requested
        if self.holding or self.last_request is None:
            return True
        return k - self.last_request >= self.period or remaining <= self.d_max + self.crossfade + 2

    def sent(self, k: int, rid: int) -> None:
        self.inflight = rid
        self.last_request = k
        if self.holding:
            self._hold_requested = True

    def received(self, rid: int, d: int | None = None) -> bool:
        """A reply with actions arrived (any epoch). Records d and frees the slot if it was in flight."""
        if d is not None:
            self._d.append(int(d))
        if rid != self.inflight:
            return False
        self.inflight = None
        return True

    def failed(self, rid: int) -> None:
        """An error reply or a lost request: free the slot so the request can be repeated."""
        if rid == self.inflight:
            self.inflight = None
            self._hold_requested = False


class Planner:
    """The per-tick chunk logic of the control loop: Stitcher + Scheduler, no I/O, no clock.

    Per tick the loop calls, in this order: appends() (does this observation enter the ring), received()
    or failed() and offer() for each reply, step() for the command, then due() and sent().
    """

    def __init__(self, mode: str = "prefetch", period: int = 16, crossfade: int = 6, exec_horizon: int = EXEC,
                 chunk: int = CHUNK):
        if exec_horizon > chunk:
            raise ValueError("exec_horizon cannot exceed chunk")
        self.stitcher = Stitcher(chunk, crossfade)
        self.scheduler = Scheduler(mode, period, crossfade, exec_horizon)

    @property
    def mode(self) -> str:
        return self.scheduler.mode

    @property
    def holding(self) -> bool:
        return self.scheduler.holding

    @property
    def remaining(self) -> int:
        return self.scheduler.remaining(self.stitcher.remaining)

    def reset(self) -> None:
        self.stitcher.reset()
        self.scheduler.reset()

    def appends(self) -> bool:
        return self.scheduler.appends()

    def offer(self, actions, anchor, d: int) -> int | None:
        """A current reply: its start index (0 after a hold, else d), or None when it is dropped."""
        holding = self.scheduler.holding
        if holding:
            self.stitcher.reset()
        start = self.stitcher.accept(actions, anchor, d, after_hold=holding)
        if start is not None:
            self.scheduler.started()
        return start

    def step(self, last_cmd) -> np.ndarray | None:
        """The next command before clamping (last command + joint delta, gripper absolute), or None to hold."""
        if self.scheduler.holding:
            return None
        nxt = self.stitcher.next()
        if nxt is None:
            self.scheduler.hold()
            return None
        delta, gripper = nxt
        cmd = np.array(last_cmd, dtype=np.float64).reshape(JOINTS + 1)
        cmd[:JOINTS] += delta
        cmd[GRIPPER] = gripper
        if self.scheduler.stepped():
            self.stitcher.reset()
        return cmd

    def due(self, k: int) -> bool:
        return self.scheduler.due(k, self.remaining)

    def sent(self, k: int, rid: int) -> None:
        self.scheduler.sent(k, rid)

    def received(self, rid: int, d: int | None = None) -> bool:
        return self.scheduler.received(rid, d)

    def failed(self, rid: int) -> None:
        self.scheduler.failed(rid)
