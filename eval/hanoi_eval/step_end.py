"""Automatic step end (decision 05): the gripper closes on a piece, reopens, then the arm rests."""

from __future__ import annotations

import numpy as np

PHASES = ("unarmed", "idle", "closing", "grasped", "released", "fired")


class AutoEndDetector:
    """Feed it every tick of one step; update() returns "auto" exactly once, after grasp, release and rest.

    - grasp: the gripper crosses the closed threshold and does not reopen past the open threshold for min_hold_s
    - release: it then crosses the open threshold
    - rest: every joint 0-4 then moves slower than rest_speed_deg_s, measured between consecutive states, for rest_s

    A gripper that starts the step closed has to open once before a grasp can count. Closing again after the
    release goes back to the grasp phase. `closed_is` says which end of the 0..100 range is closed. The detector
    is pure: the caller passes the tick time, so it has no clock and no thread.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.phase = "unarmed"
        self.events: list[tuple[str, float]] = []
        self._t_close: float | None = None
        self._close_from = "idle"
        self._t_rest: float | None = None
        self._prev: tuple[float, np.ndarray] | None = None
        self.gripper_min = float("inf")
        self.gripper_max = float("-inf")

    def closed(self, g: float) -> bool:
        c = self.cfg
        return g <= c.closed_threshold if c.closed_is == "low" else g >= c.closed_threshold

    def is_open(self, g: float) -> bool:
        c = self.cfg
        return g >= c.open_threshold if c.closed_is == "low" else g <= c.open_threshold

    @property
    def fired(self) -> bool:
        return self.phase == "fired"

    def interrupt(self) -> None:
        """Forget the rest timer and the previous sample, e.g. after a pause. The grasp phase is kept."""
        self._t_rest = None
        self._prev = None

    def observe(self, info) -> str | None:
        """update() from a so101_client TickInfo."""
        return self.update(info.t, info.state, info.cmd)

    def update(self, t: float, state, cmd=None) -> str | None:
        if self.phase == "fired":
            return None
        state = np.asarray(state, dtype=np.float64)
        source = state if (self.cfg.signal == "state" or cmd is None) else np.asarray(cmd, dtype=np.float64)
        g = float(source[5])
        if g == g:
            self.gripper_min = min(self.gripper_min, g)
            self.gripper_max = max(self.gripper_max, g)
        joints = state[:5]
        speed = None
        if self._prev is not None:
            dt = t - self._prev[0]
            if dt > 0:
                speed = float(np.max(np.abs(joints - self._prev[1]))) / dt
        self._prev = (t, joints.copy())

        phase = self.phase
        if phase == "unarmed":
            if self.is_open(g):
                self._to("idle", t)
        elif phase in ("idle", "released"):
            if self.closed(g):
                self._close_from = phase
                self._t_close = t
                self._t_rest = None
                self._to("closing", t)
            elif phase == "released" and speed is not None:
                if speed < self.cfg.rest_speed_deg_s:
                    if self._t_rest is None:
                        self._t_rest = t
                    if t - self._t_rest >= self.cfg.rest_s:
                        self._to("fired", t)
                        return "auto"
                else:
                    self._t_rest = None
        elif phase == "closing":
            if self.is_open(g):
                self._to(self._close_from, t)
            elif t - self._t_close >= self.cfg.min_hold_s:
                self._to("grasped", t)
        elif phase == "grasped":
            if self.is_open(g):
                self._t_rest = None
                self._to("released", t)
        return None

    def _to(self, phase: str, t: float) -> None:
        self.phase = phase
        self.events.append((phase, t))

    def summary(self, t0: float = 0.0) -> dict:
        """Phase changes relative to t0 plus the gripper range seen, for attempts.jsonl."""
        seen = self.gripper_min <= self.gripper_max
        return {
            "phase": self.phase,
            "events": [[phase, round(t - t0, 3)] for phase, t in self.events],
            "gripper_min": round(self.gripper_min, 2) if seen else None,
            "gripper_max": round(self.gripper_max, 2) if seen else None,
        }
