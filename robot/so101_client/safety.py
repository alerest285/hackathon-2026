"""Native-unit tripwires for the SO-101, checked on every command write (codex audit, 2026-10-17).

The control loop clamps in its own units, which are the checkpoint's convention when a joint map wraps the
arm (console/mapping.py), so a model degree can be two native degrees. These checks run below any map, in
LeRobot's degrees, against the last measured state and the last written goal. A violation raises
SafetyStop and the command is not written; the loop then stops for good (latched) and the motors keep the
last goal that did go out. Nothing here clips a command: a clipped command would differ from the one the
loop records as sent.

The gripper (index 5) is left out: a grasp holds a steady error by design, and the loop keeps it in 0..100.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import STATE_NAMES

JOINTS = 5


def load_limits(path) -> tuple | None:
    """5 x (lo, hi) native degrees from a JSON {joint: [lo, hi]} file (readout.py --save-limits, fit_reference.py).
    None for no path. Raises ValueError unless lo < hi for every joint."""
    if path is None:
        return None
    path = Path(path).expanduser()
    data = json.loads(path.read_text())
    limits = tuple((float(data[name][0]), float(data[name][1])) for name in STATE_NAMES[:JOINTS])
    bad = [name for name, (lo, hi) in zip(STATE_NAMES, limits) if not lo < hi]
    if bad:
        raise ValueError(f"{path}: each joint needs lo < hi, not for {', '.join(bad)}")
    return limits


class SafetyStop(RuntimeError):
    """A command broke a native-unit bound. The loop stops on it at once instead of retrying."""


@dataclass
class NativeGuard:
    max_error_deg: float | None = 25.0  # |goal - measured| per joint: an arm that cannot follow, a bad map
    max_step_deg: float | None = 15.0  # |goal - last written goal| per joint and write
    limits: tuple | None = None  # 5 x (lo, hi) native degrees (readout.py --save-limits), or None
    limit_margin_deg: float = 2.0

    def __post_init__(self):
        if self.limits is not None:
            lim = np.asarray(self.limits, dtype=np.float64)
            if lim.shape != (JOINTS, 2) or not (lim[:, 0] < lim[:, 1]).all():
                raise ValueError("limits must be 5 x (lo, hi) native degrees with lo < hi")
            self.limits = tuple((float(lo), float(hi)) for lo, hi in lim)

    def check(self, goal, measured=None, last_goal=None) -> None:
        """Raise SafetyStop when goal breaks a bound. measured/last_goal may be None (not known yet).

        A joint already outside its limits may move back in but not further out, the same rule as the
        loop's clamp, so an arm that starts outside its measured range is not stopped for standing still.
        """
        goal = np.asarray(goal, dtype=np.float64).reshape(-1)[:JOINTS]
        problems = []
        if measured is not None and self.max_error_deg is not None:
            err = np.abs(goal - np.asarray(measured, dtype=np.float64).reshape(-1)[:JOINTS])
            problems += [f"{STATE_NAMES[j]} goal {goal[j]:.1f} is {err[j]:.1f} deg from the measured "
                         f"{float(measured[j]):.1f} (max {self.max_error_deg:g})"
                         for j in np.flatnonzero(err > self.max_error_deg)]
        if last_goal is not None and self.max_step_deg is not None:
            step = np.abs(goal - np.asarray(last_goal, dtype=np.float64).reshape(-1)[:JOINTS])
            problems += [f"{STATE_NAMES[j]} jumps {step[j]:.1f} deg in one write (max {self.max_step_deg:g})"
                         for j in np.flatnonzero(step > self.max_step_deg)]
        if self.limits is not None:
            lim = np.asarray(self.limits, dtype=np.float64)
            lo, hi = lim[:, 0] - self.limit_margin_deg, lim[:, 1] + self.limit_margin_deg
            if measured is not None:
                m = np.asarray(measured, dtype=np.float64).reshape(-1)[:JOINTS]
                lo, hi = np.minimum(lo, m), np.maximum(hi, m)
            problems += [f"{STATE_NAMES[j]} goal {goal[j]:.1f} is outside its limits "
                         f"[{self.limits[j][0]:g}, {self.limits[j][1]:g}]"
                         for j in np.flatnonzero((goal < lo) | (goal > hi))]
        if problems:
            raise SafetyStop("native guard: " + "; ".join(problems))
