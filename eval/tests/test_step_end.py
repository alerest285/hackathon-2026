"""Synthetic 30 Hz gripper and joint traces through the grasp -> release -> rest detector."""

import numpy as np
import pytest

from hanoi_eval.config import AutoEndConfig
from hanoi_eval.step_end import AutoEndDetector

FPS = 30.0


def make_trace(segments, mirror=False):
    """segments: (duration_s, gripper_from, gripper_to, joint_speed_deg_s). Returns times and (6,) states."""
    times, states = [], []
    t, q = 0.0, np.zeros(5)
    for duration, g0, g1, speed in segments:
        n = int(round(duration * FPS))
        for i in range(n):
            g = g0 + (g1 - g0) * (i + 1) / n
            q = q + speed / FPS
            states.append(np.r_[q, 100.0 - g if mirror else g])
            times.append(t)
            t += 1.0 / FPS
    return times, states


def fires(detector, times, states, cmds=None):
    out = []
    for i, (t, s) in enumerate(zip(times, states)):
        if detector.update(t, s, None if cmds is None else cmds[i]) == "auto":
            out.append(t)
    return out


APPROACH = (0.6, 60, 60, 25)       # open, moving
CLOSE = (0.3, 60, 5, 25)           # closing on the piece
CARRY = (1.0, 5, 5, 30)            # closed, moving
OPEN = (0.3, 5, 60, 30)            # releasing
REST = (2.0, 60, 60, 0)            # still


def low_cfg(**kw):
    return AutoEndConfig(**{"closed_threshold": 15.0, "open_threshold": 25.0, "min_hold_s": 0.5, "rest_s": 1.0,
                            "rest_speed_deg_s": 10.0, **kw})


def test_fires_once_after_grasp_release_rest():
    times, states = make_trace([APPROACH, CLOSE, CARRY, OPEN, REST])
    det = AutoEndDetector(low_cfg())
    fired = fires(det, times, states)
    assert len(fired) == 1
    rest_start = APPROACH[0] + CLOSE[0] + CARRY[0] + OPEN[0]
    assert fired[0] == pytest.approx(rest_start + 1.0, abs=2.5 / FPS)
    assert [p for p, _ in det.events] == ["idle", "closing", "grasped", "released", "fired"]
    assert det.update(times[-1] + 1.0, states[-1]) is None


def test_no_fire_on_gripper_jitter():
    jitter = [(2 / FPS, 60, 5, 25), (3 / FPS, 5, 60, 25)] * 12
    times, states = make_trace([APPROACH, *jitter, REST])
    det = AutoEndDetector(low_cfg())
    assert fires(det, times, states) == []
    assert det.phase == "idle"


def test_no_fire_on_noise_around_the_closed_threshold():
    noise = [(1 / FPS, 14, 14, 20), (1 / FPS, 16, 16, 20)] * 30
    times, states = make_trace([APPROACH, *noise, (2.0, 16, 16, 0)])
    det = AutoEndDetector(low_cfg())
    assert fires(det, times, states) == []
    assert det.phase == "grasped"


def test_high_closed_direction_fires_once():
    times, states = make_trace([APPROACH, CLOSE, CARRY, OPEN, REST], mirror=True)
    det = AutoEndDetector(low_cfg(closed_is="high", closed_threshold=85.0, open_threshold=75.0))
    assert len(fires(det, times, states)) == 1
    low = AutoEndDetector(low_cfg())
    assert fires(low, times, states) == []


def test_no_fire_while_the_arm_keeps_moving_after_release():
    times, states = make_trace([APPROACH, CLOSE, CARRY, OPEN, (3.0, 60, 60, 20)])
    det = AutoEndDetector(low_cfg())
    assert fires(det, times, states) == []
    assert det.phase == "released"


def test_short_grasp_does_not_count():
    times, states = make_trace([APPROACH, (0.1, 60, 5, 25), (0.2, 5, 5, 25), OPEN, REST])
    det = AutoEndDetector(low_cfg())
    assert fires(det, times, states) == []


def test_gripper_closed_at_the_start_must_open_first():
    start_closed = (1.0, 5, 5, 0)
    times, states = make_trace([start_closed, OPEN, REST])
    det = AutoEndDetector(low_cfg())
    assert fires(det, times, states) == []
    times, states = make_trace([start_closed, OPEN, CLOSE, CARRY, OPEN, REST])
    det = AutoEndDetector(low_cfg())
    assert len(fires(det, times, states)) == 1


def test_regrasp_after_release_needs_a_new_release():
    times, states = make_trace([APPROACH, CLOSE, CARRY, OPEN, (0.5, 60, 60, 0), CLOSE, CARRY, OPEN, REST])
    det = AutoEndDetector(low_cfg())
    fired = fires(det, times, states)
    assert len(fired) == 1
    second_release_end = APPROACH[0] + 2 * (CLOSE[0] + CARRY[0] + OPEN[0]) + 0.5
    assert fired[0] == pytest.approx(second_release_end + 1.0, abs=2.5 / FPS)


def test_cmd_signal_reads_the_commanded_gripper():
    times, cmds = make_trace([APPROACH, CLOSE, CARRY, OPEN, REST])
    states = [np.r_[np.zeros(5), 60.0] for _ in cmds]   # dry run: the measured arm never moves
    assert len(fires(AutoEndDetector(low_cfg(signal="cmd")), times, states, cmds)) == 1
    assert fires(AutoEndDetector(low_cfg(signal="state")), times, states, cmds) == []


def test_interrupt_restarts_the_rest_timer():
    times, states = make_trace([APPROACH, CLOSE, CARRY, OPEN, (0.6, 60, 60, 0)])
    det = AutoEndDetector(low_cfg())
    assert fires(det, times, states) == []
    det.interrupt()
    t = times[-1] + 5.0
    assert det.update(t, states[-1]) is None          # no previous sample: no speed yet
    assert det.update(t + 0.5, states[-1]) is None    # still: the rest timer starts here
    assert det.update(t + 1.0, states[-1]) is None
    assert det.update(t + 1.5, states[-1]) == "auto"


def test_summary_reports_events_and_gripper_range():
    times, states = make_trace([APPROACH, CLOSE, CARRY, OPEN, REST])
    det = AutoEndDetector(low_cfg())
    fires(det, times, states)
    s = det.summary()
    assert s["phase"] == "fired"
    assert s["gripper_min"] == pytest.approx(5.0) and s["gripper_max"] == pytest.approx(60.0)
    assert [e[0] for e in s["events"]] == ["idle", "closing", "grasped", "released", "fired"]
