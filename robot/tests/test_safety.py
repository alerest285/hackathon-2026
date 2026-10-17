"""NativeGuard on its own, and inside SO101.write with the stub bus from test_robot_io."""

import numpy as np
import pytest
from test_robot_io import stub_robot

from so101_client.safety import NativeGuard, SafetyStop

HOME = np.array([0.0, -100.0, 90.0, 60.0, 0.0, 30.0])


def test_small_goals_pass_and_the_gripper_is_never_checked():
    guard = NativeGuard(limits=((-90, 90), (-105, 100), (-95, 95), (-100, 100), (-150, 150)))
    goal = HOME + [3.0, 4.0, -2.0, 1.0, 5.0, 0.0]
    goal[5] = 0.0  # a full squeeze on a disk keeps a large gripper error
    guard.check(goal, measured=HOME, last_goal=HOME)


def test_a_goal_far_from_the_measured_arm_stops():
    with pytest.raises(SafetyStop, match="shoulder_lift goal -70.0 is 30.0 deg from the measured -100.0"):
        NativeGuard().check(HOME + [0, 30, 0, 0, 0, 0], measured=HOME)


def test_a_jump_between_writes_stops():
    with pytest.raises(SafetyStop, match="elbow_flex jumps 16.0 deg in one write"):
        NativeGuard(max_error_deg=None).check(HOME + [0, 0, 16, 0, 0, 0], last_goal=HOME)


def test_limits_stop_beyond_the_margin_but_an_arm_outside_may_come_back():
    guard = NativeGuard(max_error_deg=None, max_step_deg=None, limits=((-90, 90),) * 5)
    guard.check(np.array([91.5, 0, 0, 0, 0, 50]))  # inside the 2 degree margin
    with pytest.raises(SafetyStop, match="shoulder_pan goal 93.0 is outside its limits"):
        guard.check(np.array([93.0, 0, 0, 0, 0, 50]))
    measured = np.array([0.0, -100.0, 0, 0, 0, 50])  # the folded lift sits below its limit
    guard.check(np.array([0.0, -99.0, 0, 0, 0, 50]), measured=measured)  # moving back in is fine
    with pytest.raises(SafetyStop, match="shoulder_lift"):
        guard.check(np.array([0.0, -101.0, 0, 0, 0, 50]), measured=measured)  # further out is not


def test_limits_need_lo_below_hi():
    with pytest.raises(ValueError, match="lo < hi"):
        NativeGuard(limits=((10, -10),) * 5)


def test_so101_write_checks_the_guard_before_the_bus(tmp_path):
    robot = stub_robot(tmp_path)
    robot.connect()  # the stub reads 1..6 and pins the goal there
    bus = robot._robot.bus
    present = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert np.allclose(robot.last_goal, present) and np.allclose(robot.last_state, present)
    n = len(bus.calls)
    robot.write(present + [5, 0, 0, 0, 0, 50])
    assert bus.calls[n:] == ["write Goal_Position"]
    assert np.allclose(robot.last_goal, present + [5, 0, 0, 0, 0, 50])
    with pytest.raises(SafetyStop):
        robot.write(present + [40, 0, 0, 0, 0, 0])
    assert bus.calls[n:] == ["write Goal_Position"]  # nothing went out
    assert np.allclose(robot.last_goal, present + [5, 0, 0, 0, 0, 50])
