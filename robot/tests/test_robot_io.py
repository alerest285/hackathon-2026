"""SO101 against the installed LeRobot 0.6.1, without an arm: the config it builds and the calibration
check that runs before the port is opened. Subprocesses, so HF_LEROBOT_CALIBRATION can point at tmp_path
(LeRobot reads it at import)."""

import os
import subprocess
import sys

PORT = "/dev/tty.so101-test-missing"


def run_python(code, tmp_path):
    env = {**os.environ, "HF_LEROBOT_CALIBRATION": str(tmp_path)}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_lerobot_config(tmp_path):
    stdout = run_python(f"""
from so101_client.robot_io import SO101
robot = SO101({PORT!r}, "hanoi_test")._lerobot()
c = robot.config
print(type(robot).__name__, c.id, c.port, c.use_degrees, c.max_relative_target, c.disable_torque_on_disconnect, c.cameras)
print([(name, m.id, m.norm_mode.value) for name, m in robot.bus.motors.items()])
""", tmp_path)
    # LeRobot 0.6.1 registers one config class as so100_follower and so101_follower; both build SOFollower
    head, motors = stdout.strip().splitlines()
    assert head == f"SOFollower hanoi_test {PORT} True None False {{}}"
    assert motors == ("[('shoulder_pan', 1, 'degrees'), ('shoulder_lift', 2, 'degrees'), ('elbow_flex', 3, 'degrees'), "
                      "('wrist_flex', 4, 'degrees'), ('wrist_roll', 5, 'degrees'), ('gripper', 6, 'range_0_100')]")


def test_missing_calibration_names_the_file_and_the_fix(tmp_path):
    stdout = run_python(f"""
from so101_client.robot_io import SO101
r = SO101({PORT!r}, "hanoi_test")
try:
    r.connect()
except RuntimeError as e:
    print(e)
print("connected", r.connected)
""", tmp_path)
    expected = tmp_path / "robots" / "so_follower" / "hanoi_test.json"
    assert str(expected) in stdout
    assert f"lerobot-calibrate --robot.type=so101_follower --robot.port={PORT} --robot.id=hanoi_test" in stdout
    assert "connected False" in stdout
