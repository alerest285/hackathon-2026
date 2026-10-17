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


class StubBus:
    """Records the calls SO101.connect makes on the bus."""

    def __init__(self):
        self.calls = []
        self.is_connected = False
        self.is_calibrated = True

    def connect(self):
        self.calls.append("connect")
        self.is_connected = True

    def sync_read(self, name, num_retry=0):
        self.calls.append(f"read {name}")
        return {"shoulder_pan": 1.0, "shoulder_lift": 2.0, "elbow_flex": 3.0, "wrist_flex": 4.0, "wrist_roll": 5.0,
                "gripper": 6.0}

    def sync_write(self, name, values):
        self.calls.append(f"write {name}")

    def disconnect(self, disable_torque):
        self.calls.append(f"disconnect torque_off={disable_torque}")
        self.is_connected = False


class StubFollower:
    def __init__(self, bus, calibration, fail_configure=False):
        self.bus, self.calibration_fpath, self.fail_configure = bus, str(calibration), fail_configure

    def configure(self):
        self.bus.calls.append("configure")
        if self.fail_configure:
            raise ConnectionError("no status packet")


def stub_robot(tmp_path, **kw):
    from so101_client.robot_io import SO101

    calibration = tmp_path / "so101_hanoi.json"
    calibration.write_text("{}")
    robot = SO101(PORT, "so101_hanoi")
    robot._robot = StubFollower(StubBus(), calibration, **kw)
    return robot


def test_connect_pins_the_goal_before_configure_turns_torque_back_on(tmp_path):
    robot = stub_robot(tmp_path)
    robot.connect()
    pin = ["read Present_Position", "write Goal_Position"]
    assert robot._robot.bus.calls == ["connect", *pin, "configure", *pin]


def test_a_failed_connect_repins_the_goal_and_keeps_torque_on(tmp_path):
    robot = stub_robot(tmp_path, fail_configure=True)
    try:
        robot.connect()
    except ConnectionError:
        pass
    else:
        raise AssertionError("connect() must raise")
    pin = ["read Present_Position", "write Goal_Position"]
    assert robot._robot.bus.calls == ["connect", *pin, "configure", *pin, "disconnect torque_off=False"]
