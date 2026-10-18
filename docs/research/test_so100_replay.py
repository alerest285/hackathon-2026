"""Replay error-path tests with fake robots only; no LeRobot/serial/camera code is imported."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("replay", Path(__file__).with_name("so100_replay.py"))
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


@pytest.fixture
def rig(monkeypatch, tmp_path):
    class FakeBus:
        def __init__(self, robot):
            self.robot = robot
            self.is_connected = False

        def connect(self, handshake=True):
            self.robot.calls.append(("connect", handshake))
            if "connect_closed" in self.robot.failures:
                raise RuntimeError("connect failed before opening")
            self.is_connected = True
            if "connect_open" in self.robot.failures:
                raise RuntimeError("connect failed after opening")

        @property
        def is_calibrated(self):
            self.robot.calls.append(("calibration",))
            if "calibration_read" in self.robot.failures:
                raise RuntimeError("calibration read failed")
            return "mismatch" not in self.robot.failures

        def disconnect(self, disable_torque=True):
            self.robot.calls.append(("disconnect", disable_torque))
            assert disable_torque is False, "cleanup must not change torque"
            if "disconnect" in self.robot.failures:
                raise RuntimeError("disconnect failed")
            self.is_connected = False

        def write(self, *args, **kwargs):
            pytest.fail("direct motor writes are forbidden")

        sync_write = enable_torque = disable_torque = write

    class FakeRobot:
        instances = []
        failures = set()

        def __init__(self, config):
            assert config.disable_torque_on_disconnect is False
            self.config = config
            self.calls = []
            self.bus = FakeBus(self)
            self.calibration_fpath = "fake-calibration.json"
            self.reads = 0
            self.instances.append(self)

        def connect(self, **kwargs):
            pytest.fail("Robot.connect configures before the calibration guard")

        def disconnect(self):
            pytest.fail("cleanup must address the bus even after partial connection")

        def configure(self):
            assert self.calls[-1] == ("calibration",)
            assert "mismatch" not in self.failures
            self.calls.append(("configure",))
            if "configure" in self.failures:
                raise RuntimeError("configure failed")

        def get_observation(self):
            self.reads += 1
            self.calls.append(("observe", self.reads))
            if self.reads == 1 and "initial_read" in self.failures:
                raise RuntimeError("initial observation failed")
            if self.reads == 2 and "read" in self.failures:
                raise RuntimeError("observation failed after send")
            if self.reads == 3 and "read_later" in self.failures:
                raise RuntimeError("observation failed after send")
            if self.reads == 2 and "interrupt" in self.failures:
                raise KeyboardInterrupt
            return {f"{joint}.pos": 0.0 for joint in replay.JOINTS}

        def send_action(self, command):
            self.calls.append(("send",))
            return command

    # Replace the entire import tree so a regression cannot reach a real hardware implementation.
    for name in ("lerobot", "lerobot.robots", "lerobot.robots.so_follower"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    follower = sys.modules["lerobot.robots.so_follower"]
    follower.SO101Follower = FakeRobot
    follower.SO101FollowerConfig = lambda **kwargs: SimpleNamespace(**kwargs)

    episode = tmp_path / "episode.npz"
    limits = tmp_path / "limits.json"
    np.savez(episode, action=np.zeros((2, 6)))
    limits.write_text(json.dumps({joint: [-90, 90] for joint in replay.JOINTS[:5]}))
    output = tmp_path / "logs" / "replay.npz"
    expanduser = Path.expanduser
    monkeypatch.setattr(Path, "expanduser", lambda p: output if str(p).startswith("~/hackathon-data/replay/") else expanduser(p))
    monkeypatch.setattr("builtins.input", lambda *_: "")
    monkeypatch.setattr(replay.time, "sleep", lambda *_: None)
    monkeypatch.setattr(sys, "argv", [
        "replay", "--port", "FORBIDDEN", "--robot-id", "fake", "--episode", str(episode),
        "--limits", str(limits), "--ramp-s", "0.034",
    ])
    return FakeRobot, replay.main, output


def test_mismatch_rejected_before_configuration_or_commands(rig):
    robots, run, output = rig
    robots.failures = {"mismatch"}
    with pytest.raises(SystemExit, match="not moving"):
        run()
    assert robots.instances[0].calls == [("connect", True), ("calibration",), ("disconnect", False)]
    assert not output.exists()


@pytest.mark.parametrize("failure", ["connect_closed", "connect_open", "calibration_read", "configure", "initial_read"])
def test_failures_close_open_bus_without_torque_writes(rig, failure):
    robots, run, output = rig
    robots.failures = {failure}
    with pytest.raises(RuntimeError):
        run()
    robot = robots.instances[0]
    closes = [call for call in robot.calls if call[0] == "disconnect"]
    assert closes == ([] if failure == "connect_closed" else [("disconnect", False)])
    assert not robot.bus.is_connected
    assert ("send",) not in robot.calls
    assert not output.exists()


@pytest.mark.parametrize("failure", ["read", "interrupt"])
def test_missing_measurement_keeps_a_paired_nan_row(rig, failure):
    robots, run, output = rig
    robots.failures = {failure}
    if failure == "read":
        with pytest.raises(RuntimeError, match="observation failed after send"):
            run()
    else:
        run()
    with np.load(output) as log:
        assert log["cmd"].shape == log["measured"].shape == (1, 6)
        assert np.isnan(log["measured"]).all()
        assert log["measured_valid"].tolist() == [False]
    assert robots.instances[0].calls[-1] == ("disconnect", False)


@pytest.mark.parametrize("secondary", ["disconnect", "save", "summary"])
def test_finalization_errors_do_not_replace_primary_error(rig, monkeypatch, capsys, secondary):
    robots, run, output = rig
    robots.failures = {"read", secondary}
    if secondary == "save":
        def save_failed(*args, **kwargs):
            raise OSError("log save failed")
        monkeypatch.setattr(replay.np, "savez", save_failed)
    elif secondary == "summary":
        # A later failed read leaves one valid row, exercising the numeric summary too.
        robots.failures = {"read_later"}
        def summary_failed(*args, **kwargs):
            raise ValueError("summary failed")
        monkeypatch.setattr(replay.np, "percentile", summary_failed)
    with pytest.raises(RuntimeError, match="observation failed after send"):
        run()
    assert "could not" in capsys.readouterr().err
    assert output.exists() is (secondary != "save")


def test_disconnect_failure_still_saves_log_and_reports_failure(rig):
    robots, run, output = rig
    robots.failures = {"disconnect"}
    with pytest.raises(RuntimeError, match="disconnect failed"):
        run()
    assert output.exists()


def test_interrupt_is_not_replaced_by_log_failure(rig, monkeypatch, capsys):
    robots, run, _ = rig
    robots.failures = {"interrupt"}
    def save_failed(*args, **kwargs):
        raise OSError("log save failed")
    monkeypatch.setattr(replay.np, "savez", save_failed)
    run()
    captured = capsys.readouterr()
    assert "stopped" in captured.out and "log save failed" in captured.err


def test_success_configures_after_check_and_saves_valid_pairs(rig):
    robots, run, output = rig
    run()
    robot = robots.instances[0]
    assert robot.calls[:3] == [("connect", True), ("calibration",), ("configure",)]
    assert robot.calls[-1] == ("disconnect", False)
    with np.load(output) as log:
        assert log["cmd"].shape == log["measured"].shape == (4, 6)
        assert log["measured_valid"].all()
