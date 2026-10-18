"""Calibration snapshot tests; all serial access is forbidden and all motor reads are faked."""
import dataclasses
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import serial

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
import lerobot.motors.feetech as feetech
from lerobot.utils import constants


NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
CALIBRATION = {
    name: MotorCalibration(i, 0, -100 + i, 200 + i, 3800 - i)
    for i, name in enumerate(NAMES, 1)
}
spec = importlib.util.spec_from_file_location("snapshot", Path(__file__).with_name("so100_save_motor_calibration.py"))
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)


@pytest.fixture(autouse=True)
def no_serial(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A calibration snapshot test attempted to open a serial port")
    monkeypatch.setattr(serial, "Serial", forbidden)


@pytest.fixture
def rig(monkeypatch, tmp_path):
    monkeypatch.setattr(constants, "HF_LEROBOT_CALIBRATION", tmp_path)

    class FakeBus:
        instances = []
        failure = None
        on_read = None

        def __init__(self, port, motors):
            self.port, self.motors = port, motors
            self.is_connected = False
            self.calls = []
            self.instances.append(self)

        def connect(self, handshake=True):
            self.calls.append(("connect", handshake))
            if self.failure == "connect_closed":
                raise RuntimeError("connect failed before opening")
            self.is_connected = True
            if self.failure == "connect_open":
                raise RuntimeError("connect failed after opening")

        def read_calibration(self):
            self.calls.append(("read",))
            if self.failure == "read":
                raise RuntimeError("read failed")
            if self.on_read:
                self.on_read()
            return CALIBRATION

        def disconnect(self, disable_torque=True):
            self.calls.append(("disconnect", disable_torque))
            assert disable_torque is False, "cleanup must never disable torque"
            if self.failure == "cleanup":
                raise RuntimeError("close failed")
            self.is_connected = False

        def write(self, *args, **kwargs):
            pytest.fail("Motor writes are forbidden")

        sync_write = enable_torque = disable_torque = write

    monkeypatch.setattr(feetech, "FeetechMotorsBus", FakeBus)

    def run(kind="leader"):
        monkeypatch.setattr(sys, "argv", ["snapshot", "--kind", kind, "--port", "FORBIDDEN", "--id", "audit"])
        snapshot.main()

    def output(kind="leader"):
        return tmp_path / ("teleoperators" if kind == "leader" else "robots") / f"so_{kind}" / "audit.json"

    return FakeBus, run, output


@pytest.mark.parametrize("kind", ["leader", "follower"])
def test_success_preserves_all_six_motors_in_correct_namespace(rig, kind):
    bus_type, run, output = rig
    run(kind)
    bus = bus_type.instances[0]
    assert bus.calls == [("connect", False), ("read",), ("disconnect", False)]
    assert not bus.is_connected
    assert tuple(bus.motors) == NAMES
    assert [m.id for m in bus.motors.values()] == list(range(1, 7))
    assert all(m.model == "sts3215" for m in bus.motors.values())
    assert bus.motors["gripper"].norm_mode == MotorNormMode.RANGE_0_100
    assert json.loads(output(kind).read_text()) == {name: dataclasses.asdict(c) for name, c in CALIBRATION.items()}


@pytest.mark.parametrize("failure", ["read", "connect_open", "connect_closed", "cleanup"])
def test_errors_never_change_torque_or_save_partial_calibration(rig, failure):
    bus_type, run, output = rig
    bus_type.failure = failure
    with pytest.raises(RuntimeError):
        run()
    bus = bus_type.instances[0]
    closes = [call for call in bus.calls if call[0] == "disconnect"]
    assert closes == ([] if failure == "connect_closed" else [("disconnect", False)])
    assert not output().exists()


def test_existing_file_refused_before_connecting(rig):
    bus_type, run, output = rig
    output().parent.mkdir(parents=True)
    output().write_bytes(b"existing calibration")
    with pytest.raises(SystemExit, match="not overwriting"):
        run()
    assert output().read_bytes() == b"existing calibration"
    assert bus_type.instances == []


def test_file_created_during_read_is_not_overwritten(rig):
    bus_type, run, output = rig
    def competing_writer():
        output().parent.mkdir(parents=True)
        output().write_bytes(b"concurrent calibration")
    bus_type.on_read = staticmethod(competing_writer)
    with pytest.raises(FileExistsError):
        run()
    assert output().read_bytes() == b"concurrent calibration"
    assert bus_type.instances[0].calls[-1] == ("disconnect", False)


def test_installed_feetech_reads_three_registers_per_motor_without_writes():
    # Bypass SDK construction entirely, then exercise the actual installed implementation.
    bus = object.__new__(FeetechMotorsBus)
    bus.motors = {name: Motor(i, "sts3215", MotorNormMode.DEGREES) for i, name in enumerate(NAMES, 1)}
    bus.protocol_version = 0
    calls = []
    fields = {"Homing_Offset": "homing_offset", "Min_Position_Limit": "range_min", "Max_Position_Limit": "range_max"}
    def read(register, motor, *, normalize):
        assert normalize is False
        calls.append((motor, register))
        return getattr(CALIBRATION[motor], fields[register])
    def forbidden(*args, **kwargs):
        pytest.fail("read_calibration attempted a motor write")
    bus.read = read
    bus.write = bus._write = forbidden
    assert bus.read_calibration() == CALIBRATION
    assert len(calls) == 18
    assert set(calls) == {(motor, register) for motor in NAMES for register in fields}
