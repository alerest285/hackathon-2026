"""The SO-101 follower over LeRobot 0.6.1: so101_follower with use_degrees=True, no cameras.

Joints 0-4 are degrees, the gripper is RANGE_0_100 (which end is closed depends on the calibration).
Differences from LeRobot's own connect(): no interactive calibration (a missing calibration file or a
mismatch with the motors is an error that names the fix), and the goal is pinned to the present position
right after configuring. LeRobot 0.6.1 disables torque on disconnect by default
(SOFollowerConfig.disable_torque_on_disconnect, motors_bus.disconnect): this client turns that off, so
the arm keeps holding its last goal unless disconnect(torque_off=True) is asked for.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import STATE_NAMES

log = logging.getLogger(__name__)


class SO101:
    state_names = STATE_NAMES

    def __init__(self, port: str, robot_id: str, use_degrees: bool = True, *, num_read_retries: int = 2):
        self.port = port
        self.robot_id = robot_id
        self.use_degrees = use_degrees
        self.num_read_retries = num_read_retries
        self._robot = None

    def _lerobot(self):
        if self._robot is None:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

            config = SO101FollowerConfig(
                port=self.port,
                id=self.robot_id,
                use_degrees=self.use_degrees,
                cameras={},
                max_relative_target=None,
                disable_torque_on_disconnect=False,
                num_read_retries=self.num_read_retries,
            )
            self._robot = SO101Follower(config)
        return self._robot

    @property
    def calibration_path(self) -> Path:
        """Where LeRobot 0.6.1 keeps the calibration of this robot id."""
        return Path(self._lerobot().calibration_fpath)

    @property
    def connected(self) -> bool:
        return self._robot is not None and self._robot.bus.is_connected

    def calibrate_command(self) -> str:
        return f"lerobot-calibrate --robot.type=so101_follower --robot.port={self.port} --robot.id={self.robot_id}"

    def connect(self) -> None:
        robot = self._lerobot()
        path = self.calibration_path
        if not path.is_file():
            raise RuntimeError(
                f"no calibration for robot id {self.robot_id!r}: LeRobot 0.6.1 expects {path}. "
                f"Create it with: {self.calibrate_command()}"
            )
        bus = robot.bus
        bus.connect()
        try:
            if not bus.is_calibrated:
                raise RuntimeError(
                    f"the motors' calibration differs from {path}. Run {self.calibrate_command()} and press ENTER "
                    "to write that file to the motors, or type c to calibrate again."
                )
            robot.configure()  # LeRobot's own: position mode, PID gains, gripper current limits
            present = bus.sync_read("Present_Position", num_retry=self.num_read_retries)
            bus.sync_write("Goal_Position", present)
        except Exception:
            bus.disconnect(disable_torque=False)
            raise
        log.info("SO-101 %s connected on %s", self.robot_id, self.port)

    def read_state(self) -> np.ndarray:
        """(6,) float32: degrees x5 + gripper %, in state_names order."""
        values = self._robot.bus.sync_read("Present_Position", num_retry=self.num_read_retries)
        return np.array([values[name] for name in STATE_NAMES], dtype=np.float32)

    def write(self, cmd: np.ndarray) -> None:
        cmd = np.asarray(cmd, dtype=np.float64).reshape(len(STATE_NAMES))
        if not np.isfinite(cmd).all():
            raise ValueError(f"refusing a non-finite command {cmd}")
        self._robot.bus.sync_write("Goal_Position", {name: float(v) for name, v in zip(STATE_NAMES, cmd)})

    def set_torque(self, enabled: bool) -> None:
        """Torque off lets the arm be moved by hand (it falls if nobody holds it). Torque on first pins the
        goal to the present position, so the arm stays where it is."""
        bus = self._robot.bus
        if enabled:
            present = bus.sync_read("Present_Position", num_retry=self.num_read_retries)
            bus.sync_write("Goal_Position", present)
            bus.enable_torque()
        else:
            bus.disable_torque()

    def disconnect(self, torque_off: bool = False) -> None:
        """Close the port. Torque stays on (the arm holds its last goal) unless torque_off=True."""
        if not self.connected:
            return
        self._robot.bus.disconnect(disable_torque=torque_off)
        log.info("SO-101 %s disconnected, torque %s", self.robot_id, "off" if torque_off else "on")
