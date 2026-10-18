"""Save the calibration an SO-100/SO-101 arm's motors hold right now as a LeRobot calibration file (read-only).

    cd robot && uv run --python 3.13 python ../docs/research/so100_save_motor_calibration.py \
        --kind leader --port <leader port> --id my_leader_e82

Use it when the motors were calibrated on another machine (here: Erfan's calibration, which the e82 and soda datasets
were recorded with) and this machine lacks the matching file. This reads the homing offset and min/max position
limits of all six motors into LeRobot's calibration schema. Motor IDs come from the standard SO arm mapping and
Feetech's read_calibration sets drive_mode=0; these two fields are not read from registers. Compare the saved values
or hash with the original file when available. Nothing is written to the motors and torque is not touched, including
on connection or read failures. It refuses to overwrite an existing file; LeRobot finds it under the same --id.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=("leader", "follower"), required=True)
    p.add_argument("--port", required=True)
    p.add_argument("--id", required=True, help="the robot/teleop id to save it under, e.g. my_leader_e82")
    a = p.parse_args()
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS, TELEOPERATORS

    # LeRobot 0.6.1 aliases SO100/101 to these shared calibration namespaces.
    namespace = TELEOPERATORS if a.kind == "leader" else ROBOTS
    out = HF_LEROBOT_CALIBRATION / namespace / f"so_{a.kind}" / f"{a.id}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting (pick another --id)")
    names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    # Use the bus directly: Robot/Teleoperator destructors can disable torque after a partial connection failure.
    bus = FeetechMotorsBus(
        port=a.port,
        motors={
            name: Motor(i, "sts3215", MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.DEGREES)
            for i, name in enumerate(names, 1)
        },
    )
    try:
        bus.connect(handshake=False)
        cal = bus.read_calibration()
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
    text = json.dumps({name: dataclasses.asdict(c) for name, c in cal.items()}, indent=4)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {out}  sha256 {hashlib.sha256(text.encode()).hexdigest()}")
    for name, c in cal.items():
        print(f"  {name:14s} homing {c.homing_offset:5d}  min {c.range_min:5d}  max {c.range_max:5d}  drive {c.drive_mode}")


if __name__ == "__main__":
    main()
