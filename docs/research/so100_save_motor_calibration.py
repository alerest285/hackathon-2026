"""Save the calibration an SO-100/SO-101 arm's motors hold right now as a LeRobot calibration file (read-only).

    cd robot && uv run --python 3.13 python ../docs/research/so100_save_motor_calibration.py \
        --kind leader --port <leader port> --id my_leader_e82

Use it when the motors were calibrated on another machine (here: Erfan's calibration, which the e82 and soda datasets
were recorded with) and this machine lacks the matching file. LeRobot's calibration file is exactly the motors'
registers (homing offset, min/max position limit, drive mode), so reading them reproduces the file byte for byte,
e.g. the follower's equals Erfan's my_follower.json (sha256 4a9562e4…). Nothing is written to the motors and torque
is not touched. It refuses to overwrite an existing file; LeRobot then finds it under --id without any calibration
prompt.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=("leader", "follower"), required=True)
    p.add_argument("--port", required=True)
    p.add_argument("--id", required=True, help="the robot/teleop id to save it under, e.g. my_leader_e82")
    a = p.parse_args()
    if a.kind == "leader":
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        device = SO101Leader(SO101LeaderConfig(port=a.port, id=a.id))
    else:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        device = SO101Follower(SO101FollowerConfig(port=a.port, id=a.id))
    out = Path(device.calibration_fpath)
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting (pick another --id)")
    bus = device.bus
    bus.connect(handshake=False)
    try:
        cal = bus.read_calibration()
    finally:
        bus.disconnect(disable_torque=False)
    text = json.dumps({name: dataclasses.asdict(c) for name, c in cal.items()}, indent=4)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out}  sha256 {hashlib.sha256(text.encode()).hexdigest()}")
    for name, c in cal.items():
        print(f"  {name:14s} homing {c.homing_offset:5d}  min {c.range_min:5d}  max {c.range_max:5d}  drive {c.drive_mode}")


if __name__ == "__main__":
    main()
