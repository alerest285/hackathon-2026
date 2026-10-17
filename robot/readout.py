"""Print the six joint values of the real arm at 10 Hz, to find gripper thresholds and joint limits.

    uv run --python 3.13 python readout.py --port /dev/tty.usbmodemXXXX --robot-id so101_hanoi
    uv run --python 3.13 python readout.py --port ... --robot-id ... --torque-off --save-limits limits.json

--torque-off releases the motors so the arm can be moved by hand; it falls if nobody holds it, and torque
stays off at exit. Ctrl-C ends the readout and prints the range seen per joint; --save-limits writes the
five joint ranges (widened by --margin) in the format run.py --limits reads.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from so101_client import STATE_NAMES

SHORT = ("pan", "lift", "elbow", "wflex", "wroll", "grip")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--port", required=True, help="serial port (lerobot-find-port)")
    p.add_argument("--robot-id", required=True, help="the id used with lerobot-calibrate")
    p.add_argument("--torque-off", action="store_true", help="release the motors to move the arm by hand")
    p.add_argument("--yes", action="store_true", help="skip the confirmation before torque goes off")
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0: until Ctrl-C)")
    p.add_argument("--save-limits", type=Path, help="write {joint: [lo, hi]} for the 5 joints")
    p.add_argument("--margin", type=float, default=0.0, help="degrees added on both sides of each saved range")
    a = p.parse_args(argv)
    if a.margin < 0:
        p.error("--margin must be >= 0 (a negative margin can leave lo >= hi)")

    from so101_client.robot_io import SO101

    robot = SO101(a.port, a.robot_id)
    try:
        robot.connect()
    except (RuntimeError, ConnectionError) as e:
        print(f"could not connect: {e}")
        return 1
    print(f"connected to {a.robot_id} on {a.port} (calibration {robot.calibration_path})")
    torque_off = False
    lo = np.full(6, np.inf)
    hi = np.full(6, -np.inf)
    try:
        if a.torque_off:
            if not a.yes:
                input("Torque goes off and the arm falls unless you hold it. Hold the arm, then press ENTER: ")
            robot.set_torque(False)
            torque_off = True
            print("torque off: move the arm by hand")
        print("      t " + " ".join(f"{s:>8s}" for s in SHORT) + "   (degrees x5, gripper %)")
        t0 = time.perf_counter()
        n = 0
        while not a.seconds or time.perf_counter() - t0 < a.seconds:
            n += 1
            state = robot.read_state()
            lo, hi = np.minimum(lo, state), np.maximum(hi, state)
            print(f"{time.perf_counter() - t0:7.1f} " + " ".join(f"{v:8.2f}" for v in state), flush=True)
            delay = t0 + n / a.hz - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        print()
    finally:
        robot.disconnect(torque_off=torque_off)
        print(f"disconnected, torque {'off (the arm is limp)' if torque_off else 'on'}")
    if np.isfinite(lo).all():
        print("\nrange seen")
        for name, a_lo, a_hi in zip(STATE_NAMES, lo, hi):
            print(f"  {name:14s} {a_lo:8.2f} .. {a_hi:8.2f}")
        if a.save_limits:
            limits = {name: [round(float(lo[i]) - a.margin, 2), round(float(hi[i]) + a.margin, 2)]
                      for i, name in enumerate(STATE_NAMES[:5])}
            limits["gripper_seen"] = [round(float(lo[5]), 2), round(float(hi[5]), 2)]
            a.save_limits.write_text(json.dumps(limits, indent=2) + "\n")
            print(f"wrote {a.save_limits} (margin {a.margin:g} deg); use it with run.py --limits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
