"""Measure OUR arm's joint range by hand and write the mapping the console uses (console/mapping.py).

    cd robot && uv run --python 3.13 python ../console/fit_mapping.py --port /dev/tty.usbmodemXXXX [--seconds 45]

1. Torque goes OFF: the arm falls unless someone holds it. Hold it before pressing ENTER.
2. For --seconds, move the arm by hand through the motions the task needs: rest pose, reaching to every place on the
   table, high and low, gripper fully open and closed, wrist rolled both ways. Cover the workspace, not the extremes.
3. The 1st/99th percentile per joint becomes ~/hackathon-data/joint_mapping.json. The console loads it at start.
Torque stays off at the end: put the arm down in its rest pose.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "robot"))
sys.path.insert(0, str(HERE))
from mapping import DEFAULT_PATH, JOINTS, MODEL_Q01, MODEL_Q99, JointMap  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True)
    p.add_argument("--robot-id", default="so101_hanoi")
    p.add_argument("--seconds", type=float, default=45.0)
    p.add_argument("--out", type=Path, default=DEFAULT_PATH)
    p.add_argument("--flip", nargs="*", default=[], choices=list(JOINTS),
                   help="joints that turn the OPPOSITE way to the checkpoint's data (seen in dry run: the command column moves "
                        "the wrong way). Refit with them listed")
    p.add_argument("--fake", action="store_true", help="test with robot/'s FakeRobot (no hardware)")
    a = p.parse_args()
    if a.fake:
        from so101_client.fakes import FakeRobot

        robot = FakeRobot()
    else:
        from so101_client.robot_io import SO101

        robot = SO101(a.port, a.robot_id)
    robot.connect()
    if not a.fake:
        input("Torque goes OFF next and the arm will fall. Hold the arm, then press ENTER. ")
        robot.set_torque(False)
    print(f"move the arm through the workspace for {a.seconds:.0f} s ...", flush=True)
    samples, t0 = [], time.monotonic()
    rng = np.random.default_rng(0)
    while time.monotonic() - t0 < a.seconds:
        s = np.asarray(robot.read_state(), float)
        if a.fake:  # the fake does not move by itself: sweep it so the fit has something to measure
            s = s + rng.uniform(-40, 40, 6) * np.array([1, 1, 1, 1, 1, 0.5]) + np.array([0, 0, 0, 0, 0, 50])
        samples.append(s)
        n = len(samples)
        if n % 30 == 0:
            print(f"  {time.monotonic() - t0:4.0f} s  " + "  ".join(f"{j[:6]} {v:7.1f}" for j, v in zip(JOINTS, s)), flush=True)
        time.sleep(1 / 30)
    robot.disconnect(torque_off=True) if not a.fake else robot.disconnect()
    X = np.stack(samples)
    lo, hi = np.percentile(X, 1, axis=0), np.percentile(X, 99, axis=0)
    m_lo, m_hi = list(MODEL_Q01), list(MODEL_Q99)
    for j in a.flip:  # reversed direction: our low end maps to the model's high end
        i = JOINTS.index(j)
        m_lo[i], m_hi[i] = MODEL_Q99[i], MODEL_Q01[i]
    jm = JointMap(lo, hi, m_lo, m_hi, meta={"flipped": a.flip, "samples": len(X), "seconds": a.seconds, "robot_id": a.robot_id,
                                "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    jm.save(a.out)
    np.save(Path(a.out).with_suffix(".samples.npy"), X)
    print(f"\n{len(X)} samples. Mapping written to {a.out}")
    print(f"{'joint':14s} {'ours q01':>9s} {'ours q99':>9s}   {'model q01':>9s} {'model q99':>9s}   scale")
    for j, l, h, ml, mh, k in zip(JOINTS, lo, hi, m_lo, m_hi, jm.k):
        warn = "  <- moved less than 10: sweep it again" if abs(h - l) < 10 else ""
        print(f"{j:14s} {l:9.1f} {h:9.1f}   {ml:9.1f} {mh:9.1f}   {k:5.2f}{warn}")
    print("Torque is OFF: rest the arm. Then start the console; it loads this mapping automatically.")


if __name__ == "__main__":
    main()
