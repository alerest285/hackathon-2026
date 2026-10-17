"""Map OUR arm's joint values into the convention flux-3-action-so101 was trained in: LeRobot's old SO-100 calibration.

    cd robot && uv run --python 3.13 python ../console/fit_reference.py --port /dev/ttyACM1 --robot-id my_follower_arm

Why (T21). The checkpoint's state statistics come from VoicAndrei/so100_kitchen (LeRobot v2.1, May 2025). It was recorded
with LeRobot's manual SO-100 calibration (feetech_calibration.py, removed by #777 on 2025-06-05):
    old degrees = 90 + s * (raw - raw_rotated) * 180 / 2048      0 at the "zero" pose, +90 at the "rotated" pose
    gripper     = 0..100 % from closed (zero pose) to fully open (rotated pose)
LeRobot 0.6.1 reads (raw - mid) * 360 / 4095 instead. Both are degrees, so per body joint: old = s * 4095/4096 * ours + b.
- s: shoulder_lift and wrist_roll reversed (flux-action issue #2). Lift and elbow also follow from the rest poses: at rest
  our lift sits at its range minimum and our elbow at its maximum, the dataset's both at their maxima. Pan and wrist_flex
  are checked by hand below.
- b: from the rest pose. Over 47 dataset episode starts it varies by under 2 deg (IQR) on every body joint.
- gripper: both conventions run 0..100 from closed to fully open: identity.
Output: ~/hackathon-data/joint_mapping.json in console/mapping.py's format (model range = the checkpoint's q01/q99), and
~/hackathon-data/limits.json: the dataset's whole joint range in our degrees, +-10 deg, for console/start.sh --limits.
Reads only. Torque goes off so the arm can be moved by hand, and stays off.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mapping import DEFAULT_PATH, JOINTS, MODEL_Q01, MODEL_Q99, JointMap  # noqa: E402

SCALE = 4095 / 4096  # LeRobot 0.6.1 ticks per degree / the old 2048 ticks per half turn
SIGN = np.array([1, -1, 1, 1, -1], float)  # body joints, ours -> old; pan and wrist_flex are confirmed by hand
# VoicAndrei/so100_kitchen, old convention: median state at 47 random episode starts (rest), and the whole-dataset range.
DATASET_REST = np.array([0.8, 182.5, 171.8, 77.3, -87.8])
DATASET_MIN = np.array([-111.1, 63.5, 32.6, -44.9, -155.2])
DATASET_MAX = np.array([67.0, 186.4, 179.5, 101.9, 75.7])
LIMIT_MARGIN = 10.0
PHOTO = "https://raw.githubusercontent.com/huggingface/lerobot/6d723c45a993d1e030dfda20d48a893a74a4fe11/media/so100/follower_{}.webp"


def fit(rest, right=None, straight=None):
    """rest/right/straight: (6,) our degrees. Returns (A, B, notes) with old = A * ours + B per joint (gripper identity)."""
    sign, notes = SIGN.copy(), []
    if right is not None:  # the old rotated pose turns the pan +90 deg to the robot's right
        d = right[0] - rest[0]
        if abs(d) < 15:
            notes.append(f"pan moved {d:+.0f} deg only: kept the default sign {sign[0]:+.0f}")
        else:
            sign[0] = 1.0 if d > 0 else -1.0
            notes.append(f"pan: robot's right reads {d:+.0f} deg in ours -> sign {sign[0]:+.0f}")
    b = DATASET_REST - sign * SCALE * rest[:5]
    if straight is not None:  # wrist in line with the forearm is 0 in the old convention
        w = sign[3] * SCALE * straight[3] + b[3]
        if abs(w) > 25:
            alt = -sign[3] * SCALE * straight[3] + (DATASET_REST[3] + sign[3] * SCALE * rest[3])
            if abs(alt) < abs(w):
                sign[3] = -sign[3]
                b[3] = DATASET_REST[3] - sign[3] * SCALE * rest[3]
                notes.append(f"wrist_flex: straight read {w:.0f} old deg, reversed reads {alt:.0f} -> sign {sign[3]:+.0f}")
            else:
                notes.append(f"wrist_flex: straight reads {w:.0f} old deg (expected ~0): check the straight pose")
        else:
            notes.append(f"wrist_flex: straight reads {w:+.0f} old deg (expected ~0): sign {sign[3]:+.0f} confirmed")
    return np.r_[sign * SCALE, 1.0], np.r_[b, 0.0], notes


def write_outputs(A, B, meta, out: Path, limits_out: Path):
    q01, q99 = np.asarray(MODEL_Q01), np.asarray(MODEL_Q99)
    jm = JointMap((q01 - B) / A, (q99 - B) / A, q01, q99, meta=meta)  # two points on old = A * ours + B
    if out.exists():
        out.rename(out.with_name(f"{out.stem}.before-{time.strftime('%Y%m%dT%H%M%S')}.json"))
    jm.save(out)
    lo, hi = (DATASET_MIN - LIMIT_MARGIN - B[:5]) / A[:5], (DATASET_MAX + LIMIT_MARGIN - B[:5]) / A[:5]
    limits = {j: sorted([round(float(a), 1), round(float(b), 1)]) for j, a, b in zip(JOINTS[:5], lo, hi)}
    limits_out.write_text(json.dumps(limits, indent=1))
    return jm, limits


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True)
    p.add_argument("--robot-id", default="so101_hanoi")
    p.add_argument("--out", type=Path, default=DEFAULT_PATH)
    p.add_argument("--limits-out", type=Path, default=DEFAULT_PATH.with_name("limits.json"))
    a = p.parse_args()
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=a.port, id=a.robot_id, use_degrees=True))
    if not Path(robot.calibration_fpath).is_file():
        sys.exit(f"no calibration file {robot.calibration_fpath}: pass the --robot-id you calibrated with")
    bus = robot.bus
    bus.connect()

    def read(n=15):
        xs = []
        for _ in range(n):
            v = bus.sync_read("Present_Position", num_retry=2)
            xs.append([v[j] for j in JOINTS])
            time.sleep(1 / 30)
        xs = np.array(xs)
        if xs.std(0).max() > 2:
            print(f"  (the arm moved while reading: spread {np.round(xs.std(0), 1).tolist()})")
        return xs.mean(0)

    try:
        if not bus.is_calibrated:
            sys.exit(f"the motors disagree with {robot.calibration_fpath}: run console/calibrate.sh first")
        input("Torque goes OFF next. Hold the arm, then press ENTER. ")
        bus.disable_torque()
        input(f"1/3  Fold the arm into its REST pose ({PHOTO.format('rest')}), let go, press ENTER. ")
        rest = read()
        input("2/3  Stand behind the robot, facing where it reaches. Turn the base about 45 deg to YOUR right, press ENTER. ")
        right = read()
        input("3/3  Lift the forearm a little and straighten the wrist: gripper in line with the forearm, press ENTER. ")
        straight = read()
    finally:
        bus.disconnect(disable_torque=False)

    A, B, notes = fit(rest, right, straight)
    meta = {"method": "old LeRobot SO-100 convention, rest-anchored (console/fit_reference.py, T21)", "robot_id": a.robot_id,
            "sign": np.sign(A[:5]).tolist(), "offset_deg": np.round(B[:5], 2).tolist(), "rest_ours": np.round(rest, 2).tolist(),
            "right_ours": np.round(right, 2).tolist(), "straight_ours": np.round(straight, 2).tolist(), "notes": notes,
            "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    jm, limits = write_outputs(A, B, meta, a.out, a.limits_out)
    print("\n" + "\n".join(notes))
    print(f"\n{'joint':14s} {'sign':>4s} {'offset':>7s}   {'ours rest':>9s} -> {'model':>6s} (dataset rest)   limits, our degrees")
    for i, j in enumerate(JOINTS):
        lim = limits.get(j, ["0", "100"])
        ds = f"{DATASET_REST[i]:6.1f}" if i < 5 else "   ~4 "
        print(f"{j:14s} {np.sign(A[i]):+4.0f} {B[i]:7.1f}   {rest[i]:9.1f} -> {jm.to_model(rest)[i]:6.1f} ({ds})   {lim}")
    print(f"\nWrote {a.out} and {a.limits_out}. Torque is OFF: put the arm back at rest. Then the dry run:")
    print(f"  console/start.sh --port {a.port} --robot-id {a.robot_id} --limits {a.limits_out} --scene <i> --wrist <j>")


if __name__ == "__main__":
    main()
