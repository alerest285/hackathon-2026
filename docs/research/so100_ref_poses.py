"""Reference-pose check (T27): measure each joint's sign and offset against the convention the checkpoint was trained in.

    cd robot && uv run --python 3.13 python ../docs/research/so100_ref_poses.py --port /dev/ttyACM1 \
        --robot-id my_follower_arm --map ~/hackathon-data/maps/so100_phys_map.json

The checkpoint's statistics come from LeRobot's old manual SO-100 calibration (see console/fit_reference.py). It
defined two poses from photos (PHOTO below): ZERO, the arm stretched straight out horizontally (every joint 0), and
ROTATED, every joint turned +90 (upper arm vertical, forearm horizontal forward, gripper pointing down and rolled, jaws
fully open). Hold the arm in each pose; the script reads our degrees and compares them with --map, then fits per body
joint
    old = s * 4095/4096 * ours + b      s = sign(rotated - zero), b = mean of the two poses' offsets
and writes that as a candidate map to --out (never over joint_mapping.json). Pan stays where it is in both poses
(its sign and offset come from --map, which fit_reference.py hand-checked); the gripper stays identity.
Reads only: torque goes OFF and stays off. After ENTER you get --countdown seconds to pose with both hands.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "console"))
from fit_reference import PHOTO, SCALE  # noqa: E402
from mapping import JOINTS, MODEL_Q01, MODEL_Q99, JointMap  # noqa: E402

# Ours must move by ~90 deg between the poses; hand posing is good to a few degrees, so allow a wide band.
SPAN_OK = (60.0, 120.0)
MAX_DISAGREE = 15.0  # deg between the two poses' offsets before a joint is flagged


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True)
    p.add_argument("--robot-id", required=True)
    p.add_argument("--map", type=Path, required=True, help="map to check, e.g. ~/hackathon-data/maps/so100_phys_map.json")
    p.add_argument("--out", type=Path, default=Path("~/hackathon-data/maps/so100_refpose_map.json"))
    p.add_argument("--countdown", type=float, default=5.0)
    a = p.parse_args()
    out = a.out.expanduser()
    if out.name == "joint_mapping.json":
        sys.exit("Not written: pick the default map yourself (console --mapping <file>); this script never replaces it")
    jm = JointMap.load(a.map.expanduser())
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=a.port, id=a.robot_id, use_degrees=True))
    cal = Path(robot.calibration_fpath)
    if not cal.is_file():
        sys.exit(f"no calibration file {cal}: pass the --robot-id you calibrated with")
    cal_sha = hashlib.sha256(cal.read_bytes()).hexdigest()
    if jm.meta.get("robot_id") != a.robot_id or jm.meta.get("calibration_sha256") != cal_sha:
        sys.exit(f"{a.map} was fitted for {jm.meta.get('robot_id')!r} under another calibration file: refit it first")
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
            print(f"  (the arm moved while reading: spread {np.round(xs.std(0), 1).tolist()}; hold it still)")
        return xs.mean(0)

    def capture(prompt):
        input(prompt + "\n     Press ENTER, then pose. ")
        for s in range(int(a.countdown), 0, -1):
            print(f"  reading in {s} ...", flush=True)
            time.sleep(1)
        x = read()
        print("  read:", "  ".join(f"{j} {v:7.1f}" for j, v in zip(JOINTS, x)))
        return x

    try:
        if not bus.is_calibrated:
            sys.exit(f"the motors disagree with {cal}: run console/calibrate.sh first")
        input("Torque goes OFF next and stays off. Hold the arm, then press ENTER. ")
        bus.disable_torque()
        zero = capture(f"1/2  ZERO pose ({PHOTO.format('zero')}): base facing forward; upper arm, forearm and gripper "
                       "in one straight horizontal line pointing forward; gripper's motor on top as in the photo; jaws "
                       "closed.")
        rot = capture(f"2/2  ROTATED pose ({PHOTO.format('rotated')}): base unchanged; upper arm vertical; forearm "
                      "horizontal pointing forward; gripper pointing straight down, rolled as in the photo; jaws fully "
                      "open.")
    finally:
        bus.disconnect(disable_torque=False)
    report(a, jm, out, cal_sha, zero, rot)


def report(a, jm: JointMap, out: Path, cal_sha: str, zero, rot):
    for name, x in (("zero", zero), ("rotated", rot)):
        if np.asarray(x).shape != (6,) or not np.isfinite(x).all():
            sys.exit(f"Not written: the {name} reading is invalid: {np.round(x, 1).tolist()}")
    map_sign = np.sign(jm.k[:5])
    map_off = jm.to_model(np.zeros(6))[:5]
    mz, mr = jm.to_model(zero), jm.to_model(rot)
    print(f"\n{a.map} in old degrees (expected: zero 0, rotated +90; pan does not move):")
    print(f"{'joint':14s} {'zero':>7s} {'rotated':>8s}  {'moved':>7s} {'sign':>5s} {'map':>4s} {'offset':>7s} {'map':>7s} "
          f"{'poses disagree':>14s}  verdict")
    sign, off, verdict = map_sign.copy(), map_off.astype(float).copy(), {}
    for i, j in enumerate(JOINTS[:5]):
        d = (rot[i] - zero[i] + 180) % 360 - 180  # wrist_roll turns fully and can wrap at +-180 between the poses
        if i == 0:
            verdict[j] = "kept from the map"
            print(f"{j:14s} {mz[i]:7.1f} {mr[i]:8.1f}  {d:+7.1f} {'':5s} {map_sign[i]:+4.0f} {off[i]:7.1f} {map_off[i]:7.1f} "
                  f"{'':14s}  {verdict[j]}")
            continue
        s = float(np.sign(d)) if d else 0.0
        b_zero, b_rot = 0.0 - s * SCALE * zero[i], 90.0 - s * SCALE * (zero[i] + d)
        sign[i], off[i] = s, (b_zero + b_rot) / 2
        if not SPAN_OK[0] <= abs(d) <= SPAN_OK[1]:
            v = f"UNCLEAR: moved {d:+.0f} deg between the poses (expected about 90), redo"
        elif abs(b_zero - b_rot) > MAX_DISAGREE:
            v = f"UNCLEAR: the poses disagree by {abs(b_zero - b_rot):.0f} deg, redo"
        elif s != map_sign[i]:
            v = "SIGN REVERSED vs the map"
        else:
            v = f"sign ok, offset {off[i] - map_off[i]:+.1f} deg vs the map"
        verdict[j] = v
        print(f"{j:14s} {mz[i]:7.1f} {mz[i] + jm.k[i] * d:8.1f}  {d:+7.1f} {s:+5.0f} {map_sign[i]:+4.0f} {off[i]:7.1f} {map_off[i]:7.1f} "
              f"{abs(b_zero - b_rot):14.1f}  {v}")
    print(f"gripper: zero (closed) {zero[5]:.1f}, rotated (fully open) {rot[5]:.1f}   (old convention: 0 closed, 100 open)")
    if any(v.startswith("UNCLEAR") for v in verdict.values()):
        sys.exit("\nNot written: at least one joint is unclear. Check the poses against the photos and run again.")
    A, B = np.r_[sign * SCALE, 1.0], np.r_[off, 0.0]
    q01, q99 = np.asarray(MODEL_Q01), np.asarray(MODEL_Q99)
    meta = {"method": "old LeRobot SO-100 convention from the zero and rotated reference poses "
                      "(docs/research/so100_ref_poses.py, T27)",
            "robot_id": a.robot_id, "calibration_sha256": cal_sha,
            "status": "experimental: hand-posed reference photos; pan and gripper kept from the checked map",
            "checked_against": str(a.map), "verdict": verdict, "sign": sign.tolist(), "offset_deg": np.round(off, 2).tolist(),
            "zero_ours": np.round(zero, 2).tolist(), "rotated_ours": np.round(rot, 2).tolist(),
            "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out.parent.mkdir(parents=True, exist_ok=True)
    JointMap((q01 - B) / A, (q99 - B) / A, q01, q99, meta=meta).save(out)
    print(f"\nWrote {out} (a candidate: the console uses it only with --mapping {out}).")
    print("Torque is OFF: put the arm back at rest.")


if __name__ == "__main__":
    main()
