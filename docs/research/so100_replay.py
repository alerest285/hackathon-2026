"""Replay a recorded episode's commands on the follower: a calibration check you can see.

    cd robot && uv run --python 3.13 python ../docs/research/so100_replay.py \
        --port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61035421-if00 --robot-id my_follower_arm_e82 \
        --episode ~/hackathon-data/replay/e82_ep010.npz --limits ~/hackathon-data/maps/so100_e82_limits.json --speed 0.5

If the motors hold the calibration the episode was recorded with, the arm retraces the demo: same reach, same height
over the table, same grasp spots (compare with <episode>_keyframes.png). A wrong calibration shows as the same motion
offset in space: the wrist bent differently, the gripper too high or into the table.

Safety: it refuses to move unless the motors match the --robot-id calibration file. It ramps from the current pose to
the episode's first command over --ramp-s, then plays at --speed (0.5 = half speed). Every command stays inside
--limits, and body joints move at most --max-step degrees per 30 Hz tick. LeRobot's own guard caps the distance from
the measured position (--max-relative). Ctrl-C stops: torque stays on and the arm holds. Keep a hand on the power
switch. No other program may hold the port (stop the console first).
Writes ~/hackathon-data/replay/replay_<stamp>.npz (commands, measured states and measured_valid;
missing measurements are NaN).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FPS = 30


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True)
    p.add_argument("--robot-id", required=True)
    p.add_argument("--episode", type=Path, required=True, help=".npz with action (N, 6) in native degrees")
    p.add_argument("--limits", type=Path, required=True, help="native limits JSON for the 5 body joints")
    p.add_argument("--speed", type=float, default=0.5)
    p.add_argument("--start-s", type=float, default=0.0)
    p.add_argument("--duration-s", type=float, default=None)
    p.add_argument("--ramp-s", type=float, default=3.0)
    p.add_argument("--max-step", type=float, default=3.0, help="deg per tick, body joints")
    p.add_argument("--max-relative", type=float, default=20.0, help="LeRobot guard: max deg from the measured position")
    a = p.parse_args()

    ep = np.load(a.episode.expanduser())
    A = np.asarray(ep["action"], float)
    i0 = int(a.start_s * FPS)
    i1 = len(A) if a.duration_s is None else min(len(A), i0 + int(a.duration_s * FPS))
    lim = json.loads(a.limits.expanduser().read_text())
    lo = np.array([lim[j][0] for j in JOINTS[:5]] + [0.0])
    hi = np.array([lim[j][1] for j in JOINTS[:5]] + [100.0])
    seg = A[i0:i1]
    outside = [(j, round(float(seg[:, k].min()), 1), round(float(seg[:, k].max()), 1)) for k, j in enumerate(JOINTS[:5])
               if seg[:, k].min() < lo[k] or seg[:, k].max() > hi[k]]
    print(f"episode {a.episode.name}: frames {i0}..{i1} ({(i1 - i0) / FPS:.1f} s recorded, "
          f"{(i1 - i0) / FPS / a.speed:.1f} s at speed {a.speed})")
    if outside:
        print("  these joints leave the limits and will be clipped:", outside)

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=a.port, id=a.robot_id, use_degrees=True,
                                              max_relative_target=a.max_relative, disable_torque_on_disconnect=False))
    cmds, meas, measured_valid = [], [], []
    interrupted = False
    try:
        # This helper has no cameras. Open only the bus: Robot.connect() would configure the motors
        # (including torque changes) before we can reject a calibration mismatch.
        robot.bus.connect()
        if not robot.bus.is_calibrated:
            raise SystemExit(f"the motors do not match {robot.calibration_fpath}: not moving")
        robot.configure()

        def state():
            obs = robot.get_observation()
            return np.array([obs[f"{j}.pos"] for j in JOINTS], float)

        def send(cmd):
            robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, cmd)})

        s0 = state()
        print("now      ", np.round(s0, 1).tolist())
        print("first cmd", np.round(A[i0], 1).tolist(), " (the ramp covers the difference)")
        input("ENTER starts the replay (Ctrl-C stops; the arm holds). Hand on the power switch. ")
        last = s0.copy()
        n_ramp = max(1, int(a.ramp_s * FPS))
        plan = [s0 + (A[i0] - s0) * (k + 1) / n_ramp for k in range(n_ramp)]
        n_play = int((i1 - i0 - 1) / a.speed)
        for n in range(n_play + 1):  # time-based index: interpolate between recorded frames
            x = i0 + n * a.speed
            k = int(x)
            w = x - k
            plan.append(A[k] if k + 1 >= i1 else (1 - w) * A[k] + w * A[k + 1])
        t_next = time.perf_counter()
        for target in plan:
            cmd = np.clip(target, lo, hi)
            step = np.clip(cmd[:5] - last[:5], -a.max_step, a.max_step)
            cmd[:5] = last[:5] + step
            send(cmd)
            last = cmd
            cmds.append(cmd)
            # Keep a row for every successful send, even if its observation fails or is interrupted.
            meas.append(np.full(len(JOINTS), np.nan))
            measured_valid.append(False)
            meas[-1] = state()
            measured_valid[-1] = True
            t_next += 1 / FPS
            time.sleep(max(0.0, t_next - time.perf_counter()))
        print("replay done; the arm holds its last command")
    except KeyboardInterrupt:
        interrupted = True
        print("\nstopped; cleanup leaves motor torque unchanged")
    finally:
        primary_error = sys.exception()
        finalization_error = None
        try:
            if robot.bus.is_connected:
                robot.bus.disconnect(disable_torque=False)
        except Exception as exc:
            finalization_error = exc
            print(f"could not close replay bus: {exc}", file=sys.stderr)
        try:
            if cmds:
                out = Path(f"~/hackathon-data/replay/replay_{time.strftime('%Y%m%dT%H%M%S')}.npz").expanduser()
                out.parent.mkdir(parents=True, exist_ok=True)
                commands, measured = np.array(cmds), np.array(meas)
                valid = np.array(measured_valid, dtype=bool)
                np.savez(out, cmd=commands, measured=measured, measured_valid=valid,
                         episode=str(a.episode), speed=a.speed, start=i0)
                if valid.any():
                    err = np.abs(commands[valid] - measured[valid])[:, :5]
                    print(f"saved {out}; |command - measured| body joints p50 {np.percentile(err, 50):.1f} p95 "
                          f"{np.percentile(err, 95):.1f} deg ({valid.sum()}/{len(valid)} measurements valid)")
                else:
                    print(f"saved {out}; no valid measurements")
        except Exception as exc:
            if finalization_error is None:
                finalization_error = exc
            print(f"could not finish replay log: {exc}", file=sys.stderr)
        if primary_error is None and not interrupted and finalization_error is not None:
            raise finalization_error


if __name__ == "__main__":
    main()
