"""Physical joint map for the CURRENT calibration, from a teleop dataset instead of hand-held poses (no arm, no GPU).

    cd robot && uv run --python 3.13 --with pandas --with pyarrow python ../docs/research/so100_phys_map_from_dataset.py \
        --dataset ~/.cache/huggingface/lerobot/erh-hack-team-ale/so100_pick_cube_20261017_185713 --robot-id my_follower_arm

Same convention and method as console/fit_reference.py (T21): old = s * 4095/4096 * ours + b per body joint, gripper
identity, signs [+1, -1, +1, +1, -1]. The offsets b come from the dataset's median episode start, matched to the
kitchen dataset's median episode start (both are teleop starts). The dataset must have been recorded with the
calibration file whose hash goes into the map; nothing here can check that.
Limits: the range Saad's teleop actually reached (state and action, min..max) +- MARGIN, cut to the calibrated travel.
Writes to --out/--limits-out (default ~/hackathon-data/maps/), never over ~/hackathon-data/joint_mapping.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CONSOLE = Path(__file__).resolve().parents[2] / "console"
sys.path.insert(0, str(CONSOLE))
from fit_reference import DATASET_REST, SCALE, SIGN, travel  # noqa: E402
from mapping import JOINTS, MODEL_Q01, MODEL_Q99, JointMap  # noqa: E402

MARGIN = 5.0  # deg beyond what the teleop reached

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--dataset", type=Path, required=True)
p.add_argument("--robot-id", required=True)
p.add_argument("--calibration", type=Path, help="default: LeRobot's robots/so_follower/<robot-id>.json")
p.add_argument("--out", type=Path, default=Path("~/hackathon-data/maps/so100_phys_map.json"))
p.add_argument("--limits-out", type=Path, default=Path("~/hackathon-data/maps/so100_phys_limits.json"))
a = p.parse_args()
cal = a.calibration or Path(f"~/.cache/huggingface/lerobot/calibration/robots/so_follower/{a.robot_id}.json")
cal, out, limits_out = cal.expanduser(), a.out.expanduser(), a.limits_out.expanduser()
if out.resolve() == Path("~/hackathon-data/joint_mapping.json").expanduser().resolve():
    sys.exit("Not written: choose the default map yourself (console --mapping <file>), this script never replaces it")

root = a.dataset.expanduser()
d = pd.concat([pd.read_parquet(f, columns=["observation.state", "action", "episode_index", "frame_index"])
               for f in sorted((root / "data").rglob("*.parquet"))]).sort_values(["episode_index", "frame_index"])
state, action = np.stack(d["observation.state"].to_numpy()).astype(float), np.stack(d["action"].to_numpy()).astype(float)
ep = d["episode_index"].to_numpy()
starts = np.stack([state[ep == e][0] for e in np.unique(ep)])
rest = np.median(starts, 0)
iqr = np.subtract(*np.percentile(starts, [75, 25], 0))

A = np.r_[SIGN * SCALE, 1.0]
B = np.r_[DATASET_REST - SIGN * SCALE * rest[:5], 0.0]
lo, hi = np.percentile(state, 1, 0), np.percentile(state, 99, 0)
jm = JointMap(lo, hi, A * lo + B, A * hi + B, meta={
    "method": "old LeRobot SO-100 convention (T21 signs), offsets from teleop episode starts "
              "(docs/research/so100_phys_map_from_dataset.py)",
    "robot_id": a.robot_id,
    "calibration_sha256": hashlib.sha256(cal.read_bytes()).hexdigest(),
    "status": "experimental: wrist_roll sign unverified; scale |k|=1 not compared with the quantile map by the model",
    "sign": SIGN.tolist(), "offset_deg": np.round(B[:5], 2).tolist(),
    "rest_ours": np.round(rest, 2).tolist(), "rest_iqr": np.round(iqr, 2).tolist(),
    "checks": {"shoulder_pan": "sign hand-checked (T21, 15:24Z)", "wrist_flex": "sign hand-checked (T21, 15:24Z)",
               "shoulder_lift": "sign confirmed by data (docs/research/2026-10-17-local-calib-check.md)",
               "elbow_flex": "sign confirmed by data (same)"},
    "not_hand_checked": ["wrist_roll sign (issue #2 only)", "offsets (teleop-start medians, not reference poses)",
                         "gripper endpoints"],
    "source_dataset": str(root), "source_episodes": int(len(starts)), "source_frames": int(len(state)),
    "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
})
both = np.vstack([state, action])
limits = {}
for i, (j, (t_lo, t_hi)) in enumerate(zip(JOINTS[:5], travel(cal))):
    lim = [round(float(max(both[:, i].min() - MARGIN, t_lo)), 1), round(float(min(both[:, i].max() + MARGIN, t_hi)), 1)]
    if not lim[0] < rest[i] < lim[1]:
        sys.exit(f"Not written: {j} limits {lim} exclude the rest pose {rest[i]:.1f}")
    limits[j] = lim

out.parent.mkdir(parents=True, exist_ok=True)
limits_out.parent.mkdir(parents=True, exist_ok=True)
jm.save(out)
limits_out.write_text(json.dumps(limits, indent=1))
X = jm.to_model(state)
inside =((X >= np.minimum(MODEL_Q01, MODEL_Q99)) & (X <= np.maximum(MODEL_Q01, MODEL_Q99))).mean(0) * 100
print(f"{len(starts)} episodes, {len(state)} frames, calibration {cal} sha256 {jm.meta['calibration_sha256'][:12]}")
print(f"{'joint':12s} {'sign':>5s} {'offset':>8s} {'rest ours':>10s} {'rest IQR':>9s} {'limits (ours)':>18s} {'in ckpt range':>14s}")
for i, j in enumerate(JOINTS):
    lim = limits.get(j, ["", ""])
    sgn = f"{SIGN[i]:+.0f}" if i < 5 else "id"
    print(f"{j:12s} {sgn:>5s} {B[i]:8.2f} {rest[i]:10.1f} {iqr[i]:9.1f} {str(lim):>18s} {inside[i]:13.0f}%")
print(f"wrote {out} and {limits_out}")
