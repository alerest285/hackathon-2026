"""Calibration check: does the joint map make the model agree with Saad's own hand on Saad's own arm and scene?

Run ON the GPU host, from the flux-action env, against a LeRobot dataset recorded on OUR arm in NATIVE units
(console/teleop.sh record, or console --record in live mode):

    cd /workspace/flux-action && uv run --extra encoders --extra data --with pandas python \
        /workspace/hackathon-2026/infra/gpu/calib_check.py --dataset /workspace/datasets/<name> \
        --map /workspace/datasets/joint_mapping.json [--per-episode 12] [--out /workspace/outputs/calib-check]

For frames sampled across every episode it feeds the real scene/wrist frames + the mapped state to the model, maps the
predicted chunk back to native units, and compares with what the operator actually commanded next. Per joint:

  corr      correlation between predicted and recorded joint motion over the chunk (pred[k]-pred[0] vs rec[k]-rec[0]).
            Clearly NEGATIVE on a joint = its sign in the map is reversed. Near zero = the model ignores/doesn't know it.
  start_gap median (pred[0] - current state), native deg. NOT an offset test: the checkpoint predicts joint deltas
            integrated onto the measured state, so pred[0] always starts at the state. An offset error shows up as
            wrong or weak motion (low corr, high err) instead. Kept only as a sanity check (should be ~0).
  err       mean |pred - recorded| over the chunk, native deg.

Also runs the same frames with --map off (raw native state) for comparison. Writes report.json + report.md to --out.
No images are written.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "console"))
from mapping import JOINTS, JointMap  # noqa: E402

from flux_action.inference.offline import load_observation  # noqa: E402
from flux_action.inference.so101 import load_policy  # noqa: E402

CKPT, REV = "black-forest-labs/flux-3-action-so101", "c9e13b2aca6a0a472b3ea03fd90cff32d3e85849"

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--dataset", type=Path, required=True)
p.add_argument("--map", type=Path, required=True)
p.add_argument("--per-episode", type=int, default=12)
p.add_argument("--max-episodes", type=int, default=0, help="0 = all")
p.add_argument("--guidance", nargs=2, type=float, metavar=("VIDEO", "ACTION"),
               help="override the checkpoint's guidance (3 3) to test e.g. 1 1 (half the compute)")
p.add_argument("--no-raw", action="store_true", help="skip the raw-state comparison (half the time)")
p.add_argument("--out", type=Path, default=Path("/workspace/outputs/calib-check"))
p.add_argument("--scene-key", default="observation.images.scene")
p.add_argument("--wrist-key", default="observation.images.wrist")
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
root = a.dataset
info = json.loads((root / "meta/info.json").read_text())
fps = info["fps"]
data = pd.concat([pd.read_parquet(f) for f in sorted((root / "data").rglob("*.parquet"))])
eps = pd.concat([pd.read_parquet(f) for f in sorted((root / "meta/episodes").rglob("*.parquet"))])
tasks = pd.read_parquet(root / "meta/tasks.parquet")
task_of = {int(r.task_index): str(i) for i, r in tasks.iterrows()} if "task_index" in tasks.columns else {}
jm = JointMap.load(a.map)
print(f"dataset {root.name}: {len(eps)} episodes, {len(data)} frames, fps {fps}; map {a.map} meta {jm.meta}")


def frame_at(ep_row, key, t_in_ep):
    chunk, fidx = int(ep_row[f"videos/{key}/chunk_index"]), int(ep_row[f"videos/{key}/file_index"])
    path = root / f"videos/{key}/chunk-{chunk:03d}/file-{fidx:03d}.mp4"
    t = float(ep_row[f"videos/{key}/from_timestamp"]) + t_in_ep
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(int(t / s.time_base), stream=s, backward=True)
        for f in c.decode(s):
            if f.time is not None and f.time >= t - 0.5 / fps:
                return np.asarray(Image.fromarray(f.to_ndarray(format="rgb24")).resize((640, 480)))
    raise RuntimeError(f"no frame at {t:.3f}s in {path}")


policy = load_policy(CKPT, revision=REV).to("cuda")
if a.guidance:
    policy.config.guidance_scale, policy.config.guidance_scale_action = a.guidance
plan_ms = []
rows = {"mapped": [], "raw": []}
t0 = time.time()
for i_ep, (_, e) in enumerate(eps.iterrows()):
    if a.max_episodes and i_ep >= a.max_episodes:
        break
    ep = int(e.episode_index)
    d = data[data.episode_index == ep].sort_values("frame_index")
    if len(d) < 60:
        continue
    task = task_of.get(int(d.task_index.iloc[0]), "") if "task_index" in d else ""
    for fi in np.linspace(5, len(d) - 45, a.per_episode).astype(int):
        r = d.iloc[fi]
        st = np.asarray(r["observation.state"], np.float32)
        rec = np.stack(d["action"].to_numpy()[fi + 1: fi + 43]).astype(np.float32)
        imgs = {"scene": frame_at(e, a.scene_key, float(r.timestamp)), "wrist": frame_at(e, a.wrist_key, float(r.timestamp))}
        for mode in (("mapped",) if a.no_raw else ("mapped", "raw")):
            s_in = jm.to_model(st) if mode == "mapped" else st
            np.savez("/tmp/calib_obs.npz", **{"images.scene": imgs["scene"], "images.wrist": imgs["wrist"], "state": s_in})
            batch = load_observation("/tmp/calib_obs.npz", policy.config, task, "cuda")
            torch.cuda.synchronize(); tp = time.perf_counter()
            with torch.no_grad():
                pred = policy.predict_action_chunk(batch)
            torch.cuda.synchronize(); plan_ms.append((time.perf_counter() - tp) * 1000)
            pred = pred.float().cpu().numpy()[0]
            if mode == "mapped":
                pred = jm.to_ours(pred)
            n = min(len(pred), len(rec))
            rows[mode].append({"ep": ep, "frame": int(fi), "task": task, "state": st.tolist(),
                               "pred_d": (pred[:n] - pred[0]).tolist(), "rec_d": (rec[:n] - rec[0]).tolist(),
                               "start_gap": (pred[0] - st).tolist(), "err": np.abs(pred[:n] - rec[:n]).mean(0).tolist()})
    print(f"episode {ep} done ({len(rows['mapped'])} samples, {time.time() - t0:.0f} s)", flush=True)


def summarize(rs):
    pd_ = np.concatenate([np.asarray(r["pred_d"]) for r in rs]); rd = np.concatenate([np.asarray(r["rec_d"]) for r in rs])
    out = {}
    for j, name in enumerate(JOINTS):
        x, y = pd_[:, j], rd[:, j]
        corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-6 and y.std() > 1e-6 else float("nan")
        gaps = np.array([r["start_gap"][j] for r in rs]); errs = np.array([r["err"][j] for r in rs])
        verdict = "SIGN reversed?" if corr < -0.2 else ("weak/unknown" if not corr > 0.2 else "ok")
        if y.std() < 2:
            verdict = "not judged (operator barely moved it)"
        out[name] = {"corr": round(corr, 2), "start_gap_median": round(float(np.median(gaps)), 1),
                     "err_mean": round(float(errs.mean()), 1), "recorded_motion_std": round(float(y.std()), 1), "verdict": verdict}
    return out


report = {"dataset": str(root), "map": str(a.map), "map_meta": jm.meta, "samples": len(rows["mapped"]),
          "guidance": list(a.guidance) if a.guidance else "checkpoint (3, 3)",
          "plan_ms_p50": round(float(np.median(plan_ms[1:] or plan_ms)), 1),
          "checkpoint": f"{CKPT}@{REV[:8]}", "mapped": summarize(rows["mapped"]), "raw": summarize(rows["raw"]) if rows["raw"] else None}
(a.out / "report.json").write_text(json.dumps(report, indent=1))
lines = [f"# Calibration check {time.strftime('%Y-%m-%d %H:%M')}", "", f"dataset `{root}`, map `{a.map}`, {report['samples']} samples, guidance {report['guidance']}, plan p50 {report['plan_ms_p50']} ms", "",
         "| joint | corr mapped | corr raw | start gap mapped | err mapped | err raw | verdict (mapped) |", "|---|---|---|---|---|---|---|"]
for name in JOINTS:
    m = report["mapped"][name]; r = (report["raw"] or {}).get(name, {"corr": "-", "err_mean": "-"})
    lines.append(f"| {name} | {m['corr']} | {r['corr']} | {m['start_gap_median']} | {m['err_mean']} | {r['err_mean']} | {m['verdict']} |")
lines += ["", "Read per joint: corr > 0.2 = model moves with the operator (map plausible); corr < -0.2 = sign reversed in the map; "
          "corr near 0 or err much worse than raw = offset/scale suspect. Compare against the raw columns: the map must not be "
          "worse than raw. start gap is a sanity check only (deltas are integrated onto the state)."]
(a.out / "report.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
