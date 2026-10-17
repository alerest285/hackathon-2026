"""Zero-shot SO-101 inference on a REAL recorded frame, compared with what the teleoperator did next.

Run ON the GPU host from the flux-action env:
    cd $FLUX_DIR && uv run --extra encoders --extra data python \
        /workspace/hackathon-2026/infra/gpu/real-so101.py [--episode 0] [--frame 60]

Data: hungho77/so101_left_arm_pick_red_cube_into_pot (LeRobot v3, 30 fps, cameras top + wrist, AV1).
Only the parquet files and the first video chunk of each camera are downloaded (~410 MB).
Mapping: top -> scene, wrist -> wrist (BFL: scene left, wrist right; never swap).
Caveat: this dataset's joint units may be LeRobot's normalized range rather than the degrees the
checkpoint was trained on; the printed state ranges make that visible.
"""
import argparse, time
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download

from flux_action.inference.offline import load_observation
from flux_action.inference.so101 import load_policy

REPO = "hungho77/so101_left_arm_pick_red_cube_into_pot"
CAMS = {"scene": "observation.images.top", "wrist": "observation.images.wrist"}

p = argparse.ArgumentParser()
p.add_argument("--episode", type=int, default=0)
p.add_argument("--frame", type=int, default=60)
p.add_argument("--out", default="/workspace/outputs/real-so101")
a = p.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

root = Path(snapshot_download(REPO, repo_type="dataset", allow_patterns=[
    "meta/*", "meta/episodes/*/*.parquet", "data/*/*.parquet",
    *[f"videos/{k}/chunk-000/file-000.mp4" for k in CAMS.values()]]))
data = pd.read_parquet(root / "data/chunk-000/file-000.parquet")
eps = pd.read_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
tasks = pd.read_parquet(root / "meta/tasks.parquet")

ep = data[data.episode_index == a.episode].sort_values("frame_index")
row = ep[ep.frame_index == a.frame].iloc[0]
task = tasks.index[tasks.task_index == row.task_index][0] if "task_index" in tasks.columns else str(tasks.index[0])
state = np.asarray(row["observation.state"], np.float32)
future = np.stack(ep[ep.frame_index > a.frame]["action"].to_numpy()[:42]).astype(np.float32)
allstate = np.stack(data["observation.state"].to_numpy())
print("task:", task)
print("state:", np.round(state, 1), "| dataset state min", np.round(allstate.min(0), 0), "max", np.round(allstate.max(0), 0))

def frame_at(cam_key):
    e = eps[eps.episode_index == a.episode].iloc[0]
    assert int(e[f"videos/{cam_key}/chunk_index"]) == 0 and int(e[f"videos/{cam_key}/file_index"]) == 0, "episode not in first video file"
    t = float(e[f"videos/{cam_key}/from_timestamp"]) + float(row.timestamp)
    with av.open(str(root / f"videos/{cam_key}/chunk-000/file-000.mp4")) as c:
        s = c.streams.video[0]
        c.seek(int(t / s.time_base), stream=s, backward=True)
        for f in c.decode(s):
            if f.time is not None and f.time >= t - 1e-3:
                return f.to_ndarray(format="rgb24")
    raise RuntimeError("frame not found")

from PIL import Image
imgs = {k: frame_at(v) for k, v in CAMS.items()}
print("raw camera shapes:", {k: v.shape for k, v in imgs.items()})
# Both streams must share one shape (policy._cameras asserts it); the live rig should match too.
imgs = {k: np.asarray(Image.fromarray(v).resize((640, 480), Image.BILINEAR)) for k, v in imgs.items()}
np.savez(out / "observation.npz", **{"images.scene": imgs["scene"], "images.wrist": imgs["wrist"], "state": state})
try:
    Image.fromarray(np.concatenate([imgs["scene"], imgs["wrist"]], 1)).save(out / "input_scene_left_wrist_right.png")
except Exception as e:
    print("png skipped:", e)

policy = load_policy("black-forest-labs/flux-3-action-so101").to("cuda")
batch = load_observation(out / "observation.npz", policy.config, task, "cuda")
torch.cuda.synchronize(); t0 = time.perf_counter()
with torch.no_grad():
    pred = policy.predict_action_chunk(batch).float().cpu().numpy()[0]
torch.cuda.synchronize(); dt = time.perf_counter() - t0
np.save(out / "pred.npy", pred); np.save(out / "recorded.npy", future)

n = min(len(pred), len(future))
err = np.abs(pred[:n] - future[:n])
names = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]
print(f"plan {dt:.2f} s, pred {pred.shape}, compared over {n} steps")
for i in (0, 10, 20, n - 1):
    print(f"step {i:2d}  pred {np.round(pred[i], 1)}  recorded {np.round(future[i], 1)}")
print("mean |pred - recorded| per joint:", dict(zip(names, np.round(err.mean(0), 2))))
print("saved:", out)
