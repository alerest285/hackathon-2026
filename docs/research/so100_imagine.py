"""Imagination viewer: decode the future video FLUX 3 Action generates alongside its actions, next to what really happened.

Run ON the GPU pod with the flux-action venv (not `uv run`: a sync would drop natten):

    cd /workspace/hackathon-2026 && HF_HOME=/workspace/hf /workspace/flux-action/.venv/bin/python \
        docs/research/so100_imagine.py --dataset /workspace/datasets/lora_so100_pick_cube_hfsim_v1 \
        --checkpoint /workspace/ckpt/flux-3-action-so101-hfsim-norm --episodes 0 20 40 --per-episode 3

The SO-101 policy denoises a future video and a 42-step action chunk jointly (policy._sample), then keeps only the
actions. This script wraps the sampler to keep its final state, unpacks the video tokens (prc_vid's "(t h w) c"),
and decodes them with the policy's own video VAE: 11 latent frames -> 41 frames of the 512x256 canvas (scene left,
wrist right). Inputs are the live window: 8 measured states, the 8 preceding commands, frames t-7 and t (256x256
per camera), the episode's caption. The dataset must be in the units the checkpoint expects (native LeRobot
degrees for the hfsim-norm base).

Writes to --out: <ep>_<t>.mp4 (top: imagined, bottom: real frames t..t+40, 30 fps), sheet_<ep>.png (columns
k = 0, 8, ..., 40; per moment an imagined row and a real row), summary.json (predicted vs recorded joint motion).
"""
from __future__ import annotations

import argparse
import json
import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from einops import rearrange
from PIL import Image, ImageDraw

from flux_action.inference import sampling
from flux_action.inference.so101 import load_policy
from flux_action.processing import packing

J = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]
CAPTURE: dict = {}


def capturing(fn):
    def wrapped(*args, **kwargs):
        out = fn(*args, **kwargs)
        CAPTURE["out"] = out
        return out
    return wrapped


sampling.euler = capturing(sampling.euler)
sampling.cosmos_unipc_order2 = capturing(sampling.cosmos_unipc_order2)


def load_dataset(root: Path):
    info = json.loads((root / "meta/info.json").read_text())
    data = pq.read_table(sorted((root / "data").rglob("*.parquet"))).to_pydict()
    eps = pq.read_table(sorted((root / "meta/episodes").rglob("*.parquet"))).to_pylist()
    tasks = {}
    for r in pq.read_table(root / "meta/tasks.parquet").to_pylist():  # v3: the task string is the pandas index
        text = r.get("task") or r.get("__index_level_0__") or next(v for k, v in r.items() if isinstance(v, str))
        tasks[int(r["task_index"])] = text
    return info, data, {int(e["episode_index"]): e for e in eps}, tasks


def video_frames(root: Path, ep: dict, key: str, t0: float, n: int, fps: int) -> list[np.ndarray]:
    """n consecutive RGB frames starting at episode time t0."""
    path = root / f"videos/{key}/chunk-{int(ep[f'videos/{key}/chunk_index']):03d}/file-{int(ep[f'videos/{key}/file_index']):03d}.mp4"
    start = float(ep[f"videos/{key}/from_timestamp"]) + t0
    out = []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(int(start / s.time_base), stream=s, backward=True)
        for f in c.decode(s):
            if f.time is None or f.time < start - 0.5 / fps:
                continue
            out.append(f.to_ndarray(format="rgb24"))
            if len(out) == n:
                break
    if len(out) < n:
        raise RuntimeError(f"{path}: only {len(out)} of {n} frames from {start:.3f}s")
    return out


def square(frame: np.ndarray, size: int = 256) -> np.ndarray:
    return np.asarray(Image.fromarray(frame).resize((size, size), Image.BILINEAR))


def write_mp4(path: Path, frames: list[np.ndarray], fps: int = 30):
    with av.open(str(path), "w") as c:
        codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
        s = c.add_stream(codec, rate=Fraction(fps))
        s.width, s.height, s.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
        for f in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def label(img: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 8 * len(text) + 6, 14), fill=(0, 0, 0))
    d.text((3, 1), text, fill=(255, 255, 0))
    return np.asarray(im)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--checkpoint", default="/workspace/ckpt/flux-3-action-so101-hfsim-norm")
    p.add_argument("--episodes", type=int, nargs="+", default=[0, 20, 40, 60])
    p.add_argument("--per-episode", type=int, default=3)
    p.add_argument("--task", help="caption override (default: the episode's own task)")
    p.add_argument("--out", type=Path, default=Path(f"/workspace/outputs/imagine/{time.strftime('%Y%m%dT%H%M%S')}"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    policy = load_policy(a.checkpoint)
    policy.set_text_encoder_offload(True)
    policy.to("cuda")
    cfg = policy.config
    keys = list(cfg.camera_keys)  # images.scene, images.wrist (canvas left, right)
    n_obs, chunk = cfg.n_obs_steps, cfg.chunk_size
    n_lat = packing.latent_frames(chunk)
    lh, lw = cfg.latent_hw
    print(f"checkpoint {a.checkpoint}: cameras {keys}, latent {n_lat}x{lh}x{lw}, sampler {cfg.sampler} "
          f"{cfg.num_inference_steps} steps, state q01 {np.round(cfg.state_normalization['q01'], 1).tolist()}", flush=True)

    info, data, eps, tasks = load_dataset(a.dataset)
    fps = int(info["fps"])
    ep_idx = np.asarray(data["episode_index"])
    states, actions = np.asarray(data["observation.state"], np.float32), np.asarray(data["action"], np.float32)
    frame_idx = np.asarray(data["frame_index"])
    summary = []
    for e in a.episodes:
        rows = np.flatnonzero(ep_idx == e)
        rows = rows[np.argsort(frame_idx[rows])]
        n = len(rows)
        S, A = states[rows], actions[rows]
        task = a.task or tasks[int(np.asarray(data["task_index"])[rows[0]])]
        # moments: evenly spaced, plus the frame where the gripper closes fastest (a grasp)
        lo, hi = n_obs, n - chunk - 1
        ts = list(np.linspace(lo, hi, a.per_episode).astype(int))
        g = np.diff(A[:, 5])
        grasp = int(np.clip(np.argmin(g[lo:hi]) + lo - 10, lo, hi))
        if all(abs(grasp - t) > 15 for t in ts):
            ts.append(grasp)
        sheet_rows = []
        for t in sorted(ts):
            cams = {}
            for k in keys:
                fr = video_frames(a.dataset, eps[e], f"observation.{k}", (t - n_obs + 1) / fps, n_obs + chunk, fps)
                cams[k] = [square(f) for f in fr]  # frames t-7 .. t+41
            order = [0] * (n_obs - 1) + [n_obs - 1]
            batch = {
                "state": torch.from_numpy(S[t - n_obs + 1: t + 1])[None].cuda(),
                "command_history": torch.from_numpy(A[t - n_obs: t])[None].cuda(),
                "task": [task],
            }
            for k in keys:
                win = np.stack([cams[k][i] for i in order]).transpose(0, 3, 1, 2)
                batch[k] = torch.from_numpy(np.ascontiguousarray(win))[None].cuda()
            t_start = time.perf_counter()
            with torch.inference_mode():
                pred = policy.predict_action_chunk(batch)[0].float().cpu().numpy()
                tokens = CAPTURE["out"]["x_video"]  # (1, n_lat*lh*lw, 96)
                lat = rearrange(tokens[0], "(t h w) c -> c t h w", t=n_lat, h=lh, w=lw)[None]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    vid = policy.video_vae.decode(lat.to(torch.bfloat16))
            vid = ((vid[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
            dt = time.perf_counter() - t_start
            real = [np.concatenate([cams[keys[0]][n_obs - 1 + i], cams[keys[1]][n_obs - 1 + i]], 1) for i in range(len(vid))]
            imag = [np.asarray(Image.fromarray(f).resize((512, 256))) for f in vid]
            frames = [np.concatenate([label(imag[i], f"imagined +{i}"), label(real[i], f"real +{i}")], 0)
                      for i in range(len(vid))]
            write_mp4(a.out / f"ep{e:03d}_t{t:04d}.mp4", frames, fps)
            cols = list(range(0, len(vid), 8))
            sheet_rows.append(np.concatenate([np.concatenate([label(imag[i], f"ep{e} t{t} imag +{i}") for i in cols], 1),
                                              np.concatenate([label(real[i], f"real +{i}") for i in cols], 1)], 0))
            rec = A[t: t + chunk]
            m = min(len(rec), len(pred))
            summary.append({"episode": e, "t": t, "task": task, "seconds": round(dt, 2), "video_frames": len(vid),
                            "pred_motion": dict(zip(J, np.round(pred[m - 1] - A[t - 1], 1).tolist())),
                            "real_motion": dict(zip(J, np.round(rec[m - 1] - A[t - 1], 1).tolist())),
                            "mean_abs_err": dict(zip(J, np.round(np.abs(pred[:m] - rec[:m]).mean(0), 1).tolist()))})
            print(f"ep {e} t {t}: {dt:.1f} s, {len(vid)} imagined frames; motion over the chunk "
                  f"pred {summary[-1]['pred_motion']} real {summary[-1]['real_motion']}", flush=True)
        Image.fromarray(np.concatenate(sheet_rows, 0)).save(a.out / f"sheet_ep{e:03d}.png")
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
