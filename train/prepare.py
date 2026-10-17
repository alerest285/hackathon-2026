"""Turn our recorded LeRobot datasets into one LoRA training set in the checkpoint's joint space.

Why this step exists: the LoRA recipe keeps flux-3-action-so101's own normalization (history quantiles in its
processor JSONs). LeRobot ignores dataset stats for this checkpoint (processor_flux3.py, conditioning="history"),
so our arm's degrees would be normalized against the training corpus's ranges and land in the wrong place.
At inference the console maps our joints into model space (console/mapping.py, JointMap.to_model); the training
data must go through the same map, from the same arm's joint_mapping.json. Model space is shared, so demos from
different arms merge after their own maps.

Per source:  load -> drop episodes (failed / unlabeled runs, recording gaps, too short, unknown prompt)
             -> map observation.state and action to model space -> recompute their stats.
Then merge all sources into --out and write <out>/prepare_manifest.json (what went in, which maps, coverage).

Run on the machine that holds the recordings, with the LeRobot that wrote them:
  uv run --python 3.13 --with "lerobot[dataset]==0.6.1" python train/prepare.py \
      --source ~/hackathon-data/so100_hanoi_demos \
      --source ~/hackathon-data/so100_hanoi_flux_zeroshot \
      --mapping ~/hackathon-data/joint_mapping.json --out ~/hackathon-data/lora_v1
A source can carry its own map: --source DIR=MAP.json (a demo set from another arm).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import tomllib
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
CAMERAS = ("observation.images.scene", "observation.images.wrist")  # the checkpoint's keys, scene left, wrist right
MAPPED = ("observation.state", "action")
SIDECARS = ("console_episodes.jsonl", "hanoi_episodes.jsonl")  # console/recorder.py, eval/hanoi_eval/dataset.py
# The checkpoint reads 8 history ticks and 42 future frames: a window spans 50 frames. BFL's standalone trainer
# drops shorter episodes; LeRobot pads them instead, which trains on copies. Keep the same floor.
MIN_FRAMES = 50
FPS = 30


def load_mapping_module():
    spec = importlib.util.spec_from_file_location("train_console_mapping", REPO / "console" / "mapping.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hanoi_prompts(config: Path) -> set[str]:
    """Every prompt eval can issue: each piece onto each place (moves = "all"), from eval/config/hanoi.toml."""
    c = tomllib.loads(config.read_text())
    return {c["prompt"].format(piece=p, place=c["places"][pl], source="{source}")
            for p in c["pieces"] for pl in c["places"]}


def read_sidecar(root: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for name in SIDECARS:
        f = root / name
        if f.exists():
            for line in f.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    i = int(d["episode_index"])
                    out[i] = {**out.get(i, {}), **d}  # merge both files; hanoi's verdict (read last) wins
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_features(meta, root: Path):
    f = meta.features
    problems = []
    if int(meta.fps) != FPS:
        problems.append(f"fps {meta.fps}, the checkpoint runs at {FPS}")
    for key in MAPPED:
        names = (f.get(key) or {}).get("names")
        if key not in f or tuple(f[key]["shape"]) != (6,):
            problems.append(f"{key} missing or not shape (6,)")
        elif list(names) != [f"{j}.pos" for j in JOINTS]:
            problems.append(f"{key} names {names}, expected {[f'{j}.pos' for j in JOINTS]}")
    for cam in CAMERAS:
        if cam not in f:
            problems.append(f"{cam} missing (cameras present: {[k for k in f if k.startswith('observation.images')]})")
    if problems:
        raise SystemExit(f"{root}: not a dataset this checkpoint can train on:\n  " + "\n  ".join(problems))


def video_shortfalls(meta, root: Path) -> dict[int, str]:
    """Episodes whose scene or wrist video does not cover their numeric rows.

    The streaming encoder can fall behind and write fewer frames than the parquet holds (seen 17.10: 12 rows,
    scene 6, wrist 12, dropped_frames 0). Two checks per camera: the episode's own span (to - from) * fps must
    equal its length, and the shared v3 mp4 must really hold frames up to the largest to_timestamp in it
    (demuxed packet count, no decode)."""
    import av

    eps = meta.episodes
    bad: dict[int, str] = {}
    file_frames: dict[Path, int] = {}
    for cam in CAMERAS:
        for i in range(meta.total_episodes):
            e = eps[i]
            n = int(e["length"])
            span = round((float(e[f"videos/{cam}/to_timestamp"]) - float(e[f"videos/{cam}/from_timestamp"])) * meta.fps)
            if abs(span - n) > 1:
                bad.setdefault(i, f"{cam.split('.')[-1]} video spans {span} frames, episode has {n} rows")
                continue
            path = root / meta.get_video_file_path(i, cam)
            if path not in file_frames:
                if not path.exists():
                    file_frames[path] = -1
                else:
                    with av.open(str(path)) as c:
                        file_frames[path] = sum(1 for pk in c.demux(c.streams.video[0]) if pk.size)
            have = file_frames[path]
            need = round(float(e[f"videos/{cam}/to_timestamp"]) * meta.fps)
            if have < 0:
                bad.setdefault(i, f"{cam.split('.')[-1]} video file {path.name} missing")
            elif have < need - 1:
                bad.setdefault(i, f"{cam.split('.')[-1]} video {path.name} holds {have} frames, episode needs up to {need}")
    return bad


def select_episodes(ds, root: Path, keep: str, min_frames: int, prompts: set[str] | None, allow_dry: bool):
    """(episode indices to drop, reasons per dropped episode, per-task count of kept episodes)."""
    side = read_sidecar(root)
    eps = ds.meta.episodes
    short_video = video_shortfalls(ds.meta, root)
    drop, why, kept = [], {}, Counter()
    for i in range(ds.meta.total_episodes):
        n = int(eps[i]["length"])
        tasks = eps[i]["tasks"]
        task = tasks[0] if isinstance(tasks, (list, tuple)) else tasks
        s = side.get(i)
        reason = None
        if n < min_frames:
            reason = f"{n} frames < {min_frames}"
        elif i in short_video:
            reason = short_video[i]
        elif s is not None and s.get("valid") is False and not (allow_dry and s.get("run_mode") == "dry"):
            reason = f"recorder marked invalid (run_mode {s.get('run_mode')}, encoder_dropped {s.get('encoder_dropped')})"
        elif s is not None and s.get("run_mode") in ("dry", "fake") and not allow_dry:
            reason = f"{s['run_mode']} run: the arm did not follow the commands, state and action disagree"
        elif s is not None and s.get("dropped_frames", 0):
            reason = f"{s['dropped_frames']} frames dropped while recording (timing gap)"
        elif s is not None and s.get("interventions"):
            reason = f"interventions {s['interventions']} (a human corrected the arm)"
        elif s is not None and any((s.get("encoder_dropped") or {}).values()):
            reason = f"encoder dropped frames {s['encoder_dropped']}"
        elif s is not None and keep == "success" and s.get("verdict") != "success":
            reason = f"verdict {s.get('verdict')}"
        elif s is not None and keep == "not-fail" and s.get("verdict") in ("fail", "error", "quit"):
            reason = f"verdict {s.get('verdict')}"
        elif prompts is not None and task not in prompts:
            reason = f"task {task!r} is not a Hanoi prompt (exact text matters: it is the caption the model sees)"
        if reason:
            drop.append(i)
            why[i] = reason
        else:
            kept[task] += 1
    return drop, why, kept


def to_model_space(root: Path, jm) -> dict:
    """Rewrite observation.state and action in every data parquet, then their per-episode and global stats."""
    import pandas as pd
    from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
    from lerobot.datasets.io_utils import load_stats, write_stats

    per_ep: dict[int, dict[str, np.ndarray]] = {}
    for p in sorted((root / "data").glob("*/*.parquet")):
        df = pd.read_parquet(p)
        for key in MAPPED:
            arr = np.stack(df[key].to_numpy()).astype(np.float64)
            df[key] = list(jm.to_model(arr).astype(np.float32))
        df.to_parquet(p, index=False)
        for ep, g in df.groupby("episode_index"):
            per_ep[int(ep)] = {key: np.stack(g[key].to_numpy()) for key in MAPPED}

    ep_stats = {ep: {k: get_feature_stats(v, axis=0, keepdims=False) for k, v in d.items()} for ep, d in per_ep.items()}
    for p in sorted((root / "meta" / "episodes").glob("*/*.parquet")):
        df = pd.read_parquet(p)
        for row, ep in enumerate(df["episode_index"]):
            for key, st in ep_stats[int(ep)].items():
                for stat, val in st.items():
                    col = f"stats/{key}/{stat}"
                    if col in df.columns:
                        df.at[row, col] = np.atleast_1d(val).tolist()
        df.to_parquet(p, index=False)
    stats = load_stats(root) or {}
    stats.update(aggregate_stats([ep_stats[e] for e in sorted(ep_stats)]))
    write_stats(stats, root)
    allv = {k: np.concatenate([per_ep[e][k] for e in sorted(per_ep)]) for k in MAPPED}
    return allv


def coverage(values: np.ndarray, mm) -> list[dict]:
    """Per joint: how much of our data sits outside the checkpoint's 1-99 % training range (model space)."""
    out = []
    for j, name in enumerate(JOINTS):
        v = values[:, j]
        lo, hi = mm.MODEL_Q01[j], mm.MODEL_Q99[j]
        out.append({"joint": name, "ours_min": round(float(v.min()), 2), "ours_max": round(float(v.max()), 2),
                    "ckpt_q01": round(lo, 2), "ckpt_q99": round(hi, 2),
                    "outside_pct": round(100 * float(np.mean((v < lo) | (v > hi))), 2)})
    return out


def parse_source(text: str, default_map: Path | None) -> tuple[Path, Path]:
    src, _, m = text.partition("=")
    mp = Path(m).expanduser() if m else default_map
    if mp is None:
        raise SystemExit(f"--source {src}: no joint map (give --mapping or --source DIR=MAP.json)")
    return Path(src).expanduser(), mp


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", required=True, help="LeRobot dataset root, optionally DIR=MAP.json")
    p.add_argument("--mapping", type=Path, default=None, help="default joint map (console/fit_reference.py output)")
    p.add_argument("--out", type=Path, required=True, help="new dataset root (must not exist)")
    p.add_argument("--repo-id", default=None, help="LeRobot repo_id of the output (default alerest285/<out name>)")
    p.add_argument("--keep", choices=("success", "not-fail", "all"), default="success",
                   help="for runs with a verdict sidecar (console/eval). Teleop demos have none and are all kept")
    p.add_argument("--min-frames", type=int, default=MIN_FRAMES)
    p.add_argument("--hanoi-only", action="store_true",
                   help="keep only episodes whose task is a prompt eval/config/hanoi.toml can issue")
    p.add_argument("--allow-dry", action="store_true", help="keep dry runs (never for training: the arm did not move)")
    p.add_argument("--no-map", action="store_true",
                   help="the sources are already in model space (never for our raw recordings)")
    a = p.parse_args(argv)

    from lerobot.datasets.dataset_tools import delete_episodes, merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = a.out.expanduser()
    if out.exists():
        raise SystemExit(f"{out} exists; pick a new --out (datasets are versioned by name: lora_v1, lora_v2, ...)")
    repo_id = a.repo_id or f"alerest285/{out.name}"
    mm = load_mapping_module()
    prompts = hanoi_prompts(REPO / "eval" / "config" / "hanoi.toml") if a.hanoi_only else None
    default_map = a.mapping.expanduser() if a.mapping else None

    work = Path(tempfile.mkdtemp(prefix="lora-prepare-", dir=out.parent if out.parent.exists() else None))
    parts, manifest_sources, values = [], [], {k: [] for k in MAPPED}
    try:
        for n, text in enumerate(a.source):
            src, mp = (Path(text).expanduser(), None) if a.no_map else parse_source(text, default_map)
            ds = LeRobotDataset(f"local/{src.name}", root=src)
            check_features(ds.meta, src)
            drop, why, kept = select_episodes(ds, src, a.keep, a.min_frames, prompts, a.allow_dry)
            dst = work / f"src{n}"
            if len(drop) == ds.meta.total_episodes:
                print(f"{src}: no episode kept, skipped")
                manifest_sources.append({"root": str(src), "kept": 0, "dropped": {str(k): v for k, v in why.items()}})
                continue
            if drop:
                delete_episodes(ds, drop, output_dir=dst, repo_id=f"local/{src.name}-kept")
            else:
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*SIDECARS))
            entry = {"root": str(src), "episodes": ds.meta.total_episodes, "kept": ds.meta.total_episodes - len(drop),
                     "kept_per_task": dict(kept), "dropped": {str(k): v for k, v in why.items()}}
            if mp is not None:
                jm = mm.JointMap.load(mp)
                vals = to_model_space(dst, jm)
                for k in MAPPED:
                    values[k].append(vals[k])
                entry["mapping"] = {"path": str(mp), "sha256": sha256(mp), "robot_id": jm.meta.get("robot_id"),
                                    "calibration_sha256": jm.meta.get("calibration_sha256"),
                                    "k": [round(float(x), 4) for x in jm.k]}
            manifest_sources.append(entry)
            parts.append(LeRobotDataset(f"local/{src.name}-kept", root=dst))
            print(f"{src}: kept {entry['kept']}/{entry['episodes']}" + (f", dropped: {Counter(why.values()).most_common(3)}" if why else ""))

        if not parts:
            raise SystemExit("nothing to train on: every episode was dropped (see the reasons above)")
        if len(parts) == 1:
            shutil.move(str(parts[0].root), out)
            merged = LeRobotDataset(repo_id, root=out)
        else:
            merged = merge_datasets(parts, output_repo_id=repo_id, output_dir=out)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    tasks = Counter()
    for i in range(merged.meta.total_episodes):
        t = merged.meta.episodes[i]["tasks"]
        tasks[t[0] if isinstance(t, (list, tuple)) else t] += 1
    cov = coverage(np.concatenate(values["observation.state"]), mm) if values["observation.state"] else None
    manifest = {"repo_id": repo_id, "episodes": merged.meta.total_episodes, "frames": merged.meta.total_frames,
                "minutes": round(merged.meta.total_frames / FPS / 60, 1), "tasks": dict(tasks),
                "space": "model (flux-3-action-so101 @ c9e13b2a joint convention, console/mapping.py)",
                "keep": a.keep, "min_frames": a.min_frames, "sources": manifest_sources, "state_coverage": cov}
    (out / "prepare_manifest.json").write_text(json.dumps(manifest, indent=1))

    print("\nThe task strings below are the captions the adapter learns: type them exactly the same at inference.")
    print(f"{out}: {manifest['episodes']} episodes, {manifest['frames']} frames ({manifest['minutes']} min)")
    for t, c in sorted(tasks.items()):
        print(f"  {c:4d}  {t}")
    if prompts:
        missing = sorted(prompts - set(tasks))
        if missing:
            print(f"  no episodes yet for {len(missing)} prompt(s): " + "; ".join(missing[:4]) + (" ..." if len(missing) > 4 else ""))
    if cov:
        print("state vs the checkpoint's 1-99 % range (model space):")
        for c in cov:
            flag = "  <- check the map" if c["outside_pct"] > 10 else ""
            print(f"  {c['joint']:<14} {c['ours_min']:8.1f} .. {c['ours_max']:8.1f}   ckpt {c['ckpt_q01']:8.1f} .. "
                  f"{c['ckpt_q99']:8.1f}   outside {c['outside_pct']:5.1f} %{flag}")
    print(f"\nNext: rsync -a {out}/ <pod>:/workspace/datasets/{out.name}/  then train/train.sh probe {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
