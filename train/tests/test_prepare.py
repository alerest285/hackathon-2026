"""End-to-end check of train/prepare.py on small datasets written by the real LeRobot 0.6.1.

  uv run --python 3.13 --with "lerobot[dataset]==0.6.1" --with pytest python -m pytest train/tests -q
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
H, W = 48, 64
GOOD = "pick up the small disk and place it on the right peg"
GOOD2 = "pick up the medium disk and place it on the middle peg"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prepare = load("train_prepare", REPO / "train" / "prepare.py")
mapping = load("train_mapping_t", REPO / "console" / "mapping.py")


def features():
    names = [f"{j}.pos" for j in JOINTS]
    cam = {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]}
    return {"observation.state": {"dtype": "float32", "shape": (6,), "names": names},
            "action": {"dtype": "float32", "shape": (6,), "names": names},
            "observation.images.scene": dict(cam), "observation.images.wrist": dict(cam)}


def make_dataset(root: Path, episodes: list[tuple[str, int]], seed: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    rng = np.random.default_rng(seed)
    ds = LeRobotDataset.create(f"local/{root.name}", fps=30, features=features(), root=root,
                               robot_type="so100_follower", use_videos=True)
    raw = []
    for task, n in episodes:
        st = rng.uniform(-60, 60, size=(n, 6)).astype(np.float32)
        st[:, 5] = rng.uniform(0, 100, size=n)
        act = st + 1.0
        for t in range(n):
            img = np.full((H, W, 3), (t * 4) % 255, np.uint8)
            ds.add_frame({"observation.state": st[t], "action": act[t], "observation.images.scene": img,
                          "observation.images.wrist": img, "task": task})
        ds.save_episode()
        raw.append((st, act))
    ds.finalize()
    return raw


def write_map(path: Path, robot_id: str, scale: float):
    lo = np.array([-50, -40, -30, -20, -10, 5], float)
    hi = lo + 100 * scale
    jm = mapping.JointMap(lo, hi, meta={"robot_id": robot_id, "calibration_sha256": "x" * 64})
    jm.save(path)
    return jm


def test_prepare_maps_filters_and_merges(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    a, b = tmp_path / "demos_a", tmp_path / "runs_b"
    raw_a = make_dataset(a, [(GOOD, 60), (GOOD, 20), ("wave hello", 60)], seed=0)
    raw_b = make_dataset(b, [(GOOD2, 55), (GOOD2, 55), (GOOD, 55)], seed=1)
    # console sidecar for b: ep0 live success, ep1 dry success, ep2 live fail
    side = [{"episode_index": 0, "verdict": "success", "run_mode": "live", "dropped_frames": 0, "encoder_dropped": {}, "valid": True},
            {"episode_index": 1, "verdict": "success", "run_mode": "dry", "dropped_frames": 0, "encoder_dropped": {}, "valid": False},
            {"episode_index": 2, "verdict": "fail", "run_mode": "live", "dropped_frames": 0, "encoder_dropped": {}, "valid": True}]
    (b / "console_episodes.jsonl").write_text("".join(json.dumps(s) + "\n" for s in side))
    # eval writes a second sidecar; it must merge, not replace (ep0 keeps run_mode live from the console file)
    (b / "hanoi_episodes.jsonl").write_text(json.dumps({"episode_index": 0, "verdict": "success", "interventions": 0}) + "\n")
    jm_a, jm_b = write_map(tmp_path / "map_a.json", "arm_a", 1.0), write_map(tmp_path / "map_b.json", "arm_b", 1.5)

    out = tmp_path / "lora_v1"
    assert prepare.main(["--source", f"{a}={tmp_path / 'map_a.json'}", "--source", f"{b}={tmp_path / 'map_b.json'}",
                         "--out", str(out), "--hanoi-only"]) == 0

    m = json.loads((out / "prepare_manifest.json").read_text())
    assert m["episodes"] == 2 and m["frames"] == 60 + 55
    assert m["tasks"] == {GOOD: 1, GOOD2: 1}
    da, db = m["sources"][0]["dropped"], m["sources"][1]["dropped"]
    assert "frames <" in da["1"] and "not a Hanoi prompt" in da["2"]
    assert "invalid" in db["1"] and "verdict fail" in db["2"]
    assert m["sources"][1]["mapping"]["robot_id"] == "arm_b"

    ds = LeRobotDataset("alerest285/lora_v1", root=out)
    st = np.stack([ds.hf_dataset[i]["observation.state"].numpy() for i in range(len(ds.hf_dataset))])
    act = np.stack([ds.hf_dataset[i]["action"].numpy() for i in range(len(ds.hf_dataset))])
    np.testing.assert_allclose(st[:60], jm_a.to_model(raw_a[0][0]), atol=1e-3)
    np.testing.assert_allclose(act[60:], jm_b.to_model(raw_b[0][1]), atol=1e-3)
    s = ds.meta.stats["observation.state"]
    np.testing.assert_allclose(s["max"], st.max(0), atol=1e-3)
    np.testing.assert_allclose(s["min"], st.min(0), atol=1e-3)
    # frames decode for both cameras at the last index
    item = ds[len(ds) - 1]
    assert item["observation.images.scene"].shape[-2:] == (H, W)


def test_short_video_is_dropped(tmp_path):
    import pandas as pd

    a = tmp_path / "demos"
    make_dataset(a, [(GOOD, 60), (GOOD, 60)], seed=2)
    # Simulate the encoder shortfall: episode 1's wrist span claims fewer frames than its rows.
    p = next((a / "meta" / "episodes").glob("*/*.parquet"))
    df = pd.read_parquet(p)
    col = "videos/observation.images.wrist/to_timestamp"
    df.loc[df["episode_index"] == 1, col] = df.loc[df["episode_index"] == 1, col] - 0.5
    df.to_parquet(p, index=False)
    write_map(tmp_path / "map.json", "arm", 1.0)
    out = tmp_path / "lora"
    assert prepare.main(["--source", str(a), "--mapping", str(tmp_path / "map.json"), "--out", str(out)]) == 0
    m = json.loads((out / "prepare_manifest.json").read_text())
    assert m["episodes"] == 1 and "wrist video spans" in m["sources"][0]["dropped"]["1"]


def test_refuses_existing_out(tmp_path):
    (tmp_path / "x").mkdir()
    with pytest.raises(SystemExit):
        prepare.main(["--source", str(tmp_path), "--mapping", "m.json", "--out", str(tmp_path / "x")])
