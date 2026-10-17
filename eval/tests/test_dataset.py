"""The real-arm wiring without an arm (joint map, native limits, native guard), the unit split between the loop and
the logs, and the LeRobotDataset of attempts, read back with LeRobot itself."""

import json

import numpy as np
import pytest
from so101_client.fakes import REST_POSE
from test_runner_fake import CONFIG, CYCLE_S, FAST, read_jsonl

from hanoi_eval import session as session_mod
from hanoi_eval.config import load_config
from hanoi_eval.dataset import console_module, load_joint_map
from hanoi_eval.keys import ScriptedKeys
from hanoi_eval.session import SessionOptions, build_rig, run_session

LIMITS = [[-110.0, 110.0], [-105.0, 105.0], [-100.0, 100.0], [-100.0, 100.0], [-160.0, 160.0]]
OFFSET = np.array([0.0, 200.0, 50.0, 0.0, 0.0, 0.0])  # model = ours + OFFSET; the gripper stays as it is


def write_map(tmp_path, offset=OFFSET):
    """A console/mapping.py JointMap with slope 1 and a per-joint offset."""
    mapping = console_module("mapping")
    lo = np.array([-50.0, -50.0, -50.0, -50.0, -50.0, 0.0])
    path = tmp_path / "joint_mapping.json"
    mapping.JointMap(lo, lo + 100.0, lo + offset, lo + 100.0 + offset).save(path)
    return path


def test_the_real_arm_is_mapped_and_guarded_in_its_own_degrees(tmp_path):
    from so101_client.robot_io import SO101

    path = write_map(tmp_path)
    cfg = load_config(CONFIG, {"robot.port": "/dev/tty.none", "robot.mapping": str(path), "loop.joint_limits": LIMITS})
    rig = build_rig(cfg, SessionOptions(live=True))  # builds, connects nothing
    assert type(rig.robot).__name__ == "MappedRobot" and isinstance(rig.robot._r, SO101)
    assert rig.robot._r.guard is rig.guard
    assert rig.guard.limits == tuple(tuple(pair) for pair in LIMITS)  # native, under the map
    assert rig.native_limits[1] == (-105.0, 105.0)
    assert rig.loop_limits[1] == pytest.approx((95.0, 305.0))  # the loop clamps in the checkpoint's convention


def test_live_refuses_without_limits_or_without_the_joint_map(tmp_path):
    base = {"robot.port": "/dev/tty.none"}
    missing = str(tmp_path / "missing.json")
    with pytest.raises(SystemExit, match="--live needs joint limits"):
        build_rig(load_config(CONFIG, {**base, "robot.mapping": str(write_map(tmp_path))}), SessionOptions(live=True))
    with pytest.raises(SystemExit, match="no joint map"):
        build_rig(load_config(CONFIG, {**base, "robot.mapping": missing, "loop.joint_limits": LIMITS}),
                  SessionOptions(live=True))
    dry = build_rig(load_config(CONFIG, {**base, "robot.mapping": missing}), SessionOptions())
    assert dry.joint_map is None and "raw LeRobot degrees" in dry.warnings[0]
    raw = build_rig(load_config(CONFIG, {**base, "robot.mapping": "", "loop.joint_limits": LIMITS}),
                    SessionOptions(live=True))
    assert raw.joint_map is None and raw.loop_limits == raw.native_limits  # "" opts out explicitly


def test_mapped_fake_run_logs_native_units_and_records_one_lerobot_episode_per_attempt(tmp_path, monkeypatch):
    path = write_map(tmp_path)
    jmap, mapped = load_joint_map(path)
    real_build = session_mod.build_rig

    def mapped_fakes(cfg, opts):  # the fakes behind the same map a real arm gets
        rig = real_build(cfg, opts)
        rig.robot, rig.joint_map, rig.joint_map_path = mapped(rig.robot, jmap), jmap, str(path)
        return rig

    monkeypatch.setattr(session_mod, "build_rig", mapped_fakes)
    root = tmp_path / "dataset"
    cfg = load_config(CONFIG, {**FAST, "record.root": str(root), "record.repo_id": "test/hanoi_eval"})
    opts = SessionOptions(fake=True, fake_latency_ms=60.0, fake_jitter_ms=10.0, fake_cycle_s=CYCLE_S, episodes=1,
                          out=tmp_path / "runs", record=True)
    lines: list[str] = []
    run_dir, runner = run_session(cfg, opts, keys_source=ScriptedKeys({"start": "\n", "verdict": "s"}),
                                  log=lines.append)
    attempts = read_jsonl(run_dir / "ep01" / "attempts.jsonl")
    assert len(attempts) == 7 and all(a["success"] and a["end_reason"] == "auto" for a in attempts), lines

    # the loop ran in the checkpoint's convention; ticks.jsonl is back in the arm's degrees, chunks are as sent
    ticks = read_jsonl(run_dir / "ep01" / "ticks.jsonl")
    lift = np.median([t["state"][1] for t in ticks])
    assert abs(lift - REST_POSE[1]) < 30
    chunks = np.load(run_dir / "ep01" / "chunks.npz")
    assert abs(np.median(chunks["actions"][:, :, 1]) - (REST_POSE[1] + OFFSET[1])) < 30
    run = json.loads((run_dir / "run.json").read_text())
    assert run["joint_map"]["path"] == str(path) and run["lerobot_dataset"]["root"] == str(root)

    # one LeRobot episode per attempt, the prompt as its task, the verdict and Hanoi context in the sidecar
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("test/hanoi_eval", root=root)
    assert ds.meta.total_episodes == 7
    side = read_jsonl(root / "hanoi_episodes.jsonl")
    assert [s["episode_index"] for s in side] == list(range(7))
    assert [s["attempt"] for s in side] == [a["attempt"] for a in attempts]
    assert [s["task"] for s in side] == [a["prompt"] for a in attempts]
    assert all(s["verdict"] == "success" and s["frames"] > 0 and s["run"] == run_dir.name for s in side)
    assert all(s["fake"] and not s["live"] for s in side)
    assert sum(s["frames"] for s in side) == ds.meta.total_frames
    states = np.stack([np.asarray(row) for row in ds.hf_dataset["observation.state"]])
    assert abs(np.median(states[:, 1]) - REST_POSE[1]) < 30  # the dataset holds the arm's own degrees


def test_a_full_writer_queue_never_blocks_the_control_thread():
    """end() may wait for a slot in the writer's queue; add() on the control thread must not wait with it."""
    import threading
    import time

    from hanoi_eval.dataset import HanoiDataset

    class Writer:  # the Recorder's interface: add() never blocks, save() blocks while the queue is full
        def __init__(self):
            self.frames, self.release = 0, threading.Event()

        def add(self, *args):
            self.frames += 1

        def save(self, verdict):
            self.release.wait(5.0)

    ds = HanoiDataset.__new__(HanoiDataset)
    ds.writer, ds._task, ds._lock, ds.frames_added = Writer(), None, threading.Lock(), 0
    ds.begin("task a")
    ender = threading.Thread(target=ds.end, args=("success", {}))
    ender.start()
    time.sleep(0.05)  # end() is now stuck in save()
    t0 = time.perf_counter()
    ds.add(np.zeros(6), np.zeros(6), np.zeros((4, 4, 3)), np.zeros((4, 4, 3)), "task a")
    assert time.perf_counter() - t0 < 0.02 and ds.frames_added == 0  # returned at once; the attempt was closed
    ds.writer.release.set()
    ender.join(1.0)
    assert not ender.is_alive()
