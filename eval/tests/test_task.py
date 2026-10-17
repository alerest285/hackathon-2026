"""Scored task runs (run_task.py) without hardware: FakeRobot + FakeCameras + FakeNet through the real ControlLoop,
the same session code as the CLI, scripted operator keys."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_runner_fake import read_jsonl

from hanoi_eval import summary as summary_mod
from hanoi_eval.config import ConfigError, load_config
from hanoi_eval.keys import KEYMAP, ScriptedKeys
from hanoi_eval.runner import STEP_EVENTS
from hanoi_eval.session import SessionOptions, run_session

EVAL_DIR = Path(__file__).resolve().parents[1]
CONFIG = EVAL_DIR / "config" / "task.toml"
FAST = {"cameras.width": 160, "cameras.height": 120, "cameras.image_size": 64, "task.steps": 4,
        "task.attempt_timeout_s": 1.5}


def run_task(tmp_path, answers, attempts=3, record=None, delay_s=0.0, **overrides):
    cfg = load_config(CONFIG, {**FAST, **overrides})
    lines: list[str] = []
    opts = SessionOptions(fake=True, fake_latency_ms=60.0, fake_jitter_ms=10.0, fake_cycle_s=1.2, task=True,
                          attempts=attempts, record=record, out=tmp_path / "runs")
    run_dir, runner = run_session(cfg, opts, keys_source=ScriptedKeys(answers, delay_s=delay_s), log=lines.append)
    return run_dir, runner, lines


def test_three_attempts_count_steps_undo_timeout_pause_and_quit(tmp_path):
    answers = {
        "start": "\n",
        # one key per prompt; the runner re-arms the step prompt after every step and pause for a script
        "step": ["+", "+", "+", "+", "s",       # attempt 1: four steps, success
                 "+", "+", "-", "+", None,      # attempt 2: 2 steps after the undo, then nothing until the timeout
                 " ", " ", "q"],                # attempt 3: pause, resume, quit
        "verdict": ["f"],                       # attempt 2's timeout verdict
    }
    run_dir, runner, lines = run_task(tmp_path, answers)
    assert run_dir.name.endswith("-zero-shot-task")
    attempts = read_jsonl(run_dir / "attempts.jsonl")
    assert [a["attempt"] for a in attempts] == [1, 2, 3]
    a1, a2, a3 = attempts
    assert (a1["steps_completed"], a1["success"], a1["complete"], a1["end_reason"]) == (4, True, True, "key")
    assert len(a1["step_times_s"]) == 4 and a1["step_times_s"] == sorted(a1["step_times_s"])
    assert (a2["steps_completed"], a2["success"], a2["complete"], a2["end_reason"]) == (2, False, False, "timeout")
    assert a2["active_s"] >= 1.5
    assert (a3["success"], a3["quit"], a3["pauses"], a3["interventions"]) == (None, True, 1, 1)
    assert any("resumed, history reset" in line for line in lines)
    for name in ("attempt.json", "ticks.jsonl", "chunks.npz", "scene.mp4", "wrist.mp4"):
        assert (run_dir / "att01" / name).is_file(), name
    assert json.loads((run_dir / "att01" / "attempt.json").read_text())["recording"]["ticks"]["n"] > 0
    chunks = np.load(run_dir / "att01" / "chunks.npz")
    assert chunks["actions"].shape[1:] == (42, 6) and chunks["accepted"].any()

    run = json.loads((run_dir / "run.json").read_text())
    assert run["kind"] == "task" and run["task"]["steps_total"] == 4 and run["task"]["attempts"] == 3
    totals = json.loads((run_dir / "summary.json").read_text())["totals"]
    assert totals["attempts"] == 3 and totals["valid_attempts"] == 2 and totals["quit"] == 1
    assert totals["successes"] == 1 and totals["success_rate"] == 0.5
    assert (totals["steps_completed"], totals["mean_steps"], totals["max_steps"]) == (6, 3.0, 4)
    assert totals["complete_attempts"] == 1 and totals["interventions"] == 0  # the quit attempt is not counted
    assert "| 1 | 4/4 | success | key |" in (run_dir / "summary.md").read_text()


def test_status_table_lists_task_runs(tmp_path, capsys):
    run_dir, _, _ = run_task(tmp_path, {"start": "\n", "step": ["+", "+", "s"]}, attempts=1)
    summary_mod.main(["--status", str(run_dir)])
    out = capsys.readouterr().out
    assert "| Run | Label | Rig | Attempts (valid) |" in out
    assert f"| {run_dir.name} | zero-shot | fake robot and policy | 1 (1) | 1/1 | 2 / 2.0 / 2 of 4 | 0 | 0 |" in out


def test_recorded_attempts_become_lerobot_episodes_with_steps_in_the_sidecar(tmp_path):
    root = tmp_path / "dataset"
    # 0.4 s per key: each attempt lasts long enough to be saved (MIN_EPISODE_FRAMES)
    run_dir, _, _ = run_task(tmp_path, {"start": "\n", "step": ["+", "s", "+", "+", "f"]}, attempts=2, record=True,
                             delay_s=0.4, **{"record.root": str(root), "record.repo_id": "test/task_eval"})
    side = read_jsonl(root / "hanoi_episodes.jsonl")
    assert [s["attempt"] for s in side] == [1, 2]
    assert [(s["verdict"], s["steps_completed"], s["steps_total"]) for s in side] == [("success", 1, 4), ("fail", 2, 4)]
    assert all(s["kind"] == "task" and s["run_mode"] == "fake" and s["run"] == run_dir.name for s in side)
    checks = read_jsonl(root / "console_episodes.jsonl")
    assert all(c["valid"] is False for c in checks)  # a fake run is never training data
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("test/task_eval", root=root)
    assert ds.meta.total_episodes == 2
    assert {t for t in ds.meta.tasks.index} == {load_config(CONFIG).task.instruction}


def test_task_config_is_checked(tmp_path):
    cfg = load_config(CONFIG)
    assert cfg.task.instruction.startswith("Put the green objects") and cfg.robot.mapping == ""
    with pytest.raises(ConfigError, match="task.steps"):
        load_config(CONFIG, {"task.steps": -1})
    assert load_config(EVAL_DIR / "config" / "hanoi.toml").task.instruction == ""  # optional there


def test_step_keys_map_to_step_events_that_hanoi_prompts_never_accept():
    assert KEYMAP["+"] == KEYMAP["="] == "step" and KEYMAP["-"] == "unstep"
    assert not {"step", "unstep"} & set(STEP_EVENTS)


def run_cli(*args):
    return subprocess.run([sys.executable, str(EVAL_DIR / "run_task.py"), *args], capture_output=True, text=True,
                          timeout=120, cwd=EVAL_DIR)


def test_cli_print_warmup_and_refusals():
    out = run_cli("--print-warmup", "--instruction", "put the cube in the box")
    assert out.returncode == 0 and "--warmup-task 'put the cube in the box'" in out.stdout
    assert "--engine lerobot --adapter" in out.stdout
    assert run_cli("--live", "--fake").returncode == 2
    assert run_cli("--unattended").returncode == 2
    assert run_cli("--fake", "--fake-hardware").returncode == 2
    empty = run_cli("--print-warmup", "--instruction", " ")
    assert empty.returncode == 2 and "task.instruction is empty" in empty.stderr


def test_an_attempt_too_short_to_save_is_discarded_not_saved():
    """LeRobot 0.6.1 cannot finalize a dataset holding a 1-frame episode, so accidental starts are discarded."""
    import threading

    from hanoi_eval.dataset import MIN_EPISODE_FRAMES, HanoiDataset

    class Writer:
        def __init__(self):
            self.calls = []

        def add(self, *args):
            pass

        def save(self, verdict):
            self.calls.append("save")

        def discard(self):
            self.calls.append("discard")

    ds = HanoiDataset.__new__(HanoiDataset)
    ds.writer, ds._task, ds._lock, ds.frames_added, ds._open_frames, ds.discarded = Writer(), None, threading.Lock(), 0, 0, 0
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    for frames in (1, MIN_EPISODE_FRAMES):
        ds.begin("task")
        for _ in range(frames):
            ds.add(np.zeros(6), np.zeros(6), img, img, "task")
        ds.end("success", {})
    assert ds.writer.calls == ["discard", "save"] and ds.discarded == 1
