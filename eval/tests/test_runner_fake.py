"""End to end without hardware: FakeRobot + FakeCameras + FakeNet(pickplace) through the real ControlLoop,
assembled by the same session code as run_hanoi.py --fake, with scripted operator keys."""

import json
from pathlib import Path

import numpy as np

from hanoi_eval.config import load_config
from hanoi_eval.keys import ScriptedKeys
from hanoi_eval.session import SessionOptions, run_session

CONFIG = Path(__file__).resolve().parents[1] / "config" / "hanoi.toml"

# FakeNet pickplace with a 1.2 s cycle: the gripper is open, closes for about 0.5 s, then reopens. Its joints
# swing all the time (up to about 21 deg/s at this cycle), so the rest threshold is raised for the fake.
CYCLE_S = 1.2
FAST = {
    "step_timeout_s": 8.0,
    "auto_end.min_hold_s": 0.15,
    "auto_end.rest_s": 0.25,
    "auto_end.rest_speed_deg_s": 60.0,
    "cameras.width": 160,
    "cameras.height": 120,
    "cameras.image_size": 64,
}


def run_fake(tmp_path, answers, episodes=1, sanity=None, delay_s=0.0, **overrides):
    cfg = load_config(CONFIG, {**FAST, **overrides})
    keys = answers if hasattr(answers, "read") else ScriptedKeys(answers, delay_s=delay_s)
    lines: list[str] = []
    opts = SessionOptions(fake=True, fake_latency_ms=60.0, fake_jitter_ms=10.0, fake_cycle_s=CYCLE_S,
                          episodes=episodes, sanity=sanity, out=tmp_path)
    run_dir, runner = run_session(cfg, opts, keys_source=keys, log=lines.append)
    return run_dir, runner, keys, lines


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_full_strict_episode_with_fakes(tmp_path):
    run_dir, runner, keys, lines = run_fake(tmp_path, {"start": "\n", "verdict": "s"})
    ep_dir = run_dir / "ep01"
    for name in ("run.json", "summary.md", "summary.json"):
        assert (run_dir / name).is_file(), name
    for name in ("episode.json", "attempts.jsonl", "ticks.jsonl", "chunks.npz", "scene.mp4", "wrist.mp4",
                 "scene.times.csv", "wrist.times.csv"):
        assert (ep_dir / name).is_file(), name

    attempts = read_jsonl(ep_dir / "attempts.jsonl")
    assert len(attempts) == 7
    assert [a["end_reason"] for a in attempts] == ["auto"] * 7, lines
    assert all(a["success"] and a["verdict_by"] == "operator" for a in attempts)
    assert attempts[0]["prompt"] == "pick up the small disk and place it on the right peg"
    assert attempts[3]["prompt"] == "pick up the large disk and place it on the right peg"
    assert [a["reset_history"] for a in attempts] == [True] + [False] * 6
    assert all(a["chunks"] > 0 and a["auto_end"]["phase"] == "fired" for a in attempts)
    assert keys.pressed.count(("verdict", "s")) == 7

    episode = json.loads((ep_dir / "episode.json").read_text())
    assert episode["success"] and episode["end"] == "done"
    assert episode["steps_completed"] == 7 and episode["steps_attempted"] == 7 and episode["interventions"] == 0
    assert episode["tracked_state"] == {"left": [], "middle": [], "right": [3, 2, 1]}
    assert episode["video"]["scene"]["frames"] > 0 and episode["video"]["wrist"]["frames"] > 0

    chunks = np.load(ep_dir / "chunks.npz")
    assert chunks["actions"].ndim == 3 and chunks["actions"].shape[1:] == (42, 6)
    assert len(chunks["id"]) == len(chunks["actions"]) > 0 and chunks["accepted"].any()

    ticks = read_jsonl(ep_dir / "ticks.jsonl")
    assert len(ticks) > 60
    assert {"k", "t", "attempt", "state", "cmd", "held", "lateness_ms", "frame_age_ms", "remaining", "task"} <= set(ticks[0])
    assert any(t.get("chunk") for t in ticks)

    summary = (run_dir / "summary.md").read_text()
    assert "| Episode | Steps in a row | Steps attempted | Attempts | Success | Duration | Interventions | End |" in summary
    assert "| ep01 | 7/7 | 7 | 7 | yes |" in summary
    totals = json.loads((run_dir / "summary.json").read_text())["totals"]
    assert totals["episode_successes"] == 1 and totals["successful_attempts"] == 7 and totals["completed_steps"] == 7

    run = json.loads((run_dir / "run.json").read_text())
    assert run["fake"] and run["kind"] == "episodes" and run["protocol"] == "strict"
    assert run["server_hello"]["protocol"] == "so101.v1"
    assert run["loop"]["dry_run"] is False
    assert run["config"]["pieces"] == ["large disk", "medium disk", "small disk"]


def test_failed_step_stops_a_strict_episode(tmp_path):
    run_dir, runner, keys, lines = run_fake(tmp_path, {"start": "\n", "verdict": ["s", "s", "f"]})
    attempts = read_jsonl(run_dir / "ep01" / "attempts.jsonl")
    assert [a["success"] for a in attempts] == [True, True, False]
    episode = json.loads((run_dir / "ep01" / "episode.json").read_text())
    assert not episode["success"] and episode["end"] == "failed"
    assert episode["failed_step"] == 3 and episode["steps_completed"] == 2 and episode["steps_attempted"] == 3
    totals = json.loads((run_dir / "summary.json").read_text())["totals"]
    assert (totals["successful_attempts"], totals["total_attempts"], totals["completed_steps"]) == (2, 3, 2)
    assert "step 3 failed (auto)" in (run_dir / "summary.md").read_text()


def test_keys_end_steps_and_recover_retries_once(tmp_path):
    steps = ["s", "f", "s", "s", "s", "s", "s", "s"]
    run_dir, runner, keys, lines = run_fake(
        tmp_path, {"start": "\n", "step": steps, "restore": "\n"}, delay_s=0.2, protocol="recover"
    )
    assert run_dir.name.endswith("-recover")
    attempts = read_jsonl(run_dir / "ep01" / "attempts.jsonl")
    assert len(attempts) == 8
    assert all(a["end_reason"] == "key" for a in attempts)
    assert [(a["step"], a["try"], a["success"]) for a in attempts[:3]] == [(1, 0, True), (2, 0, False), (2, 1, True)]
    assert attempts[2]["reset_history"] and attempts[2]["restored"] and attempts[2]["interventions"] == 1
    episode = json.loads((run_dir / "ep01" / "episode.json").read_text())
    assert episode["end"] == "done" and episode["steps_completed"] == 7 and episode["first_try_in_row"] == 1
    assert episode["interventions"] == 1 and not episode["success"]


def test_demo_mode_counts_auto_ends_and_carries_the_tower_back(tmp_path):
    run_dir, runner, keys, lines = run_fake(
        tmp_path, {"start": "\n", "verdict": "q"}, episodes=2, mode="demo", pieces=["large disk", "small disk"]
    )
    assert run_dir.name.endswith("-demo")
    assert [e["end"] for e in runner.episodes] == ["done", "done"], lines
    assert all(a["end_reason"] == "auto" and a["verdict_by"] == "auto" for a in runner.attempts)
    assert keys.pressed == [("start", "\n")]
    first, second = runner.episodes
    assert first["initial_state"] == {"left": [2, 1], "middle": [], "right": []}
    assert second["initial_state"] == {"left": [], "middle": [], "right": [2, 1]}
    assert second["tracked_state"] == {"left": [2, 1], "middle": [], "right": []}
    assert second["moves"][0]["prompt"] == "pick up the small disk and place it on the middle peg"


class TerminalCloses(ScriptedKeys):
    """Answers the start prompt, then the terminal goes away during the first step."""

    def read(self, timeout, prompt):
        if prompt is not None and prompt.context == "step":
            raise EOFError("the terminal closed")
        return super().read(timeout, prompt)


def test_lost_keys_end_the_step_and_the_run(tmp_path):
    run_dir, runner, keys, lines = run_fake(tmp_path, TerminalCloses({"start": "\n"}), episodes=3)
    assert runner.quit and len(runner.episodes) == 1
    episode = runner.episodes[0]
    assert episode["end"] == "quit" and episode["attempts"] == 1 and episode["active_s"] < 2.0
    attempts = read_jsonl(run_dir / "ep01" / "attempts.jsonl")
    assert len(attempts) == 1 and attempts[0]["quit"] and attempts[0]["end_reason"] == "quit"


def test_quit_at_the_start_prompt_writes_an_empty_run(tmp_path):
    run_dir, runner, keys, lines = run_fake(tmp_path, {"start": "q"}, episodes=3)
    assert runner.quit and runner.episodes == []
    assert not (run_dir / "ep01").exists()
    assert (run_dir / "summary.md").is_file() and json.loads((run_dir / "run.json").read_text())["episodes_run"] == 0


def test_sanity_run_is_one_step_episodes(tmp_path):
    run_dir, runner, keys, lines = run_fake(
        tmp_path, {"start": "\n", "step": "s"}, sanity=1, delay_s=0.1, **{"sanity.moves": "solution"}
    )
    assert run_dir.name.endswith("-sanity")
    assert len(runner.episodes) == 7 and all(e["steps_total"] == 1 and e["success"] for e in runner.episodes)
    summary = (run_dir / "summary.md").read_text()
    assert "## Per move" in summary and "small disk: left peg -> right peg" in summary
