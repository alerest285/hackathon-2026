"""End to end without hardware: FakeRobot + FakeCameras + FakeNet(pickplace) through the real ControlLoop,
assembled by the same session code as run_hanoi.py --fake, with scripted operator keys."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import run_hanoi
import so101_client.fakes
from so101_client.fakes import FakeRobot
from so101_client.loop import ControlLoop

from hanoi_eval.config import load_config
from hanoi_eval.keys import ScriptedKeys
from hanoi_eval.recorder import EpisodeRecorder
from hanoi_eval.runner import InfraError
from hanoi_eval.session import SessionOptions, run_session, unwarmed_prompts

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


def run_fake(tmp_path, answers, episodes=1, sanity=None, delay_s=0.0, latency_ms=60.0, **overrides):
    cfg = load_config(CONFIG, {**FAST, **overrides})
    keys = answers if hasattr(answers, "read") else ScriptedKeys(answers, delay_s=delay_s)
    lines: list[str] = []
    opts = SessionOptions(fake=True, fake_latency_ms=latency_ms, fake_jitter_ms=10.0, fake_cycle_s=CYCLE_S,
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


class UnpluggedRobot(FakeRobot):
    """State reads fail for good 0.3 s after connect, like a pulled USB cable; the loop stops 30 ticks later."""

    def connect(self):
        super().connect()
        self.t_unplug = time.perf_counter() + 0.3

    def read_state(self):
        if time.perf_counter() > self.t_unplug:
            raise OSError("[Errno 6] Device not configured")
        return super().read_state()


def test_a_dead_control_loop_ends_the_episode_as_an_error_not_a_failed_step(tmp_path, monkeypatch):
    monkeypatch.setattr(so101_client.fakes, "FakeRobot", UnpluggedRobot)
    t0 = time.perf_counter()
    with pytest.raises(InfraError, match="control loop stopped"):
        run_fake(tmp_path, {"start": "\n", "verdict": "f"}, episodes=3, **{"auto_end.enabled": False})
    assert time.perf_counter() - t0 < FAST["step_timeout_s"]  # not booked as a timeout
    run_dir = next(tmp_path.iterdir())
    attempts = read_jsonl(run_dir / "ep01" / "attempts.jsonl")
    assert len(attempts) == 1 and attempts[0]["end_reason"] == "error" and attempts[0]["success"] is None
    assert "state read failed" in attempts[0]["error"]
    episode = json.loads((run_dir / "ep01" / "episode.json").read_text())
    assert episode["end"] == "error" and "state read failed" in episode["error"]
    assert not (run_dir / "ep02").exists()  # the run stops
    totals = json.loads((run_dir / "summary.json").read_text())["totals"]
    assert (totals["episodes"], totals["errors"], totals["total_attempts"], totals["completed_steps"]) == (0, 1, 0, 0)
    summary = (run_dir / "summary.md").read_text()
    assert "infrastructure error at step 1" in summary and "**Control loop error:**" in summary


class PauseResumeKeys:
    """Enter at the start prompt; in the step, space (pause), space (resume), then s, 0.3 s apart."""

    interactive = False

    def __init__(self):
        self.step_keys = [" ", " ", "s"]
        self.started = False
        self.pressed: list[tuple[str, str]] = []

    def read(self, timeout, prompt):
        time.sleep(min(timeout, 0.005))
        if prompt is None:
            return None
        key = None
        if prompt.context == "start" and not self.started:
            self.started, key = True, "\n"
        elif prompt.context == "step" and self.step_keys:
            if time.perf_counter() - prompt.t > 0.3 * (4 - len(self.step_keys)):
                key = self.step_keys.pop(0)
        if key is not None:
            self.pressed.append((prompt.context, key))
        return key


def test_resuming_from_a_pause_resets_the_history(tmp_path, monkeypatch):
    calls = []
    set_task = ControlLoop.set_task
    monkeypatch.setattr(ControlLoop, "set_task", lambda self, task, reset_history: (
        calls.append(reset_history), set_task(self, task, reset_history))[1])
    run_dir, runner, keys, lines = run_fake(tmp_path, PauseResumeKeys(), pieces=["disk"], **{"auto_end.enabled": False})
    attempt = read_jsonl(run_dir / "ep01" / "attempts.jsonl")[0]
    assert keys.pressed == [("start", "\n"), ("step", " "), ("step", " "), ("step", "s")]
    assert attempt["pauses"] == 1 and attempt["interventions"] == 1 and attempt["success"]
    assert calls == [True, True]  # the episode start, then the resume: a pause is an intervention


def test_strict_mode_ignores_r_during_a_step(tmp_path):
    run_dir, runner, keys, lines = run_fake(tmp_path, {"start": "\n", "step": "r", "verdict": "s"}, delay_s=0.1,
                                            pieces=["disk"])
    attempt = read_jsonl(run_dir / "ep01" / "attempts.jsonl")[0]
    assert ("step", "r") in keys.pressed
    assert attempt["end_reason"] == "auto" and attempt["success"] and not attempt["retry_requested"]


def test_every_chunk_is_logged_with_the_attempt_that_requested_it(tmp_path):
    # 600 ms turnaround (18 ticks, under the 21 that starve): requests go back to back, so one is in flight at
    # every step end, and with no verdict prompt (demo mode) the next step starts at once. Before, those
    # replies were logged with the next attempt, and the episode's last one not at all.
    run_dir, runner, keys, lines = run_fake(tmp_path, {"start": "\n", "verdict": "q"}, latency_ms=600.0, mode="demo",
                                            pieces=["large disk", "small disk"])
    assert [e["end"] for e in runner.episodes] == ["done"], lines
    ticks = read_jsonl(run_dir / "ep01" / "ticks.jsonl")
    attempt_at = {t["k"]: -1 if t["attempt"] is None else t["attempt"] for t in ticks}  # chunks.npz: -1 = none
    chunks = np.load(run_dir / "ep01" / "chunks.npz")
    assert len(chunks["id"]) > 3 and (chunks["reason"] == "stale").any()
    assert [int(a) for a in chunks["attempt"]] == [attempt_at[int(k)] for k in chunks["tick"]]
    stats = runner.run_info["loop_stats"]
    assert len(chunks["id"]) == stats["plans"] + stats["stale"] + stats["dropped"] == stats["requests"]
    attempts = read_jsonl(run_dir / "ep01" / "attempts.jsonl")
    assert [a["chunks"] for a in attempts] == [int((chunks["attempt"] == a["attempt"]).sum()) for a in attempts]


def test_attempt_at_maps_a_tick_to_the_attempt_running_then(tmp_path):
    recorder = EpisodeRecorder(tmp_path / "ep01")
    for k, attempt in [(10, None), (11, 0), (12, 0), (13, None), (14, 1)]:
        recorder.attempt = attempt
        recorder.on_tick(SimpleNamespace(k=k, t=k / 30, state=np.zeros(6), cmd=np.zeros(6), held=False,
                                         lateness_ms=0.0, frame_age_ms={}, remaining=0, task="t", chunk=None))
    recorder.close()
    assert [recorder.attempt_at(k) for k in (5, 10, 11, 12, 13, 14, 99)] == [None, None, 0, 0, None, 1, 1]


def test_print_warmup_lists_every_prompt_once(capsys):
    assert run_hanoi.main(["--print-warmup"]) == 0
    out = capsys.readouterr().out
    tasks = [line.split("--warmup-task ", 1)[1].rstrip(" \\").strip("'") for line in out.splitlines() if "--warmup-task" in line]
    assert len(tasks) == len(set(tasks)) == 9  # 3 pieces x 3 places
    assert tasks[0] == "pick up the small disk and place it on the right peg"  # step 1 of the solution
    assert "bash infra/gpu/so101/serve.sh" in out
    cfg = load_config(CONFIG)
    assert unwarmed_prompts({"warm_tasks": tasks[:2]}, cfg) == tasks[2:]
    assert unwarmed_prompts({"warm_tasks": tasks}, cfg) == [] and unwarmed_prompts({}, cfg) == []
