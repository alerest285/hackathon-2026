"""End to end over a real socket, no hardware: the so101.v1 server (infra/gpu/so101/server.py, fake engine) in a
subprocess, and the eval runner with FakeRobot + FakeCameras and the real NetClient, assembled by the same session
code as run_hanoi.py --fake-hardware. One strict episode in eval mode; the test-only unattended operator confirms
success after each auto end (run_hanoi.py --unattended). The server's pickplace cycle is shortened to 3 s, and its
300 ms latency leaves a request in flight at most step changes, so stale replies cross the socket too.
"""

import contextlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import run_hanoi
from hanoi_eval.config import load_config
from hanoi_eval.keys import unattended_keys
from hanoi_eval.session import SessionOptions, run_session

EVAL = Path(__file__).resolve().parents[1]
SERVER = EVAL.parent / "infra" / "gpu" / "so101" / "server.py"
CONFIG = EVAL / "config" / "hanoi.toml"

PERIOD_S = 3.0  # fake pickplace cycle: grasp at about 0.8 s, release at about 1.45 s, rest from 2.2 s
SERVER_ARGS = ["--fake-mode", "pickplace", "--fake-period-s", str(PERIOD_S), "--fake-latency-ms", "300",
               "--fake-jitter-ms", "50"]
# Thresholds for the 3 s cycle: its return swing reaches about 25 deg/s, so rest is anything below 40.
FAST = {
    "step_timeout_s": 10.0,
    "auto_end.min_hold_s": 0.15,
    "auto_end.rest_s": 0.25,
    "auto_end.rest_speed_deg_s": 40.0,
    "cameras.width": 320,
    "cameras.height": 240,
}
LAYOUT_RUN = ("run.json", "summary.md", "summary.json")
LAYOUT_EPISODE = ("episode.json", "attempts.jsonl", "ticks.jsonl", "chunks.npz", "scene.mp4", "wrist.mp4",
                  "scene.times.csv", "wrist.times.csv")


@contextlib.contextmanager
def fake_server(tmp_path: Path, *args: str):
    """server.py --engine fake on 127.0.0.1:<port chosen by the OS>; yields the port. Output goes to a file, so a
    full pipe can never stall the server."""
    out_path = tmp_path / "server.out"
    cmd = [sys.executable, str(SERVER), "--engine", "fake", "--port", "0", *args]
    with open(out_path, "w") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 30
            while True:
                text = out_path.read_text()
                found = re.search(r"listening on ws://127\.0\.0\.1:(\d+)", text)
                if found:
                    break
                assert proc.poll() is None, f"server exited:\n{text}"
                assert time.monotonic() < deadline, f"server did not start:\n{text}"
                time.sleep(0.05)
            yield int(found.group(1))
        finally:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(5)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def holds_after_first_chunk(ticks: list[dict]) -> dict:
    """Per attempt: ticks held with a task set after that attempt's first accepted chunk."""
    out: dict = {}
    seen: set = set()
    for t in ticks:
        a = t["attempt"]
        if a is None:
            continue
        out.setdefault(a, 0)
        if (t.get("chunk") or {}).get("accepted"):
            seen.add(a)
        elif a in seen and t["held"] and t["task"] is not None:
            out[a] += 1
    return out


def test_strict_episode_over_the_real_socket(tmp_path):
    server_log = tmp_path / "server-log"
    lines: list[str] = []
    keys = unattended_keys()
    with fake_server(tmp_path, *SERVER_ARGS, "--log-dir", str(server_log)) as port:
        url = f"ws://127.0.0.1:{port}"
        cfg = load_config(CONFIG, {**FAST, "server.url": url})
        opts = SessionOptions(fake_hardware=True, episodes=1, out=tmp_path / "runs")
        run_dir, runner = run_session(cfg, opts, keys_source=keys, log=lines.append)
    server_out = (tmp_path / "server.out").read_text()

    for name in LAYOUT_RUN:
        assert (run_dir / name).is_file(), name
    ep_dir = run_dir / "ep01"
    for name in LAYOUT_EPISODE:
        assert (ep_dir / name).is_file() and (ep_dir / name).stat().st_size > 0, name

    run = json.loads((run_dir / "run.json").read_text())
    assert run["fake"] is False and run["fake_hardware"] is True and run["server_url"] == url
    assert run["loop"]["dry_run"] is False  # the fake arm takes the commands, so the measured gripper moves
    hello = run["server_hello"]
    assert hello["protocol"] == "so101.v1" and hello["engine"] == "fake" and hello["fake"]["period_s"] == PERIOD_S

    attempts = read_jsonl(ep_dir / "attempts.jsonl")
    assert [a["end_reason"] for a in attempts] == ["auto"] * 7, lines
    assert all(a["success"] and a["verdict_by"] == "operator" for a in attempts)
    assert keys.pressed == [("start", "\n")] + [("verdict", "s")] * 7
    assert [a["reset_history"] for a in attempts] == [True] + [False] * 6
    episode = json.loads((ep_dir / "episode.json").read_text())
    assert episode["success"] and episode["end"] == "done" and episode["interventions"] == 0
    assert episode["steps_completed"] == 7
    assert episode["tracked_state"] == {"left": [], "middle": [], "right": [3, 2, 1]}

    stats = run["loop_stats"]
    print("e2e socket:", {k: stats[k] for k in ("lateness_p99_ms", "starved", "plans", "stale", "dropped",
                                                "turnaround_p50_ms", "turnaround_p95_ms", "server_p50_ms")})
    assert stats["error"] is None and stats["errors"] == 0 and stats["timeouts"] == 0 and stats["unmatched"] == 0
    assert stats["plans"] >= 7 and stats["starved"] == 0, stats
    assert stats["lateness_p99_ms"] < 10.0, stats
    assert 0 < stats["turnaround_p50_ms"] < 1000.0, stats
    ticks = read_jsonl(ep_dir / "ticks.jsonl")
    assert holds_after_first_chunk(ticks) == dict.fromkeys(range(7), 0)

    # Every chunk the client recorded is the one the server logged for that request: same id, tick and actions.
    chunks = np.load(ep_dir / "chunks.npz")
    assert chunks["actions"].shape[1:] == (42, 6) and chunks["accepted"].sum() >= 7
    assert np.isfinite(chunks["server_ms"]).all() and (chunks["server_ms"] > 0).all()
    assert not chunks["accepted"][chunks["reason"] == "stale"].any()  # replies to the previous step are never run
    events = [json.loads(line) for f in sorted(server_log.glob("so101-*.jsonl")) for line in f.read_text().splitlines()]
    served = {e["id"]: e for e in events if e["event"] == "infer"}
    assert [e["event"] for e in events].count("connect") == 1, server_out  # one connection, no reconnect
    for rid, tick, actions in zip(chunks["id"], chunks["tick"], chunks["actions"], strict=True):
        logged = served[int(rid)]
        assert logged["tick"] == int(tick)
        np.testing.assert_allclose(np.array(logged["actions"]), actions, atol=2e-3)
    assert set(e["task"] for e in served.values()) == {a["prompt"] for a in attempts}


@pytest.mark.parametrize(
    "argv",
    [["--unattended"], ["--fake", "--fake-hardware"], ["--live", "--fake-hardware"], ["--auto-confirm", "--live"]],
)
def test_cli_refuses_unsafe_flag_combinations(argv, capsys):
    assert run_hanoi.main(argv) == 2
    assert capsys.readouterr().err
