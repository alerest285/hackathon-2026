#!/usr/bin/env python3
"""Reproduce SO-101 history/failure handling entirely offline.

Uses the repository's actual ControlLoop._tick() with inert, local Python stubs.
Never calls start(), connects a device/socket, imports robot_io, or runs inference.
Synthetic +2-degree deltas illustrate the integration mechanism; they are NOT
actual FLUX predictions or an estimate of Saad's real-arm trajectory.

Run from the repo root using the robot environment:
  robot/.venv/bin/python docs/research/so101_calibration_audit.py

This is a historical bug reproducer, not a safety acceptance test. Its assertions
describe the audited behavior and should fail once that behavior is corrected.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True

import numpy as np


class StubRobot:
    """Fixed arm, instant-following simulation, or always-failing transport."""

    def __init__(self, behavior: str):
        self.behavior = behavior
        self.state = np.array([0, -101.8, 0, 0, 0, 50], dtype=np.float32)
        self.initial = self.state.copy()
        self.attempted_writes = 0
        self.successful_writes = 0

    def read_state(self):
        return self.state.copy()

    def write(self, cmd):
        self.attempted_writes += 1
        if self.behavior == "fail":
            raise OSError("synthetic failed write; no device exists")
        if self.behavior == "follow":
            self.state = np.asarray(cmd, dtype=np.float32).copy()
        self.successful_writes += 1


class StubCameras:
    names = ("scene", "wrist")

    def latest(self):
        # Encoded data never leaves this process and is never image-decoded.
        return {name: SimpleNamespace(t=time.perf_counter(), encoded=b"stub-frame")
                for name in self.names}


class StubNet:
    connected = True
    hello = {}

    def __init__(self, delta: float):
        self.delta = delta
        self.rid = 0
        self.replies = []
        self.windows = []

    def submit(self, window, task, tick):
        self.rid += 1
        self.windows.append((window.state.copy(), window.cmd_hist.copy()))
        actions = np.tile(window.cmd_hist[-1], (42, 1))
        actions[:, 1] += np.arange(1, 43) * self.delta
        now = time.perf_counter()
        self.replies.append(SimpleNamespace(
            id=self.rid, actions=actions, error=None, timing={},
            recv_t=now, sent_t=now,
        ))
        return self.rid

    def poll(self):
        replies, self.replies = self.replies, []
        return replies


def git_head(path: Path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
    ).strip()


def run_case(ControlLoop, LoopConfig, *, label, mode, dry_run, behavior,
             ticks, delta, expected_command, expect_history_mismatch):
    robot, cameras, net = StubRobot(behavior), StubCameras(), StubNet(delta)
    loop = ControlLoop(robot, cameras, net, LoopConfig(
        mode=mode, dry_run=dry_run, caffeinate=False,
    ))
    loop.set_task("Synthetic constant lift delta", reset_history=True)
    # No thread, timing sleeps, hardware, network, or filesystem outputs.
    for tick in range(ticks):
        loop._tick(tick, time.perf_counter(), 0.0)

    observed, history = net.windows[-1]
    mismatch = float(np.max(np.abs(observed - history)))
    np.testing.assert_allclose(loop._last_cmd[1], expected_command, atol=0.001, rtol=0)
    assert loop.error is None
    assert loop._counts["ticks"] == ticks
    assert loop._counts["errors"] == 0
    assert loop._counts["bus_errors"] == 0
    if dry_run:
        assert robot.attempted_writes == robot.successful_writes == 0
        np.testing.assert_array_equal(robot.state, robot.initial)
    elif behavior == "fail":
        assert robot.attempted_writes == ticks
        assert robot.successful_writes == 0
        assert loop._counts["write_errors"] == ticks
        np.testing.assert_array_equal(robot.state, robot.initial)
    elif behavior == "follow":
        assert robot.attempted_writes == robot.successful_writes == ticks
        assert loop._counts["write_errors"] == 0
        np.testing.assert_array_equal(robot.state, loop._last_cmd)
    if expect_history_mismatch:
        assert mismatch > 100.0
    else:
        np.testing.assert_array_equal(observed, history)
    return {
        "case": label, "mode": mode, "dry_run": dry_run, "ticks": ticks,
        "synthetic_delta_deg": delta,
        "measured_lift": round(float(robot.state[1]), 3),
        "last_command_lift": round(float(loop._last_cmd[1]), 3),
        "last_request_measured_lift": round(float(observed[-1, 1]), 3),
        "last_request_command_anchor": round(float(history[-1, 1]), 3),
        "max_window_state_command_gap": round(mismatch, 3),
        "attempted_writes": robot.attempted_writes,
        "successful_writes": robot.successful_writes,
        "write_errors": loop._counts["write_errors"],
        "assertions": "PASS",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path,
                        default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "robot"))
    from so101_client.loop import ControlLoop, LoopConfig

    source_paths = ["robot/so101_client/loop.py", "robot/so101_client/chunks.py",
                    "robot/so101_client/history.py", "robot/so101_client/robot_io.py"]
    provenance = {
        "repo": str(repo), "commit": git_head(repo),
        "source_sha256": {path: hashlib.sha256((repo / path).read_bytes()).hexdigest()
                          for path in source_paths},
        "python": sys.version.split()[0], "numpy": np.__version__,
    }
    try:
        provenance["installed_lerobot_version"] = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError:
        pass
    print(json.dumps({"provenance": provenance}, sort_keys=True))

    cases = [
        # Controls: a zero-delta policy causes no history divergence.
        dict(label="zero_delta_control_sync", mode="sync", dry_run=True,
             behavior="fixed", ticks=390, delta=0, expected_command=-101.8,
             expect_history_mismatch=False),
        dict(label="zero_delta_control_prefetch", mode="prefetch", dry_run=True,
             behavior="fixed", ticks=390, delta=0, expected_command=-101.8,
             expect_history_mismatch=False),
        # Demonstrates accumulation even in sync mode without chunk crossfades.
        dict(label="dry_run_sync", mode="sync", dry_run=True,
             behavior="fixed", ticks=390, delta=2, expected_command=654.2,
             expect_history_mismatch=True),
        dict(label="dry_run_prefetch", mode="prefetch", dry_run=True,
             behavior="fixed", ticks=390, delta=2, expected_command=676.2,
             expect_history_mismatch=True),
        # Control: a following simulated arm has consistent measured/command history.
        dict(label="following_simulation_control", mode="sync", dry_run=False,
             behavior="follow", ticks=100, delta=2, expected_command=90.2,
             expect_history_mismatch=False),
        # Actual failure-handling bug: errors do not stop candidate integration.
        dict(label="every_write_fails", mode="sync", dry_run=False,
             behavior="fail", ticks=100, delta=2, expected_command=90.2,
             expect_history_mismatch=True),
    ]
    for case in cases:
        print(json.dumps(run_case(ControlLoop, LoopConfig, **case), sort_keys=True))
    print("PASS: six offline cases; no hardware, camera, network, or model was used.")


if __name__ == "__main__":
    main()
