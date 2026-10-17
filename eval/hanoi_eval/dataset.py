"""Hanoi attempts as a native LeRobotDataset, plus the two console/ modules eval reuses.

One step attempt = one LeRobot episode, with the move's prompt as its task, so the episodes train like the teleop
demos (console/teleop.sh) and load anywhere LeRobot runs. The writer is console/recorder.py's Recorder (features
of lerobot-record's so100_follower: observation.state and action float32 (6,) in the arm's own degrees, scene and
wrist video at 480x640). LeRobot's metadata stays untouched; the verdict and the Hanoi context of each episode go
in <root>/hanoi_episodes.jsonl, one line per saved episode, keyed by episode_index.

console/mapping.py (T20/T21) owns the joint map format, so eval loads that module instead of copying it.
Both are loaded by file path under private names, so nothing called `mapping` or `recorder` lands on sys.path.
"""

from __future__ import annotations

import importlib.util
import json
import threading
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
CONSOLE_DIR = REPO_DIR / "console"


def console_module(name: str):
    """console/<name>.py as a module named hanoi_console_<name>."""
    key = f"hanoi_console_{name}"
    path = CONSOLE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(key, path)
    if spec is None or not path.is_file():
        raise ImportError(f"{path} not found: eval needs console/{name}.py from the same checkout")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_joint_map(path):
    """(JointMap, MappedRobot class) for the map file at path (console/mapping.py's JSON)."""
    mapping = console_module("mapping")
    return mapping.JointMap.load(Path(path).expanduser()), mapping.MappedRobot


def verdict_word(attempt: dict) -> str:
    if attempt["success"] is True:
        return "success"
    if attempt["success"] is False:
        return "fail"
    return "error" if attempt["end_reason"] == "error" else "quit"


class HanoiDataset:
    """begin(prompt) / add(...) per tick / end(verdict, meta). add() never blocks the control thread, and the lock
    makes sure no frame of one attempt can land in the next episode."""

    def __init__(self, repo_id: str, root, robot_type: str = "so100_follower", fps: int = 30):
        recorder = console_module("recorder")

        class Writer(recorder.Recorder):
            def _save(self, verdict):  # runs on the recorder's writer thread, in queue order
                word, meta = verdict if isinstance(verdict, tuple) else (verdict, {})
                before = self.ds.meta.total_episodes
                frames, task = self.frames, self.task
                dropped = getattr(self, "_ep_dropped", None)
                super()._save(word)
                if self.ds.meta.total_episodes > before:
                    line = {"episode_index": before, "task": task, "frames": frames, "verdict": word or "unlabeled",
                            "dropped_frames": dropped, **meta}
                    with open(self.root / "hanoi_episodes.jsonl", "a") as f:
                        f.write(json.dumps(line) + "\n")

        self.writer = Writer(repo_id, Path(root).expanduser(), fps=fps, robot_type=robot_type)
        self.repo_id, self.root = repo_id, self.writer.root
        self._task: str | None = None
        self._lock = threading.Lock()
        self.frames_added = 0

    def begin(self, task: str) -> None:
        with self._lock:
            self._task = task

    def add(self, state, action, scene_bgr, wrist_bgr, task: str | None) -> None:
        """Control thread. Only frames of the open attempt's task are kept."""
        with self._lock:
            if self._task is None or task != self._task or scene_bgr is None or wrist_bgr is None:
                return
            self.writer.add(state, action, scene_bgr, wrist_bgr, task)
            self.frames_added += 1

    def end(self, verdict: str, meta: dict) -> None:
        """Close the open attempt as one episode (nothing is saved when it got no frame)."""
        with self._lock:
            self._task = None
            self.writer.save((verdict, meta))

    def pause(self) -> None:
        """Stop taking frames (a human is in the scene); begin() resumes."""
        with self._lock:
            self._task = None

    def status(self) -> dict:
        return {**self.writer.status(), "frames_added": self.frames_added}

    def close(self) -> None:
        self.writer.close()
