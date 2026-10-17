"""Record console runs as native LeRobot datasets (the same layout lerobot-record writes for an SO-101).

Features, matching LeRobot's so100_follower / so101_follower (same class in LeRobot 0.6.1) + two OpenCV cameras named scene and wrist:
  observation.state  float32 (6,)  shoulder_pan.pos ... gripper.pos   measured joints
  action             float32 (6,)  same names                          the command sent (or that a dry run would send)
  observation.images.scene / observation.images.wrist   video 480x640x3 RGB
One console run = one episode, with the instruction as its task. Each episode's verdict (success / fail) goes in
<root>/console_episodes.jsonl; LeRobot's own metadata stays untouched, so the dataset loads anywhere LeRobot runs.

All dataset calls happen on one writer thread: the control loop and camera threads only enqueue.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("console.recorder")
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
H, W = 480, 640


def features() -> dict:
    names = [f"{j}.pos" for j in JOINTS]
    cam = {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]}
    return {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": names},
        "action": {"dtype": "float32", "shape": (6,), "names": names},
        "observation.images.scene": dict(cam),
        "observation.images.wrist": dict(cam),
    }


class Recorder:
    def __init__(self, repo_id: str, root: Path, fps: int = 30, robot_type: str = "so100_follower"):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.root = Path(root).expanduser()
        if (self.root / "meta" / "info.json").exists():
            self.ds = LeRobotDataset.resume(repo_id, root=self.root) if hasattr(LeRobotDataset, "resume") else None
            if self.ds is None:
                raise RuntimeError(f"{self.root} exists and this LeRobot cannot resume; pick another --record-root")
        else:
            self.root.parent.mkdir(parents=True, exist_ok=True)
            self.ds = LeRobotDataset.create(repo_id, fps=fps, features=features(), root=self.root,
                                            robot_type=robot_type, use_videos=True, image_writer_threads=4)
        self.repo_id, self.fps = repo_id, fps
        self.q: queue.Queue = queue.Queue(maxsize=600)
        self.frames = 0  # frames in the open episode
        self.task = None
        self.saved = 0
        self.dropped = 0  # total this session
        self._ep_dropped = 0  # in the open episode
        self.last_error = None
        self._t = threading.Thread(target=self._run, name="console-recorder", daemon=True)
        self._t.start()

    # -- producers (any thread) -------------------------------------------------------------
    def add(self, state, action, scene_bgr, wrist_bgr, task: str):
        try:
            self.q.put_nowait(("frame", np.asarray(state, np.float32).copy(), np.asarray(action, np.float32).copy(),
                               scene_bgr, wrist_bgr, task))
        except queue.Full:  # the writer fell behind: never block the control loop, but never hide it either
            self.dropped += 1
            self._ep_dropped += 1
            self.last_error = f"writer behind: {self.dropped} frames dropped (episode marked in console_episodes.jsonl)"

    def save(self, verdict: str | None = None):
        self.q.put(("save", verdict))

    def discard(self):
        self.q.put(("discard",))

    def close(self):
        self.q.put(("close",))
        self._t.join(timeout=120)

    def status(self) -> dict:
        return {"repo_id": self.repo_id, "root": str(self.root), "episode_frames": self.frames,
                "episodes_saved": self.saved, "dropped": self.dropped, "queue": self.q.qsize(),
                "error": self.last_error}

    # -- writer thread ------------------------------------------------------------------------
    @staticmethod
    def _rgb(bgr):
        if bgr.shape[0] != H or bgr.shape[1] != W:
            bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _run(self):
        while True:
            item = self.q.get()
            try:
                kind = item[0]
                if kind == "frame":
                    _, state, action, scene, wrist, task = item
                    if self.task is not None and task != self.task and self.frames:
                        self._save(None)  # instruction changed mid-run without an explicit save: close the episode
                    self.task = task
                    self.ds.add_frame({"observation.state": state, "action": action,
                                       "observation.images.scene": self._rgb(scene),
                                       "observation.images.wrist": self._rgb(wrist), "task": task})
                    self.frames += 1
                elif kind == "save":
                    self._save(item[1])
                elif kind == "discard":
                    if self.frames:
                        self.ds.clear_episode_buffer()
                    self.frames, self.task, self._ep_dropped = 0, None, 0
                elif kind == "close":
                    if self.frames:
                        self._save(None)
                    self.ds.finalize()
                    return
            except Exception as e:  # keep the robot loop alive whatever the dataset does
                self.last_error = f"{type(e).__name__}: {e}"
                log.exception("recorder")

    def _save(self, verdict):
        if not self.frames:
            return
        n, task = self.frames, self.task
        self.ds.save_episode()
        idx = self.ds.meta.total_episodes - 1
        with open(self.root / "console_episodes.jsonl", "a") as f:
            f.write(json.dumps({"episode_index": idx, "task": task, "frames": n, "seconds": round(n / self.fps, 1),
                                "verdict": verdict or "unlabeled", "dropped_frames": self._ep_dropped, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
        self.saved += 1
        self.frames, self.task, self._ep_dropped = 0, None, 0
