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
    def __init__(self, repo_id: str, root: Path, fps: int = 30, robot_type: str = "so100_follower",
                 run_mode: str = "dry"):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.root = Path(root).expanduser()
        if (self.root / "meta" / "info.json").exists():
            self.ds = LeRobotDataset.resume(repo_id, root=self.root, streaming_encoding=True, encoder_threads=2) \
                if hasattr(LeRobotDataset, "resume") else None
            if self.ds is None:
                raise RuntimeError(f"{self.root} exists and this LeRobot cannot resume; pick another --record-root")
        else:
            self.root.parent.mkdir(parents=True, exist_ok=True)
            # streaming_encoding: frames go straight into the video encoder. Without it LeRobot writes every frame of
            # both cameras as a PNG first (~30 MB/s), which filled Saad's disk.
            self.ds = LeRobotDataset.create(repo_id, fps=fps, features=features(), root=self.root,
                                            robot_type=robot_type, use_videos=True, image_writer_threads=0,
                                            streaming_encoding=True, encoder_threads=2)
        self.repo_id, self.fps, self.run_mode = repo_id, fps, run_mode  # run_mode: live | dry | fake
        # One ORDERED queue for frames and controls, unbounded so put() never blocks the control thread. Frames beyond
        # MAX_BACKLOG are dropped and counted; controls are never dropped, and apply exactly at their place in the order.
        self.q: queue.Queue = queue.Queue()
        self.backlog = 0
        self.low_disk = False  # set by the writer thread below MIN_FREE_MB; the console reads it
        self.frames = 0  # frames in the open episode
        self.task = None
        self.saved = 0
        self.dropped = 0  # total this session
        self._ep_dropped = 0  # in the open episode
        self.last_error = None
        self.close_error = None
        self._closing = False
        self._t = threading.Thread(target=self._run, name="console-recorder", daemon=True)
        self._t.start()

    # -- producers (any thread) -------------------------------------------------------------
    def free_mb(self) -> int:
        import shutil

        return int(shutil.disk_usage(self.root).free / 1e6)

    MAX_BACKLOG = 600

    def add(self, state, action, scene_bgr, wrist_bgr, task: str):
        if self.backlog >= self.MAX_BACKLOG:  # the writer fell behind: never block the control loop, never hide it
            self.dropped += 1
            self.q.put(("dropped",))
            self.last_error = f"writer behind: {self.dropped} frames dropped (episode marked in console_episodes.jsonl)"
            return
        self.backlog += 1
        self.q.put(("frame", np.asarray(state, np.float32).copy(), np.asarray(action, np.float32).copy(),
                    scene_bgr, wrist_bgr, task))

    def save(self, verdict: str | None = None):
        self.q.put(("save", verdict))

    def discard(self):
        self.q.put(("discard",))

    def close(self) -> str | None:
        """Save the open episode and finalize. Returns None on success, else the error (never claims success)."""
        self.q.put(("close",))
        self._t.join(timeout=180)
        if self._t.is_alive():
            return "writer did not finish within 180 s; dataset may be incomplete"
        return self.close_error

    def status(self) -> dict:
        return {"repo_id": self.repo_id, "root": str(self.root), "episode_frames": self.frames, "free_mb": self.free_mb(),
                "run_mode": self.run_mode, "low_disk": self.low_disk, "encoder_dropped": self._encoder_drops(),
                "episodes_saved": self.saved, "dropped": self.dropped, "queue": self.q.qsize(),
                "error": self.last_error}

    # -- writer thread ------------------------------------------------------------------------
    @staticmethod
    def _rgb(bgr):
        if bgr.shape[0] != H or bgr.shape[1] != W:
            bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    MIN_FREE_MB = 2000

    def _run(self):
        n = 0
        while True:
            item = self.q.get()
            try:
                kind = item[0]
                if kind == "frame":
                    self.backlog -= 1
                    n += 1
                    if n % 300 == 0 and self.free_mb() < self.MIN_FREE_MB:
                        self.low_disk = True
                        self._save("stopped_low_disk")
                    self._frame(item)
                elif kind == "dropped":
                    self._ep_dropped += 1
                elif kind == "save":
                    self._save(item[1])
                elif kind == "discard":
                    if self.frames:
                        self.ds.clear_episode_buffer()
                    self.frames, self.task, self._ep_dropped = 0, None, 0
                elif kind == "close":
                    self._closing = True
                    if self.frames:
                        self._save(None)
                    self.ds.finalize()
                    return
            except Exception as e:  # keep the robot loop alive whatever the dataset does
                self.last_error = f"{type(e).__name__}: {e}"
                log.exception("recorder")
                if self._closing:
                    self.close_error = self.last_error
                    return

    def _frame(self, item):
        _, state, action, scene, wrist, task = item
        if self.low_disk:
            return
        if self.task is not None and task != self.task and self.frames:
            self._save(None)  # instruction changed mid-run without an explicit save: close the episode
        self.task = task
        self.ds.add_frame({"observation.state": state, "action": action,
                           "observation.images.scene": self._rgb(scene),
                           "observation.images.wrist": self._rgb(wrist), "task": task})
        self.frames += 1

    def _encoder_drops(self) -> dict:
        """LeRobot's streaming encoder drops frames silently when its queue is full; read its own counter."""
        enc = getattr(getattr(self.ds, "writer", None), "_streaming_encoder", None)
        return dict(getattr(enc, "_dropped_frames", {}) or {})

    MIN_FRAMES = 10  # LeRobot 0.6.1 cannot finalize a dataset holding a 1-frame episode (ArrowInvalid in stats)

    def _save(self, verdict):
        if not self.frames:
            return
        if self.frames < self.MIN_FRAMES:  # too short to keep: drop it, note it, never break the dataset
            n, task = self.frames, self.task
            self.ds.clear_episode_buffer()
            with open(self.root / "console_episodes.jsonl", "a") as f:
                f.write(json.dumps({"episode_index": None, "task": task, "frames": n, "verdict": "discarded_too_short",
                                    "requested_verdict": verdict, "run_mode": self.run_mode,
                                    "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
            self.last_error = f"episode of {n} frames discarded (minimum {self.MIN_FRAMES})"
            self.frames, self.task, self._ep_dropped = 0, None, 0
            return
        n, task = self.frames, self.task
        enc_drops = self._encoder_drops()  # read before save_episode: the next episode resets it
        self.ds.save_episode()
        idx = self.ds.meta.total_episodes - 1
        with open(self.root / "console_episodes.jsonl", "a") as f:
            f.write(json.dumps({"episode_index": idx, "task": task, "frames": n, "seconds": round(n / self.fps, 1),
                                "verdict": verdict or "unlabeled", "run_mode": self.run_mode,
                                "dropped_frames": self._ep_dropped, "encoder_dropped": enc_drops,
                                "valid": self._ep_dropped == 0 and not any(enc_drops.values()) and self.run_mode == "live",
                                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
        self.saved += 1
        self.frames, self.task, self._ep_dropped = 0, None, 0
