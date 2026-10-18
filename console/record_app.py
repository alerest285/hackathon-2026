"""Recording dashboard: collect teleop demonstrations from a browser page instead of arrow keys.

The leader arm drives the follower all the time (so resets are easy). Buttons start, save or redo an episode.
Recording mirrors LeRobot 0.6.1's lerobot-record exactly: same robot/teleop classes, default processors,
aggregate_pipeline_dataset_features + build_dataset_frame, the teleop action as `action`, LeRobotDataset with
streaming video encoding. The result is the same dataset lerobot-record would write (verified schema: codex 18.10).

Start it with console/record.sh (it opens http://127.0.0.1:8766).
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets import LeRobotDataset, aggregate_pipeline_dataset_features, create_initial_features
from lerobot.processor import make_default_processors
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts

import lerobot.robots.so_follower  # noqa: F401  (registers so100_follower / so101_follower)
import lerobot.teleoperators.so_leader  # noqa: F401  (registers so100_leader / so101_leader)

log = logging.getLogger("record_app")
MIN_FRAMES = 15  # LeRobot 0.6.1 cannot finalize a dataset with a 1-frame episode; anything this short is a mistake


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--follower", required=True)
    p.add_argument("--leader", required=True)
    p.add_argument("--robot-id", default="my_follower_arm_e82")
    p.add_argument("--leader-id", default="my_leader_e82")
    p.add_argument("--robot-type", default="so100_follower")
    p.add_argument("--teleop-type", default="so100_leader")
    p.add_argument("--scene", type=int, default=0)
    p.add_argument("--wrist", type=int, default=1)
    p.add_argument("--task", required=True)
    p.add_argument("--dataset", default="so100_soda_ale")
    p.add_argument("--repo-owner", default="alerest285")
    p.add_argument("--episode-s", type=float, default=20.0, help="recording stops itself after this long; then Save or Redo")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--http-port", type=int, default=8766)
    p.add_argument("--warmup-s", type=int, default=3)
    return p.parse_args()


class Recorder:
    def __init__(self, a):
        self.a = a
        cam = lambda i: OpenCVCameraConfig(index_or_path=i, width=640, height=480, fps=a.fps, warmup_s=a.warmup_s)
        rcfg = RobotConfig.get_choice_class(a.robot_type)(
            port=a.follower, id=a.robot_id, use_degrees=True, cameras={"scene": cam(a.scene), "wrist": cam(a.wrist)})
        tcfg = TeleoperatorConfig.get_choice_class(a.teleop_type)(port=a.leader, id=a.leader_id, use_degrees=True)
        self.robot = make_robot_from_config(rcfg)
        self.teleop = make_teleoperator_from_config(tcfg)
        self.t_proc, self.r_proc, self.o_proc = make_default_processors()
        features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(pipeline=self.t_proc,
                                                initial_features=create_initial_features(action=self.robot.action_features),
                                                use_videos=True),
            aggregate_pipeline_dataset_features(pipeline=self.o_proc,
                                                initial_features=create_initial_features(observation=self.robot.observation_features),
                                                use_videos=True))
        root = Path("~/hackathon-data").expanduser() / a.dataset
        repo = f"{a.repo_owner}/{a.dataset}"
        if (root / "meta" / "info.json").exists():
            self.ds = LeRobotDataset.resume(repo, root=root, streaming_encoding=True, encoder_threads=2)
        else:
            root.parent.mkdir(parents=True, exist_ok=True)
            self.ds = LeRobotDataset.create(repo, a.fps, root=root, robot_type=self.robot.name, features=features,
                                            use_videos=True, image_writer_threads=0, streaming_encoding=True,
                                            encoder_threads=2)
        self.root = root
        self.lock = threading.Lock()
        self.cmd = None  # "start" | "save" | "redo" | "stop", consumed by the control thread
        self.state = "connecting"  # connecting | idle | recording | timeup | saving | stopped | error
        self.frames = 0
        self.ep_start = None
        self.saved_this_session = 0
        self.message = ""
        self.jpeg = {}
        self.loop_hz = 0.0
        self._last_jpeg = 0.0
        self.thread = threading.Thread(target=self._run, name="record-loop", daemon=True)

    # ---------------------------------------------------------------- control thread (all robot + dataset calls)
    def _run(self):
        try:
            self.teleop.connect()  # leader first, like lerobot-record
            self.robot.connect()
        except Exception as e:
            self.state, self.message = "error", f"connect failed: {type(e).__name__}: {e}"
            log.exception("connect")
            return
        self.state, self.message = "idle", "Teleop live. Press Start episode."
        period = 1.0 / self.a.fps
        n, t_hz = 0, time.perf_counter()
        try:
            while True:
                t0 = time.perf_counter()
                with self.lock:
                    cmd, self.cmd = self.cmd, None
                if cmd and self._handle(cmd):
                    break
                obs = self.robot.get_observation()
                obs_p = self.o_proc(obs)
                act = self.teleop.get_action()
                act_p = self.t_proc((act, obs))
                self.robot.send_action(self.r_proc((act_p, obs)))
                if self.state == "recording":
                    frame = {**build_dataset_frame(self.ds.features, obs_p, prefix=OBS_STR),
                             **build_dataset_frame(self.ds.features, act_p, prefix=ACTION), "task": self.a.task}
                    self.ds.add_frame(frame)
                    self.frames += 1
                    if time.perf_counter() - self.ep_start >= self.a.episode_s:
                        self.state, self.message = "timeup", "Time is up: Save or Redo."
                self._preview(obs)
                n += 1
                if n % 30 == 0:
                    now = time.perf_counter(); self.loop_hz = 30 / (now - t_hz); t_hz = now
                dt = time.perf_counter() - t0
                if dt < period:
                    time.sleep(period - dt)
        except Exception as e:
            self.state, self.message = "error", f"{type(e).__name__}: {e}. Episode in progress was not saved."
            log.exception("loop")
            try:
                if self.frames:
                    self.ds.clear_episode_buffer()
                self.ds.finalize()
            except Exception:
                log.exception("finalize after error")

    def _handle(self, cmd) -> bool:
        """Returns True when the loop should end."""
        if cmd == "start" and self.state == "idle":
            self.frames, self.ep_start = 0, time.perf_counter()
            self.state, self.message = "recording", "Recording..."
        elif cmd == "save" and self.state in ("recording", "timeup"):
            if self.frames < MIN_FRAMES:
                self.message = f"Only {self.frames} frames: too short to save. Redo instead."
                return False
            self.state, self.message = "saving", "Saving episode..."
            self.ds.save_episode()
            self.saved_this_session += 1
            self.state, self.message = "idle", f"Saved ({self.frames} frames). Reset the scene, then Start episode."
            self.frames = 0
        elif cmd == "redo" and self.state in ("recording", "timeup"):
            self.ds.clear_episode_buffer()
            self.state, self.message = "idle", "Discarded. Reset the scene, then Start episode."
            self.frames = 0
        elif cmd == "stop":
            if self.state in ("recording", "timeup") and self.frames:
                self.ds.clear_episode_buffer()  # an unconfirmed episode is never saved
            self.state, self.message = "saving", "Finalizing dataset..."
            self.ds.finalize()
            self.robot.disconnect()
            self.teleop.disconnect()
            self.state, self.message = "stopped", f"Finalized: {self.root}. You can close this page."
            return True
        return False

    def _preview(self, obs):
        now = time.perf_counter()
        if now - self._last_jpeg < 0.2:
            return
        self._last_jpeg = now
        for cam in ("scene", "wrist"):
            img = obs.get(cam)
            if img is not None:
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    self.jpeg[cam] = buf.tobytes()

    def status(self):
        el = (time.perf_counter() - self.ep_start) if self.state in ("recording", "timeup") and self.ep_start else 0.0
        return {"state": self.state, "message": self.message, "frames": self.frames, "elapsed_s": round(el, 1),
                "episode_s": self.a.episode_s, "episodes_total": int(self.ds.meta.total_episodes),
                "saved_this_session": self.saved_this_session, "task": self.a.task, "dataset": str(self.root),
                "loop_hz": round(self.loop_hz, 1), "fps": self.a.fps}


def make_handler(rec: Recorder):
    page = (Path(__file__).parent / "record.html").read_bytes()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if self.path == "/status":
                return self._send(200, json.dumps(rec.status()).encode())
            if self.path.startswith("/cam/"):
                jpg = rec.jpeg.get(self.path.split("/")[2].split(".")[0].split("?")[0])
                return self._send(200, jpg, "image/jpeg") if jpg else self._send(404, b"", "text/plain")
            self._send(404, b"", "text/plain")

        def do_POST(self):
            cmd = self.path.strip("/")
            if cmd in ("start", "save", "redo", "stop"):
                with rec.lock:
                    rec.cmd = cmd
                return self._send(200, b'{"ok": true}')
            self._send(404, b"", "text/plain")

    return H


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = parse_args()
    rec = Recorder(a)
    rec.thread.start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.http_port), make_handler(rec))
    import signal

    def _stop(signum, frame):  # Ctrl-C = the Stop button: finalize, never lose saved episodes
        with rec.lock:
            rec.cmd = "stop"
        threading.Thread(target=lambda: (rec.thread.join(timeout=120), srv.shutdown()), daemon=True).start()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    print(f"recording dashboard on http://127.0.0.1:{a.http_port}  (dataset {rec.root}). Ctrl-C finalizes.", flush=True)
    try:
        srv.serve_forever()
    finally:
        srv.server_close()
        rec.thread.join(timeout=120)
        print(f"done: {rec.status()}", flush=True)


if __name__ == "__main__":
    main()
