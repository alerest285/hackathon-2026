"""Arm console: a localhost page to drive the SO-101 with FLUX by typing instructions.

Runs on the Mac at the arm, inside robot/'s venv, and wraps the Hanoi agent's ControlLoop (robot/so101_client)
without changing it. Start it with console/start.sh (tunnel + server + this), or by hand:

    cd robot && uv run --python 3.13 python ../console/app.py --port /dev/tty.usbmodemXXXX --robot-id so101_hanoi \
        --scene 0 --wrist 1 [--live] [--limits limits.json]
    open http://127.0.0.1:8765

Dry run is the default: the arm's state is read and the commands are only shown. --live writes them to the motors.
No hardware? --fake-robot --fake-cameras (and --fake-net for no server) exercise the whole page.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import os  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ROBOT_DIR = Path(os.environ.get("HACK_ROBOT_DIR", ROOT / "robot"))  # override only for testing another checkout
sys.path.insert(0, str(ROBOT_DIR))

import cv2  # noqa: E402  (robot/ venv ships opencv-python-headless)
import numpy as np  # noqa: E402

import run as robot_run  # noqa: E402  robot/run.py: build() and load_limits()
from so101_client import STATE_NAMES  # noqa: E402
from so101_client.loop import ControlLoop, LoopConfig  # noqa: E402

PRESETS = [l.strip() for l in (Path(__file__).parent / "presets.txt").read_text().splitlines()
           if l.strip() and not l.startswith("#")]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--http-port", type=int, default=8765)
    p.add_argument("--server", default="ws://localhost:8000")
    p.add_argument("--live", action="store_true", help="write commands to the real arm (default: dry run)")
    p.add_argument("--port", help="SO-101 serial port (lerobot-find-port)")
    p.add_argument("--robot-id", default="so101_hanoi")
    p.add_argument("--scene", default="0")
    p.add_argument("--wrist", default="1")
    p.add_argument("--limits", type=Path)
    p.add_argument("--mode", choices=("prefetch", "sync"), default="prefetch")
    p.add_argument("--step-max", type=float, default=6.0)
    p.add_argument("--fake-robot", action="store_true")
    p.add_argument("--fake-cameras", action="store_true")
    p.add_argument("--fake-net", nargs="?", const="pickplace", choices=("hold", "sine", "pickplace"), default=None)
    a = p.parse_args(argv)
    if not a.fake_robot and not a.port:
        p.error("--port is required unless --fake-robot (find it with: uv run --python 3.13 lerobot-find-port)")
    return a


class Console:
    def __init__(self, a):
        self.a = a
        # robot/run.py's build() takes its own argparse namespace; give it the fields it reads.
        ns = argparse.Namespace(
            fake_robot=a.fake_robot, fake_cameras=a.fake_cameras, fake_net=a.fake_net, port=a.port,
            robot_id=a.robot_id, scene=a.scene, wrist=a.wrist, server=a.server, image_size=256,
            image_format="jpeg", jpeg_quality=95, latency_ms=400.0, jitter_ms=50.0)
        robot, cameras, net = robot_run.build(ns)
        cfg = LoopConfig(mode=a.mode, step_max_deg=a.step_max, joint_limits=robot_run.load_limits(a.limits),
                         dry_run=not a.live)
        self.loop = ControlLoop(robot, cameras, net, cfg)
        self.lock = threading.Lock()
        self.jpeg: dict[str, bytes] = {}
        self._last_enc: dict[str, float] = {}
        self.info = None
        self.last_chunk = None
        self.history: list[dict] = []  # instructions sent, newest last
        self.started = False
        self.error = None
        self.loop.add_observer(self._on_tick)
        self.loop.add_frame_tap(self._on_frame)

    # called on the control thread every tick: keep it cheap
    def _on_tick(self, info):
        with self.lock:
            self.info = info
            if info.chunk:
                c = dict(info.chunk)
                c.pop("actions", None)
                self.last_chunk = c

    # called on camera threads with full-resolution BGR frames: encode at most 6 Hz per camera
    def _on_frame(self, cam, frame_bgr, t):
        now = time.monotonic()
        if now - self._last_enc.get(cam, 0.0) < 0.16:
            return
        self._last_enc[cam] = now
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self.lock:
                self.jpeg[cam] = buf.tobytes()

    def start(self):
        if not self.started:
            self.loop.start()  # connects arm, cameras, server; holds until the first instruction
            self.started = True

    def status(self) -> dict:
        with self.lock:
            info, chunk = self.info, self.last_chunk
        st = {"live": bool(self.a.live), "started": self.started, "running": self.loop.running,
              "task": self.loop.task, "error": self.error, "history": self.history[-8:], "presets": PRESETS,
              "hello": {k: v for k, v in (getattr(self.loop, "hello", None) or {}).items()
                        if isinstance(v, (str, int, float, bool))}}
        if info is not None:
            st.update(tick=info.k, held=bool(info.held), remaining=int(info.remaining),
                      state=dict(zip(STATE_NAMES, np.round(np.asarray(info.state, float), 1).tolist())),
                      cmd=dict(zip(STATE_NAMES, np.round(np.asarray(info.cmd, float), 1).tolist())),
                      lateness_ms=round(float(info.lateness_ms), 1),
                      frame_age_ms={k: round(float(v), 0) for k, v in (info.frame_age_ms or {}).items()})
        st["last_chunk"] = chunk
        # The checkpoint was trained on one arm calibration. Joints far outside its q01..q99 range mean ours differs,
        # and zero-shot cannot work until the calibration (or an offset) matches.
        hello = getattr(self.loop, "hello", None) or {}
        lo, hi = hello.get("state_q01"), hello.get("state_q99")
        if info is not None and lo and hi:
            s_ = np.asarray(info.state, float)
            st["trained_range"] = {n: [round(l, 0), round(h, 0)] for n, l, h in zip(STATE_NAMES, lo, hi)}
            st["out_of_range"] = [n for n, v, l, h in zip(STATE_NAMES, s_, lo, hi) if v < l - 10 or v > h + 10]
        try:
            s = self.loop.stats()
            st["stats"] = {k: v for k, v in s.items() if isinstance(v, (int, float, str, bool)) or
                           (isinstance(v, dict) and all(isinstance(x, (int, float)) for x in v.values()))}
        except Exception as e:  # stats are best-effort
            st["stats_error"] = str(e)
        return st


def make_handler(con: Console):
    page = (Path(__file__).parent / "index.html").read_bytes()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, default=str).encode())

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if self.path == "/status":
                return self._json(con.status())
            if self.path.startswith("/cam/"):
                cam = self.path.split("/")[2].split(".")[0].split("?")[0]
                with con.lock:
                    jpg = con.jpeg.get(cam)
                return self._send(200, jpg, "image/jpeg") if jpg else self._send(404, b"no frame yet", "text/plain")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            try:
                if self.path == "/task":
                    task = str(body.get("task", "")).strip()
                    if not task:
                        return self._json({"error": "empty instruction"}, 400)
                    con.start()
                    reset = bool(body.get("reset", True))
                    con.loop.set_task(task, reset_history=reset)
                    con.history.append({"t": time.strftime("%H:%M:%S"), "task": task, "reset": reset})
                    con.error = None
                    return self._json({"ok": True})
                if self.path == "/pause":
                    con.loop.pause() if con.started else None
                    return self._json({"ok": True})
                if self.path == "/connect":
                    con.start()
                    return self._json({"ok": True})
                self._json({"error": "unknown"}, 404)
            except Exception as e:
                con.error = f"{type(e).__name__}: {e}"
                self._json({"error": con.error}, 500)

    return H


def main(argv=None):
    a = parse_args(argv)
    if a.live:
        print("LIVE MODE: commands go to the motors. Someone must stand at the arm with a hand on the power switch.")
        input("Press ENTER to continue, Ctrl-C to abort. ")
    con = Console(a)
    srv = ThreadingHTTPServer(("127.0.0.1", a.http_port), make_handler(con))
    print(f"arm console on http://127.0.0.1:{a.http_port}  ({'LIVE' if a.live else 'dry run'}). Ctrl-C stops; the arm holds.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        if con.started:
            con.loop.stop()
        print("stopped: last command stays the motors' goal (arm holds), port closed with torque on")


if __name__ == "__main__":
    main()
