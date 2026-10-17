"""Stand-ins with the same API as SO101, Cameras and NetClient, for tests and dry runs without hardware.

FakeRobot tracks its commands with a first-order lag. FakeCameras draws a moving pattern at 30 Hz and
encodes it like the real cameras. FakeNet answers in-process, without sockets: each reply is computed from
the request's window (joints integrated from cmd_hist[-1], gripper absolute) and handed out by poll()
after the simulated latency.
"""

from __future__ import annotations

import heapq
import itertools
import math
import threading
import time

import numpy as np

from . import CAMERAS, CHUNK, EXEC, FPS, N_OBS, PROTOCOL, STATE_NAMES, protocol
from .cameras import Frame, encode_image
from .net import Reply

REST_POSE = (0.0, -100.0, 95.0, 70.0, 0.0, 2.0)


class FakeRobot:
    state_names = STATE_NAMES

    def __init__(self, port: str = "fake", robot_id: str = "fake", use_degrees: bool = True, *,
                 start_pose=REST_POSE, tau_s: float = 0.1, noise: float = 0.05, seed: int = 0):
        self.port, self.robot_id, self.use_degrees = port, robot_id, use_degrees
        self.tau_s = tau_s
        self.noise = noise
        self.connected = False
        self.torque = True
        self.writes = 0
        self.last_write: np.ndarray | None = None
        self._x = np.array(start_pose, dtype=np.float64).reshape(len(STATE_NAMES))
        self._goal = self._x.copy()
        self._t = time.perf_counter()
        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()

    def connect(self) -> None:
        with self._lock:
            self._t = time.perf_counter()
            self.connected = True

    def _advance(self) -> None:
        now = time.perf_counter()
        dt, self._t = now - self._t, now
        if self.torque and self.tau_s > 0:
            self._x += (self._goal - self._x) * (1.0 - math.exp(-dt / self.tau_s))

    def read_state(self) -> np.ndarray:
        if not self.connected:
            raise RuntimeError("FakeRobot is not connected")
        with self._lock:
            self._advance()
            state = self._x + self._rng.normal(0.0, self.noise, self._x.shape)
        state[5] = min(100.0, max(0.0, state[5]))
        return state.astype(np.float32)

    def write(self, cmd: np.ndarray) -> None:
        if not self.connected:
            raise RuntimeError("FakeRobot is not connected")
        cmd = np.asarray(cmd, dtype=np.float64).reshape(len(STATE_NAMES))
        if not np.isfinite(cmd).all():
            raise ValueError(f"refusing a non-finite command {cmd}")
        with self._lock:
            self._advance()
            self._goal = cmd.copy()
            self.writes += 1
            self.last_write = cmd.astype(np.float32)

    def set_torque(self, enabled: bool) -> None:
        with self._lock:
            self._advance()
            self.torque = bool(enabled)
            if enabled:
                self._goal = self._x.copy()

    def disconnect(self, torque_off: bool = False) -> None:
        self.connected = False
        self.torque = not torque_off


class FakeCameras:
    """Synthetic frames: a per-camera gradient with a moving disc, at fps, encoded like Cameras."""

    def __init__(self, sources: dict | None = None, width: int = 640, height: int = 480, fps: int = FPS,
                 image_size: int = 256, image_format: str = "jpeg", jpeg_quality: int = 95):
        self.sources = dict(sources) if sources else {name: i for i, name in enumerate(CAMERAS)}
        self.names = tuple(self.sources)
        self.width, self.height, self.fps = width, height, fps
        self.image_size = image_size
        self.image_format = image_format
        self.jpeg_quality = jpeg_quality
        self.modes = {name: (width, height, float(fps)) for name in self.names}
        self.running = False
        self._latest: dict[str, Frame] = {}
        self._lock = threading.Lock()
        self._taps: list = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        x = np.linspace(0, 255, width, dtype=np.float32)
        y = np.linspace(0, 255, height, dtype=np.float32)
        self._bases = []
        for i in range(len(self.names)):
            base = np.empty((height, width, 3), dtype=np.uint8)
            base[..., 0] = x[None, :]
            base[..., 1] = y[:, None]
            base[..., 2] = (60 + 120 * i) % 256
            self._bases.append(base)

    def start(self, timeout_s: float = 5.0) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="fake-cameras", daemon=True)
        self._thread.start()
        self.running = True
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            with self._lock:
                if all(name in self._latest for name in self.names):
                    return
            time.sleep(0.005)
        raise RuntimeError("FakeCameras produced no frames")

    def latest(self) -> dict[str, Frame]:
        with self._lock:
            return dict(self._latest)

    def add_tap(self, fn) -> None:
        self._taps = [*self._taps, fn]

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None
        self.running = False

    def pattern(self, i: int, s: float) -> np.ndarray:
        import cv2

        img = self._bases[i].copy()
        cx = int((0.5 + 0.35 * math.sin(2 * math.pi * 0.25 * s + i)) * self.width)
        cy = int((0.5 + 0.35 * math.cos(2 * math.pi * 0.2 * s + 2 * i)) * self.height)
        cv2.circle(img, (cx, cy), max(4, self.height // 8), (255, 255, 255), -1)
        return img

    def _run(self) -> None:
        period = 1.0 / self.fps
        t0 = time.perf_counter()
        n = 0
        while not self._stop.is_set():
            n += 1
            delay = t0 + n * period - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            t = time.perf_counter()
            for i, name in enumerate(self.names):
                bgr = self.pattern(i, t - t0)
                frame = Frame(bgr=bgr, t=t, seq=n,
                              encoded=encode_image(bgr, self.image_size, self.image_format, self.jpeg_quality))
                with self._lock:
                    self._latest[name] = frame
                for fn in self._taps:
                    try:
                        fn(name, bgr, t)
                    except Exception:
                        pass


class FakeNet:
    """In-process so101.v1 peer. mode: hold (zero deltas), sine (every joint swings), pickplace (a gripper
    close-hold-open cycle every cycle_s with a small arm motion). latency_ms is the whole turnaround."""

    MODES = ("hold", "sine", "pickplace")

    def __init__(self, mode: str = "pickplace", latency_ms: float = 400.0, jitter_ms: float = 50.0, *,
                 fps: float = FPS, chunk: int = CHUNK, rtt_ms: float = 40.0, cycle_s: float = 4.0, seed: int = 0,
                 check_wire: bool = True):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.mode = mode
        self.latency_ms = latency_ms
        self.jitter_ms = jitter_ms
        self.fps = fps
        self.chunk = chunk
        self.rtt_ms = rtt_ms
        self.cycle_s = cycle_s
        self.check_wire = check_wire
        self.hello: dict | None = None
        self.requests: list[dict] = []
        self._connected = False
        self._ids = itertools.count(1)
        self._heap: list = []
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(seed)

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> dict:
        self._connected = True
        self.hello = {
            "type": "hello", "protocol": PROTOCOL, "engine": f"fake-net:{self.mode}", "checkpoint": None,
            "revision": None, "chunk": self.chunk, "exec": EXEC, "fps": int(self.fps), "n_obs": N_OBS,
            "snapshots": [0, N_OBS - 1], "cameras": list(CAMERAS), "state_names": list(STATE_NAMES),
            "guidance": None, "steps": None, "warm": True,
        }
        return dict(self.hello)

    def submit(self, window, task: str, tick: int) -> int:
        rid = next(self._ids)
        now = time.perf_counter()
        if not self._connected:
            with self._lock:
                heapq.heappush(self._heap, (now, rid, Reply(rid, int(tick), None, {}, "not connected", now, now)))
            return rid
        state, cmd = np.asarray(window.state, np.float32), np.asarray(window.cmd_hist, np.float32)
        if self.check_wire:
            msg = protocol.unpack(protocol.pack(protocol.infer_message(rid, tick, task, window)))
            state, cmd = msg["state"], msg["cmd_hist"]
            if state.shape != (N_OBS, 6) or cmd.shape != (N_OBS, 6) or state.dtype != np.float32:
                raise ValueError(f"bad window: state {state.shape} {state.dtype}, cmd_hist {cmd.shape}")
            if sorted(msg["images"]) != sorted(window.images) or any(len(v) != 2 for v in msg["images"].values()):
                raise ValueError("bad window: images must hold [i0, i7] per camera")
        actions = self.plan(cmd[-1], int(tick))
        latency = max(0.0, self.latency_ms + self._rng.normal(0.0, self.jitter_ms)) / 1000.0
        infer_ms = 0.8 * latency * 1000.0
        timing = {"queue_ms": 0.0, "decode_ms": 0.5, "infer_ms": infer_ms, "total_ms": infer_ms + 0.5}
        reply = Reply(rid, int(tick), actions, timing, None, now, now + latency)
        with self._lock:
            heapq.heappush(self._heap, (now + latency, rid, reply))
        self.requests.append({"id": rid, "tick": int(tick), "task": task, "t": now})
        return rid

    def poll(self) -> list[Reply]:
        now = time.perf_counter()
        out = []
        with self._lock:
            while self._heap and self._heap[0][0] <= now:
                out.append(heapq.heappop(self._heap)[2])
        return out

    def ping(self, n: int = 10) -> list[float]:
        return [max(0.1, self.rtt_ms + self._rng.normal(0.0, self.jitter_ms / 10.0)) for _ in range(n)]

    def close(self) -> None:
        """Requests still in flight fail with an error, as with the real client."""
        self._connected = False
        now = time.perf_counter()
        with self._lock:
            pending = [r for _, _, r in self._heap]
            self._heap = [(now, r.id, Reply(r.id, r.tick, None, {}, "disconnected", r.sent_t, now)) for r in pending]
            heapq.heapify(self._heap)

    def plan(self, anchor, tick: int) -> np.ndarray:
        """(chunk, 6) absolute commands for ticks tick..tick+chunk-1, joints integrated from the anchor."""
        anchor = np.asarray(anchor, dtype=np.float64).reshape(6)
        n = self.chunk
        tt = (tick + np.arange(n)) / self.fps
        out = np.empty((n, 6))
        if self.mode == "hold":
            out[:] = anchor
            return out.astype(np.float32)
        w = 2 * math.pi / self.cycle_s
        amp = np.array([8.0, 6.0, 6.0, 8.0, 10.0]) if self.mode == "sine" else np.array([4.0, 3.0, 3.0, 4.0, 0.0])
        velocity = amp[None, :] * w * np.cos(w * tt)[:, None] / self.fps
        out[:, :5] = anchor[:5] + np.cumsum(velocity, axis=0)
        if self.mode == "sine":
            out[:, 5] = 50.0 + 30.0 * np.sin(w * tt)
        else:
            phase = (tt % self.cycle_s) / self.cycle_s
            out[:, 5] = np.interp(phase, [0.0, 0.25, 0.35, 0.75, 0.85, 1.0], [60.0, 60.0, 5.0, 5.0, 60.0, 60.0])
        return out.astype(np.float32)
