"""The 30 Hz control loop (docs/architecture.md, "Client loop").

Each tick k, at an absolute deadline (sleep until spin_margin before it, then spin):

1. read the state, take the newest frame of each camera, and append (state_k, cmd_{k-1}, frames_k) to the
   ring, except while sync mode waits for its reply;
2. handle replies: d = k - r; a current reply is joined by the Planner, a reply to a request sent before
   the last set_task or pause is ignored;
3. pick the command (the next delta added to the last sent command, or a hold that resends it), clamp it,
   and write it unless this is a dry run;
4. send the next request when the scheduler says so;
5. hand a TickInfo to the observers.

Camera capture, encoding and socket I/O run on their own threads; this thread never waits for them.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from . import CAMERAS, CHUNK, EXEC, N_OBS
from .chunks import MODES, Planner, clamp
from .history import Ring

log = logging.getLogger(__name__)

TICK_HISTORY = 216_000  # lateness samples kept for stats(): 2 h at 30 Hz
PLAN_HISTORY = 20_000
COUNTERS = ("ticks", "overruns", "holds", "starved", "requests", "plans", "dropped", "stale", "errors", "timeouts",
            "unmatched", "clamped", "bus_errors", "write_errors", "observer_errors", "stale_frames")


@dataclass
class LoopConfig:
    fps: float = 30.0
    mode: str = "prefetch"  # "prefetch" | "sync"
    period: int = 16  # P
    crossfade: int = 6  # K
    exec_horizon: int = EXEC
    chunk: int = CHUNK
    step_max_deg: float = 6.0  # per-tick joint clamp
    joint_limits: tuple | None = None  # 5 x (lo, hi) degrees; gripper always 0..100
    dry_run: bool = True
    spin_margin_ms: float = 6.0
    # Additions to the shared signature, all optional:
    request_timeout_s: float = 10.0  # a request with no reply after this long is given up (a new caption is slow)
    max_frame_age_ms: float = 1000.0  # no new request while a camera's newest frame is older than this
    error_backoff_ticks: int = 15  # after an error reply, wait this long before the next request
    caffeinate: bool = True  # on macOS, run `caffeinate -dimsu -w <pid>` while the loop runs


@dataclass
class TickInfo:
    k: int
    t: float
    state: np.ndarray  # (6,) measured
    cmd: np.ndarray  # (6,) sent, or the one a dry run would have sent
    held: bool
    lateness_ms: float
    frame_age_ms: dict
    remaining: int
    task: str | None
    chunk: dict | None = None  # on the tick a reply is handled: id, tick, d, accepted, turnaround_ms, server_ms, actions


class ControlLoop:
    """Runs the control thread. start() brings up whatever robot, cameras and net are not up yet, and
    stop() releases those again (torque stays on, so the arm holds its last command)."""

    def __init__(self, robot, cameras, net, cfg: LoopConfig | None = None):
        cfg = cfg or LoopConfig()
        if cfg.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if cfg.joint_limits is not None and np.asarray(cfg.joint_limits, dtype=float).shape != (5, 2):
            raise ValueError("joint_limits must be 5 x (lo, hi) degrees")
        self.cfg = cfg
        self.robot, self.cameras, self.net = robot, cameras, net
        self.cams = tuple(getattr(cameras, "names", CAMERAS))
        self.hello: dict | None = getattr(net, "hello", None)
        self.error: str | None = None
        self.k = -1
        self._period = 1.0 / cfg.fps
        self._ring = Ring(N_OBS, image_format=getattr(cameras, "image_format", "jpeg"))
        self._planner = Planner(cfg.mode, cfg.period, cfg.crossfade, cfg.exec_horizon, cfg.chunk)
        self._lock = threading.Lock()
        self._pending: list[tuple] = []
        self._observers: list = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._owned: set[str] = set()
        self._caffeinate: subprocess.Popen | None = None
        self._task: str | None = None
        self._epoch = 0
        self._requests: dict[int, dict] = {}
        self._last_cmd: np.ndarray | None = None
        self._last_state: np.ndarray | None = None
        self._first_chunk = False
        self._backoff_until = 0
        self._bus_failures = 0
        self._stats_lock = threading.Lock()
        self._lateness: deque[float] = deque(maxlen=TICK_HISTORY)
        self._turnaround: deque[float] = deque(maxlen=PLAN_HISTORY)
        self._server: deque[float] = deque(maxlen=PLAN_HISTORY)
        self._rtt: deque[float] = deque(maxlen=PLAN_HISTORY)
        self._network: deque[float] = deque(maxlen=PLAN_HISTORY)
        self._counts = dict.fromkeys(COUNTERS, 0)

    # ---------------------------------------------------------------- public API

    @property
    def task(self) -> str | None:
        return self._task

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Threads up. The loop holds (resends the last command, no requests) until set_task."""
        if self._thread is not None:
            raise RuntimeError("ControlLoop.start() was already called")
        try:
            if not getattr(self.robot, "connected", False):
                self.robot.connect()
                self._owned.add("robot")
            if not getattr(self.cameras, "running", False):
                self.cameras.start()
                self._owned.add("cameras")
            if not getattr(self.net, "connected", False):
                self.hello = self.net.connect()
                self._owned.add("net")
            else:
                self.hello = getattr(self.net, "hello", None)
        except Exception:
            self._release()
            raise
        if sys.getswitchinterval() > 0.001:
            sys.setswitchinterval(0.001)  # bounds how long another thread can keep the control thread waiting
        if self.cfg.caffeinate and sys.platform == "darwin":
            try:
                self._caffeinate = subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())],
                                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError as e:
                log.warning("caffeinate did not start: %s", e)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="so101-control", daemon=True)
        self._thread.start()

    def set_task(self, task: str, reset_history: bool) -> None:
        """Applied at the next tick: drops the queued actions and ignores replies in flight. With
        reset_history the ring is refilled with the current observation and the last command becomes the
        measured state (episode start, after an intervention)."""
        with self._lock:
            self._pending.append(("task", str(task), bool(reset_history)))

    def pause(self) -> None:
        """Hold position, send no requests, drop the active chunk."""
        with self._lock:
            self._pending.append(("pause",))

    def add_observer(self, fn) -> None:
        """fn(TickInfo) on the control thread every tick. Must return fast; exceptions are caught and logged."""
        with self._lock:
            self._observers = [*self._observers, fn]

    def add_frame_tap(self, fn) -> None:
        """fn(camera, frame_bgr, t) with full-resolution frames, called on the camera threads."""
        self.cameras.add_tap(fn)

    def stop(self) -> None:
        """End the control thread; the last command stays the motors' goal, so the arm holds."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._release()

    def stats(self) -> dict:
        with self._stats_lock:
            lateness = np.array(self._lateness)
            turnaround = np.array(self._turnaround)
            server = np.array(self._server)
            rtt = np.array(self._rtt)
            network = np.array(self._network)
            counts = dict(self._counts)

        def pct(a: np.ndarray, q: float) -> float | None:
            return round(float(np.percentile(a, q)), 3) if len(a) else None

        def top(a: np.ndarray) -> float | None:
            return round(float(a.max()), 3) if len(a) else None

        return {
            "mode": self.cfg.mode,
            "dry_run": self.cfg.dry_run,
            **counts,
            "lateness_p50_ms": pct(lateness, 50),
            "lateness_p99_ms": pct(lateness, 99),
            "lateness_max_ms": top(lateness),
            "turnaround_p50_ms": pct(turnaround, 50),
            "turnaround_p95_ms": pct(turnaround, 95),
            "turnaround_max_ms": top(turnaround),
            "server_p50_ms": pct(server, 50),
            "server_p95_ms": pct(server, 95),
            "rtt_p50_ms": pct(rtt, 50),
            "rtt_p95_ms": pct(rtt, 95),
            "network_p50_ms": pct(network, 50),
            "network_p95_ms": pct(network, 95),
            "d_max": self._planner.scheduler.d_max,
            "error": self.error,
        }

    # ---------------------------------------------------------------- control thread

    def _release(self) -> None:
        steps = (("net", lambda: self.net.close()), ("cameras", lambda: self.cameras.stop()),
                 ("robot", lambda: self.robot.disconnect(torque_off=False)))
        for name, fn in steps:
            if name in self._owned:
                try:
                    fn()
                except Exception:
                    log.exception("releasing %s failed", name)
                self._owned.discard(name)
        if self._caffeinate is not None:
            self._caffeinate.terminate()
            self._caffeinate = None

    def _run(self) -> None:
        try:
            self._loop()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            log.exception("control loop stopped")

    def _loop(self) -> None:
        period = self._period
        margin = self.cfg.spin_margin_ms / 1000.0
        t0 = time.perf_counter() + period
        k = 0
        while not self._stop.is_set():
            deadline = t0 + k * period
            delay = deadline - time.perf_counter() - margin
            if delay > 0:
                time.sleep(delay)
            while time.perf_counter() < deadline:
                pass
            t = time.perf_counter()
            late = t - deadline
            if late > period:
                self._counts["overruns"] += 1
                t0 = t - k * period  # start counting deadlines from now instead of bursting to catch up
            self._tick(k, t, late * 1000.0)
            k += 1

    def _tick(self, k: int, t: float, lateness_ms: float) -> None:
        cfg = self.cfg
        self.k = k
        bus_ok = True
        try:
            state = np.asarray(self.robot.read_state(), dtype=np.float32).reshape(6)
            self._bus_failures = 0
        except Exception as e:
            self._bus_failures += 1
            self._counts["bus_errors"] += 1
            if self._bus_failures >= 30:
                raise RuntimeError(f"state read failed {self._bus_failures} ticks in a row: {e}") from e
            if self._bus_failures == 1:
                log.warning("state read failed: %s", e)
            if self._last_state is None:
                return
            state, bus_ok = self._last_state, False
        self._last_state = state
        if self._last_cmd is None:
            self._last_cmd = state.copy()
        frames = self.cameras.latest()
        frames_ok = all(cam in frames for cam in self.cams)
        frame_age = {cam: (t - frame.t) * 1000.0 for cam, frame in frames.items()}

        reset_now = self._apply_pending(state, frames, frames_ok)
        last = self._last_cmd

        # 1. history
        if self._ring.ready and frames_ok and not reset_now and self._planner.appends():
            self._ring.append(state, last, frames)

        # 2. replies
        chunk_info = None
        for reply in self.net.poll():
            info = self._handle_reply(reply, k)
            if info is not None and (chunk_info is None or info["accepted"] or not chunk_info["accepted"]):
                chunk_info = info
        self._expire_requests(k, t)

        # 3. command
        nxt = self._planner.step(last) if self._task is not None and bus_ok else None
        held = nxt is None
        if held:
            cmd = last
            if self._task is not None:
                self._counts["holds"] += 1
                if self._first_chunk and cfg.mode == "prefetch":
                    self._counts["starved"] += 1
        else:
            cmd = clamp(nxt, last, cfg.joint_limits, cfg.step_max_deg).astype(np.float32)
            if np.abs(cmd - nxt).max() > 1e-4:
                self._counts["clamped"] += 1
        if not cfg.dry_run:
            try:
                self.robot.write(cmd)
            except Exception as e:
                self._counts["write_errors"] += 1
                if self._counts["write_errors"] in (1, 30):
                    log.warning("command write failed: %s", e)
        self._last_cmd = cmd

        # 4. request
        if (self._task is not None and bus_ok and frames_ok and self._ring.ready and k >= self._backoff_until
                and self.net.connected and self._planner.due(k)):
            if max(frame_age.values()) <= cfg.max_frame_age_ms:
                self._submit(k, t)
            else:
                self._counts["stale_frames"] += 1

        # 5. observers
        info = TickInfo(k=k, t=t, state=state, cmd=cmd, held=held, lateness_ms=lateness_ms, frame_age_ms=frame_age,
                        remaining=self._planner.remaining, task=self._task, chunk=chunk_info)
        with self._stats_lock:
            self._lateness.append(lateness_ms)
            self._counts["ticks"] += 1
        for fn in self._observers:
            try:
                fn(info)
            except Exception:
                self._counts["observer_errors"] += 1
                if self._counts["observer_errors"] in (1, 100):
                    log.exception("observer failed (%d so far)", self._counts["observer_errors"])

    def _apply_pending(self, state: np.ndarray, frames: dict, frames_ok: bool) -> bool:
        """Apply set_task/pause calls in order. Returns True when the ring was reset at this tick."""
        with self._lock:
            if not self._pending:
                return False
            pending, self._pending = self._pending, []
        reset_now = False
        for i, item in enumerate(pending):
            needs_reset = item[0] == "task" and (item[2] or not self._ring.ready)
            if needs_reset and not frames_ok:
                with self._lock:  # no frame from every camera yet: retry at the next tick
                    self._pending[:0] = pending[i:]
                break
            self._epoch += 1
            self._planner.reset()
            self._first_chunk = False
            if item[0] == "pause":
                self._task = None
                continue
            self._task = item[1]
            if needs_reset:
                self._ring.reset(state, frames)
                self._last_cmd = state.copy()
                reset_now = True
        return reset_now

    def _submit(self, k: int, t: float) -> None:
        window = self._ring.window()
        rid = self.net.submit(window, self._task, k)
        self._requests[rid] = {"epoch": self._epoch, "tick": k, "t": t, "frame_t": window.frame_t,
                               "anchor": window.cmd_hist[-1].copy()}
        self._planner.sent(k, rid)
        self._counts["requests"] += 1

    def _handle_reply(self, reply, k: int) -> dict | None:
        req = self._requests.pop(reply.id, None)
        if req is None:  # a request already given up on (timeout), or an id the server made up
            self._counts["unmatched"] += 1
            return None
        if reply.error is not None or reply.actions is None:
            self._planner.failed(reply.id)
            self._counts["errors"] += 1
            if req["epoch"] == self._epoch:
                self._backoff_until = k + self.cfg.error_backoff_ticks
            if self._counts["errors"] <= 5 or self._counts["errors"] % 100 == 0:
                log.warning("request %s failed: %s", reply.id, reply.error)
            return None
        d = k - req["tick"]
        self._planner.received(reply.id, d)
        frame_t = min(req["frame_t"].values()) if req["frame_t"] else req["t"]
        turnaround_ms = (reply.recv_t - frame_t) * 1000.0
        server_ms = reply.timing.get("total_ms")
        rtt_ms = (reply.recv_t - reply.sent_t) * 1000.0
        with self._stats_lock:
            self._turnaround.append(turnaround_ms)
            if math.isfinite(rtt_ms):
                self._rtt.append(rtt_ms)
            if server_ms is not None:
                self._server.append(float(server_ms))
                if math.isfinite(rtt_ms):
                    self._network.append(rtt_ms - float(server_ms))
        start = None
        if req["epoch"] != self._epoch or self._task is None:
            reason = "stale"
            self._counts["stale"] += 1
        else:
            start = self._planner.offer(reply.actions, req["anchor"], d)
            if start is None:
                reason = "late"
                self._counts["dropped"] += 1
            else:
                reason = "ok"
                self._counts["plans"] += 1
                self._first_chunk = True
        return {"id": reply.id, "tick": req["tick"], "d": d, "accepted": start is not None, "start": start,
                "reason": reason, "turnaround_ms": turnaround_ms, "server_ms": server_ms, "rtt_ms": rtt_ms,
                "actions": reply.actions}

    def _expire_requests(self, k: int, t: float) -> None:
        rid = self._planner.scheduler.inflight
        req = self._requests.get(rid) if rid is not None else None
        if req is not None and t - req["t"] > self.cfg.request_timeout_s:
            self._requests.pop(rid, None)
            self._planner.failed(rid)
            self._counts["timeouts"] += 1
            log.warning("request %s got no reply within %.1f s; giving up on it", rid, self.cfg.request_timeout_s)
        if k % 300 == 0 and len(self._requests) > 1:
            horizon = t - 4 * self.cfg.request_timeout_s
            for old in [r for r, v in self._requests.items() if v["t"] < horizon and r != rid]:
                del self._requests[old]
