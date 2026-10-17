"""Episode files (ticks.jsonl, chunks.npz, attempts.jsonl, episode.json) and run-level helpers."""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import queue
import subprocess
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

CHUNK_META =("id", "tick", "d", "accepted", "start", "reason", "turnaround_ms", "server_ms", "rtt_ms")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def to_jsonable(obj):
    """json.dumps default: numpy scalars and arrays, paths, sets."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (Path, set, frozenset, tuple)):
        return str(obj) if isinstance(obj, Path) else list(obj)
    if isinstance(obj, (bytes, bytearray)):
        return f"<{len(obj)} bytes>"
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=to_jsonable) + "\n")
    tmp.replace(path)


def make_run_dir(base: str | Path, label: str) -> Path:
    """eval/runs/<UTC stamp>-<label>/, with a numeric suffix if that second is taken."""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    stem = f"{utc_stamp()}-{label}"
    path = base / stem
    i = 1
    while path.exists():
        i += 1
        path = base / f"{stem}-{i}"
    path.mkdir()
    return path


def git_info(repo: str | Path) -> dict:
    """Commit, branch and whether the work tree has uncommitted changes (read-only git calls)."""

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def _percentile(values: list[float], q: float) -> float | None:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    return round(float(np.percentile(clean, q)), 2) if clean else None


def _floats(values, digits: int) -> list[float]:
    return [round(float(v), digits) for v in np.asarray(values, dtype=np.float64).reshape(-1)]


class EpisodeRecorder:
    """Writes one episode directory. on_tick() runs on the control thread and only copies and queues;
    a background thread turns ticks into ticks.jsonl. Chunks are kept in memory and saved on close()."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.attempt: int | None = None
        self.tick_count = 0
        self.tick_errors = 0
        self.ticks_per_attempt: dict[int, int] = {}
        self.chunks: list[dict] = []
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._lateness: list[float] = []
        self._frame_age: dict[str, list[float]] = {}
        self._held = 0
        self._idle = 0
        self._closed = False
        self._stats: dict = {}
        self._attempts_file = open(self.dir / "attempts.jsonl", "a")
        self._thread = threading.Thread(target=self._write_ticks, name=f"ticks-{self.dir.name}", daemon=True)
        self._thread.start()

    def on_tick(self, info) -> None:
        if self._closed:
            return
        attempt = self.attempt
        self.tick_count += 1
        if attempt is not None:
            self.ticks_per_attempt[attempt] = self.ticks_per_attempt.get(attempt, 0) + 1
        chunk = getattr(info, "chunk", None)
        meta = None
        if chunk:
            meta = {key: chunk.get(key) for key in CHUNK_META}
            actions = chunk.get("actions")
            self.chunks.append(
                {
                    **meta,
                    "k": info.k,
                    "attempt": attempt,
                    "actions": None if actions is None else np.array(actions, dtype=np.float32, copy=True),
                }
            )
        self._q.put(
            (
                attempt,
                info.k,
                info.t,
                np.array(info.state, dtype=np.float32, copy=True),
                np.array(info.cmd, dtype=np.float32, copy=True),
                bool(info.held),
                info.lateness_ms,
                dict(info.frame_age_ms or {}),
                info.remaining,
                info.task,
                meta,
            )
        )

    def _write_ticks(self) -> None:
        with open(self.dir / "ticks.jsonl", "w") as f:
            while True:
                item = self._q.get()
                if item is None:
                    break
                attempt, k, t, state, cmd, held, lateness, frame_age, remaining, task, meta = item
                try:
                    record = {
                        "k": k,
                        "t": round(float(t), 6),
                        "attempt": attempt,
                        "state": _floats(state, 4),
                        "cmd": _floats(cmd, 4),
                        "held": held,
                        "lateness_ms": None if lateness is None else round(float(lateness), 3),
                        "frame_age_ms": {cam: round(float(v), 2) for cam, v in frame_age.items() if v is not None},
                        "remaining": remaining,
                        "task": task,
                    }
                    if meta is not None:
                        record["chunk"] = meta
                    f.write(json.dumps(record, default=to_jsonable) + "\n")
                except (TypeError, ValueError) as e:
                    self.tick_errors += 1
                    if self.tick_errors == 1:
                        log.error("ticks.jsonl: tick %s not written: %r", k, e)
                    continue
                if task is None:
                    self._idle += 1
                elif held:
                    self._held += 1
                if lateness is not None:
                    self._lateness.append(float(lateness))
                for cam, v in frame_age.items():
                    if v is not None:
                        self._frame_age.setdefault(cam, []).append(float(v))
                if self._q.empty():
                    f.flush()

    def chunks_in(self, attempt: int) -> int:
        return sum(1 for c in self.chunks if c["attempt"] == attempt)

    def write_attempt(self, record: dict) -> None:
        self._attempts_file.write(json.dumps(record, default=to_jsonable) + "\n")
        self._attempts_file.flush()

    def write_episode(self, record: dict) -> None:
        write_json(self.dir / "episode.json", record)

    def save_chunks(self, chunk_len: int = 42, dims: int = 6) -> None:
        chunks = self.chunks
        actions = np.zeros((len(chunks), chunk_len, dims), dtype=np.float32)
        has_actions = np.zeros(len(chunks), dtype=bool)
        for i, c in enumerate(chunks):
            a = c["actions"]
            if a is not None and a.shape == (chunk_len, dims):
                actions[i] = a
                has_actions[i] = True

        def column(key, dtype, missing):
            return np.array([missing if c.get(key) is None else c[key] for c in chunks], dtype=dtype)

        np.savez_compressed(
            self.dir / "chunks.npz",
            id=column("id", np.int64, -1),
            tick=column("tick", np.int64, -1),
            d=column("d", np.int64, -1),
            k=column("k", np.int64, -1),
            accepted=column("accepted", bool, False),
            start=column("start", np.int64, -1),
            reason=np.array([str(c.get("reason") or "") for c in chunks], dtype=str),
            turnaround_ms=column("turnaround_ms", np.float32, np.nan),
            server_ms=column("server_ms", np.float32, np.nan),
            rtt_ms=column("rtt_ms", np.float32, np.nan),
            attempt=column("attempt", np.int64, -1),
            has_actions=has_actions,
            actions=actions,
        )

    def close(self, chunk_len: int = 42) -> dict:
        """Finish ticks.jsonl, save chunks.npz and return the episode's tick and chunk stats."""
        if self._closed:
            return self._stats
        self._closed = True
        self._q.put(None)
        self._thread.join(timeout=30)
        self._attempts_file.close()
        self.save_chunks(chunk_len)
        accepted = sum(1 for c in self.chunks if c.get("accepted"))
        reasons: dict[str, int] = {}
        for c in self.chunks:
            key = str(c.get("reason") or ("ok" if c.get("accepted") else "not accepted"))
            reasons[key] = reasons.get(key, 0) + 1
        self._stats = {
            "ticks": {
                "n": self.tick_count,
                "not_written": self.tick_errors,
                "held": self._held,
                "idle": self._idle,
                "lateness_ms_p50": _percentile(self._lateness, 50),
                "lateness_ms_p99": _percentile(self._lateness, 99),
                "lateness_ms_max": round(max(self._lateness), 2) if self._lateness else None,
                "frame_age_ms_p50": {cam: _percentile(v, 50) for cam, v in self._frame_age.items()},
                "frame_age_ms_p95": {cam: _percentile(v, 95) for cam, v in self._frame_age.items()},
            },
            "chunks": {
                "received": len(self.chunks),
                "accepted": accepted,
                "by_reason": reasons,
                "turnaround_ms_p50": _percentile([c.get("turnaround_ms") for c in self.chunks], 50),
                "turnaround_ms_p95": _percentile([c.get("turnaround_ms") for c in self.chunks], 95),
            },
        }
        return self._stats
