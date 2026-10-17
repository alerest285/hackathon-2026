"""WebSocket client for so101.v1. All network I/O runs on one thread with its own asyncio loop, so
submit() and poll() never block the control loop.

The client reconnects with backoff (0.5 s doubling to 5 s). When the connection drops, every request in
flight comes back as a Reply with an error, so the caller's one-in-flight slot is freed.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import itertools
import logging
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from . import protocol

log = logging.getLogger(__name__)


@dataclass
class Reply:
    id: int
    tick: int
    actions: np.ndarray | None  # (42, 6) float32 absolute, None on error
    timing: dict  # server timing: queue_ms, decode_ms, infer_ms, total_ms
    error: str | None
    sent_t: float  # client perf_counter when the request left
    recv_t: float  # client perf_counter when the reply arrived


def _set_nodelay(ws) -> None:
    transport = getattr(ws, "transport", None)
    sock = transport.get_extra_info("socket") if transport is not None else None
    if sock is not None:
        with contextlib.suppress(OSError):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


class NetClient:
    def __init__(self, url: str = "ws://localhost:8000", connect_timeout_s: float = 10.0, *,
                 max_queue_age_s: float = 1.0):
        self.url = url
        self.connect_timeout_s = connect_timeout_s
        self.max_queue_age_s = max_queue_age_s  # a request still unsent after this long is failed, not sent
        self.hello: dict | None = None
        self.reconnects = 0
        self.last_error: str | None = None
        self.server_t: float | None = None
        self._replies: deque[Reply] = deque()
        self._ids = itertools.count(1)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._main_task: asyncio.Task | None = None
        self._ws = None
        self._send_q: asyncio.Queue | None = None
        self._pending: dict[int, tuple[int, float]] = {}
        self._pongs: dict[float, asyncio.Future] = {}
        self._connected = threading.Event()
        self._fatal = threading.Event()
        self._closing = False

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def connect(self) -> dict:
        """Start the network thread and wait for the server's hello."""
        if self._thread is not None:
            raise RuntimeError("NetClient.connect() was already called")
        self._closing = False
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="so101-net", daemon=True)
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._main(), self._loop)
        deadline = time.perf_counter() + self.connect_timeout_s
        while not self._connected.is_set() and not self._fatal.is_set() and time.perf_counter() < deadline:
            time.sleep(0.01)
        if not self._connected.is_set():
            reason = self.last_error or "timeout"
            self.close()
            raise ConnectionError(
                f"no so101.v1 server at {self.url} ({reason}). Is the ssh tunnel up? infra/aws/ssh.sh forwards 8000."
            )
        return dict(self.hello)

    def submit(self, window, task: str, tick: int) -> int:
        """Queue one infer request; returns its id at once. Failures come back through poll()."""
        rid = next(self._ids)
        now = time.perf_counter()
        if not self.connected or self._loop is None or self._send_q is None:
            self._replies.append(Reply(rid, int(tick), None, {}, "not connected", now, now))
            return rid
        try:
            self._loop.call_soon_threadsafe(self._send_q.put_nowait, ("infer", (rid, window, str(task), int(tick)), now))
        except RuntimeError:
            self._replies.append(Reply(rid, int(tick), None, {}, "not connected", now, now))
        return rid

    def poll(self) -> list[Reply]:
        out = []
        while True:
            try:
                out.append(self._replies.popleft())
            except IndexError:
                return out

    def ping(self, n: int = 10, timeout_s: float = 5.0) -> list[float]:
        """Round trips in ms, one ping at a time. Blocking; not for the control loop."""
        if not self.connected or self._loop is None:
            raise ConnectionError("not connected")
        return [asyncio.run_coroutine_threadsafe(self._ping_once(timeout_s), self._loop).result(timeout_s + 1.0)
                for _ in range(n)]

    def close(self) -> None:
        loop = self._loop
        if loop is None:
            return
        self._closing = True

        async def stop():
            ws = self._ws
            if ws is not None:
                with contextlib.suppress(Exception):
                    await ws.close()
            task = self._main_task
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task

        if loop.is_running():
            with contextlib.suppress(Exception):
                asyncio.run_coroutine_threadsafe(stop(), loop).result(3.0)
            loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if not loop.is_running():
            loop.close()
        self._connected.clear()
        self._loop = None
        self._thread = None

    # ---------------------------------------------------------------- network thread

    async def _main(self) -> None:
        from websockets.asyncio.client import connect as ws_connect

        self._main_task = asyncio.current_task()
        self._send_q = asyncio.Queue()
        kwargs = dict(compression=None, max_size=None, open_timeout=self.connect_timeout_s, ping_interval=20,
                      ping_timeout=20, close_timeout=1)
        if "proxy" in inspect.signature(ws_connect).parameters:
            kwargs["proxy"] = None  # the tunnel is local; a system proxy must not intercept it
        backoff = 0.5
        while not self._closing:
            was_connected = False
            try:
                async with ws_connect(self.url, **kwargs) as ws:
                    _set_nodelay(ws)
                    raw = await asyncio.wait_for(ws.recv(), timeout=self.connect_timeout_s)
                    hello = protocol.unpack(raw) if isinstance(raw, (bytes, bytearray)) else {}
                    if protocol.message_type(hello) != "hello":
                        raise ConnectionError(f"expected hello, got {protocol.message_type(hello)!r}")
                    notes = protocol.check_hello(hello)
                    if hello.get("protocol") != protocol.PROTOCOL:
                        self.last_error = notes[0]
                        self._fatal.set()
                        return
                    for note in notes:
                        log.warning("server hello: %s", note)
                    self.hello = hello
                    self._ws = ws
                    self.last_error = None
                    backoff = 0.5
                    was_connected = True
                    self._connected.set()
                    log.info("connected to %s: engine %s, warm %s", self.url, hello.get("engine"), hello.get("warm"))
                    sender = asyncio.create_task(self._sender(ws))
                    try:
                        await self._receiver(ws)
                    finally:
                        sender.cancel()
                        with contextlib.suppress(BaseException):
                            await sender
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
            finally:
                self._connected.clear()
                self._ws = None
                self._fail_pending("disconnected" if was_connected else "not connected")
                if was_connected and not self._closing:
                    log.warning("connection to %s lost (%s); reconnecting", self.url,
                                self.last_error or "closed by the server")
            if self._closing:
                break
            await asyncio.sleep(backoff)
            backoff = min(2 * backoff, 5.0)
            self.reconnects += 1

    async def _sender(self, ws) -> None:
        while True:
            kind, payload, queued_t = await self._send_q.get()
            if kind != "infer":
                await ws.send(payload)
                continue
            rid, window, task, tick = payload
            now = time.perf_counter()
            if now - queued_t > self.max_queue_age_s:
                self._replies.append(Reply(rid, tick, None, {}, "expired before sending", now, now))
                continue
            try:
                data = protocol.pack(protocol.infer_message(rid, tick, task, window))
            except Exception as e:
                self._replies.append(Reply(rid, tick, None, {}, f"could not pack the request: {e}", now, now))
                continue
            self._pending[rid] = (tick, time.perf_counter())
            await ws.send(data)

    async def _receiver(self, ws) -> None:
        async for raw in ws:
            recv_t = time.perf_counter()
            if not isinstance(raw, (bytes, bytearray)):
                log.warning("ignoring a text frame from the server: %.200s", raw)
                continue
            try:
                msg = protocol.unpack(raw)
            except Exception as e:
                log.warning("ignoring an undecodable message: %s", e)
                continue
            kind = protocol.message_type(msg)
            if kind == "chunk":
                self._on_chunk(msg, recv_t)
            elif kind == "error":
                self._on_error(msg, recv_t)
            elif kind == "pong":
                self._on_pong(msg, recv_t)
            elif kind == "ping":  # through the queue, so the sender task stays the only writer
                self._send_q.put_nowait(("raw", protocol.pack(protocol.pong_message(msg.get("t"), time.time())), recv_t))
            elif kind == "hello":
                self.hello = msg
            else:
                log.warning("ignoring message type %r", kind)

    def _on_chunk(self, msg: dict, recv_t: float) -> None:
        rid = msg.get("id")
        tick, sent_t = self._pending.pop(rid, (msg.get("tick", -1), float("nan")))
        tick = int(msg.get("tick", tick))
        try:
            actions, error = protocol.parse_actions(msg), None
        except Exception as e:
            actions, error = None, f"bad chunk: {e}"
        timing = msg.get("timing") or {}
        self._replies.append(Reply(int(rid) if rid is not None else -1, tick, actions, dict(timing), error, sent_t, recv_t))

    def _on_error(self, msg: dict, recv_t: float) -> None:
        rid = msg.get("id")
        if rid is None and len(self._pending) == 1:
            rid = next(iter(self._pending))
        text = str(msg.get("error"))
        if rid is None:
            log.warning("server error without a request id: %s", text)
            return
        tick, sent_t = self._pending.pop(rid, (-1, float("nan")))
        self._replies.append(Reply(int(rid), tick, None, {}, text, sent_t, recv_t))

    def _on_pong(self, msg: dict, recv_t: float) -> None:
        fut = self._pongs.pop(msg.get("t"), None)
        if fut is None and self._pongs:
            fut = self._pongs.pop(next(iter(self._pongs)))
        if fut is not None and not fut.done():
            fut.set_result((recv_t, msg))

    async def _ping_once(self, timeout_s: float) -> float:
        t = time.perf_counter()
        fut = asyncio.get_running_loop().create_future()
        self._pongs[t] = fut
        self._send_q.put_nowait(("raw", protocol.pack(protocol.ping_message(t)), t))
        try:
            recv_t, msg = await asyncio.wait_for(fut, timeout_s)
        finally:
            self._pongs.pop(t, None)
        self.server_t = msg.get("server_t")
        return (recv_t - t) * 1000.0

    def _fail_pending(self, reason: str) -> None:
        now = time.perf_counter()
        for rid, (tick, sent_t) in self._pending.items():
            self._replies.append(Reply(rid, tick, None, {}, reason, sent_t, now))
        self._pending.clear()
        if self._send_q is not None:
            while not self._send_q.empty():
                kind, payload, _ = self._send_q.get_nowait()
                if kind == "infer":
                    rid, _, _, tick = payload
                    self._replies.append(Reply(rid, tick, None, {}, reason, now, now))
        for fut in self._pongs.values():
            if not fut.done():
                fut.set_exception(ConnectionError(reason))
        self._pongs.clear()
