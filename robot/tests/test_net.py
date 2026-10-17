"""NetClient against a minimal so101.v1 peer on 127.0.0.1 (real sockets, no GPU)."""

import asyncio
import socket
import threading
import time

import numpy as np
import pytest

from so101_client import CAMERAS, N_OBS, PROTOCOL, STATE_NAMES, protocol
from so101_client.history import Window
from so101_client.net import NetClient

HELLO = {"type": "hello", "protocol": PROTOCOL, "engine": "mini", "checkpoint": "test", "revision": "0", "chunk": 42,
         "exec": 32, "fps": 30, "n_obs": N_OBS, "snapshots": [0, 7], "cameras": list(CAMERAS),
         "state_names": list(STATE_NAMES), "guidance": [3, 3], "steps": 4, "warm": True}


class MiniServer:
    """task "fail" -> error message, task "drop" -> the server closes the connection, else a chunk."""

    def __init__(self, delay_s=0.05):
        self.delay_s = delay_s
        self.received = []
        self.pongs = []
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.port = asyncio.run_coroutine_threadsafe(self._start(), self.loop).result(5)

    async def _start(self):
        from websockets.asyncio.server import serve

        self.server = await serve(self._handler, "127.0.0.1", 0, compression=None, max_size=None)
        return self.server.sockets[0].getsockname()[1]

    async def _handler(self, ws):
        await ws.send(protocol.pack(HELLO))
        await ws.send(protocol.pack({"type": "ping", "t": 123.5}))  # the client must answer with a pong
        async for raw in ws:
            msg = protocol.unpack(raw)
            if msg["type"] == "pong":
                self.pongs.append(msg)
            elif msg["type"] == "ping":
                await ws.send(protocol.pack({"type": "pong", "t": msg["t"], "server_t": time.time()}))
            elif msg["type"] == "infer":
                self.received.append(msg)
                if msg["task"] == "fail":
                    await ws.send(protocol.pack({"type": "error", "id": msg["id"], "error": "engine failed"}))
                elif msg["task"] == "drop":
                    await ws.close()
                    return
                else:
                    await asyncio.sleep(self.delay_s)
                    actions = np.repeat(np.asarray(msg["cmd_hist"])[-1:], 42, axis=0).astype(np.float32)
                    await ws.send(protocol.pack({
                        "type": "chunk", "id": msg["id"], "tick": msg["tick"], "actions": actions,
                        "timing": {"queue_ms": 0.0, "decode_ms": 1.0, "infer_ms": 40.0, "total_ms": 41.0}}))

    def close(self):
        async def stop():
            self.server.close()
            await self.server.wait_closed()

        asyncio.run_coroutine_threadsafe(stop(), self.loop).result(5)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)


def window():
    state = np.arange(48, dtype=np.float32).reshape(8, 6)
    return Window(state=state, cmd_hist=state + 100, images={"scene": [b"s0", b"s7"], "wrist": [b"w0", b"w7"]},
                  image_format="jpeg", frame_t={"scene": 0.0, "wrist": 0.0})


def wait_reply(net, rid, timeout_s=5.0):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        for reply in net.poll():
            if reply.id == rid:
                return reply
        time.sleep(0.002)
    raise AssertionError(f"no reply for request {rid}")


@pytest.fixture
def server():
    s = MiniServer()
    yield s
    s.close()


def test_hello_ping_infer_and_error(server):
    net = NetClient(f"ws://127.0.0.1:{server.port}", connect_timeout_s=5.0)
    try:
        hello = net.connect()
        assert hello["protocol"] == PROTOCOL and hello["engine"] == "mini" and net.connected
        rtts = net.ping(3)
        assert len(rtts) == 3 and all(0 < r < 1000 for r in rtts)
        w = window()
        rid = net.submit(w, "stack", tick=77)
        reply = wait_reply(net, rid)
        assert reply.error is None and reply.tick == 77
        assert reply.actions.shape == (42, 6) and reply.actions.dtype == np.float32
        np.testing.assert_array_equal(reply.actions[0], w.cmd_hist[-1])
        assert reply.timing["total_ms"] == 41.0 and reply.recv_t > reply.sent_t
        got = server.received[-1]
        np.testing.assert_array_equal(got["state"], w.state)
        np.testing.assert_array_equal(got["cmd_hist"], w.cmd_hist)
        assert got["images"] == w.images and got["image_format"] == "jpeg"
        assert (got["id"], got["tick"], got["task"]) == (rid, 77, "stack")
        rid = net.submit(w, "fail", tick=78)
        reply = wait_reply(net, rid)
        assert reply.actions is None and reply.error == "engine failed"
        assert server.pongs and server.pongs[0]["t"] == 123.5 and "server_t" in server.pongs[0]
    finally:
        net.close()
    assert not net.connected


def test_disconnect_fails_the_request_in_flight_then_reconnects(server):
    net = NetClient(f"ws://127.0.0.1:{server.port}", connect_timeout_s=5.0)
    try:
        net.connect()
        rid = net.submit(window(), "drop", tick=1)
        reply = wait_reply(net, rid)
        assert reply.actions is None and reply.error == "disconnected"
        deadline = time.perf_counter() + 5.0
        while not net.connected and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert net.connected and net.reconnects >= 1
        rid = net.submit(window(), "stack", tick=2)
        assert wait_reply(net, rid).error is None
    finally:
        net.close()


def test_connect_without_a_server_raises_and_submit_fails_softly():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    net = NetClient(f"ws://127.0.0.1:{port}", connect_timeout_s=1.0)
    with pytest.raises(ConnectionError):
        net.connect()
    rid = net.submit(window(), "stack", tick=0)
    replies = net.poll()
    assert [(r.id, r.error) for r in replies] == [(rid, "not connected")]
