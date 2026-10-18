"""so101.v1 server tests: fake engine only, real WebSockets on free localhost ports. No torch, no GPU.

    uv run --python 3.13 --with pytest --with websockets --with msgpack --with numpy --with pillow \
        pytest infra/gpu/so101/tests -q
"""

import asyncio
import contextlib
import inspect
import json
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import protocol  # noqa: E402
import server  # noqa: E402
from engines import make_engine  # noqa: E402
from engines.base import (  # noqa: E402
    load_norm_stats,
    override_normalization,
    override_sampling,
    pin_revision,
    synthetic_frames,
    window_order,
)
from engines.fake import PERIOD_S, FakeEngine, pickplace_pose  # noqa: E402
from websockets.sync.client import connect  # noqa: E402

TASK = "pick up the small red disk and place it on the right peg"


# ---- helpers ------------------------------------------------------------------------------------------
@contextlib.contextmanager
def running(engine):
    """PolicyServer on 127.0.0.1:<free port> in a background thread; yields the port."""
    srv = server.PolicyServer(engine)
    ready, box = threading.Event(), {}

    def on_ready(port):
        box["port"] = port
        ready.set()

    def run():
        loop = asyncio.new_event_loop()
        box["loop"], box["stop"] = loop, asyncio.Event()
        try:
            loop.run_until_complete(srv.serve("127.0.0.1", 0, ready=on_ready, stop=box["stop"]))
        finally:
            loop.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(10), "server did not start"
    try:
        yield box["port"]
    finally:
        box["loop"].call_soon_threadsafe(box["stop"].set)
        thread.join(10)
        srv.executor.shutdown(wait=False, cancel_futures=True)


def client(port):
    kw = {"compression": None, "max_size": protocol.MAX_MESSAGE_BYTES, "open_timeout": 5}
    if "proxy" in inspect.signature(connect).parameters:
        kw["proxy"] = None
    return connect(f"ws://127.0.0.1:{port}", **kw)


def recv(ws, timeout: float = 5.0) -> dict:
    return protocol.unpackb(ws.recv(timeout=timeout))


def warm(engine):
    engine.warmup(TASK)
    return engine


def window(seed: int = 0):
    rng = np.random.default_rng(seed)
    state = rng.uniform(-30, 30, (8, 6)).astype(np.float32)
    state[:, 5] = rng.uniform(0, 100, 8)
    return state, (state + rng.normal(0, 0.5, (8, 6))).astype(np.float32)


def images(fmt: str = "jpeg", size: int = 256) -> dict:
    frames = synthetic_frames(size)
    return {cam: [server.encode_image(f, fmt) for f in frames] for cam in protocol.CAMERAS}


def infer_msg(id: int = 1, tick: int = 100, fmt: str = "jpeg"):
    state, cmd = window(id)
    return protocol.infer(id, tick, TASK, state, cmd, images(fmt), fmt), cmd


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---- codec and helpers -----------------------------------------------------------------------------
def test_codec_round_trip_uses_the_nd_map_and_rejects_bad_arrays():
    msg = {"a": np.arange(12, dtype=np.float32).reshape(3, 4), "b": np.zeros((2, 2, 3), np.uint8), "c": [np.float32(1.5)]}
    wire = protocol.packb(msg)
    plain = msgpack.unpackb(wire)
    assert plain["a"] == {"__nd__": True, "dtype": "<f4", "shape": [3, 4], "data": msg["a"].tobytes()}
    assert plain["b"]["dtype"] == "|u1"
    out = protocol.unpackb(wire)
    assert out["a"].dtype == np.float32 and np.array_equal(out["a"], msg["a"])
    assert out["b"].dtype == np.uint8 and out["b"].shape == (2, 2, 3)
    assert out["c"] == [1.5]
    with pytest.raises(protocol.ProtocolError, match="dtype"):
        protocol.unpackb(msgpack.packb({"x": {"__nd__": True, "dtype": "|O", "shape": [1], "data": b"x"}}))
    with pytest.raises(protocol.ProtocolError, match="bytes"):
        protocol.unpackb(msgpack.packb({"x": {"__nd__": True, "dtype": "<f4", "shape": [2], "data": b"abc"}}))
    with pytest.raises(protocol.ProtocolError):
        protocol.unpackb(b"\xc1")
    with pytest.raises(TypeError):
        protocol.packb({"x": np.array([object()])})


def test_window_order_and_revision_pins():
    assert window_order(8, [0, 7]) == [0, 0, 0, 0, 0, 0, 0, 1]
    rev = "62878e2925e59b7a89ec14463ce89932624c490d"
    spec = "black-forest-labs/flux-3-action-base:video_vae.safetensors"
    assert pin_revision(spec, rev)[0] == f"{spec}@{rev}"  # the config leaves it open: our pin
    assert pin_revision(f"{spec}@{rev}", rev)[0] == f"{spec}@{rev}"
    # The config pins another commit: its pin wins, so a different --base-revision is an error unless forced.
    with pytest.raises(ValueError, match="--base-revision ''"):
        pin_revision(f"{spec}@abc123", rev)
    pinned, note = pin_revision(f"{spec}@abc123", rev, force=True)
    assert pinned == f"{spec}@{rev}" and note.startswith("WARNING")
    assert pin_revision(f"{spec}@abc123", None) == (f"{spec}@abc123", "")  # --base-revision '': the config's own
    assert pin_revision(spec, None) == (spec, "")


def test_override_sampling_sets_the_config_and_warns_on_a_change():
    cfg = SimpleNamespace(guidance_scale=3.0, guidance_scale_action=None, num_inference_steps=4)
    assert override_sampling(cfg) == [] and cfg.guidance_scale_action is None
    assert override_sampling(cfg, (3.0, 3.0), 4) == []  # the same values: no warning
    notes = override_sampling(cfg, (1.0, 1.0), 2)
    assert (cfg.guidance_scale, cfg.guidance_scale_action, cfg.num_inference_steps) == (1.0, 1.0, 2)
    assert len(notes) == 2 and all(n.startswith("WARNING") for n in notes) and "3/3" in notes[0]
    with pytest.raises(ValueError):
        override_sampling(cfg, steps=0)


def test_server_imports_without_torch_or_gpu_engines():
    assert "torch" not in sys.modules
    assert "engines.flux_action" not in sys.modules and "engines.lerobot" not in sys.modules


# ---- over the socket -------------------------------------------------------------------------------
def test_hello_has_the_architecture_fields():
    with running(warm(FakeEngine(mode="hold"))) as port, client(port) as ws:
        hello = recv(ws)
    assert hello["type"] == "hello" and hello["protocol"] == "so101.v1"
    assert hello["engine"] == "fake" and hello["checkpoint"] == "fake" and "revision" in hello
    assert (hello["chunk"], hello["exec"], hello["fps"], hello["n_obs"]) == (42, 32, 30, 8)
    assert hello["snapshots"] == [0, 7] and hello["cameras"] == ["scene", "wrist"]
    assert hello["state_names"] == list(protocol.STATE_NAMES)
    assert hello["guidance"] == [3.0, 3.0] and hello["steps"] == 4 and hello["warm"] is True
    assert hello["warm_tasks"] == [TASK]


def test_guidance_and_steps_flags_reach_the_engine_and_hello():
    args = server.parse_args(["--engine", "fake", "--guidance", "1", "1", "--steps", "2"])
    engine = make_engine("fake", **server.engine_kwargs(args))
    hello = server.PolicyServer(engine).hello_message()
    assert hello["guidance"] == [1.0, 1.0] and hello["steps"] == 2
    gpu = server.engine_kwargs(server.parse_args(["--force-base-revision"]))
    assert gpu["force_base_revision"] is True and gpu["guidance"] is None and gpu["steps"] is None
    for bad in (["--steps", "0"], ["--guidance", "nan", "1"]):
        with pytest.raises(SystemExit):
            server.parse_args(bad)


def test_ping_gets_pong_with_the_client_time_echoed():
    with running(FakeEngine(mode="hold")) as port, client(port) as ws:
        recv(ws)
        t = time.perf_counter()
        ws.send(protocol.packb(protocol.ping(t)))
        pong = recv(ws)
    assert pong["type"] == "pong" and pong["t"] == t
    assert abs(pong["server_t"] - time.time()) < 5


@pytest.mark.parametrize("fmt", ["jpeg", "png", "raw"])
def test_infer_returns_42x6_float32_and_hold_repeats_the_last_command(fmt):
    with running(FakeEngine(mode="hold")) as port, client(port) as ws:
        recv(ws)
        msg, cmd = infer_msg(7, 1234, fmt)
        ws.send(protocol.packb(msg))
        reply = recv(ws)
    assert reply["type"] == "chunk" and reply["id"] == 7 and reply["tick"] == 1234
    actions = reply["actions"]
    assert actions.dtype == np.float32 and actions.shape == (42, 6)
    np.testing.assert_array_equal(actions, np.tile(cmd[-1], (42, 1)))
    timing = reply["timing"]
    assert set(timing) == {"queue_ms", "decode_ms", "infer_ms", "total_ms"}
    assert all(v >= 0 for v in timing.values()) and timing["total_ms"] >= timing["infer_ms"]


def test_malformed_requests_get_errors_and_the_connection_stays_usable():
    def bad(id, **changes):
        msg, _ = infer_msg(id)
        msg.update(changes)
        return msg

    small = {cam: [server.encode_image(f, "jpeg") for f in synthetic_frames(128)] for cam in protocol.CAMERAS}
    cases = [
        (b"\xc1 not msgpack", None, "msgpack"),
        ("a text frame", None, "binary"),
        (protocol.packb(bad(3, state=np.zeros((7, 6), np.float32))), 3, "(8, 6)"),
        (protocol.packb(bad(4, cmd_hist=np.zeros((8, 6), np.float64))), 4, "float32"),
        (protocol.packb(bad(5, images={"scene": images()["scene"]})), 5, "images"),
        (protocol.packb(bad(6, image_format="webp")), 6, "image_format"),
        (protocol.packb(bad(7, images={"scene": images()["scene"], "wrist": [b"no jpeg", b"x"]})), 7, "decode"),
        (protocol.packb(bad(8, images={"scene": images()["scene"], "wrist": small["wrist"]})), 8, "one size"),
        (protocol.packb(bad(9, tick="soon")), 9, "tick"),
        (protocol.packb({"type": "launch", "id": 10}), 10, "unknown message type"),
        (protocol.packb({"id": 11}), 11, "unknown message type"),
    ]
    with running(FakeEngine(mode="hold")) as port, client(port) as ws:
        recv(ws)
        for frame, rid, words in cases:
            ws.send(frame)
            reply = recv(ws)
            assert reply["type"] == "error" and reply["id"] == rid, reply
            assert words in reply["error"], reply["error"]
        good, cmd = infer_msg(12)
        ws.send(protocol.packb(good))
        reply = recv(ws)
    assert reply["type"] == "chunk" and reply["id"] == 12
    np.testing.assert_array_equal(reply["actions"][0], cmd[-1])


def test_pong_is_not_delayed_by_a_running_inference():
    with running(FakeEngine(mode="hold", latency_ms=300)) as port, client(port) as ws:
        recv(ws)
        msg, _ = infer_msg(1)
        t_infer = time.perf_counter()
        ws.send(protocol.packb(msg))
        time.sleep(0.02)
        rtts = []
        for _ in range(5):
            t = time.perf_counter()
            ws.send(protocol.packb(protocol.ping(t)))
            pong = recv(ws)
            assert pong["type"] == "pong" and pong["t"] == t
            rtts.append((time.perf_counter() - t) * 1000)
            time.sleep(0.02)
        assert time.perf_counter() - t_infer < 0.28, "the pings must overlap the running plan"
        reply = recv(ws)
    assert reply["type"] == "chunk" and reply["timing"]["infer_ms"] >= 290
    assert max(rtts) < 50, rtts


def test_two_infers_in_a_row_are_processed_in_order():
    with running(FakeEngine(mode="hold", latency_ms=100)) as port, client(port) as ws:
        recv(ws)
        (a, cmd_a), (b, cmd_b) = infer_msg(1, 10), infer_msg(2, 11)
        ws.send(protocol.packb(a))
        ws.send(protocol.packb(b))
        first, second = recv(ws), recv(ws)
    assert (first["id"], first["tick"], second["id"], second["tick"]) == (1, 10, 2, 11)
    np.testing.assert_array_equal(first["actions"][0], cmd_a[-1])
    np.testing.assert_array_equal(second["actions"][0], cmd_b[-1])
    assert second["timing"]["queue_ms"] >= 80, second["timing"]


def test_a_flooding_client_gets_busy_errors_beyond_the_queue_limit():
    n = server.MAX_PENDING + 2
    frames = [protocol.packb(infer_msg(i)[0]) for i in range(n)]
    with running(FakeEngine(mode="hold", latency_ms=100)) as port, client(port) as ws:
        recv(ws)
        for frame in frames:
            ws.send(frame)
        replies = [recv(ws) for _ in range(n)]
    busy = [r["id"] for r in replies if r["type"] == "error"]
    done = [r["id"] for r in replies if r["type"] == "chunk"]
    assert busy == [n - 2, n - 1] and done == list(range(server.MAX_PENDING))


# ---- fake engine motion ----------------------------------------------------------------------------
def test_pickplace_closes_reopens_then_rests_and_restarts_on_a_new_task():
    clock = {"t": 0.0}
    engine = FakeEngine(mode="pickplace", clock=lambda: clock["t"])
    state, cmd = window(0)
    anchor, chunks = cmd[-1].copy(), []
    for k in range(int(PERIOD_S * 30 / 42) + 1):  # consecutive chunks, stitched like the client does
        clock["t"] = k * 42 / 30
        chunk = engine.infer(state, np.tile(anchor, (8, 1)), None, "task A")
        chunks.append(chunk)
        anchor = chunk[-1]
    traj = np.concatenate(chunks)[: int(PERIOD_S * 30)]
    grip, joints = traj[:, 5], traj[:, :5]
    opened, closed = engine.gripper_open, engine.gripper_closed
    i_closed = int(np.argmax(grip <= closed))
    i_reopened = i_closed + int(np.argmax(grip[i_closed:] >= opened))
    assert grip[0] == opened and grip[i_closed] == closed and i_reopened > i_closed
    assert np.abs(np.diff(joints, axis=0)).max() < 1.0  # degrees per tick
    assert np.abs(joints - cmd[-1][:5]).max() < 20
    rest = joints[int(6.6 * 30) :]
    assert len(rest) >= 60 and np.ptp(rest, axis=0).max() < 1e-4 and (grip[int(6.6 * 30) :] == opened).all()
    np.testing.assert_allclose(joints[-1], joints[0], atol=1e-3)  # back where the cycle started
    clock["t"] += 3.0
    restart = engine.infer(state, np.tile(anchor, (8, 1)), None, "task B")
    assert (restart[:, 5] == opened).all() and np.abs(restart[:, :5] - anchor[:5]).max() < 8.5


def test_pickplace_period_scales_every_phase():
    for t in np.linspace(0.0, 2 * PERIOD_S, 97):
        short, g_short = pickplace_pose(t / 3, 60.0, 10.0, PERIOD_S / 3)
        full, g_full = pickplace_pose(t, 60.0, 10.0)
        np.testing.assert_allclose(short, full, atol=1e-9)
        assert g_short == pytest.approx(g_full)
    clock = {"t": 0.0}
    engine = FakeEngine(mode="pickplace", clock=lambda: clock["t"], period_s=3.0)
    assert engine.meta["fake"]["period_s"] == 3.0
    state, cmd = window(0)
    engine.infer(state, cmd, None, "task A")  # the cycle starts here
    clock["t"] = 1.0  # one third in: transport, gripper closed
    assert (engine.infer(state, cmd, None, "task A")[:5, 5] == engine.gripper_closed).all()
    with pytest.raises(ValueError):
        FakeEngine(period_s=0)


def test_sine_moves_joints_and_keeps_the_gripper():
    engine = FakeEngine(mode="sine")
    state, cmd = window(1)
    chunk = engine.infer(state, cmd, None, TASK)
    assert (chunk[:, 5] == cmd[-1][5]).all()
    assert 0 < np.abs(chunk[:, :4] - cmd[-1][:4]).max() < 12


# ---- the CLI ----------------------------------------------------------------------------------------------
def test_selftest_cli_passes_with_the_fake_engine():
    out = subprocess.run(
        [sys.executable, str(HERE / "server.py"), "--engine", "fake", "--selftest", "--n", "5"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.strip().endswith("PASS") and "infer ms p50" in out.stdout
    assert "first warmup plan" in out.stdout


def test_cli_server_serves_logs_and_stops_on_ctrl_c(tmp_path):
    port = free_port()
    cmd = [sys.executable, str(HERE / "server.py"), "--engine", "fake", "--fake-mode", "hold"]
    cmd += ["--port", str(port), "--log-dir", str(tmp_path)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                ws = client(port)
                break
            except OSError:
                assert time.monotonic() < deadline and proc.poll() is None, "server did not come up"
                time.sleep(0.1)
        with ws:
            hello = recv(ws)
            msg, cmd_hist = infer_msg(5, 77)
            ws.send(protocol.packb(msg))
            reply = recv(ws)
        assert hello["engine"] == "fake" and hello["warm"] is True
        assert reply["type"] == "chunk" and reply["id"] == 5
        proc.send_signal(signal.SIGINT)
        output = proc.communicate(timeout=20)[0]
    finally:
        if proc.poll() is None:
            proc.kill()
    assert f"listening on ws://127.0.0.1:{port}" in output and "infer id=5 tick=77" in output
    assert "stopped" in output
    assert hello["warm_tasks"] == [server.DEFAULT_TASK] and TASK != server.DEFAULT_TASK
    assert output.count("WARNING: caption not warmed up") == 1  # TASK was never warmed up
    events = [json.loads(line) for f in tmp_path.glob("so101-*.jsonl") for line in f.read_text().splitlines()]
    infer = [e for e in events if e["event"] == "infer"]
    assert [e["event"] for e in events][0] == "start" and len(infer) == 1
    assert infer[0]["id"] == 5 and len(infer[0]["actions"]) == 42 and "images" not in infer[0]


# ---- engine logic that runs without torch (keep after test_server_imports_without_torch_or_gpu_engines) ----
def test_flux_action_compile_falls_back_to_eager_when_the_compiled_warmup_fails():
    from engines.base import Engine
    from engines.flux_action import FluxActionEngine  # torch is imported only by __init__, which is skipped

    class Dit:
        def forward(self):
            return "eager"

    dit = Dit()
    dit.forward = lambda: "compiled"  # an instance attribute, as the engine sets torch.compile's wrapper
    engine = FluxActionEngine.__new__(FluxActionEngine)
    Engine.__init__(engine)
    engine.policy, engine.compiled, engine.device = SimpleNamespace(dit=dit), True, SimpleNamespace(type="cpu")
    engine.meta = {"chunk": 42, "cameras": ["scene", "wrist"], "compile": "dit (experimental)"}

    def infer(state, cmd_hist, frames, task):
        if dit.forward() == "compiled":
            raise RuntimeError("dynamo failed")
        return np.tile(cmd_hist[-1], (42, 1))

    engine.infer = infer
    engine.warmup(TASK)
    assert engine.compiled is False and dit.forward() == "eager"
    assert engine.meta["compile"] is False and "dynamo failed" in engine.meta["compile_failed"]
    assert engine.warm and engine.warm_tasks == [TASK] and engine.first_plan_ms is not None


def test_pin_revision_leaves_local_snapshot_paths_alone():
    """lerobot-train writes absolute snapshot paths into an adapter's config (T23), sometimes with a :file part."""
    rev = "62878e2925e59b7a89ec14463ce89932624c490d"
    for spec in ("/workspace/hf/hub/models--black-forest-labs--flux-3-action-base/snapshots/62878e29:vae",
                 "~/hf/snapshots/62878e29/text_encoder", "./weights:vae"):
        assert pin_revision(spec, rev) == (spec, "")


NEW_CAL = {  # new-calibration-like stats: LeRobot degrees around 0, gripper 0..100
    "state_q01": [-90.0, -100.0, -95.0, -90.0, -150.0, 0.5],
    "state_q99": [90.0, 100.0, 95.0, 90.0, 150.0, 60.0],
    "action_q01": [-3.0, -3.0, -3.0, -3.0, -5.0, 0.5],
    "action_q99": [3.0, 3.0, 3.0, 3.0, 5.0, 60.0],
}


def test_norm_stats_load_flat_or_nested_and_refuse_bad_files(tmp_path):
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps(NEW_CAL))
    stats, digest = load_norm_stats(flat)
    assert sorted(stats) == ["action", "state"] and stats["state"]["q99"][1] == 100.0 and len(digest) == 64
    nested = tmp_path / "nested.json"
    nested.write_text(json.dumps({"state": {"q01": NEW_CAL["state_q01"], "q99": NEW_CAL["state_q99"]}}))
    only_state, _ = load_norm_stats(nested)
    assert list(only_state) == ["state"]  # the action stream keeps the checkpoint's
    wrapped = tmp_path / "HFSIM_NORMALIZATION.json"  # saad-calib's file: the quantiles under "stats", plus notes
    wrapped.write_text(json.dumps({"source": "multimodalart/flux-3-action-so101-sim@09edebb",
                                   "stats": {"state": {"q01": NEW_CAL["state_q01"], "q99": NEW_CAL["state_q99"]},
                                             "action": {"q01": NEW_CAL["action_q01"], "q99": NEW_CAL["action_q99"]}}}))
    both, _ = load_norm_stats(wrapped)
    assert sorted(both) == ["action", "state"] and both["action"]["q99"][4] == 5.0
    for bad, why in (({"state_q01": [0] * 5, "state_q99": [1] * 5}, "6 finite"),
                     ({"state_q01": [0] * 6, "state_q99": [0] * 6}, "q99 > q01"),
                     ({"other": 1}, "no state or action")):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match=why):
            load_norm_stats(path)


def test_override_normalization_sets_the_flux_action_config_and_warns_on_a_change():
    cfg = SimpleNamespace(state_normalization={"q01": [0.0] * 6, "q99": [1.0] * 6}, action_normalization=None)
    stats = {"state": {"q01": NEW_CAL["state_q01"], "q99": NEW_CAL["state_q99"]}}
    notes = override_normalization(cfg, stats)
    assert cfg.state_normalization["q99"][1] == 100.0 and cfg.action_normalization is None
    assert len(notes) == 1 and notes[0].startswith("WARNING: state normalization")
    assert override_normalization(cfg, stats) == []  # the same values again: no warning


def test_norm_stats_flag_reaches_the_gpu_engines_and_the_hello(tmp_path):
    path = tmp_path / "newcal.json"
    path.write_text(json.dumps(NEW_CAL))
    stats, _ = load_norm_stats(path)
    kw = server.engine_kwargs(server.parse_args(["--norm-stats", str(path)]), stats)
    assert kw["norm_stats"] == stats  # flux-action, and lerobot alike
    out = subprocess.run(
        [sys.executable, str(HERE / "server.py"), "--engine", "fake", "--selftest", "--n", "2", "--norm-stats", str(path)],
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "norm stats:" in out.stdout and "ignored (fake engine)" in out.stdout
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    out = subprocess.run(
        [sys.executable, str(HERE / "server.py"), "--engine", "fake", "--selftest", "--norm-stats", str(bad)],
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 2 and "ERROR: --norm-stats" in out.stdout and out.stdout.strip().endswith("FAIL")
