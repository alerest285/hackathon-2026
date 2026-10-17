"""SO-101 policy server: so101.v1 over one WebSocket per client (docs/architecture.md, decision 06).

Stateless: every infer request carries its own 8-tick window and the Mac owns the history. One
inference runs at a time on a worker thread while the event loop keeps answering pings. Binds
127.0.0.1 only; the Mac reaches it through ``ssh -L 8000:localhost:8000``.

    python server.py --engine flux-action|lerobot|fake [--port 8000] [--selftest] [--log-dir DIR] ...
    bash infra/gpu/so101/serve.sh [same args]        picks the right venv on the GPU host

Imports without torch: the engines import it when they are built.
"""

import argparse
import asyncio
import io
import json
import math
import socket
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import protocol
from engines import NAMES, make_engine
from engines.base import load_norm_stats, synthetic_window

DEFAULT_CHECKPOINT = "black-forest-labs/flux-3-action-so101"
DEFAULT_REVISION = "c9e13b2aca6a0a472b3ea03fd90cff32d3e85849"
DEFAULT_BASE_REVISION = "62878e2925e59b7a89ec14463ce89932624c490d"
# Step 1 of eval/config/hanoi.toml. Warm up every instruction: `eval/run_hanoi.py --print-warmup` prints the flags.
DEFAULT_TASK = "pick up the small disk and place it on the right peg"
FAKE_MODES = ("hold", "sine", "pickplace")
MAX_PENDING = 8  # queued infer requests per connection before the server answers "busy"
HELLO_DEFAULTS = {
    "engine": None,
    "checkpoint": None,
    "revision": None,
    "chunk": protocol.CHUNK,
    "exec": 32,
    "fps": 30,
    "n_obs": protocol.N_OBS,
    "snapshots": list(protocol.SNAPSHOTS),
    "cameras": list(protocol.CAMERAS),
    "state_names": list(protocol.STATE_NAMES),
    "guidance": [3.0, 3.0],
    "steps": 4,
}


def say(msg: str) -> None:
    print(f"{datetime.now(timezone.utc):%H:%M:%S.%f}"[:-3] + " " + msg, flush=True)


# ---- images -------------------------------------------------------------------------------------
_codec: tuple | None = None


def _image_codec() -> tuple:
    """Pillow if importable, else OpenCV; imported on first use."""
    global _codec
    if _codec is None:
        try:
            from PIL import Image

            _codec = ("pil", Image)
        except ImportError:
            try:
                import cv2

                _codec = ("cv2", cv2)
            except ImportError:
                _codec = ("none", None)
    return _codec


def decode_image(data: bytes, where: str) -> np.ndarray:
    """jpeg/png bytes -> uint8 (H, W, 3) RGB."""
    kind, lib = _image_codec()
    if kind == "pil":
        try:
            with lib.open(io.BytesIO(data)) as im:
                if max(im.size) > protocol.MAX_FRAME_SIDE:  # header only; the pixels are not decoded yet
                    raise ValueError(f"frame size {im.size} exceeds {protocol.MAX_FRAME_SIDE}")
                return np.asarray(im.convert("RGB"))
        except Exception as e:
            raise protocol.ProtocolError(f"cannot decode {where}: {type(e).__name__}: {e}") from None
    if kind == "cv2":
        bgr = lib.imdecode(np.frombuffer(data, np.uint8), lib.IMREAD_COLOR)
        if bgr is None:
            raise protocol.ProtocolError(f"cannot decode {where}")
        return bgr[..., ::-1]
    raise RuntimeError("neither Pillow nor OpenCV is installed, so jpeg/png frames cannot be decoded")


def encode_image(rgb: np.ndarray, image_format: str, quality: int = 95):
    """uint8 (H, W, 3) RGB -> what an infer message carries: jpeg/png bytes, or the array for raw."""
    if image_format == "raw":
        return np.ascontiguousarray(rgb, dtype=np.uint8)
    jpeg = image_format == "jpeg"
    kind, lib = _image_codec()
    if kind == "pil":
        buf = io.BytesIO()
        lib.fromarray(rgb).save(buf, format="JPEG" if jpeg else "PNG", **({"quality": quality} if jpeg else {}))
        return buf.getvalue()
    if kind == "cv2":
        params = [lib.IMWRITE_JPEG_QUALITY, quality] if jpeg else []
        ok, enc = lib.imencode(".jpg" if jpeg else ".png", np.ascontiguousarray(rgb[..., ::-1]), params)
        if not ok:
            raise RuntimeError(f"cannot encode {image_format}")
        return enc.tobytes()
    raise RuntimeError("neither Pillow nor OpenCV is installed, so jpeg/png frames cannot be encoded")


def codec_summary() -> str:
    """Import and exercise the jpeg/png codec once, so the first request does not pay for it."""
    kind, lib = _image_codec()
    if kind == "none":
        return "nothing (neither Pillow nor OpenCV is installed): only image_format raw works"
    decode_image(encode_image(np.full((32, 32, 3), 128, np.uint8), "jpeg"), "codec check")
    if kind == "pil":
        import PIL

        return f"Pillow {PIL.__version__}"
    return f"OpenCV {lib.__version__}"


def decode_frames(req: protocol.InferRequest, cameras) -> dict:
    """{camera: uint8 (2, H, W, 3) RGB [i0, i7]}. All frames must share one size (the model stacks them)."""
    out, shape = {}, None
    for cam in cameras:
        frames = []
        for k, item in enumerate(req.images[cam]):
            if req.image_format == "raw":
                img = item
            else:
                img = decode_image(item, f"{req.image_format} frame {cam}[{k}]")
            if shape is None:
                shape = img.shape
                side = protocol.MAX_FRAME_SIDE
                if not (16 <= shape[0] <= side and 16 <= shape[1] <= side):
                    raise protocol.ProtocolError(f"unsupported frame size {shape[:2]}")
            elif img.shape != shape:
                raise protocol.ProtocolError(
                    f"all frames must share one size, got {shape[:2]} and {img.shape[:2]} ({cam}[{k}])"
                )
            frames.append(img)
        out[cam] = np.stack(frames).astype(np.uint8, copy=False)
    return out


# ---- one request ----------------------------------------------------------------------------------
def run_request(engine, req: protocol.InferRequest, cameras, chunk: int, t_recv: float):
    """Decode and infer one request on the worker thread. Returns (actions (chunk, 6) f32, timing, (H, W))."""
    t0 = time.perf_counter()
    frames = decode_frames(req, cameras)
    t1 = time.perf_counter()
    actions = np.asarray(engine.infer(req.state, req.cmd_hist, frames, req.task), dtype=np.float32)
    t2 = time.perf_counter()
    if actions.shape != (chunk, 6):
        raise RuntimeError(f"engine returned actions of shape {actions.shape}, expected ({chunk}, 6)")
    if not np.isfinite(actions).all():
        raise RuntimeError("engine returned non-finite actions")
    timing = {"queue_ms": (t0 - t_recv) * 1e3, "decode_ms": (t1 - t0) * 1e3, "infer_ms": (t2 - t1) * 1e3}
    return actions, timing, frames[cameras[0]].shape[1:3]


def make_hello(engine) -> dict:
    return protocol.hello(**{**HELLO_DEFAULTS, "engine": engine.name, **engine.meta})


class JsonlLog:
    """One JSON line per event in DIR/so101-<UTC stamp>.jsonl. No images."""

    def __init__(self, directory):
        root = Path(directory).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"so101-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.jsonl"
        self._file = self.path.open("a", buffering=1)

    def write(self, event: str, **fields) -> None:
        line = {"t": round(time.time(), 3), "event": event, **fields}
        self._file.write(json.dumps(line, default=_jsonable) + "\n")


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return np.round(obj.astype(np.float64), 4).tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


class PolicyServer:
    """so101.v1 over WebSocket: hello on connect, pong at once, infer on a single worker thread, replies in order."""

    def __init__(self, engine, executor: ThreadPoolExecutor | None = None, log: JsonlLog | None = None):
        self.engine = engine
        self.executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="so101-infer")
        self.log = log
        self.hello = make_hello(engine)
        self.cameras = tuple(self.hello["cameras"])
        self.n_obs, self.chunk = int(self.hello["n_obs"]), int(self.hello["chunk"])
        self.n_frames = len(self.hello["snapshots"])
        self.seen_tasks = set(getattr(engine, "warm_tasks", ()))  # captions whose text context is cached

    def hello_message(self) -> dict:
        warm_tasks = list(getattr(self.engine, "warm_tasks", ()))
        return {**self.hello, "warm": bool(self.engine.warm), "warm_tasks": warm_tasks}

    async def serve(self, host: str, port: int, ready=None, stop: asyncio.Event | None = None) -> None:
        """Listen until ``stop`` is set (forever without one). ``ready(port)`` is called once bound."""
        from websockets.asyncio.server import serve

        async with serve(
            self.handle, host, port, compression=None, max_size=protocol.MAX_MESSAGE_BYTES
        ) as server:
            bound = [s.getsockname()[:2] for s in server.sockets]
            urls = ", ".join(f"ws://{h}:{p}" for h, p in bound)
            say(f"listening on {urls} (engine {self.engine.name}, one inference at a time; Ctrl-C stops)")
            if ready is not None:
                ready(bound[0][1])
            if stop is None:
                await server.serve_forever()
            else:
                await stop.wait()

    async def handle(self, ws) -> None:
        from websockets.exceptions import ConnectionClosed

        peer = _peer(ws)
        sock = ws.transport.get_extra_info("socket") if ws.transport is not None else None
        if sock is not None:
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
        say(f"connect {peer}")
        self._jsonl("connect", peer=peer)
        loop = asyncio.get_running_loop()
        pending: set[asyncio.Task] = set()
        served = 0
        try:
            await ws.send(protocol.packb(self.hello_message()))
            async for raw in ws:
                t_recv = time.perf_counter()
                try:
                    msg = protocol.unpackb(raw)
                except protocol.ProtocolError as e:
                    await self._error(ws, None, str(e), peer)
                    continue
                kind = msg.get("type")
                rid = msg.get("id") if protocol.is_int(msg.get("id")) else None
                if kind == "ping":
                    await ws.send(protocol.packb(protocol.pong(msg.get("t"), time.time())))
                elif kind == "infer":
                    if len(pending) >= MAX_PENDING:
                        await self._error(ws, rid, f"server busy: {len(pending)} requests queued", peer)
                        continue
                    try:
                        req = protocol.parse_infer(
                            msg, n_obs=self.n_obs, cameras=self.cameras, n_frames=self.n_frames
                        )
                    except protocol.ProtocolError as e:
                        await self._error(ws, rid, str(e), peer)
                        continue
                    fut = loop.run_in_executor(
                        self.executor, run_request, self.engine, req, self.cameras, self.chunk, t_recv
                    )
                    task = asyncio.create_task(self._reply(ws, req, fut, t_recv, peer, len(raw)))
                    pending.add(task)
                    task.add_done_callback(pending.discard)
                    served += 1
                else:
                    await self._error(ws, rid, f"unknown message type {kind!r}", peer)
        except ConnectionClosed:
            pass
        finally:
            for task in list(pending):
                task.cancel()  # drops queued work of a client that left; a running plan finishes
            say(f"disconnect {peer} ({served} infer requests)")
            self._jsonl("disconnect", peer=peer, requests=served)

    async def _reply(self, ws, req: protocol.InferRequest, fut, t_recv: float, peer: str, nbytes: int) -> None:
        from websockets.exceptions import ConnectionClosed

        try:
            actions, timing, hw = await fut
        except asyncio.CancelledError:
            raise
        except protocol.ProtocolError as e:
            await self._error(ws, req.id, str(e), peer)
            return
        except Exception as e:
            traceback.print_exc()
            await self._error(ws, req.id, f"{type(e).__name__}: {e}"[:2000], peer)
            return
        timing["total_ms"] = (time.perf_counter() - t_recv) * 1e3
        try:
            await ws.send(protocol.packb(protocol.chunk(req.id, req.tick, actions, timing)))
        except ConnectionClosed:
            return
        say(
            f"infer id={req.id} tick={req.tick} queue {timing['queue_ms']:.1f} decode {timing['decode_ms']:.1f} "
            f"infer {timing['infer_ms']:.1f} total {timing['total_ms']:.1f} ms  {hw[0]}x{hw[1]} "
            f"{req.image_format} {nbytes / 1024:.0f} KB  {peer}  {req.task[:60]!r}"
        )
        if req.task not in self.seen_tasks:
            self.seen_tasks.add(req.task)
            say(
                f"WARNING: caption not warmed up, so this plan encoded it ({timing['infer_ms']:.0f} ms): "
                f"{req.task[:80]!r}. Start the server with --warmup-task for every instruction "
                "(eval/run_hanoi.py --print-warmup prints them)"
            )
        self._jsonl(
            "infer",
            peer=peer,
            id=req.id,
            tick=req.tick,
            task=req.task,
            image_format=req.image_format,
            frame_hw=list(hw),
            request_bytes=nbytes,
            timing={k: round(v, 3) for k, v in timing.items()},
            state=req.state,
            cmd_hist=req.cmd_hist,
            actions=actions,
        )

    async def _error(self, ws, rid, message: str, peer: str) -> None:
        from websockets.exceptions import ConnectionClosed

        say(f"error id={rid} {peer}: {message}")
        self._jsonl("error", peer=peer, id=rid, error=message)
        try:
            await ws.send(protocol.packb(protocol.error(rid, message)))
        except ConnectionClosed:
            pass

    def _jsonl(self, event: str, **fields) -> None:
        if self.log is not None:
            self.log.write(event, **fields)


def _peer(ws) -> str:
    addr = getattr(ws, "remote_address", None)
    return f"{addr[0]}:{addr[1]}" if addr else "?"


# ---- self-test ------------------------------------------------------------------------------------
def selftest(engine, executor: ThreadPoolExecutor, n: int, task: str) -> bool:
    """N plans of one synthetic window through the codec, the JPEG decode and the engine, as the server runs them."""
    hello = make_hello(engine)
    cameras, chunk = tuple(hello["cameras"]), int(hello["chunk"])
    state, cmd_hist, frames = synthetic_window(hello)
    images = {cam: [encode_image(f, "jpeg") for f in frames[cam]] for cam in cameras}
    raw = protocol.packb(protocol.infer(0, 0, task, state, cmd_hist, images, "jpeg"))
    req = protocol.parse_infer(
        protocol.unpackb(raw), n_obs=int(hello["n_obs"]), cameras=cameras, n_frames=len(hello["snapshots"])
    )
    keys = ("steps", "guidance", "sampler", "seed", "dtype", "compile", "offload_text_encoder", "device", "gpu")
    say(f"selftest: {engine.name} {json.dumps({k: hello[k] for k in keys if k in hello})}")
    if "video_vae_id" in hello:
        say(
            f"selftest: encoders {hello['video_vae_id']} and {hello['text_encoder_id']} (checkpoint config: "
            f"{hello.get('config_video_vae_id')} and {hello.get('config_text_encoder_id')})"
        )
    if hello.get("compile_failed"):
        say(f"selftest: WARNING: --compile failed in warmup and the engine runs eager: {hello['compile_failed']}")
    say(f"selftest: {n} plans, request {len(raw) / 1024:.0f} KB (4 JPEG q95 frames 256x256), task {task!r}")
    torch = sys.modules.get("torch")
    cuda = torch is not None and torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        peak_load = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    infer_ms, decode_ms, outs, ok = [], [], [], True
    for _ in range(n):
        try:
            actions, timing, _ = executor.submit(
                run_request, engine, req, cameras, chunk, time.perf_counter()
            ).result()
        except Exception:
            traceback.print_exc()
            ok = False
            break
        infer_ms.append(timing["infer_ms"])
        decode_ms.append(timing["decode_ms"])
        outs.append(actions)
    if infer_ms:
        p50, p95 = np.percentile(infer_ms, [50, 95])
        first = getattr(engine, "first_plan_ms", None)
        say(
            f"selftest: infer ms p50 {p50:.1f}  p95 {p95:.1f}  max {max(infer_ms):.1f}  "
            f"(decode p50 {np.percentile(decode_ms, 50):.2f} ms, {len(infer_ms)} plans"
            + (f"; first warmup plan {first:.1f} ms, with compile, kernels and the caption)" if first else ")")
        )
    if cuda:
        gib = 2**30
        say(
            f"selftest: GPU {torch.cuda.get_device_name()}: peak {peak_load / gib:.2f} GiB during load and warmup, "
            f"peak {torch.cuda.max_memory_allocated() / gib:.2f} GiB during plans, "
            f"{torch.cuda.memory_allocated() / gib:.2f} GiB allocated"
        )
    if outs:
        last = outs[-1]
        spread = max(float(np.abs(o - outs[0]).max()) for o in outs)
        say(f"selftest: actions {last.shape} {last.dtype}, max |plan k - plan 1| {spread:.4f}")
        first, final = (np.round(last[i].astype(np.float64), 2).tolist() for i in (0, -1))
        say(f"selftest: action 0 {first}, action {chunk - 1} {final}")
    ok = ok and len(outs) == n and all(o.shape == (chunk, 6) and np.isfinite(o).all() for o in outs)
    print("PASS" if ok else "FAIL", flush=True)
    return ok


# ---- CLI ----------------------------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SO-101 policy server (so101.v1, docs/architecture.md).")
    p.add_argument("--engine", choices=NAMES, default="flux-action")
    p.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1 and reach it through ssh -L")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Hub repo or local policy directory")
    p.add_argument("--revision", default=DEFAULT_REVISION, help="Hub commit of the checkpoint ('' = main)")
    p.add_argument(
        "--base-revision",
        default=DEFAULT_BASE_REVISION,
        help="commit of black-forest-labs/flux-3-action-base for encoder specs the checkpoint config leaves "
        "unpinned; a config that pins another commit is an error ('' = whatever the checkpoint config says)",
    )
    p.add_argument(
        "--force-base-revision",
        action="store_true",
        help="load --base-revision even where the checkpoint config pins a different commit",
    )
    p.add_argument(
        "--guidance",
        nargs=2,
        type=float,
        metavar=("VIDEO", "ACTION"),
        help="override the checkpoint's guidance (3 3); 1 1 skips the unconditional pass, about half the compute",
    )
    p.add_argument("--steps", type=int, help="override the checkpoint's sampling steps (4)")
    p.add_argument(
        "--norm-stats",
        metavar="JSON",
        help="state/action q01/q99 (hello keys state_q01 ... or {state: {q01, q99}, action: {...}}) that replace the "
        "checkpoint's normalization, e.g. new-calibration stats for an arm in native LeRobot degrees. With --adapter "
        "it only checks that they equal the adapter's own and refuses otherwise",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--compile",
        action="store_true",
        help="flux-action: torch.compile the DiT (experimental; eager again if warmup fails); lerobot: compile_model",
    )
    p.add_argument("--adapter", help="PEFT/LoRA adapter directory (lerobot engine)")
    p.add_argument("--text-encoder-on-gpu", action="store_true", help="keep Qwen on the GPU (default: offloaded)")
    p.add_argument("--fake-mode", choices=FAKE_MODES, default="pickplace")
    p.add_argument("--fake-latency-ms", type=float, default=0.0)
    p.add_argument("--fake-jitter-ms", type=float, default=0.0)
    p.add_argument("--fake-period-s", type=float, help="pickplace cycle in seconds (default 9); every phase scales")
    p.add_argument("--log-dir", help="also write one JSON line per request to DIR (no images)")
    p.add_argument(
        "--warmup-task",
        action="append",
        help=f"caption to warm up and cache; repeat it for each instruction (default {DEFAULT_TASK!r})",
    )
    p.add_argument("--selftest", action="store_true", help="time --n plans on a synthetic window, print PASS/FAIL")
    p.add_argument("--n", type=int, default=50, help="plans in --selftest")
    args = p.parse_args(argv)
    if args.steps is not None and args.steps < 1:
        p.error("--steps must be >= 1")
    if args.guidance is not None and not all(math.isfinite(g) for g in args.guidance):
        p.error("--guidance takes two finite numbers")
    return args


def engine_kwargs(args: argparse.Namespace, norm_stats: dict | None = None) -> dict:
    sampling = {"guidance": tuple(args.guidance) if args.guidance else None, "steps": args.steps}
    if args.engine == "fake":
        kw = {"mode": args.fake_mode, "latency_ms": args.fake_latency_ms, "jitter_ms": args.fake_jitter_ms, **sampling}
        if args.fake_period_s is not None:
            kw["period_s"] = args.fake_period_s
        return kw
    kw = {
        "checkpoint": args.checkpoint,
        "revision": args.revision or None,
        "base_revision": args.base_revision or None,
        "force_base_revision": args.force_base_revision,
        "device": args.device,
        "compile": args.compile,
        "offload_text_encoder": not args.text_encoder_on_gpu,
        "norm_stats": norm_stats,
        **sampling,
    }
    if args.engine == "lerobot":
        kw["adapter"] = args.adapter
    return kw


def main(argv=None) -> int:
    args = parse_args(argv)
    tasks = args.warmup_task or [DEFAULT_TASK]
    if args.adapter and args.engine != "lerobot":
        print("ERROR: --adapter needs --engine lerobot", file=sys.stderr)
        return 2
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        say(f"WARNING: binding {args.host}; this server is meant to stay behind the ssh tunnel")
    if args.engine == "fake":
        say(f"so101 server: engine fake ({args.fake_mode})")
    else:
        rev = f"@{args.revision[:8]}" if args.revision else ""
        say(f"so101 server: engine {args.engine}, {args.checkpoint}{rev}, device {args.device}")
    norm_stats, norm_info = None, None
    if args.norm_stats:
        try:
            norm_stats, digest = load_norm_stats(args.norm_stats)
        except (OSError, ValueError) as e:
            say(f"ERROR: --norm-stats: {e}")
            if args.selftest:
                print("FAIL", flush=True)
            return 2
        norm_info = {"path": args.norm_stats, "sha256": digest, "streams": sorted(norm_stats),
                     "mode": "ignored (fake engine)" if args.engine == "fake"
                     else ("checked against the adapter" if args.adapter else "override")}
        say(f"norm stats: {args.norm_stats} sha256 {digest[:12]} ({', '.join(sorted(norm_stats))}), {norm_info['mode']}")
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="so101-infer")
    try:
        engine = make_engine(args.engine, **engine_kwargs(args, norm_stats))
        engine.meta["norm_stats"] = norm_info  # None: the checkpoint's own normalization
        for task in tasks:  # on the worker thread, where every later plan runs
            executor.submit(engine.warmup, task).result()
    except Exception as e:
        traceback.print_exc()
        say(f"ERROR: could not build or warm up engine {args.engine}: {type(e).__name__}: {e}")
        if isinstance(e, ImportError):
            say(
                "hint: this Python lacks the engine's packages. Start through infra/gpu/so101/serve.sh, and build "
                "the env with infra/gpu/bootstrap.sh (flux-action) or infra/gpu/so101/bootstrap-lerobot.sh (lerobot)"
            )
        if args.selftest:
            print("FAIL", flush=True)
        return 1
    say(f"images: jpeg/png decoded with {codec_summary()}")
    if args.selftest:
        return 0 if selftest(engine, executor, args.n, tasks[0]) else 1
    log = JsonlLog(args.log_dir) if args.log_dir else None
    server = PolicyServer(engine, executor, log)
    say("hello " + json.dumps(server.hello_message()))
    if log is not None:
        say(f"logging requests to {log.path}")
        log.write("start", hello=server.hello_message(), argv=sys.argv[1:])
    try:
        asyncio.run(server.serve(args.host, args.port))
    except KeyboardInterrupt:
        say("stopped (Ctrl-C)")
    except OSError as e:
        say(f"ERROR: cannot listen on {args.host}:{args.port}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
