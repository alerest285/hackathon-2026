"""Builds the robot pieces, the ControlLoop and the runner, runs, and always tears down in order.

Shared by run_hanoi.py and the fake end-to-end test, so both take the same path.
"""

from __future__ import annotations

import logging
import platform
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .hanoi import all_prompts
from .keys import KeyListener
from .recorder import git_info, make_run_dir, utc_now
from .runner import HanoiRunner

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = EVAL_DIR.parent


@dataclass
class SessionOptions:
    fake: bool = False
    fake_hardware: bool = False           # FakeRobot + FakeCameras with the real NetClient to server.url
    fake_latency_ms: float = 400.0
    fake_jitter_ms: float = 50.0
    fake_cycle_s: float | None = None     # FakeNet pickplace cycle; None keeps the fake's default
    live: bool = False
    episodes: int | None = None
    sanity: int | None = None             # repeats per move; set = a sanity run
    out: Path = field(default_factory=lambda: EVAL_DIR / "runs")
    argv: list[str] | None = None

    @property
    def dry_run(self) -> bool:
        """Commands reach the motors only with --live. A fake robot always takes them (nothing moves)."""
        return not (self.live or self.fake or self.fake_hardware)


def loop_config(cfg, dry_run: bool):
    """so101_client LoopConfig from the [loop] table."""
    from so101_client.loop import LoopConfig

    lp = cfg.loop
    values = {
        "fps": lp.fps,
        "mode": lp.mode,
        "period": lp.period,
        "crossfade": lp.crossfade,
        "exec_horizon": lp.exec_horizon,
        "chunk": lp.chunk,
        "step_max_deg": lp.step_max_deg,
        "joint_limits": None if lp.joint_limits is None else tuple(tuple(pair) for pair in lp.joint_limits),
        "dry_run": dry_run,
        "spin_margin_ms": lp.spin_margin_ms,
    }
    known = {f.name for f in fields(LoopConfig)}
    missing = sorted(set(values) - known)
    if missing:
        logger.warning("LoopConfig has no field(s) %s; they are not applied", missing)
    return LoopConfig(**{k: v for k, v in values.items() if k in known})


def unwarmed_prompts(hello: dict, cfg) -> list[str]:
    """Prompts of this config the server did not warm up (hello `warm_tasks`); [] when it does not say."""
    warm = hello.get("warm_tasks")
    if warm is None:
        return []
    return [text for text in all_prompts(cfg) if text not in set(warm)]


def build_rig(cfg, opts: SessionOptions):
    """(robot, cameras, net): all fakes with opts.fake; FakeRobot + FakeCameras with the real server with
    opts.fake_hardware; otherwise the SO-101, the two cameras and the server."""
    cam = cfg.cameras
    sources = {"scene": cam.scene, "wrist": cam.wrist}
    camera_args = dict(width=cam.width, height=cam.height, fps=cam.fps, image_size=cam.image_size,
                       image_format=cam.image_format, jpeg_quality=cam.jpeg_quality)
    if opts.fake or opts.fake_hardware:
        from so101_client.fakes import FakeCameras, FakeNet, FakeRobot

        robot, cameras = FakeRobot(), FakeCameras(sources, **camera_args)
        if opts.fake_hardware:
            from so101_client.net import NetClient

            return robot, cameras, NetClient(cfg.server.url)
        net_args = {"cycle_s": opts.fake_cycle_s} if opts.fake_cycle_s else {}
        return (
            robot,
            cameras,
            FakeNet(mode="pickplace", latency_ms=opts.fake_latency_ms, jitter_ms=opts.fake_jitter_ms, **net_args),
        )
    if not cfg.robot.port:
        raise SystemExit("no serial port: set robot.port in the config or pass --port (or use --fake)")
    from so101_client.cameras import Cameras
    from so101_client.net import NetClient
    from so101_client.robot_io import SO101

    return SO101(cfg.robot.port, cfg.robot.id, use_degrees=True), Cameras(sources, **camera_args), NetClient(cfg.server.url)


def run_session(cfg, opts: SessionOptions, keys_source=None, log=print) -> tuple[Path | None, HanoiRunner | None]:
    """Connect, run the episodes or the sanity run, tear down. Returns the run directory and the runner."""
    from so101_client.loop import ControlLoop

    loop_cfg = loop_config(cfg, opts.dry_run)
    robot, cameras, net = build_rig(cfg, opts)
    keys = KeyListener(keys_source)
    loop = runner = run_dir = None
    connected: list = []
    try:
        keys.start()  # first: without a terminal for the keys there is no point connecting anything
        robot.connect()
        connected.append(("robot", robot))
        cameras.start()
        connected.append(("cameras", cameras))
        hello = net.connect()
        connected.append(("net", net))
        if opts.fake:
            what = "fake robot and policy"
        elif opts.fake_hardware:
            what = f"fake robot and cameras, real server at {cfg.server.url}"
        else:
            what = "LIVE arm: commands are written" if opts.live else "dry run: commands are logged, not written"
        log(f"connected: {what}; server {hello.get('engine')} {hello.get('checkpoint') or ''}".rstrip())
        cold = unwarmed_prompts(hello, cfg)
        if cold:
            log(f"WARNING: the server did not warm up {len(cold)} of this config's prompts (first: {cold[0]!r}), so "
                "the first plan of each encodes its caption. Restart it with the command from run_hanoi.py "
                "--print-warmup")
        label = cfg.label + ("-sanity" if opts.sanity else ("-recover" if cfg.protocol == "recover" else "")) + (
            "-demo" if cfg.mode == "demo" else ""
        )
        run_dir = make_run_dir(opts.out, label)
        loop = ControlLoop(robot, cameras, net, loop_cfg)
        run_info = {
            "name": run_dir.name,
            "created_utc": utc_now(),
            "argv": opts.argv,
            "fake": opts.fake,
            "fake_hardware": opts.fake_hardware,
            "live": opts.live,
            "dry_run": opts.dry_run,
            "fake_net": {"latency_ms": opts.fake_latency_ms, "jitter_ms": opts.fake_jitter_ms, "cycle_s": opts.fake_cycle_s}
            if opts.fake else None,
            "git": git_info(REPO_DIR),
            "server_hello": hello,
            "server_url": None if opts.fake else cfg.server.url,
            "loop": asdict(loop_cfg),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
        runner = HanoiRunner(loop, keys, cfg, run_dir, run_info=run_info, log=log)
        log(f"run directory: {run_dir}")
        loop.start()
        if opts.sanity:
            runner.run_sanity(opts.sanity)
        else:
            runner.run_episodes(opts.episodes)
    finally:
        keys.stop()
        if loop is not None:
            try:
                loop.stop()
            except Exception:  # noqa: BLE001
                logger.exception("loop.stop failed")
        for name, part in reversed(connected):
            try:
                if name == "robot":
                    part.disconnect()
                elif name == "cameras":
                    part.stop()
                else:
                    part.close()
            except Exception:  # noqa: BLE001
                logger.exception("%s teardown failed", name)
    return run_dir, runner
