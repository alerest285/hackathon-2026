"""Builds the robot pieces, the ControlLoop and the runner, runs, and always tears down in order.

Shared by run_hanoi.py and the fake end-to-end test, so both take the same path.

The real arm gets the same safety stack as robot/run.py and console/: SO101 with a NativeGuard in the arm's own
degrees (limits from loop.joint_limits / --limits, required for --live), wrapped in console/mapping.py's
MappedRobot when robot.mapping names a joint map (required for --live), so the loop, the history and the server
work in the checkpoint's convention. The loop's limits are mapped too; the runner maps state and cmd back to the
arm's degrees for the step-end thresholds, ticks.jsonl and the dataset. Chunks stay as the server sent them.
"""

from __future__ import annotations

import logging
import platform
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .dataset import HanoiDataset, load_joint_map, map_identity_problems
from .hanoi import all_prompts
from .keys import KeyListener
from .recorder import git_info, make_run_dir, utc_now
from .runner import HanoiRunner
from .task import TaskRunner

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
    record: bool | None = None            # LeRobotDataset: None = live runs when record.lerobot; True/False force it
    task: bool = False                    # run_task.py: TaskRunner on cfg.task instead of the Hanoi sequencer
    attempts: int | None = None           # task runs: None = task.attempts
    out: Path = field(default_factory=lambda: EVAL_DIR / "runs")
    argv: list[str] | None = None

    @property
    def dry_run(self) -> bool:
        """Commands reach the motors only with --live. A fake robot always takes them (nothing moves)."""
        return not (self.live or self.fake or self.fake_hardware)


def loop_config(cfg, dry_run: bool, joint_limits="config"):
    """so101_client LoopConfig from the [loop] table. joint_limits overrides loop.joint_limits (the loop's units)."""
    from so101_client.loop import LoopConfig

    lp = cfg.loop
    if joint_limits == "config":
        joint_limits = None if lp.joint_limits is None else tuple(tuple(pair) for pair in lp.joint_limits)
    values = {
        "fps": lp.fps,
        "mode": lp.mode,
        "period": lp.period,
        "crossfade": lp.crossfade,
        "exec_horizon": lp.exec_horizon,
        "chunk": lp.chunk,
        "step_max_deg": lp.step_max_deg,
        "joint_limits": joint_limits,
        "dry_run": dry_run,
        "spin_margin_ms": lp.spin_margin_ms,
    }
    known = {f.name for f in fields(LoopConfig)}
    missing = sorted(set(values) - known)
    if missing:
        logger.warning("LoopConfig has no field(s) %s; they are not applied", missing)
    return LoopConfig(**{k: v for k, v in values.items() if k in known})


def unwarmed_prompts(hello: dict, cfg, prompts: list[str] | None = None) -> list[str]:
    """Prompts of this config (or `prompts`) the server did not warm up (hello `warm_tasks`); [] when it does not
    say."""
    warm = hello.get("warm_tasks")
    if warm is None:
        return []
    return [text for text in (all_prompts(cfg) if prompts is None else prompts) if text not in set(warm)]


@dataclass
class Rig:
    robot: object
    cameras: object
    net: object
    joint_map: object = None              # console/mapping.py JointMap when the arm is mapped
    joint_map_path: str | None = None
    native_limits: tuple | None = None    # 5 x (lo, hi) in the arm's own degrees
    loop_limits: tuple | None = None      # the same in the loop's units (the checkpoint's convention when mapped)
    guard: object = None                  # the SO101's NativeGuard
    warnings: list = field(default_factory=list)


def build_rig(cfg, opts: SessionOptions) -> Rig:
    """All fakes with opts.fake; FakeRobot + FakeCameras with the real server with opts.fake_hardware; otherwise
    the SO-101 behind its native guard and joint map, the two cameras and the server."""
    cam = cfg.cameras
    sources = {"scene": cam.scene, "wrist": cam.wrist}
    camera_args = dict(width=cam.width, height=cam.height, fps=cam.fps, image_size=cam.image_size,
                       image_format=cam.image_format, jpeg_quality=cam.jpeg_quality)
    limits = None if cfg.loop.joint_limits is None else tuple(tuple(map(float, p)) for p in cfg.loop.joint_limits)
    if opts.fake or opts.fake_hardware:
        from so101_client.fakes import FakeCameras, FakeNet, FakeRobot

        robot, cameras = FakeRobot(), FakeCameras(sources, **camera_args)
        if opts.fake_hardware:
            from so101_client.net import NetClient

            net = NetClient(cfg.server.url)
        else:
            net_args = {"cycle_s": opts.fake_cycle_s} if opts.fake_cycle_s else {}
            net = FakeNet(mode="pickplace", latency_ms=opts.fake_latency_ms, jitter_ms=opts.fake_jitter_ms, **net_args)
        return Rig(robot, cameras, net, native_limits=limits, loop_limits=limits)
    if not cfg.robot.port:
        raise SystemExit("no serial port: set robot.port in the config or pass --port (or use --fake)")
    if opts.live and limits is None:
        raise SystemExit("--live needs joint limits in the arm's own degrees: --limits <file> (robot/readout.py "
                         "--save-limits or console/fit_reference.py) or loop.joint_limits in the config")
    from so101_client.cameras import Cameras
    from so101_client.net import NetClient
    from so101_client.robot_io import SO101
    from so101_client.safety import NativeGuard

    rig = Rig(None, None, None, native_limits=limits, loop_limits=limits, guard=NativeGuard(limits=limits))
    robot = SO101(cfg.robot.port, cfg.robot.id, use_degrees=True, guard=rig.guard)
    if cfg.robot.mapping:
        path = Path(cfg.robot.mapping).expanduser()
        if path.is_file():
            jmap, mapped = load_joint_map(path)
            problems = map_identity_problems(jmap.meta, cfg.robot.id, robot.calibration_path)
            if problems and opts.live:
                raise SystemExit(f"the joint map {path} does not fit this arm: {'; '.join(problems)}")
            rig.warnings += [f"the joint map {path} may not fit this arm: {problem}" for problem in problems]
            rig.joint_map, rig.joint_map_path = jmap, str(path)
            rig.loop_limits = jmap.limits_to_model(limits)
            robot = mapped(robot, jmap)
        elif opts.live:
            raise SystemExit(f"no joint map at {path}: run console/fit_reference.py (T21) first. robot.mapping = \"\" "
                             "sends raw LeRobot degrees, which the checkpoint was not trained on")
        else:
            rig.warnings.append(f"no joint map at {path}: this dry run sends raw LeRobot degrees")
    rig.robot = robot
    rig.cameras, rig.net = Cameras(sources, **camera_args), NetClient(cfg.server.url)
    return rig


def open_dataset(cfg, opts: SessionOptions) -> HanoiDataset | None:
    """The LeRobotDataset writer when this run records (live, or forced with --record)."""
    record = opts.record if opts.record is not None else (cfg.record.lerobot and opts.live)
    if not record:
        return None
    r = cfg.record
    run_mode = "fake" if opts.fake or opts.fake_hardware else ("live" if opts.live else "dry")
    return HanoiDataset(r.repo_id, r.root, robot_type=r.robot_type, fps=round(cfg.loop.fps), run_mode=run_mode)


def run_session(cfg, opts: SessionOptions, keys_source=None, log=print) -> tuple[Path | None, HanoiRunner | None]:
    """Connect, run the episodes or the sanity run, tear down. Returns the run directory and the runner."""
    from so101_client.loop import ControlLoop

    rig = build_rig(cfg, opts)
    robot, cameras, net = rig.robot, rig.cameras, rig.net
    loop_cfg = loop_config(cfg, opts.dry_run, joint_limits=rig.loop_limits)
    keys = KeyListener(keys_source)
    loop = runner = run_dir = dataset = None
    connected: list = []
    try:
        keys.start()  # first: without a terminal for the keys there is no point connecting anything
        for warning in rig.warnings:
            log(f"WARNING: {warning}")
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
        cold = unwarmed_prompts(hello, cfg, [cfg.task.instruction] if opts.task else None)
        if cold:
            log(f"WARNING: the server did not warm up {len(cold)} of this config's prompts (first: {cold[0]!r}), so "
                "the first plan of each encodes its caption. Restart it with the command from run_hanoi.py "
                "--print-warmup")
        if opts.task:
            label = cfg.label + "-task"
        else:
            label = cfg.label + ("-sanity" if opts.sanity else ("-recover" if cfg.protocol == "recover" else "")) + (
                "-demo" if cfg.mode == "demo" else ""
            )
        run_dir = make_run_dir(opts.out, label)
        dataset = open_dataset(cfg, opts)
        if dataset is not None:
            log(f"recording LeRobot episodes to {dataset.root} ({dataset.repo_id})")
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
            "joint_map": None if rig.joint_map is None else {"path": rig.joint_map_path, **rig.joint_map.to_json()},
            "native_limits": rig.native_limits,
            "native_guard": None if rig.guard is None else asdict(rig.guard),
            "units": "ticks.jsonl state/cmd in the arm's own degrees; chunks.npz actions as the server sent them"
            + (" (the checkpoint's convention, see joint_map)" if rig.joint_map is not None else ""),
            "lerobot_dataset": None if dataset is None else {"repo_id": dataset.repo_id, "root": str(dataset.root)},
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
        to_native = rig.joint_map.to_ours if rig.joint_map is not None else None
        runner_cls = TaskRunner if opts.task else HanoiRunner
        runner = runner_cls(loop, keys, cfg, run_dir, run_info=run_info, log=log, to_native=to_native, dataset=dataset)
        log(f"run directory: {run_dir}")
        loop.start()
        if opts.task:
            runner.run(opts.attempts)
        elif opts.sanity:
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
        if dataset is not None:
            try:
                dataset.close()  # saves nothing new: every attempt already ended its episode; finalizes the files
                log(f"LeRobot dataset: {dataset.status()}")
            except Exception:  # noqa: BLE001
                logger.exception("dataset close failed")
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
