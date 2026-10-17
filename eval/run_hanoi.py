"""Run Tower of Hanoi episodes with FLUX 3 Action on the SO-101 and write eval/runs/<UTC stamp>-<label>/.

    uv run --project eval python eval/run_hanoi.py --config eval/config/hanoi.toml --fake --episodes 1
    uv run --project eval python eval/run_hanoi.py --config eval/config/hanoi.toml --sanity 2 --port /dev/tty.usbmodem...
    uv run --project eval python eval/run_hanoi.py --config eval/config/hanoi.toml --live --port /dev/tty.usbmodem...

Without --live nothing is written to the motors (dry run); --fake swaps in FakeRobot, FakeCameras and
FakeNet(pickplace); --fake-hardware swaps in FakeRobot and FakeCameras and keeps the real server (--server).
Keys: s success, f fail, r retry, space pause/hold, q quit, enter continue.
Protocol and outputs: eval/README.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from pathlib import Path

from so101_client import STATE_NAMES

from hanoi_eval.config import ConfigError, load_config
from hanoi_eval.keys import unattended_keys
from hanoi_eval.session import EVAL_DIR, SessionOptions, run_session


def camera_source(text: str) -> int | str:
    return int(text) if text.strip().isdigit() else text


def read_limits(path: Path) -> list[list[float]]:
    """{joint: [lo, hi]} for the five joints, as robot/readout.py --save-limits writes it."""
    data = json.loads(path.read_text())
    return [[float(data[name][0]), float(data[name][1])] for name in STATE_NAMES[:5]]


def parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=str(EVAL_DIR / "config" / "hanoi.toml"))
    p.add_argument("--episodes", type=int, help="number of full episodes (config: episodes)")
    p.add_argument("--mode", choices=["eval", "demo"], help="eval: operator verdicts | demo: auto end = success")
    p.add_argument("--protocol", choices=["strict", "recover"])
    p.add_argument("--sanity", type=int, metavar="N", help="sanity run: every legal single move N times")
    p.add_argument("--label", help="run directory suffix (config: label)")
    p.add_argument("--fake", action="store_true", help="FakeRobot + FakeCameras + FakeNet(pickplace), no hardware")
    p.add_argument("--fake-hardware", action="store_true",
                   help="FakeRobot + FakeCameras with the real so101.v1 server (--server): the socket path, no arm")
    p.add_argument("--fake-latency-ms", type=float, default=400.0)
    p.add_argument("--fake-cycle-s", type=float, help="FakeNet pickplace cycle length (default: the fake's)")
    p.add_argument("--live", action="store_true", help="write commands to the motors (default: dry run)")
    p.add_argument("--server", help="so101.v1 server URL (config: server.url)")
    p.add_argument("--port", help="SO-101 serial port (config: robot.port)")
    p.add_argument("--robot-id", help="LeRobot calibration id (config: robot.id)")
    p.add_argument("--scene", type=camera_source, help="scene camera index or path (config: cameras.scene)")
    p.add_argument("--wrist", type=camera_source, help="wrist camera index or path (config: cameras.wrist)")
    p.add_argument("--loop-mode", choices=["prefetch", "sync"], help="client loop mode (config: loop.mode)")
    p.add_argument("--limits", type=Path, help="joint limits JSON from robot/readout.py --save-limits (config: loop.joint_limits)")
    p.add_argument("--no-video", action="store_true", help="skip scene.mp4 / wrist.mp4")
    p.add_argument("--out", default=str(EVAL_DIR / "runs"), help="parent directory of the run directories")
    p.add_argument("--unattended", "--auto-confirm", dest="unattended", action="store_true",
                   help="test-only, with --fake or --fake-hardware: enter at every prompt, verdict success after an "
                        "auto end and fail after a timeout (runs without a terminal)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    a = parse(raw_argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if a.unattended and not (a.fake or a.fake_hardware):
        print("--unattended confirms auto ends without a person, so it only runs with --fake or --fake-hardware",
              file=sys.stderr)
        return 2
    if a.fake and a.fake_hardware:
        print("--fake already replaces the server; --fake-hardware keeps the real one: pick one", file=sys.stderr)
        return 2
    if a.live and (a.fake or a.fake_hardware):
        print("--live excludes --fake and --fake-hardware", file=sys.stderr)
        return 2
    overrides = {
        "episodes": a.episodes,
        "mode": a.mode,
        "protocol": a.protocol,
        "label": a.label,
        "server.url": a.server,
        "robot.port": a.port,
        "robot.id": a.robot_id,
        "cameras.scene": a.scene,
        "cameras.wrist": a.wrist,
        "loop.mode": a.loop_mode,
        "video.enabled": False if a.no_video else None,
    }
    try:
        if a.limits:
            overrides["loop.joint_limits"] = read_limits(a.limits)
        cfg = load_config(a.config, overrides)
    except (ConfigError, OSError, KeyError, ValueError) as e:
        print(f"{a.config}: {e}", file=sys.stderr)
        return 2
    if cfg.loop.joint_limits:
        print(f"joint limits (deg): {cfg.loop.joint_limits}{f' from {a.limits}' if a.limits else ''}")
    print(f"config {a.config}: {cfg.n} pieces {cfg.pieces}, {cfg.places[cfg.start]} -> {cfg.places[cfg.goal]}, "
          f"mode {cfg.mode}, protocol {'strict (sanity)' if a.sanity else cfg.protocol}, loop {cfg.loop.mode}")

    def on_term(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_term)
    keys = unattended_keys() if a.unattended else None
    opts = SessionOptions(
        fake=a.fake,
        fake_hardware=a.fake_hardware,
        fake_latency_ms=a.fake_latency_ms,
        fake_cycle_s=a.fake_cycle_s,
        live=a.live,
        episodes=a.episodes,
        sanity=a.sanity,
        out=Path(a.out),
        argv=["run_hanoi.py", *raw_argv],
    )
    try:
        run_dir, runner = run_session(cfg, opts, keys_source=keys)
    except KeyboardInterrupt:
        print("interrupted; the hardware was released", file=sys.stderr)
        return 130
    except (RuntimeError, OSError) as e:  # OSError includes ConnectionError: no server behind the URL
        print(f"error: {e}", file=sys.stderr)
        return 1
    if run_dir is None or runner is None:
        return 1
    print()
    print((run_dir / "summary.md").read_text())
    n_ep, n_at = len(runner.episodes), len(runner.attempts)
    print(f"wrote {run_dir} ({n_ep} episode{'s' * (n_ep != 1)}, {n_at} attempt{'s' * (n_at != 1)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
