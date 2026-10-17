"""One instruction, closed loop, for sanity runs and the T06 dry run. Dry run unless --live.

    uv run --python 3.13 python run.py --task "pick up the small red disk and place it on the right peg" \
        --port /dev/tty.usbmodemXXXX --robot-id so101_hanoi --scene 0 --wrist 1

Fakes stand in for any part: --fake-robot, --fake-cameras, --fake-net [hold|sine|pickplace]. With
--fake-robot the commands go to the fake arm, which follows them. Each run writes runs/<stamp>/ticks.jsonl,
plans.jsonl and summary.json, prints the timing summary, and ends with PASS/FAIL against: tick lateness
p99 <= 2 ms, zero holds after the first chunk (prefetch), turnaround p95 <= 533 ms.
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import time
from pathlib import Path

from so101_client import STATE_NAMES
from so101_client.loop import ControlLoop, LoopConfig

HERE = Path(__file__).resolve().parent
LATENESS_P99_MS = 2.0
TURNAROUND_P95_MS = 533.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", required=True, help="the instruction, e.g. 'pick up the small red disk and place it on the right peg'")
    p.add_argument("--server", default="ws://localhost:8000", help="so101.v1 server (ssh -L 8000 tunnel)")
    p.add_argument("--mode", choices=("prefetch", "sync"), default="prefetch")
    p.add_argument("--live", action="store_true", help="write commands to the real arm (default: dry run)")
    p.add_argument("--yes", action="store_true", help="skip the ENTER confirmation before a live run")
    p.add_argument("--fake-robot", action="store_true")
    p.add_argument("--fake-cameras", action="store_true")
    p.add_argument("--fake-net", nargs="?", const="pickplace", choices=("hold", "sine", "pickplace"), default=None)
    p.add_argument("--latency-ms", type=float, default=400.0, help="fake net turnaround")
    p.add_argument("--jitter-ms", type=float, default=50.0, help="fake net jitter")
    p.add_argument("--port", help="SO-101 serial port (lerobot-find-port)")
    p.add_argument("--robot-id", help="the id used with lerobot-calibrate")
    p.add_argument("--scene", default="0", help="scene camera index or path")
    p.add_argument("--wrist", default="1", help="wrist camera index or path")
    p.add_argument("--image-format", choices=("jpeg", "png", "raw"), default="jpeg")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--period", type=int, default=16, help="P: ticks between prefetch requests")
    p.add_argument("--crossfade", type=int, default=6, help="K: seam crossfade in ticks")
    p.add_argument("--step-max", type=float, default=6.0, help="per-tick joint clamp, degrees")
    p.add_argument("--limits", type=Path,
                   help="JSON {joint: [lo, hi]} for the 5 joints (readout.py --save-limits); required with --live")
    p.add_argument("--log-dir", type=Path, help="default: robot/runs/<stamp>/")
    a = p.parse_args(argv)
    if not a.fake_robot and not (a.port and a.robot_id):
        p.error("--port and --robot-id are required unless --fake-robot")
    if a.live and not a.fake_robot and a.limits is None:
        p.error("--live needs --limits: measure them first with readout.py --save-limits (codex audit, 2026-10-17)")
    return a


def load_limits(path: Path | None):
    """{joint: [lo, hi]} for the 5 joints (readout.py --save-limits). Raises ValueError unless lo < hi."""
    if path is None:
        return None
    data = json.loads(path.read_text())
    limits = tuple((float(data[name][0]), float(data[name][1])) for name in STATE_NAMES[:5])
    bad = [name for name, (lo, hi) in zip(STATE_NAMES, limits) if not lo < hi]
    if bad:
        raise ValueError(f"{path}: each joint needs lo < hi, not for {', '.join(bad)}")
    return limits


def build(a):
    if a.fake_robot:
        from so101_client.fakes import FakeRobot

        robot = FakeRobot()
    else:
        from so101_client.robot_io import SO101
        from so101_client.safety import NativeGuard

        # native-unit tripwires below any joint map (safety.py); the limits also bound the guard
        robot = SO101(a.port, a.robot_id, guard=NativeGuard(limits=load_limits(getattr(a, "limits", None))))
    camera_args = dict(image_size=a.image_size, image_format=a.image_format, jpeg_quality=a.jpeg_quality)
    if a.fake_cameras:
        from so101_client.fakes import FakeCameras

        cameras = FakeCameras(**camera_args)
    else:
        from so101_client.cameras import Cameras

        cameras = Cameras({"scene": a.scene, "wrist": a.wrist}, **camera_args)
    if a.fake_net:
        from so101_client.fakes import FakeNet

        net = FakeNet(a.fake_net, latency_ms=a.latency_ms, jitter_ms=a.jitter_ms)
    else:
        from so101_client.net import NetClient

        net = NetClient(a.server)
    return robot, cameras, net


def tick_record(info) -> dict:
    chunk = info.chunk
    return {
        "k": info.k, "t": round(info.t, 6), "state": [round(float(v), 4) for v in info.state],
        "cmd": [round(float(v), 4) for v in info.cmd], "held": info.held, "written": info.written,
        "lateness_ms": round(info.lateness_ms, 3),
        "frame_age_ms": {cam: round(v, 2) for cam, v in info.frame_age_ms.items()}, "remaining": info.remaining,
        "task": info.task, "chunk_id": chunk["id"] if chunk else None,
        "chunk_accepted": chunk["accepted"] if chunk else None,
    }


def plan_record(info) -> dict:
    c = info.chunk
    return {
        "k": info.k, "id": c["id"], "tick": c["tick"], "d": c["d"], "accepted": c["accepted"], "start": c["start"],
        "reason": c["reason"], "turnaround_ms": round(c["turnaround_ms"], 2),
        "server_ms": None if c["server_ms"] is None else round(float(c["server_ms"]), 2),
        "rtt_ms": round(c["rtt_ms"], 2), "actions": [[round(float(v), 4) for v in row] for row in c["actions"]],
    }


def evaluate(stats: dict, mode: str) -> list[tuple[str, bool | None, str]]:
    lat, ta = stats["lateness_p99_ms"], stats["turnaround_p95_ms"]
    checks = [
        (f"tick lateness p99 <= {LATENESS_P99_MS:g} ms", lat is not None and lat <= LATENESS_P99_MS, f"{lat} ms"),
        ("zero holds after the first chunk",
         None if mode == "sync" else stats["starved"] == 0,
         "n/a: sync holds by design" if mode == "sync" else f"{stats['starved']} held ticks"),
        (f"turnaround p95 <= {TURNAROUND_P95_MS:g} ms", ta is not None and ta <= TURNAROUND_P95_MS, f"{ta} ms"),
        ("at least one chunk accepted", stats["plans"] > 0, f"{stats['plans']} plans"),
        ("control loop ran without error", stats["error"] is None, str(stats["error"])),
    ]
    return checks


def main(argv=None) -> int:
    a = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    try:
        limits = load_limits(a.limits)
    except (OSError, KeyError, ValueError) as e:
        print(f"--limits: {type(e).__name__}: {e}")
        return 2
    log_dir = a.log_dir or HERE / "runs" / stamp
    log_dir.mkdir(parents=True, exist_ok=True)
    dry_run = not (a.live or a.fake_robot)
    cfg = LoopConfig(mode=a.mode, period=a.period, crossfade=a.crossfade, step_max_deg=a.step_max,
                     joint_limits=limits, dry_run=dry_run)
    robot, cameras, net = build(a)
    where = "fake robot" if a.fake_robot else f"SO-101 {a.robot_id} on {a.port}"
    print(f"run: task {a.task!r}, mode {a.mode}, {where}, "
          f"{'fake cameras' if a.fake_cameras else f'cameras scene={a.scene} wrist={a.wrist}'}, "
          f"{f'fake net {a.fake_net}' if a.fake_net else f'server {a.server}'}")
    if a.fake_robot:
        print("commands go to the fake robot, which follows them")
    elif dry_run:
        print("DRY RUN: commands are logged, not written to the arm")
    else:
        print("LIVE: the arm will move. Keep a hand on the power switch. Ctrl-C holds and exits with torque on.")
        if not a.yes:
            try:
                input("Press ENTER to start, Ctrl-C to abort: ")
            except KeyboardInterrupt:
                print("\naborted before start")
                return 1

    ticks: queue.SimpleQueue = queue.SimpleQueue()
    loop = ControlLoop(robot, cameras, net, cfg)
    loop.add_observer(ticks.put)
    try:
        loop.start()
    except Exception as e:
        print(f"could not start: {type(e).__name__}: {e}")
        return 1
    print(f"server hello: {loop.hello}")
    loop.set_task(a.task, reset_history=True)
    n_ticks = n_plans = 0
    interrupted = False
    with open(log_dir / "ticks.jsonl", "w") as ft, open(log_dir / "plans.jsonl", "w") as fp:
        def drain():
            nonlocal n_ticks, n_plans
            while True:
                try:
                    info = ticks.get_nowait()
                except queue.Empty:
                    return
                ft.write(json.dumps(tick_record(info)) + "\n")
                n_ticks += 1
                if info.chunk is not None:
                    fp.write(json.dumps(plan_record(info)) + "\n")
                    n_plans += 1

        t_end = time.perf_counter() + a.seconds
        next_print = time.perf_counter() + 5.0
        try:
            while time.perf_counter() < t_end and loop.running:
                time.sleep(0.1)
                drain()
                if time.perf_counter() >= next_print:
                    s = loop.stats()
                    print(f"  {n_ticks} ticks, {s['plans']} plans, {s['holds']} held ticks, "
                          f"turnaround p50 {s['turnaround_p50_ms']} ms, lateness p99 {s['lateness_p99_ms']} ms")
                    next_print += 5.0
        except KeyboardInterrupt:
            interrupted = True
            print("\nCtrl-C: holding the last command and stopping")
        loop.stop()
        drain()

    stats = loop.stats()
    checks = evaluate(stats, a.mode)
    passed = all(ok is not False for _, ok, _ in checks)
    summary = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, "dry_run": dry_run,
               "hello": loop.hello, "stats": stats, "interrupted": interrupted,
               "checks": [{"check": name, "ok": ok, "value": value} for name, ok, value in checks], "pass": passed}
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\ntiming summary")
    for key in ("ticks", "overruns", "lateness_p50_ms", "lateness_p99_ms", "lateness_max_ms", "requests", "plans",
                "holds", "starved", "bus_holds", "dropped", "stale", "errors", "timeouts", "clamped", "d_max",
                "turnaround_p50_ms",
                "turnaround_p95_ms", "turnaround_max_ms", "server_p50_ms", "server_p95_ms", "rtt_p50_ms", "rtt_p95_ms",
                "network_p50_ms", "network_p95_ms", "bus_errors", "write_errors", "stale_frames"):
        print(f"  {key:20s} {stats[key]}")
    print(f"\nwrote {n_ticks} ticks to {log_dir / 'ticks.jsonl'}")
    print(f"wrote {n_plans} plans to {log_dir / 'plans.jsonl'}")
    print(f"wrote {log_dir / 'summary.json'}")
    print()
    for name, ok, value in checks:
        print(f"  {'n/a ' if ok is None else ('ok  ' if ok else 'FAIL')}  {name}: {value}")
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
