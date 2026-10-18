"""Score one instruction over attempts with FLUX 3 Action on the SO-101/SO-100: attempts, successes, completed steps.

    uv run --project eval python eval/run_task.py --fake --attempts 2                       # no hardware
    uv run --project eval python eval/run_task.py --port /dev/ttyACM1 --robot-id my_follower_arm   # dry run
    uv run --project eval python eval/run_task.py --live --limits ~/hackathon-data/maps/so100_phys_limits.json \\
        --port /dev/ttyACM1 --robot-id my_follower_arm --label zero-shot
    uv run --project eval python eval/run_task.py --print-warmup     # the GPU serve commands for this instruction

Keys: enter starts an attempt; during it + (or =) one step done, - undo, space pause/resume, s success, f fail,
q quit. Without --live nothing is written to the motors (dry run); --fake swaps in FakeRobot, FakeCameras and
FakeNet; --fake-hardware keeps the real server. Protocol and outputs: eval/README.md, "Scored task runs".
"""

from __future__ import annotations

import argparse
import logging
import shlex
import signal
import sys
from pathlib import Path

from run_hanoi import camera_source, read_limits

from hanoi_eval.config import ConfigError, load_config
from hanoi_eval.keys import ScriptedKeys
from hanoi_eval.session import EVAL_DIR, SessionOptions, run_session


def serve_commands(instruction: str) -> str:
    """The GPU server commands that warm this instruction up: zero-shot and a LoRA adapter."""
    task = shlex.quote(instruction)
    return "\n".join([
        "# zero-shot (flux-action engine; --norm-stats <json> swaps the normalization, --checkpoint <dir> another base)",
        f"bash infra/gpu/so101/serve.sh --compile --warmup-task {task} --log-dir /workspace/so101-logs",
        "# a LoRA adapter (lerobot engine: the base and the normalization come with the adapter)",
        "bash infra/gpu/so101/serve.sh --engine lerobot --adapter <run>/checkpoints/last/pretrained_model --compile \\",
        f"  --warmup-task {task} --log-dir /workspace/so101-logs",
    ])


def unattended_task_keys(steps: int) -> ScriptedKeys:
    """Test-only operator (--unattended, fakes only): enter at every start, + until the task's steps are counted,
    then s; fail after a timeout."""
    pressed: dict = {}

    def step(prompt):
        k = pressed.get(prompt.info.get("attempt"), 0)
        pressed[prompt.info.get("attempt")] = k + 1
        return "+" if k < steps else "s"

    return ScriptedKeys({"start": "\n", "step": step, "verdict": "f"})


def parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=str(EVAL_DIR / "config" / "task.toml"))
    p.add_argument("--instruction", help="the exact caption the policy gets (config: task.instruction)")
    p.add_argument("--steps", type=int, help="countable sub-goals per attempt, e.g. objects to sort; 0 = not counted")
    p.add_argument("--attempts", type=int, help="attempts in this run (config: task.attempts)")
    p.add_argument("--attempt-timeout-s", type=float, help="active seconds before the verdict prompt")
    p.add_argument("--label", help="run directory suffix, e.g. zero-shot, lora-raw, lora-ema (config: label)")
    p.add_argument("--fake", action="store_true", help="FakeRobot + FakeCameras + FakeNet(pickplace), no hardware")
    p.add_argument("--fake-hardware", action="store_true",
                   help="FakeRobot + FakeCameras with the real so101.v1 server (--server): the socket path, no arm")
    p.add_argument("--fake-latency-ms", type=float, default=400.0)
    p.add_argument("--fake-cycle-s", type=float, help="FakeNet pickplace cycle length (default: the fake's)")
    p.add_argument("--live", action="store_true", help="write commands to the motors (default: dry run)")
    p.add_argument("--server", help="so101.v1 server URL (config: server.url)")
    p.add_argument("--port", help="serial port of the follower (config: robot.port)")
    p.add_argument("--robot-id", help="LeRobot calibration id (config: robot.id)")
    p.add_argument("--mapping", help="joint map JSON (config: robot.mapping; '' = raw LeRobot degrees)")
    p.add_argument("--limits", type=Path, help="joint limits JSON in the arm's own degrees (required with --live)")
    p.add_argument("--scene", type=camera_source, help="scene camera index or path (config: cameras.scene)")
    p.add_argument("--wrist", type=camera_source, help="wrist camera index or path (config: cameras.wrist)")
    p.add_argument("--loop-mode", choices=["prefetch", "sync"], help="client loop mode (config: loop.mode)")
    p.add_argument("--no-video", action="store_true", help="skip scene.mp4 / wrist.mp4")
    p.add_argument("--out", default=str(EVAL_DIR / "runs"), help="parent directory of the run directories")
    rec = p.add_mutually_exclusive_group()
    rec.add_argument("--record", dest="record", action="store_const", const=True,
                     help="record a LeRobotDataset episode per attempt even in a dry or fake run (default: live only)")
    rec.add_argument("--no-record", dest="record", action="store_const", const=False,
                     help="no LeRobotDataset, even live (config: record.lerobot)")
    p.add_argument("--unattended", action="store_true",
                   help="test-only, with --fake or --fake-hardware: enter at every prompt, + until the steps are "
                        "counted, then s; fail after a timeout")
    p.add_argument("--print-warmup", action="store_true", help="print the GPU serve commands for this instruction")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    a = parse(raw_argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if a.unattended and not (a.fake or a.fake_hardware):
        print("--unattended counts and confirms without a person, so it only runs with --fake or --fake-hardware",
              file=sys.stderr)
        return 2
    if a.fake and a.fake_hardware:
        print("--fake already replaces the server; --fake-hardware keeps the real one: pick one", file=sys.stderr)
        return 2
    if a.live and (a.fake or a.fake_hardware):
        print("--live excludes --fake and --fake-hardware", file=sys.stderr)
        return 2
    overrides = {
        "task.instruction": a.instruction,
        "task.steps": a.steps,
        "task.attempts": a.attempts,
        "task.attempt_timeout_s": a.attempt_timeout_s,
        "label": a.label,
        "server.url": a.server,
        "robot.port": a.port,
        "robot.id": a.robot_id,
        "robot.mapping": a.mapping,
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
    task = cfg.task
    if not task.instruction.strip():
        print(f"{a.config}: task.instruction is empty: set it in [task] or pass --instruction", file=sys.stderr)
        return 2
    if a.print_warmup:
        print(f"# run on the GPU host from the repo root; instruction from {a.config}:")
        print(serve_commands(task.instruction))
        return 0
    if cfg.loop.joint_limits:
        print(f"joint limits (deg): {cfg.loop.joint_limits}{f' from {a.limits}' if a.limits else ''}")
    print(f"config {a.config}: {task.attempts if a.attempts is None else a.attempts} attempts of {task.instruction!r}, "
          f"{task.steps or 'uncounted'} steps each, timeout {task.attempt_timeout_s:g} s, label {cfg.label}, "
          f"loop {cfg.loop.mode}")

    def on_term(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_term)
    keys = unattended_task_keys(task.steps) if a.unattended else None
    opts = SessionOptions(
        fake=a.fake,
        fake_hardware=a.fake_hardware,
        fake_latency_ms=a.fake_latency_ms,
        fake_cycle_s=a.fake_cycle_s,
        live=a.live,
        record=a.record,
        task=True,
        attempts=a.attempts,
        out=Path(a.out),
        argv=["run_task.py", *raw_argv],
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
    n = len(runner.attempts)
    print(f"wrote {run_dir} ({n} attempt{'s' * (n != 1)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
