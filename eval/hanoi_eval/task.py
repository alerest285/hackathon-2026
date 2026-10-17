"""Scored runs of one instruction with live step counting (run_task.py), on the Hanoi runner's machinery.

The hackathon counts successful attempts, total attempts and completed task steps; longer chains score higher. A
task such as "Put the green objects at the right of the black line and the orange at the left of the black line"
has countable steps (one object placed on its side), so the operator counts them while the arm works:

    enter   start the attempt (set_task with reset_history=True: every attempt is a fresh episode)
    + or =  one step done (its active time is kept)        -   undo the last step
    space   pause / resume: an intervention, and the history restarts on resume
    s / f   end the attempt: success / fail                q   quit the run

After task.attempt_timeout_s of active time (pauses excluded) the arm holds and the operator gives the verdict.
Infrastructure faults (the control loop stopped, the server gone for NET_LOSS_S) end the attempt as "error";
like quits they are kept in the files but left out of the totals, and the run stops.

Each attempt gets attNN/ with attempt.json, ticks.jsonl, chunks.npz, scene.mp4, wrist.mp4 and *.times.csv
(EpisodeRecorder + VideoTaps, as a Hanoi episode), and one LeRobot episode when the run records. The run gets
run.json, attempts.jsonl, summary.md and summary.json.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .dataset import verdict_word
from .recorder import EpisodeRecorder, to_jsonable, utc_now, write_json
from .runner import NET_LOSS_S, HanoiRunner
from .summary import fmt_duration, rig_text

logger = logging.getLogger(__name__)

TASK_EVENTS = frozenset({"step", "unstep", "pause", "success", "fail", "quit"})


class TaskRunner(HanoiRunner):
    """K attempts of cfg.task.instruction. Shares the tick observer, video taps, dataset hooks, unit mapping and
    fault checks with HanoiRunner; only the protocol and the summary differ."""

    def run(self, attempts: int | None = None) -> list[dict]:
        task = self.cfg.task
        n = task.attempts if attempts is None else attempts
        self.run_info["task"] = {"instruction": task.instruction, "steps_total": task.steps, "attempts": n,
                                 "attempt_timeout_s": task.attempt_timeout_s}
        self._begin_run("task", "task")
        try:
            for i in range(1, n + 1):
                if self._one(i, n) is None or self.quit:
                    break
        finally:
            self._end_run()
        return self.attempts

    def write_summary(self) -> dict:
        return write_task_summary(self.run_dir, self.run_info, self.attempts)

    # one attempt ------------------------------------------------------------------------------------------

    def _one(self, index: int, total: int) -> dict | None:
        cfg, task = self.cfg, self.cfg.task
        name = f"att{index:02d}"
        self.log(f"\n{name} (attempt {index}/{total}): {task.instruction}")
        message = "  set up the scene, then press enter (q quits)"
        if self._ask("start", {"continue", "quit"}, {"attempt": index}, message) != "continue":
            self.quit = True
            self.log("  quit before the attempt started")
            return None
        recorder = EpisodeRecorder(self.run_dir / name)
        if self.video is not None:
            self.video.open(recorder.dir)
        self._recorder = recorder
        attempt = None
        try:
            attempt = self._attempt_task(index, recorder)
        finally:  # the files are finished even when the attempt raised; the exception then goes on
            self._safe_pause()
            self._recorder = None
            video = self.video.close() if self.video is not None else {}
            recording = recorder.close(cfg.loop.chunk)
        attempt.update(video=video, recording=recording)
        write_json(recorder.dir / "attempt.json", attempt)
        with open(self.run_dir / "attempts.jsonl", "a") as f:
            f.write(json.dumps(attempt, default=to_jsonable) + "\n")
        self.attempts.append(attempt)
        self.write_summary()
        if attempt["end_reason"] == "error":
            self.quit = True
            verdict = "infrastructure error (not counted)"
        elif attempt["success"] is None:
            self.quit = True
            verdict = "quit (not counted)"
        else:
            verdict = "success" if attempt["success"] else "fail"
        steps = attempt["steps_completed"]
        self.log(f"{name}: {verdict}, {steps}{f'/{task.steps}' if task.steps else ''} steps, "
                 f"{fmt_duration(attempt['active_s'])} robot time")
        return attempt

    def _attempt_task(self, index: int, recorder: EpisodeRecorder) -> dict:
        cfg, task = self.cfg, self.cfg.task
        n, text = task.steps, task.instruction
        info = {"attempt": index}
        recorder.attempt = index
        self._attach(None)  # no automatic step end: the operator ends a task attempt
        # A person's keys all carry the prompt on screen, so one prompt takes many + presses. A script answers one
        # key per prompt, so for it the prompt is re-armed after each step and pause (keys.ScriptedKeys).
        rearm = not getattr(getattr(self.keys, "source", None), "interactive", True)
        self.keys.expect("step", TASK_EVENTS, info)
        counted = f"steps 0/{n}" if n else "steps not counted"
        self.log(f"  + step done, - undo, space pause, s success, f fail, q quit ({counted})")
        utc_start, t_start = utc_now(), self.clock()
        steps: list[float] = []
        end_reason, key_verdict, problem = None, None, None
        interrupted = False
        pauses, paused_s, pause_t, net_lost_t = 0, 0.0, None, None

        def active(now: float) -> float:
            return now - t_start - paused_s - ((now - pause_t) if pause_t is not None else 0.0)

        try:
            problem = self._loop_problem()
            if problem is None:
                self.loop.set_task(text, reset_history=True)
                if self.dataset is not None:
                    self.dataset.begin(text)
            while problem is None:
                event = self.keys.poll()
                now = self.clock()
                problem = self._loop_problem()
                if problem is None and pause_t is None:  # a pause may be spent fixing the tunnel
                    if self._net_connected():
                        net_lost_t = None
                    elif net_lost_t is None:
                        net_lost_t = now
                    elif now - net_lost_t >= NET_LOSS_S:
                        problem = f"no connection to the server for {NET_LOSS_S:g} s"
                if problem is not None:
                    break
                if event in ("step", "unstep"):
                    if event == "step":
                        steps.append(round(active(now), 3))
                    elif steps:
                        steps.pop()
                    self.log(f"    steps {len(steps)}/{n}" if n else f"    steps {len(steps)}")
                    if n and len(steps) > n:
                        self.log(f"    more steps than the task has ({n}): check the count, - undoes one")
                    if rearm:
                        self.keys.expect("step", TASK_EVENTS, info)
                elif event == "pause":
                    if pause_t is None:
                        if self.dataset is not None:
                            self.dataset.pause()  # a person may be in the scene: no frames until resume
                        self.loop.pause()
                        pause_t, pauses = now, pauses + 1
                        self.log("    paused: space resumes (the history restarts), s/f end the attempt")
                    else:
                        paused_s += now - pause_t
                        pause_t, net_lost_t = None, None
                        self.loop.set_task(text, reset_history=True)  # an intervention: the full reset
                        if self.dataset is not None:
                            self.dataset.begin(text)
                        self.log("    resumed, history reset")
                    if rearm:
                        self.keys.expect("step", TASK_EVENTS, info)
                elif event in ("success", "fail"):
                    end_reason, key_verdict = "key", event == "success"
                    break
                elif event == "quit" or self._keys_lost():
                    end_reason = "quit"
                    break
                if pause_t is None and active(now) >= task.attempt_timeout_s:
                    end_reason = "timeout"
                    break
                time.sleep(self.poll_s)
            if problem is not None:
                end_reason = "error"
        except KeyboardInterrupt:
            end_reason, interrupted = "quit", True
        finally:
            self._safe_pause()
        t_end = self.clock()
        if pause_t is not None:
            paused_s += t_end - pause_t
        recorder.attempt = None
        utc_end = utc_now()

        quit_now = end_reason == "quit"
        verdict_by = "operator"
        success: bool | None
        if quit_now or end_reason == "error":
            success, verdict_by = None, None
            if problem is not None:
                self.log(f"    -> infrastructure error, not a failed attempt: {problem}")
        elif end_reason == "key":
            success = key_verdict
        else:  # timeout
            message = (f"    time is up after {task.attempt_timeout_s:g} s with {len(steps)}"
                       f"{f'/{n}' if n else ''} steps. Success? s/f (q quits)")
            answer = self._ask("verdict", {"success", "fail", "quit"},
                               {**info, "end_reason": end_reason, "steps": len(steps)}, message)
            if answer in (None, "quit"):
                success, verdict_by, quit_now = None, None, True
            else:
                success = answer == "success"
        if end_reason != "error" and not quit_now:
            try:
                self._await_replies()  # the reply to the attempt's last request lands in its files
            except KeyboardInterrupt:
                quit_now = interrupted = True
        result = {
            "run": self.run_dir.name,
            "kind": "task",
            "label": cfg.label,
            "attempt": index,
            "instruction": text,
            "steps_completed": len(steps),
            "steps_total": n,
            "step_times_s": steps,
            "complete": bool(n) and len(steps) >= n,
            "success": success,
            "verdict_by": verdict_by,
            "end_reason": end_reason,
            "error": problem,
            "utc_start": utc_start,
            "utc_end": utc_end,
            "t_start": t_start,
            "t_end": t_end,
            "duration_s": round(t_end - t_start, 3),
            "active_s": round(t_end - t_start - paused_s, 3),
            "paused_s": round(paused_s, 3),
            "pauses": pauses,
            "interventions": pauses,
            "quit": quit_now,
            "interrupted": interrupted,
            "chunks": recorder.chunks_in(index),
            "ticks": recorder.ticks_per_attempt.get(index, 0),
        }
        if self.dataset is not None:
            meta = {key: result[key] for key in ("attempt", "label", "instruction", "steps_completed", "steps_total",
                                                 "complete", "end_reason", "pauses", "interventions", "verdict_by")}
            meta.update(run=self.run_dir.name, kind="task", live=self.run_info.get("live"),
                        dry_run=self.run_info.get("dry_run"),
                        fake=bool(self.run_info.get("fake") or self.run_info.get("fake_hardware")))
            self.dataset.end(verdict_word(result), meta)
        return result


# summary ------------------------------------------------------------------------------------------------------

TASK_STATUS_HEADER = (
    "| Run | Label | Rig | Attempts (valid) | Successes | Steps completed (sum / mean / max of N) | Complete | Interventions |\n"
    "|---|---|---|---|---|---|---|---|"
)


def counted(attempts: list[dict]) -> list[dict]:
    """Attempts with a verdict: quits and infrastructure errors are no result of the policy."""
    return [a for a in attempts if a.get("success") is not None and a.get("end_reason") != "error"]


def summarize_task(run: dict, attempts: list[dict]) -> dict:
    valid = counted(attempts)
    steps = [a["steps_completed"] for a in valid]
    wins = [a for a in valid if a["success"]]
    n = (run.get("task") or {}).get("steps_total") or 0
    totals = {
        "attempts": len(attempts),
        "valid_attempts": len(valid),
        "quit": sum(1 for a in attempts if a.get("success") is None and a.get("end_reason") != "error"),
        "errors": sum(1 for a in attempts if a.get("end_reason") == "error"),
        "successes": len(wins),
        "success_rate": round(len(wins) / len(valid), 3) if valid else None,
        "steps_per_attempt": n,
        "steps_completed": sum(steps),
        "mean_steps": round(sum(steps) / len(steps), 2) if steps else None,
        "max_steps": max(steps) if steps else None,
        "complete_attempts": sum(1 for a in valid if a.get("complete")),
        "mean_success_active_s": round(sum(a["active_s"] for a in wins) / len(wins), 1) if wins else None,
        "interventions": sum(a.get("interventions", 0) for a in valid),
    }
    status_row = {"run": run.get("name"), "label": run.get("label"), "rig": rig_text(run), **totals}
    return {"run": run, "totals": totals, "attempts": attempts, "status_row": status_row}


def task_status_line(row: dict) -> str:
    n = row.get("steps_per_attempt") or 0
    mean = "-" if row["mean_steps"] is None else row["mean_steps"]
    best = "-" if row["max_steps"] is None else row["max_steps"]
    of = f" of {n}" if n else ""
    return (f"| {row['run']} | {row['label']} | {row['rig']} | {row['attempts']} ({row['valid_attempts']}) | "
            f"{row['successes']}/{row['valid_attempts']} | {row['steps_completed']} / {mean} / {best}{of} | "
            f"{row['complete_attempts']} | {row['interventions']} |")


def render_task_markdown(summary: dict) -> str:
    run, totals = summary["run"], summary["totals"]
    task = run.get("task") or {}
    hello = run.get("server_hello") or {}
    git = run.get("git") or {}
    commit = (git.get("commit") or "unknown")[:7] + (" (dirty)" if git.get("dirty") else "")
    n = task.get("steps_total") or 0
    lines = [
        f"# {run.get('name', 'task run')}",
        "",
        f"- Instruction: {task.get('instruction')!r}",
        f"- Steps per attempt: {n or 'not counted'}; attempt timeout {task.get('attempt_timeout_s')} s",
        f"- Label: {run.get('label')}; rig: {rig_text(run)}; commit {commit}",
        f"- Server: {hello.get('engine', '-')} {hello.get('checkpoint') or ''}"
        + (f", adapter {hello['adapter']}" if hello.get("adapter") else "")
        + (f", norm stats {hello['norm_stats'].get('path')}" if isinstance(hello.get("norm_stats"), dict) else ""),
        f"- Joint map: {(run.get('joint_map') or {}).get('path') or 'none (raw LeRobot degrees)'}",
        "",
        "## Totals (quits and infrastructure errors are not counted)",
        "",
        f"- Attempts: {totals['attempts']} ({totals['valid_attempts']} counted, {totals['quit']} quit, "
        f"{totals['errors']} errors)",
        f"- Successes: {totals['successes']}/{totals['valid_attempts']}"
        + (f" ({totals['success_rate']:.0%})" if totals["success_rate"] is not None else ""),
        f"- Steps completed: {totals['steps_completed']} in all, mean {totals['mean_steps'] if totals['mean_steps'] is not None else '-'}, "
        f"longest {totals['max_steps'] if totals['max_steps'] is not None else '-'}{f' of {n}' if n else ''}; "
        f"complete attempts {totals['complete_attempts']}",
        f"- Mean robot time of a success: {fmt_duration(totals['mean_success_active_s'])}; "
        f"interventions {totals['interventions']}",
        "",
        "## Attempts",
        "",
        "| # | Steps | Verdict | End | Robot time | Pauses |",
        "|---|---|---|---|---|---|",
    ]
    for a in summary["attempts"]:
        verdict = "-" if a.get("success") is None else ("success" if a["success"] else "fail")
        steps = f"{a['steps_completed']}/{n}" if n else str(a["steps_completed"])
        lines.append(f"| {a['attempt']} | {steps} | {verdict} | {a['end_reason']} | {fmt_duration(a['active_s'])} | "
                     f"{a.get('pauses', 0)} |")
    lines += ["", "## Status row (paste into docs/status)", "", TASK_STATUS_HEADER,
              task_status_line(summary["status_row"]), ""]
    return "\n".join(lines)


def write_task_summary(run_dir: str | Path, run: dict, attempts: list[dict]) -> dict:
    run_dir = Path(run_dir)
    summary = summarize_task(run, attempts)
    (run_dir / "summary.md").write_text(render_task_markdown(summary))
    write_json(run_dir / "summary.json", summary)
    return summary


def load_task_run(run_dir: str | Path) -> tuple[dict, list[dict]]:
    """run.json and attempts.jsonl of a task run directory."""
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    path = run_dir / "attempts.jsonl"
    attempts = [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []
    return run, attempts
