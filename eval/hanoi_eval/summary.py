"""summary.md and summary.json for a run, from its episode and attempt records.

    uv run --project eval python -m hanoi_eval.summary eval/runs/<run>            # rewrite one run's summary
    uv run --project eval python -m hanoi_eval.summary --status eval/runs/*       # one status row per run
                                                    (Hanoi runs and run_task.py runs in two tables)

Episodes that ended with q (aborted) or with an infrastructure error (the control loop stopped, the server
went away) are listed but left out of the totals: neither is a result of the policy.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .recorder import write_json

NOT_COUNTED = ("quit", "error")  # episode ends left out of the totals


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(round(seconds))
    return f"{seconds // 60}m {seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def end_text(ep: dict) -> str:
    if ep["end"] == "done":
        return "all steps done" if ep["interventions"] == 0 else "all steps done, with interventions"
    if ep["end"] == "failed":
        return f"step {ep['failed_step']} failed ({ep['failed_reason']})"
    if ep["end"] == "quit":
        return f"aborted (q) at step {ep['steps_completed'] + 1}"
    if ep["end"] == "error":
        return f"infrastructure error at step {ep['steps_completed'] + 1}: {ep.get('error')}"
    return ep["end"]


def run_kind(run: dict) -> str:
    """strict | recover | sanity, plus demo when the auto end counted as success."""
    kind = "sanity" if run.get("kind") == "sanity" else run.get("protocol", "strict")
    return f"{kind}, demo" if run.get("mode") == "demo" else kind


def summarize(run: dict, episodes: list[dict], attempts: list[dict]) -> dict:
    counted = [e for e in episodes if e["end"] not in NOT_COUNTED]
    counted_ids = {e["episode"] for e in counted}
    counted_attempts = [a for a in attempts if a["episode"] in counted_ids and a.get("success") is not None]
    steps_total = sum(e["steps_total"] for e in counted)
    totals = {
        "episodes": len(counted),
        "aborted": sum(1 for e in episodes if e["end"] == "quit"),
        "errors": sum(1 for e in episodes if e["end"] == "error"),
        "episode_successes": sum(1 for e in counted if e["success"]),
        "successful_attempts": sum(1 for a in counted_attempts if a["success"]),
        "total_attempts": len(counted_attempts),
        "completed_steps": sum(e["steps_completed"] for e in counted),
        "steps_possible": steps_total,
        "first_try_steps_in_row": sum(e["first_try_in_row"] for e in counted),
        "mean_steps_in_row": round(sum(e["steps_completed"] for e in counted) / len(counted), 2) if counted else None,
        "interventions": sum(e["interventions"] for e in counted),
        "active_s": round(sum(e["active_s"] for e in counted), 1),
        "end_reasons": dict(Counter(a["end_reason"] for a in counted_attempts)),
    }
    rows = [
        {
            "episode": e["name"],
            "steps_in_row": e["steps_completed"],
            "steps_total": e["steps_total"],
            "steps_attempted": e["steps_attempted"],
            "attempts": e["attempts"],
            "success": e["success"],
            "active_s": e["active_s"],
            "wall_s": e["wall_s"],
            "interventions": e["interventions"],
            "end": end_text(e),
            "counted": e["end"] not in NOT_COUNTED,
        }
        for e in episodes
    ]
    by_move: dict[tuple, dict] = {}
    groups = defaultdict(list)
    for a in counted_attempts:
        move = (a["move"]["disk"], a["move"]["src"], a["move"]["dst"])
        # Sanity groups by move; episodes by step and move (a demo carries the tower back, so step 1 differs).
        groups[move if run.get("kind") == "sanity" else (a["step"], *move)].append(a)
    for key in sorted(groups):
        group = groups[key]
        first = group[0]
        by_move[key] = {
            "step": first["step"] if run.get("kind") != "sanity" else None,
            "move": f"{first['piece']}: {first['from']} -> {first['to']}",
            "prompt": first["prompt"],
            "attempts": len(group),
            "successes": sum(1 for a in group if a["success"]),
            "ends": dict(Counter(a["end_reason"] for a in group)),
        }
    status_row = {
        "run": run.get("name"),
        "kind": run_kind(run),
        "episodes": totals["episodes"],
        "episode_successes": totals["episode_successes"],
        "completed_steps": totals["completed_steps"],
        "steps_possible": totals["steps_possible"],
        "successful_attempts": totals["successful_attempts"],
        "total_attempts": totals["total_attempts"],
        "mean_steps_in_row": totals["mean_steps_in_row"],
    }
    return {"run": run, "totals": totals, "episodes": rows, "moves": list(by_move.values()), "status_row": status_row}


STATUS_HEADER = (
    "| Run | Kind | Episodes | Episode successes | Completed steps | Successful / total attempts | Mean steps in a row |\n"
    "|---|---|---|---|---|---|---|"
)


def status_line(row: dict) -> str:
    return (
        f"| {row['run']} | {row['kind']} | {row['episodes']} | {row['episode_successes']}/{row['episodes']} | "
        f"{row['completed_steps']}/{row['steps_possible']} | {row['successful_attempts']}/{row['total_attempts']} | "
        f"{row['mean_steps_in_row'] if row['mean_steps_in_row'] is not None else '-'} |"
    )


def rig_text(run: dict) -> str:
    if run.get("fake"):
        return "fake robot and policy"
    if run.get("fake_hardware"):
        return "fake robot and cameras, real server"
    return "live arm" if run.get("live") else "dry run"


def render_markdown(summary: dict) -> str:
    run, totals = summary["run"], summary["totals"]
    hello = {k: ("-" if v is None else v) for k, v in (run.get("server_hello") or {}).items()}
    git = run.get("git") or {}
    commit = (git.get("commit") or "unknown")[:7] + (" (dirty)" if git.get("dirty") else "")
    lines = [
        f"# Hanoi run {run.get('name', '')}",
        "",
        f"Kind {run_kind(run)} · mode {run.get('mode')} · protocol {run.get('protocol')} · "
        f"{rig_text(run)} · "
        f"loop {run.get('loop', {}).get('mode', '?')} · git {commit}",
        "",
        f"Engine {hello.get('engine', '?')} · checkpoint {hello.get('checkpoint', '?')}"
        f"{'@' + str(hello['revision'])[:7] if hello.get('revision', '-') != '-' else ''} · "
        f"guidance {hello.get('guidance', '?')} · steps {hello.get('steps', '?')}",
        "",
        "| Episode | Steps in a row | Steps attempted | Attempts | Success | Duration | Interventions | End |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in summary["episodes"]:
        lines.append(
            f"| {r['episode']}{'' if r['counted'] else ' (not counted)'} | {r['steps_in_row']}/{r['steps_total']} | "
            f"{r['steps_attempted']} | {r['attempts']} | {'yes' if r['success'] else 'no'} | "
            f"{fmt_duration(r['active_s'])} | {r['interventions']} | {r['end']} |"
        )
    t = totals
    lines += [
        "",
        f"**Totals:** episode successes {t['episode_successes']}/{t['episodes']} · "
        f"successful attempts {t['successful_attempts']}/{t['total_attempts']} · "
        f"completed steps {t['completed_steps']}/{t['steps_possible']} · "
        f"mean steps in a row {t['mean_steps_in_row'] if t['mean_steps_in_row'] is not None else '-'} · "
        f"interventions {t['interventions']}"
        + (f" · aborted episodes not counted: {t['aborted']}" if t["aborted"] else "")
        + (f" · episodes ended by an infrastructure error, not counted: {t['errors']}" if t.get("errors") else ""),
        "",
        "Duration is robot time: the sum of the step attempts, without pauses and operator prompts.",
        "",
    ]
    loop_error = (run.get("loop_stats") or {}).get("error")
    if loop_error:
        lines += [f"**Control loop error:** {loop_error}", ""]
    if summary["moves"]:
        sanity = run.get("kind") == "sanity"
        lines += [
            "## Per move" if sanity else "## Per step",
            "",
            ("| Move | Prompt | Trials | Successes | Ends |" if sanity else "| Step | Move | Prompt | Attempts | Successes | Ends |"),
            ("|---|---|---|---|---|" if sanity else "|---|---|---|---|---|---|"),
        ]
        for m in summary["moves"]:
            ends = ", ".join(f"{k} {v}" for k, v in sorted(m["ends"].items()))
            if sanity:
                lines.append(f"| {m['move']} | {m['prompt']} | {m['attempts']} | {m['successes']} | {ends} |")
            else:
                lines.append(
                    f"| {m['step']} | {m['move']} | {m['prompt']} | {m['attempts']} | {m['successes']} | {ends} |"
                )
        lines.append("")
    lines += ["## Status row (paste into docs/status)", "", STATUS_HEADER, status_line(summary["status_row"]), ""]
    return "\n".join(lines)


def write_summary(run_dir: str | Path, run: dict, episodes: list[dict], attempts: list[dict]) -> dict:
    run_dir = Path(run_dir)
    summary = summarize(run, episodes, attempts)
    (run_dir / "summary.md").write_text(render_markdown(summary))
    write_json(run_dir / "summary.json", summary)
    return summary


def load_run(run_dir: str | Path) -> tuple[dict, list[dict], list[dict]]:
    """run.json, every epNN/episode.json and every epNN/attempts.jsonl of a run directory."""
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    episodes, attempts = [], []
    for ep_dir in sorted(p for p in run_dir.glob("ep*") if p.is_dir()):
        if (ep_dir / "episode.json").exists():
            episodes.append(json.loads((ep_dir / "episode.json").read_text()))
        if (ep_dir / "attempts.jsonl").exists():
            attempts += [json.loads(line) for line in (ep_dir / "attempts.jsonl").read_text().splitlines() if line]
    return run, episodes, attempts


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Rewrite a run's summary.md/summary.json, or print status rows.")
    p.add_argument("runs", nargs="+", help="eval/runs/<run> directories")
    p.add_argument("--status", action="store_true", help="print one docs/status table row per run instead")
    a = p.parse_args(argv)
    rows, task_rows = [], []
    for path in a.runs:
        if json.loads((Path(path) / "run.json").read_text()).get("kind") == "task":  # run_task.py
            from .task import load_task_run, summarize_task, task_status_line, write_task_summary

            run, attempts = load_task_run(path)
            if a.status:
                task_rows.append(task_status_line(summarize_task(run, attempts)["status_row"]))
                continue
            t = write_task_summary(path, run, attempts)["totals"]
            print(f"{path}: wrote summary.md and summary.json; successes {t['successes']}/{t['valid_attempts']}, "
                  f"steps completed {t['steps_completed']} (longest {t['max_steps']})")
            continue
        run, episodes, attempts = load_run(path)
        if a.status:
            rows.append(status_line(summarize(run, episodes, attempts)["status_row"]))
            continue
        summary = write_summary(path, run, episodes, attempts)
        t = summary["totals"]
        print(
            f"{path}: wrote summary.md and summary.json; episode successes {t['episode_successes']}/{t['episodes']}, "
            f"successful attempts {t['successful_attempts']}/{t['total_attempts']}, "
            f"completed steps {t['completed_steps']}/{t['steps_possible']}"
        )
    if a.status:
        if rows or not task_rows:
            print(STATUS_HEADER)
            print("\n".join(rows))
        if task_rows:
            from .task import TASK_STATUS_HEADER

            if rows:
                print()
            print(TASK_STATUS_HEADER)
            print("\n".join(task_rows))


if __name__ == "__main__":
    main()
