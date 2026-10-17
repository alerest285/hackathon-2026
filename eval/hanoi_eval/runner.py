"""HanoiRunner: one instruction per move through the robot ControlLoop (decisions 05 and 06).

Each step: set_task(prompt, reset_history) -> run until the operator's key, the automatic end or the
timeout -> loop.pause() -> verdict. History is reset at an episode start and after an intervention (a
restore, or resuming from a pause); between steps the queued actions are dropped and the history kept
(`between_steps`). In eval mode the operator's key is the ground truth; in demo mode an automatic end counts
as success. The strict protocol stops an episode at its first failed step; recover lets the operator restore
the state and retry up to max_retries. Sanity runs are one-step episodes, one per legal single move, each
repeated.

A control loop that stopped, or a server connection lost for NET_LOSS_S, is not a policy failure: the
attempt ends with end_reason "error", the episode ends as "error" and stays out of the totals, and the run
stops with the reason.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .dataset import verdict_word
from .hanoi import Move, TowerState, other_place, piece_name, prompt, sanity_moves, solve
from .keys import describe
from .recorder import EpisodeRecorder, utc_now, write_json
from .step_end import AutoEndDetector
from .summary import fmt_duration, write_summary
from .video import VideoTaps

logger = logging.getLogger(__name__)

STEP_EVENTS = frozenset({"success", "fail", "retry", "pause", "quit"})
NET_LOSS_S = 5.0  # no server connection for this long during a step: an infrastructure error
REPLY_WAIT_S = 3.0  # at most this long after a step for the reply to its last request


class QuitRun(Exception):
    """The operator pressed q (or Ctrl-C): finish the current files and stop."""


class InfraError(RuntimeError):
    """The control loop stopped or the server went away: the episode ends as an error, not a failed step."""


@dataclass
class EpisodePlan:
    index: int
    name: str
    moves: list[Move]
    initial: TowerState
    kind: str = "episode"      # episode | sanity
    title: str = ""


class HanoiRunner:
    """Drives a ControlLoop through Hanoi episodes and writes the run directory.

    The runner registers one observer and one frame tap on the loop when it is created (the loop API cannot
    remove them), so create it before loop.start(). `keys` is a started KeyListener.
    """

    def __init__(
        self,
        loop,
        keys,
        cfg,
        run_dir: str | Path,
        *,
        run_info: dict | None = None,
        log=print,
        clock=time.perf_counter,
        poll_s: float = 0.02,
        to_native=None,
        dataset=None,
    ):
        """to_native: maps the loop's state/cmd (the checkpoint's convention when the arm is mapped) back to the
        arm's own degrees for the step-end detector, ticks.jsonl and the dataset; None when they are the same.
        dataset: a HanoiDataset that records each attempt as one LeRobot episode, or None."""
        self.loop = loop
        self.to_native = to_native
        self.dataset = dataset
        self._frames: dict = {}  # newest full-resolution BGR frame per camera, for the dataset
        self.keys = keys
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_info = dict(run_info or {})
        self.run_info.setdefault("name", self.run_dir.name)
        self._log = log
        self.clock = clock
        self.poll_s = poll_s
        self.episodes: list[dict] = []
        self.attempts: list[dict] = []
        self.quit = False
        self.observer_errors = 0
        self._recorder: EpisodeRecorder | None = None
        self._detector: AutoEndDetector | None = None
        self._det_lock = threading.Lock()
        self._auto = threading.Event()
        self.video = VideoTaps(("scene", "wrist"), fps=cfg.video.fps, codec=cfg.video.codec) if cfg.video.enabled else None
        loop.add_observer(self._on_tick)
        if self.video is not None:
            loop.add_frame_tap(self.video)
        if self.dataset is not None:
            loop.add_frame_tap(self._on_frame)

    # control thread ---------------------------------------------------------------------------------------

    def _on_frame(self, cam, bgr, t) -> None:
        self._frames[cam] = bgr

    def _on_tick(self, info) -> None:
        try:
            if self.to_native is not None:
                info = dataclasses.replace(info, state=self.to_native(info.state), cmd=self.to_native(info.cmd))
            if self.dataset is not None:
                self.dataset.add(info.state, info.cmd, self._frames.get("scene"), self._frames.get("wrist"), info.task)
            recorder = self._recorder
            if recorder is not None:
                recorder.on_tick(info)
            with self._det_lock:
                detector = self._detector
                if detector is not None and detector.observe(info) == "auto":
                    self._auto.set()
        except Exception:  # noqa: BLE001 - recording must never break the control thread
            self.observer_errors += 1
            if self.observer_errors <= 3:
                logger.exception("hanoi observer failed")

    def _attach(self, detector: AutoEndDetector | None) -> None:
        with self._det_lock:
            self._detector = detector

    def log(self, text: str) -> None:
        """Operator output. A terminal that went away must not break the recording, so write errors are dropped."""
        try:
            self._log(text)
        except OSError:
            pass

    def _keys_lost(self) -> bool:
        error = getattr(self.keys, "error", None)
        if error is not None:
            logger.error("operator keys lost (%r): ending the step and the run", error)
        return error is not None

    def _loop_problem(self) -> str | None:
        """Why the control loop cannot run a step, or None."""
        error = getattr(self.loop, "error", None)
        if error:
            return f"control loop stopped: {error}"
        if not getattr(self.loop, "running", True):
            return "control loop stopped"
        return None

    def _net_connected(self) -> bool:
        return bool(getattr(getattr(self.loop, "net", None), "connected", True))

    def _await_replies(self) -> None:
        """Wait, at most REPLY_WAIT_S, until the loop has no request in flight, so the reply to the step's
        last request lands in this episode's files. The next step could not request before it anyway."""
        deadline = time.perf_counter() + REPLY_WAIT_S
        while getattr(self.loop, "pending", 0) and self._loop_problem() is None and time.perf_counter() < deadline:
            time.sleep(self.poll_s)

    # runs -------------------------------------------------------------------------------------------------

    def run_episodes(self, n: int | None = None) -> list[dict]:
        """N full episodes from the configured start to the goal. Demo mode carries the tower back and forth."""
        cfg = self.cfg
        n = cfg.episodes if n is None else n
        self._begin_run("episodes", cfg.protocol)
        start, goal = cfg.start, cfg.goal
        ask_setup = True
        try:
            for i in range(1, n + 1):
                moves = solve(cfg.n, start, goal, other_place(start, goal))
                plan = EpisodePlan(i, f"ep{i:02d}", moves, TowerState.tower(cfg.n, start), "episode", f"episode {i}/{n}")
                ep = self._episode(plan, ask_setup)
                if self.quit or ep is None:
                    break
                ask_setup = True
                if cfg.mode == "demo" and ep["end"] == "done":
                    start, goal = goal, start
                    ask_setup = False
                    self.log(f"demo: the next episode carries the tower back from {cfg.places[start]} to {cfg.places[goal]}")
        finally:
            self._end_run()
        return self.episodes

    def run_sanity(self, repeats: int | None = None) -> list[dict]:
        """Every legal single move (`sanity.moves`), each `repeats` times, as one-step strict episodes."""
        cfg = self.cfg
        repeats = cfg.sanity.repeats if repeats is None else repeats
        items = sanity_moves(cfg.n, cfg.sanity.moves, cfg.start, cfg.goal)
        total = len(items) * repeats
        self._begin_run("sanity", "strict")
        self.log(f"sanity: {len(items)} moves x {repeats} = {total} one-step episodes ({cfg.sanity.moves})")
        i = 0
        try:
            for move, setup in items:
                for r in range(repeats):
                    i += 1
                    title = f"sanity {i}/{total}, repeat {r + 1}/{repeats}"
                    ep = self._episode(EpisodePlan(i, f"ep{i:02d}", [move], setup, "sanity", title), ask_setup=True)
                    if self.quit or ep is None:
                        break
                if self.quit:
                    break
        finally:
            self._end_run()
        return self.episodes

    def _begin_run(self, kind: str, protocol: str) -> None:
        self.run_info.update(
            {
                "kind": kind,
                "mode": self.cfg.mode,
                "protocol": protocol,
                "label": self.cfg.label,
                "started_utc": self.run_info.get("started_utc") or utc_now(),
                "config": self.cfg.to_dict(),
            }
        )
        write_json(self.run_dir / "run.json", self.run_info)

    def _end_run(self) -> None:
        self.run_info["ended_utc"] = utc_now()
        self.run_info["episodes_run"] = len(self.episodes)
        self.run_info["loop_stats"] = self._loop_stats()
        self.run_info["observer_errors"] = self.observer_errors
        write_json(self.run_dir / "run.json", self.run_info)
        self.write_summary()

    def write_summary(self) -> dict:
        return write_summary(self.run_dir, self.run_info, self.episodes, self.attempts)

    def _loop_stats(self) -> dict | None:
        try:
            return self.loop.stats()
        except Exception as e:  # noqa: BLE001
            return {"error": repr(e)}

    def _ask(self, context: str, allowed, info: dict, message: str) -> str | None:
        """Register the prompt, then show it, then wait: a key pressed as soon as the text appears counts."""
        try:
            pending = self.keys.expect(context, allowed, info)
            self.log(message)
            return self.keys.wait(pending)
        except KeyboardInterrupt:
            return "quit"
        except RuntimeError:  # the key listener died, e.g. the terminal closed
            self._keys_lost()
            return "quit"

    # one episode ------------------------------------------------------------------------------------------

    def _episode(self, plan: EpisodePlan, ask_setup: bool) -> dict | None:
        cfg = self.cfg
        protocol = "strict" if plan.kind == "sanity" else cfg.protocol
        layout = plan.initial.describe(cfg.pieces, cfg.places)
        self.log(f"\n{plan.name} ({plan.title}): {layout}")
        if ask_setup:
            message = "  set the pieces up like this, then press enter (q quits)"
            if self._ask("start", {"continue", "quit"}, {"episode": plan.index}, message) != "continue":
                self.quit = True
                self.log("  quit before the episode started")
                return None
        recorder = EpisodeRecorder(self.run_dir / plan.name)
        if self.video is not None:
            self.video.open(recorder.dir)
        self._recorder = recorder
        utc_start, t0 = utc_now(), self.clock()
        state = plan.initial
        attempts: list[dict] = []
        completed = first_try_in_row = interventions = 0
        first_try_alive = True
        end, failed_step, failed_reason = "done", None, None
        reset_next = True
        error: BaseException | None = None
        try:
            for step, move in enumerate(plan.moves, 1):
                tries = 0
                while True:
                    a = self._attempt(plan, recorder, step, move, tries, state, reset_next, len(attempts))
                    recorder.write_attempt(a)  # the file first: an interrupt here must not count an unwritten attempt
                    attempts.append(a)
                    self.attempts.append(a)
                    interventions += a["interventions"]
                    reset_next = cfg.between_steps == "reset_history"
                    if a["end_reason"] == "error":
                        raise InfraError(a["error"])
                    if a["quit"]:
                        raise QuitRun
                    if a["success"]:
                        state = state.apply(move)
                        completed += 1
                        if tries == 0 and first_try_alive:
                            first_try_in_row += 1
                        break
                    first_try_alive = False
                    if protocol == "recover" and tries < cfg.max_retries:
                        choice = self._restore(plan, step, move, state, tries)
                        if choice in ("continue", "retry"):
                            tries += 1
                            reset_next = True
                            continue
                        if choice != "fail":
                            raise QuitRun
                    end, failed_step, failed_reason = "failed", step, a["end_reason"]
                    break
                if end == "failed":
                    break
        except (QuitRun, KeyboardInterrupt):
            end = "quit"
            self.quit = True
        except BaseException as e:  # noqa: BLE001 - files are finished below, then it is raised again
            end = "error"
            error = e
            self.quit = True
        finally:
            self._attach(None)
            self._safe_pause()
            self._recorder = None
            video = self.video.close() if self.video is not None else {}
            stats = recorder.close(cfg.loop.chunk)
        ep = {
            "episode": plan.index,
            "name": plan.name,
            "kind": plan.kind,
            "title": plan.title,
            "mode": cfg.mode,
            "protocol": protocol,
            "label": cfg.label,
            "initial_state": plan.initial.to_dict(),
            "tracked_state": state.to_dict(),
            "moves": [{"step": i, **m.to_dict(), "prompt": prompt(m, cfg)} for i, m in enumerate(plan.moves, 1)],
            "steps_total": len(plan.moves),
            "steps_completed": completed,
            "first_try_in_row": first_try_in_row,
            "steps_attempted": len({a["step"] for a in attempts}),
            "attempts": len(attempts),
            "successful_attempts": sum(1 for a in attempts if a["success"]),
            "interventions": interventions,
            "success": end == "done" and completed == len(plan.moves) and interventions == 0,
            "end": end,
            "failed_step": failed_step,
            "failed_reason": failed_reason,
            "error": (str(error) if isinstance(error, InfraError) else repr(error)) if error else None,
            "utc_start": utc_start,
            "utc_end": utc_now(),
            "t0_perf": t0,
            "wall_s": round(self.clock() - t0, 3),
            "active_s": round(sum(a["active_s"] for a in attempts), 3),
            "video": video,
            **stats,
            "loop_stats": self._loop_stats(),
        }
        recorder.write_episode(ep)
        self.episodes.append(ep)
        self.write_summary()
        verdict = "success" if ep["success"] else end_word(ep)
        self.log(
            f"{plan.name}: {completed}/{len(plan.moves)} steps, {verdict}, {len(attempts)} attempts, "
            f"{fmt_duration(ep['active_s'])} robot time"
        )
        if error is not None:
            raise error
        return ep

    def _restore(self, plan: EpisodePlan, step: int, move: Move, state: TowerState, tries: int) -> str | None:
        cfg = self.cfg
        message = (
            f"    restore the state before step {step}: {state.describe(cfg.pieces, cfg.places)}\n"
            f"    then press enter to retry ({tries + 1}/{cfg.max_retries}), f ends the episode, q quits"
        )
        return self._ask("restore", {"continue", "retry", "fail", "quit"}, {"episode": plan.index, "step": step}, message)

    def _safe_pause(self) -> None:
        try:
            self.loop.pause()
        except Exception:  # noqa: BLE001
            logger.exception("loop.pause failed")

    # one step attempt -------------------------------------------------------------------------------------

    def _attempt(
        self,
        plan: EpisodePlan,
        recorder: EpisodeRecorder,
        step: int,
        move: Move,
        tries: int,
        state: TowerState,
        reset_history: bool,
        index: int,
    ) -> dict:
        cfg = self.cfg
        protocol = "strict" if plan.kind == "sanity" else cfg.protocol
        text = prompt(move, cfg)
        info = {"episode": plan.index, "step": step, "try": tries}
        detector = AutoEndDetector(cfg.auto_end) if cfg.auto_end.enabled else None
        n = len(plan.moves)
        # Strict mode has no retry, so r is no key during a step there: it would end the episode as a fail.
        step_events = STEP_EVENTS if protocol == "recover" else STEP_EVENTS - {"retry"}
        end_keys = "/".join(k for k, e in (("s", "success"), ("f", "fail"), ("r", "retry"), ("q", "quit"))
                            if e in step_events)
        self._auto.clear()
        recorder.attempt = index
        self._attach(detector)
        self.keys.expect("step", step_events, info)
        self.log(
            f"  step {step}/{n}{f' retry {tries}' if tries else ''}: {text}  "
            f"[history {'reset' if reset_history else 'kept'}]"
        )
        utc_start, t_start = utc_now(), self.clock()
        end_reason, key_verdict, problem = None, None, None
        retry_requested = interrupted = False
        pauses, paused_s, pause_t, net_lost_t = 0, 0.0, None, None
        try:
            problem = self._loop_problem()
            if problem is None:
                self.loop.set_task(text, reset_history=reset_history)
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
                if event == "pause":
                    if pause_t is None:
                        self._attach(None)
                        if self.dataset is not None:
                            self.dataset.pause()  # a person may be in the scene: no frames until resume
                        self.loop.pause()
                        pause_t, pauses = now, pauses + 1
                        self.log(f"    paused: space resumes, {end_keys} end the step")
                    else:
                        paused_s += now - pause_t
                        pause_t, net_lost_t = None, None
                        if detector is not None:
                            detector.interrupt()
                        self._attach(detector)
                        # A pause is an intervention: the full reset (docs/architecture.md, decision 06).
                        self.loop.set_task(text, reset_history=True)
                        if self.dataset is not None:
                            self.dataset.begin(text)
                        self.log("    resumed, history reset")
                elif event in ("success", "fail", "retry"):
                    end_reason, key_verdict, retry_requested = "key", event == "success", event == "retry"
                    break
                elif event == "quit" or self._keys_lost():
                    end_reason = "quit"
                    break
                if pause_t is None:
                    if self._auto.is_set():
                        end_reason = "auto"
                        break
                    if now - t_start - paused_s >= cfg.step_timeout_s:
                        end_reason = "timeout"
                        break
                time.sleep(self.poll_s)
            if problem is not None:
                end_reason = "error"
        except KeyboardInterrupt:
            end_reason, interrupted = "quit", True
        finally:
            self._attach(None)
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
                self.log(f"    -> infrastructure error, not a failed step: {problem}")
        elif end_reason == "key":
            success = key_verdict
        elif cfg.mode == "demo" and end_reason == "auto":
            success, verdict_by = True, "auto"
        else:
            allowed = {"success", "fail", "quit"} | ({"retry"} if protocol == "recover" else set())
            message = f"    ended by {end_reason} after {t_end - t_start - paused_s:.1f} s. Success? {describe(allowed)}"
            answer = self._ask("verdict", allowed, {**info, "end_reason": end_reason}, message)
            if answer in (None, "quit"):
                success, verdict_by, quit_now = None, None, True
            else:
                success = answer == "success"
                retry_requested = answer == "retry"
        if success is not None:
            who = "operator" if verdict_by == "operator" else "demo: an auto end counts as success"
            self.log(f"    -> {'success' if success else 'fail'} ({who})")
        if end_reason != "error" and not quit_now:
            try:
                self._await_replies()  # usually done during the verdict prompt; counted in "chunks" below
            except KeyboardInterrupt:
                quit_now = interrupted = True
        restored = tries > 0
        result = {
            "episode": plan.index,
            "episode_name": plan.name,
            "kind": plan.kind,
            "step": step,
            "steps_total": n,
            "try": tries,
            "attempt": index,
            "move": move.to_dict(),
            "piece": piece_name(move.disk, cfg.pieces),
            "from": cfg.places[move.src],
            "to": cfg.places[move.dst],
            "prompt": text,
            "reset_history": reset_history,
            "restored": restored,
            "utc_start": utc_start,
            "utc_end": utc_end,
            "t_start": t_start,
            "t_end": t_end,
            "duration_s": round(t_end - t_start, 3),
            "active_s": round(t_end - t_start - paused_s, 3),
            "paused_s": round(paused_s, 3),
            "end_reason": end_reason,
            "error": problem,
            "success": success,
            "verdict_by": verdict_by,
            "retry_requested": retry_requested,
            "pauses": pauses,
            "interventions": pauses + int(restored),
            "quit": quit_now,
            "interrupted": interrupted,
            "auto_end": detector.summary(t_start) if detector is not None else None,
            "chunks": recorder.chunks_in(index),
            "ticks": recorder.ticks_per_attempt.get(index, 0),
            "state_before": state.to_dict(),
            "state_after": state.apply(move).to_dict() if success else None,
        }
        if self.dataset is not None:
            meta = {key: result[key] for key in ("episode", "step", "try", "attempt", "kind", "end_reason", "pauses",
                                                 "interventions", "verdict_by", "move")}
            meta.update(run=self.run_dir.name, mode=cfg.mode, protocol=protocol, live=self.run_info.get("live"),
                        dry_run=self.run_info.get("dry_run"), fake=bool(self.run_info.get("fake")
                                                                       or self.run_info.get("fake_hardware")))
            self.dataset.end(verdict_word(result), meta)
        return result


def end_word(ep: dict) -> str:
    if ep["end"] == "done":
        return "all steps done with interventions"
    if ep["end"] == "failed":
        return f"failed at step {ep['failed_step']} ({ep['failed_reason']})"
    if ep["end"] == "error":
        return f"infrastructure error, not counted: {ep['error']}"
    return ep["end"]
