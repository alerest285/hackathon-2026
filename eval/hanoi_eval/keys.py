"""Operator keys on a background thread: s success, f fail, r retry, space pause/hold, q quit, enter continue,
and for run_task.py + (or =) one task step done, - undo it.

The runner announces what it waits for with expect()/ask() (a Prompt: context + allowed events). Keys
that arrive for an earlier prompt, or that the current prompt does not allow, are dropped, so a stray
key never answers the wrong question. The default source reads the terminal in cbreak mode and restores
it on stop() and at exit. Tests and unattended fake runs pass ScriptedKeys instead.
"""

from __future__ import annotations

import atexit
import os
import queue
import select
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field

KEYMAP = {
    "s": "success",
    "f": "fail",
    "r": "retry",
    " ": "pause",
    "q": "quit",
    "\n": "continue",
    "\r": "continue",
    "+": "step",  # run_task.py: one step of the task done (= is + without shift)
    "=": "step",
    "-": "unstep",
}
EVENT_KEYS = {"success": "s", "fail": "f", "retry": "r", "pause": "space", "quit": "q", "continue": "enter",
              "step": "+", "unstep": "-"}


@dataclass(frozen=True)
class Prompt:
    seq: int
    context: str                 # start | step | verdict | restore
    allowed: frozenset[str]
    info: dict = field(default_factory=dict)
    t: float = 0.0


def describe(allowed) -> str:
    """`s=success f=fail q=quit` for the events allowed, in a fixed order."""
    order = ["continue", "success", "fail", "retry", "pause", "quit"]
    return " ".join(f"{EVENT_KEYS[e]}={e}" for e in order if e in allowed)


class TtySource:
    """Single key presses from the terminal in cbreak mode (no echo, no enter needed)."""

    interactive = True

    def __init__(self, stream=None):
        self.stream = stream or sys.stdin
        self.fd: int | None = None
        self._saved = None

    def open(self) -> None:
        import termios
        import tty

        fd = self.stream.fileno()
        if not os.isatty(fd):
            raise RuntimeError(
                "stdin is not a terminal: operator keys need a TTY (or --unattended with --fake or --fake-hardware)"
            )
        self.fd = fd
        self._saved = termios.tcgetattr(fd)
        tty.setcbreak(fd)

    def read(self, timeout: float, prompt: Prompt | None) -> str | None:
        if self.fd is None:
            time.sleep(timeout)
            return None
        ready, _, _ = select.select([self.fd], [], [], timeout)
        if not ready:
            return None
        data = os.read(self.fd, 1)
        if not data:
            raise EOFError("the terminal closed")
        return data.decode("utf-8", errors="ignore") or None

    def close(self) -> None:
        if self.fd is not None and self._saved is not None:
            import termios

            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)
            except termios.error:
                pass
        self._saved = None
        self.fd = None


class ScriptedKeys:
    """Key source for tests and unattended fake runs: presses keys only in answer to prompts.

    `answers` maps a prompt context (start, step, verdict, restore) to one key, used for every prompt of
    that context, to a list of keys used in order, one per prompt (None = press nothing for that prompt),
    or to a function of the Prompt that returns the key. A context without an entry gets nothing. When a
    list runs out, `exhausted` is pressed (q by default), so a test cannot hang on a prompt it did not
    script. `delay_s` waits that long after each prompt.
    """

    interactive = False

    def __init__(self, answers: dict, delay_s: float = 0.0, exhausted: str | None = "q"):
        self.answers = {k: (deque(v) if isinstance(v, (list, tuple)) else v) for k, v in answers.items()}
        self.delay_s = delay_s
        self.exhausted = exhausted
        self.pressed: list[tuple[str, str]] = []
        self._done = -1

    def read(self, timeout: float, prompt: Prompt | None) -> str | None:
        time.sleep(min(timeout, 0.005))
        if prompt is None or prompt.seq == self._done:
            return None
        if time.perf_counter() - prompt.t < self.delay_s:
            return None
        self._done = prompt.seq
        answer = self.answers.get(prompt.context)
        if isinstance(answer, deque):
            key = answer.popleft() if answer else self.exhausted
        elif callable(answer):
            key = answer(prompt)
        else:
            key = answer
        if key is not None:
            self.pressed.append((prompt.context, key))
        return key


def unattended_keys() -> ScriptedKeys:
    """Test-only operator for runs without a person (run_hanoi.py --unattended, with fake hardware only):
    enter at every start and restore prompt; the verdict follows the end reason, success after an auto end
    and fail after a timeout."""
    return ScriptedKeys(
        {"start": "\n", "restore": "\n", "verdict": lambda p: "s" if p.info.get("end_reason") == "auto" else "f"}
    )


class KeyListener:
    """Reads keys on its own thread and hands the runner the events its current prompt allows."""

    def __init__(self, source=None, poll_s: float = 0.05):
        self.source = source if source is not None else TtySource()
        self.poll_s = poll_s
        self._events: queue.Queue = queue.Queue()
        self._prompt: Prompt | None = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None

    def __enter__(self) -> "KeyListener":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        if self._thread is not None:
            return
        opener = getattr(self.source, "open", None)
        if opener:
            opener()
        atexit.register(self.stop)
        self._thread = threading.Thread(target=self._run, name="keys", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        closer = getattr(self.source, "close", None)
        if closer:
            closer()
        atexit.unregister(self.stop)

    def _run(self) -> None:
        answers_prompt = not getattr(self.source, "interactive", True)
        try:
            while not self._stop.is_set():
                before = self._prompt
                key = self.source.read(self.poll_s, before)
                if not key:
                    continue
                event = KEYMAP.get(key.lower() if key.isalpha() else key)
                if event is None:
                    continue
                # A person answers what is on screen when the key lands; a script answers the prompt it saw.
                prompt = before if answers_prompt else self._prompt
                self._events.put((prompt.seq if prompt else -1, event))
        except BaseException as e:  # noqa: BLE001 - surfaced to the runner through ask()
            self.error = e
        finally:
            closer = getattr(self.source, "close", None)
            if closer:
                closer()

    @property
    def prompt(self) -> Prompt | None:
        return self._prompt

    def expect(self, context: str, allowed, info: dict | None = None) -> Prompt:
        """Start a new prompt and drop keys pressed before it."""
        with self._lock:
            self._seq += 1
            self._prompt = Prompt(self._seq, context, frozenset(allowed), dict(info or {}), time.perf_counter())
            while True:
                try:
                    self._events.get_nowait()
                except queue.Empty:
                    break
            return self._prompt

    def poll(self) -> str | None:
        """The next allowed event for the current prompt, without blocking."""
        prompt = self._prompt
        while True:
            try:
                seq, event = self._events.get_nowait()
            except queue.Empty:
                return None
            if prompt is not None and seq == prompt.seq and event in prompt.allowed:
                return event

    def ask(self, context: str, allowed, info: dict | None = None, timeout: float | None = None) -> str | None:
        """Start a prompt and block until one of its allowed events arrives (None on timeout)."""
        return self.wait(self.expect(context, allowed, info), timeout)

    def wait(self, prompt: Prompt, timeout: float | None = None) -> str | None:
        """Block until an allowed event for `prompt` arrives (None on timeout or stop). Use after expect()
        when something has to be shown in between, so a fast key cannot land before the prompt exists."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self.error is not None:
                raise RuntimeError(f"key listener failed: {self.error!r}")
            slice_s = 0.1 if deadline is None else max(0.0, min(0.1, deadline - time.monotonic()))
            try:
                seq, event = self._events.get(timeout=slice_s)
            except queue.Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    return None
                if self._stop.is_set():
                    return None
                continue
            if seq == prompt.seq and event in prompt.allowed:
                return event
