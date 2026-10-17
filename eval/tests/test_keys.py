import queue
import threading
import time

from hanoi_eval.keys import KeyListener, ScriptedKeys, describe, unattended_keys


class QueueSource:
    """An interactive source fed by the test, like a person at the terminal."""

    interactive = True

    def __init__(self):
        self.q = queue.Queue()

    def read(self, timeout, prompt):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None


def test_scripted_keys_answer_by_context_in_order():
    keys = ScriptedKeys({"start": "\n", "verdict": ["s", "f"]})
    with KeyListener(keys, poll_s=0.01) as listener:
        assert listener.ask("start", {"continue", "quit"}, timeout=2) == "continue"
        assert listener.ask("verdict", {"success", "fail"}, timeout=2) == "success"
        assert listener.ask("verdict", {"success", "fail"}, timeout=2) == "fail"
        assert listener.ask("verdict", {"success", "fail", "quit"}, timeout=2) == "quit"   # list ran out
        assert listener.ask("restore", {"continue"}, timeout=0.2) is None                  # nothing scripted
    assert keys.pressed == [("start", "\n"), ("verdict", "s"), ("verdict", "f"), ("verdict", "q")]


def test_keys_the_prompt_does_not_allow_are_dropped():
    source = QueueSource()
    with KeyListener(source, poll_s=0.01) as listener:
        listener.expect("step", {"success", "fail", "pause", "quit"})
        source.q.put("x")
        source.q.put("\n")
        source.q.put("S")
        deadline = time.time() + 2
        event = None
        while event is None and time.time() < deadline:
            event = listener.poll()
            time.sleep(0.01)
        assert event == "success"


def test_keys_pressed_before_a_prompt_do_not_answer_it():
    source = QueueSource()
    with KeyListener(source, poll_s=0.01) as listener:
        listener.expect("step", {"success", "fail"})
        source.q.put("f")
        time.sleep(0.1)
        listener.expect("verdict", {"success", "fail"})
        assert listener.poll() is None
        threading.Timer(0.1, source.q.put, args=("s",)).start()   # pressed after the prompt appears
        assert listener.ask("verdict", {"success", "fail"}, timeout=2) == "success"


def test_describe_lists_keys_in_a_fixed_order():
    assert describe({"quit", "success", "fail"}) == "s=success f=fail q=quit"
    assert describe({"continue", "retry"}) == "enter=continue r=retry"


def test_unattended_operator_confirms_only_auto_ends():
    keys = unattended_keys()
    with KeyListener(keys, poll_s=0.01) as listener:
        allowed = {"success", "fail", "quit"}
        assert listener.ask("start", {"continue", "quit"}, timeout=2) == "continue"
        assert listener.ask("verdict", allowed, {"end_reason": "auto"}, timeout=2) == "success"
        assert listener.ask("verdict", allowed, {"end_reason": "timeout"}, timeout=2) == "fail"
        assert listener.ask("restore", {"continue", "fail"}, timeout=2) == "continue"
    assert keys.pressed == [("start", "\n"), ("verdict", "s"), ("verdict", "f"), ("restore", "\n")]
