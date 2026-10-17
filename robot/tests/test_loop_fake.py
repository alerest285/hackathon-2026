"""ControlLoop with FakeRobot + FakeCameras + FakeNet. Timing asserts are loose on purpose: a busy Mac
running the suite must not make these flaky."""

import threading
import time

import numpy as np
import pytest

from so101_client.fakes import FakeCameras, FakeNet, FakeRobot
from so101_client.loop import BUS_HOLD_AFTER, ControlLoop, LoopConfig

TASK_A = "pick up the small red disk and place it on the right peg"
TASK_B = "pick up the medium blue disk and place it on the middle peg"


class Recorder:
    def __init__(self):
        self.ticks = []
        self.lock = threading.Lock()

    def __call__(self, info):
        with self.lock:
            self.ticks.append(info)

    def snapshot(self):
        with self.lock:
            return list(self.ticks)


def wait_for(predicate, timeout_s):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def accepted(ticks, task=None):
    return [i.chunk for i in ticks if i.chunk and i.chunk["accepted"] and (task is None or i.task == task)]


def test_prefetch_loop_task_switch_and_pause():
    robot = FakeRobot()
    cameras = FakeCameras(width=320, height=240)
    net = FakeNet("pickplace", latency_ms=400.0, jitter_ms=50.0)
    loop = ControlLoop(robot, cameras, net, LoopConfig(dry_run=False, caffeinate=False))
    rec = Recorder()
    loop.add_observer(rec)
    loop.add_observer(lambda info: 1 / 0)  # a broken observer must not stop the loop
    taps = []
    loop.add_frame_tap(lambda cam, bgr, t: taps.append((cam, bgr.shape)))
    loop.start()
    try:
        assert loop.hello["protocol"] == "so101.v1"
        loop.set_task(TASK_A, reset_history=True)
        assert wait_for(lambda: accepted(rec.snapshot()), 3.0)
        time.sleep(1.2)  # a few prefetch seams
        n = len(net.requests)
        assert wait_for(lambda: len(net.requests) > n, 2.0)  # a request is in flight now
        stale_id = net.requests[-1]["id"]
        loop.set_task(TASK_B, reset_history=False)
        assert wait_for(lambda: accepted(rec.snapshot(), TASK_B), 3.0)
        time.sleep(0.3)
        loop.pause()
        assert wait_for(lambda: rec.snapshot()[-1].task is None, 1.0)
        time.sleep(0.6)
    finally:
        loop.stop()

    ticks = rec.snapshot()
    assert loop.error is None
    assert not loop.running
    ks = [i.k for i in ticks]
    assert ks == list(range(ks[0], ks[0] + len(ks)))  # observers saw every tick

    # task A: holds only before the first chunk, which starts at 0; later chunks join at d > 0
    a = [i for i in ticks if i.task == TASK_A]
    first = next(j for j, i in enumerate(a) if i.chunk and i.chunk["accepted"])
    assert first > 0 and all(i.held for i in a[:first])
    assert not any(i.held for i in a[first:])
    assert a[first].chunk["start"] == 0
    assert any(c["start"] > 0 for c in accepted(a))
    steps = np.abs(np.diff(np.array([i.cmd for i in a[first:]])[:, :5], axis=0))
    assert steps.max() <= 6.0 + 1e-4

    # set_task(B): the chunk is dropped at once and the reply that was in flight is ignored
    b_start = next(j for j, i in enumerate(ticks) if i.task == TASK_B)
    assert ticks[b_start].held
    stale = [i.chunk for i in ticks if i.chunk and i.chunk["id"] == stale_id]
    assert len(stale) == 1 and not stale[0]["accepted"] and stale[0]["reason"] == "stale"
    b_chunks = accepted(ticks, TASK_B)
    assert b_chunks[0]["id"] > stale_id and b_chunks[0]["start"] == 0
    task_of = {r["id"]: r["task"] for r in net.requests}
    assert all(task_of[c["id"]] == TASK_B for c in b_chunks)

    # pause: holds, sends no request, the command stays put
    p_start = next(j for j in range(b_start, len(ticks)) if ticks[j].task is None)
    paused = ticks[p_start:]
    assert len(paused) > 5
    assert all(i.held and i.remaining == 0 for i in paused)
    assert all(np.array_equal(i.cmd, paused[0].cmd) for i in paused)
    assert not [r for r in net.requests if r["t"] > paused[0].t]

    # the fake arm followed the written commands
    assert robot.writes >= len(ticks) - 2
    assert np.abs(ticks[-1].state[:5] - ticks[-1].cmd[:5]).max() < 2.0
    assert taps and taps[0][1] == (240, 320, 3)

    s = loop.stats()
    assert s["plans"] >= 4 and s["stale"] >= 1 and s["errors"] == 0
    assert s["starved"] == 0
    assert s["observer_errors"] == len(ticks)
    assert s["lateness_p99_ms"] < 25.0
    assert s["turnaround_p50_ms"] > 300.0


def test_sync_loop_executes_32_then_holds_and_replans():
    net = FakeNet("sine", latency_ms=150.0, jitter_ms=0.0)
    loop = ControlLoop(FakeRobot(), FakeCameras(width=320, height=240), net,
                       LoopConfig(mode="sync", dry_run=False, caffeinate=False))
    rec = Recorder()
    loop.add_observer(rec)
    loop.start()
    try:
        loop.set_task(TASK_A, reset_history=True)
        assert wait_for(lambda: len(accepted(rec.snapshot())) >= 3, 8.0)
    finally:
        loop.stop()
    assert loop.error is None
    ticks = [i for i in rec.snapshot() if i.task == TASK_A]
    assert all(c["start"] == 0 for c in accepted(ticks))
    runs, run, last_exec = [], 0, []
    for prev, cur in zip(ticks, ticks[1:]):
        if not cur.held:
            run += 1
        if prev.held is False and cur.held:
            runs.append(run)
            last_exec.append(prev.k)
            run = 0
    assert runs[:2] == [32, 32]
    request_ticks = {r["tick"] for r in net.requests}
    assert all(k + 1 in request_ticks for k in last_exec)  # the request goes out on the first hold tick


def test_dry_run_writes_nothing():
    robot = FakeRobot()
    loop = ControlLoop(robot, FakeCameras(width=160, height=120), FakeNet("sine", latency_ms=100.0, jitter_ms=0.0),
                       LoopConfig(dry_run=True, caffeinate=False))
    rec = Recorder()
    loop.add_observer(rec)
    loop.start()
    try:
        loop.set_task(TASK_A, reset_history=True)
        assert wait_for(lambda: len(accepted(rec.snapshot())) >= 2, 4.0)
    finally:
        loop.stop()
    assert robot.writes == 0
    moved = [i for i in rec.snapshot() if not i.held]
    assert moved and np.abs(moved[-1].cmd[:5] - moved[0].cmd[:5]).max() > 0.1  # commands were still computed


class GlitchyRobot(FakeRobot):
    """FakeRobot whose next `fail` state reads raise, like Feetech status packets that come back corrupted."""

    fail = 0

    def read_state(self):
        if self.fail > 0:
            self.fail -= 1
            raise OSError("corrupted status packet")
        return super().read_state()


def test_a_failed_state_read_keeps_the_chunk_going_and_is_not_starvation():
    robot = GlitchyRobot()
    loop = ControlLoop(robot, FakeCameras(width=160, height=120), FakeNet("sine", latency_ms=100.0, jitter_ms=0.0),
                       LoopConfig(dry_run=False, caffeinate=False))
    rec = Recorder()
    loop.add_observer(rec)
    loop.start()
    try:
        loop.set_task(TASK_A, reset_history=True)
        assert wait_for(lambda: len(accepted(rec.snapshot())) >= 2, 4.0)
        robot.fail = 1  # one glitch: the chunk goes on
        assert wait_for(lambda: loop.stats()["bus_errors"] == 1, 1.0)
        time.sleep(0.3)
        robot.fail = BUS_HOLD_AFTER  # that many in a row: the last one holds
        assert wait_for(lambda: loop.stats()["bus_errors"] == 1 + BUS_HOLD_AFTER, 1.0)
        time.sleep(0.3)
        assert loop.pending in (0, 1)
    finally:
        loop.stop()
    assert loop.error is None
    s = loop.stats()
    assert s["starved"] == 0 and s["bus_holds"] == 1
    ticks = rec.snapshot()
    first = next(j for j, i in enumerate(ticks) if i.chunk and i.chunk["accepted"])
    assert sum(i.held for i in ticks[first:]) == 1


def test_joint_limits_need_lo_below_hi():
    with pytest.raises(ValueError, match="lo < hi"):
        ControlLoop(FakeRobot(), FakeCameras(), FakeNet(), LoopConfig(joint_limits=((10.0, -10.0),) * 5))
