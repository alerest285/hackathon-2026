import itertools

import numpy as np

from so101_client.chunks import Planner, Scheduler


def test_prefetch_trigger_and_one_in_flight():
    s = Scheduler("prefetch", period=16, crossfade=6)
    assert s.due(0, 0)  # holding at the start
    s.sent(0, 1)
    assert not s.due(1, 0) and not s.due(40, 0)  # one request in flight, whatever happens
    assert s.received(1, d=12)
    s.started()
    assert not s.due(12, 30)  # 12 ticks since the request, 30 actions left
    assert s.due(16, 30)  # k - last_request >= P
    assert s.d_max == 12
    assert s.due(13, 20) and not s.due(13, 21)  # remaining <= d_max + K + 2 = 20


def test_reply_for_another_request_does_not_free_the_slot():
    s = Scheduler("prefetch")
    s.sent(0, 7)
    assert not s.received(3, d=5)
    assert s.inflight == 7
    s.failed(7)
    assert s.inflight is None


def test_d_max_is_the_worst_of_the_last_100_plans():
    s = Scheduler("prefetch")
    for d in [40] + [10] * 99:
        s.received(0, d=d)
    assert s.d_max == 40
    s.received(0, d=11)
    assert s.d_max == 11


def test_sync_requests_only_while_holding():
    s = Scheduler("sync", exec_horizon=32)
    assert s.due(0, 0) and s.appends()
    s.sent(0, 1)
    assert not s.appends()  # waiting for the reply: not in the history
    s.received(1, d=5)
    s.started()
    assert s.appends() and not s.due(5, 32)
    for _ in range(31):
        assert not s.stepped()
    assert s.stepped()  # 32nd action, on tick 36: the rest of the chunk is dropped
    assert s.remaining(10) == 0 and not s.due(36, 0)
    s.hold()  # tick 37: nothing left, the hold starts
    assert s.due(37, 0) and s.appends()


def simulate(mode, latency, n_ticks, velocity=0.25, period=16):
    """Drive a Planner the way ControlLoop does, with a server that answers after `latency` ticks."""
    planner = Planner(mode, period=period)
    ids = itertools.count(1)
    inflight = {}  # rid -> (request tick, anchor, due tick)
    last = np.zeros(6)
    log = []
    for k in range(n_ticks):
        anchor_now = last.copy()  # cmd_hist[-1] of a window built at this tick
        appended = planner.appends()
        accepted = None
        for rid in [r for r, v in inflight.items() if v[2] == k]:
            r, anchor, _ = inflight.pop(rid)
            planner.received(rid, k - r)
            actions = np.zeros((42, 6), np.float32)
            actions[:, :5] = anchor[:5] + velocity * (np.arange(42)[:, None] + 1)
            start = planner.offer(actions, anchor, k - r)
            accepted = {"id": rid, "tick": r, "d": k - r, "start": start}
        cmd = planner.step(last)
        if cmd is not None:
            last = cmd
        requested = None
        if planner.due(k):
            requested = next(ids)
            planner.sent(k, requested)
            inflight[requested] = (k, anchor_now, k + latency)
        log.append({"k": k, "appended": appended, "held": cmd is None, "accepted": accepted,
                    "requested": requested, "cmd": last.copy()})
    return log


def test_prefetch_cadence_never_starves_after_the_first_chunk():
    log = simulate("prefetch", latency=12, n_ticks=300)
    requests = [e["k"] for e in log if e["requested"]]
    assert requests[:4] == [0, 16, 32, 48]
    assert all(b - a == 16 for a, b in zip(requests, requests[1:]))
    first = next(e["k"] for e in log if e["accepted"])
    assert first == 12
    assert all(e["held"] for e in log[:first]) and not any(e["held"] for e in log[first:])
    starts = [e["accepted"]["start"] for e in log if e["accepted"]]
    assert starts[0] == 0 and set(starts[1:]) == {12}
    assert all(e["appended"] for e in log)
    steps = np.abs(np.diff(np.array([e["cmd"] for e in log])[:, :5], axis=0))
    assert steps.max() <= 0.25 + 1e-9


def test_prefetch_requests_early_when_the_remaining_actions_run_low():
    # P out of the way: only remaining <= d_max + K + 2 (= 20 with d = 12) triggers a request
    log = simulate("prefetch", latency=12, n_ticks=200, period=1000)
    requests = [e["k"] for e in log if e["requested"]]
    assert requests[:4] == [0, 33, 54, 75]
    first = next(e["k"] for e in log if e["accepted"])
    assert not any(e["held"] for e in log[first:])


def test_prefetch_with_one_request_in_flight_starves_beyond_21_ticks():
    # each chunk joined at d leaves 42 - d actions while replies come every d ticks
    assert not any(e["held"] for e in simulate("prefetch", latency=21, n_ticks=300)[21:])
    assert any(e["held"] for e in simulate("prefetch", latency=25, n_ticks=300)[25:])


def test_prefetch_drops_a_reply_that_is_too_late_and_holds():
    log = simulate("prefetch", latency=40, n_ticks=200)
    accepted = [e["accepted"] for e in log if e["accepted"]]
    # first chunk after the hold starts at 0; the next arrives with d = 40 > 35 while it runs: dropped
    assert accepted[0]["start"] == 0
    assert any(a["start"] is None for a in accepted)


def test_sync_cadence_execute_32_hold_request_start_at_zero():
    latency = 5
    log = simulate("sync", latency=latency, n_ticks=200)
    requests = [e["k"] for e in log if e["requested"]]
    # request at 0, reply at 5, 32 actions on ticks 5..36, request at 37 from the window of tick 37, ...
    assert requests[:3] == [0, 37, 74]
    assert all(b - a == 32 + latency for a, b in zip(requests, requests[1:]))
    accepted = [e for e in log if e["accepted"]]
    assert all(e["accepted"]["start"] == 0 for e in accepted)
    assert [e["k"] for e in accepted][:3] == [5, 42, 79]
    executed = [e["k"] for e in log if not e["held"]]
    assert executed[:32] == list(range(5, 37)) and executed[32] == 42
    by_k = {e["k"]: e for e in log}
    assert by_k[37]["appended"]  # the request tick enters the history
    assert not any(by_k[k]["appended"] for k in range(38, 43))  # the wait and the reply tick do not
    assert by_k[43]["appended"]
