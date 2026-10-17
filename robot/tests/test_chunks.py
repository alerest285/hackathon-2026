import numpy as np
import pytest

from so101_client.chunks import Planner, Stitcher, clamp, crossfade_weights, integrate, to_deltas


def ramp(anchor, velocity, n=42, offset=0.0, gripper=None):
    """Absolute chunk whose joints move at a constant velocity from anchor + offset."""
    anchor = np.asarray(anchor, dtype=np.float64)
    out = np.empty((n, 6))
    out[:, :5] = anchor[:5] + offset + velocity * (np.arange(n)[:, None] + 1)
    out[:, 5] = anchor[5] if gripper is None else gripper
    return out.astype(np.float32)


def run(stitcher, last, n):
    """Execute n ticks the way the loop does: add each joint delta to the last command, gripper absolute."""
    cmds = []
    for _ in range(n):
        step = stitcher.next()
        if step is None:
            break
        delta, g = step
        last = last.copy()
        last[:5] += delta
        last[5] = g
        cmds.append(last)
    return np.array(cmds), last


def test_reintegration_from_the_anchor_reproduces_the_chunk():
    rng = np.random.default_rng(0)
    anchor = rng.uniform(-90, 90, 6).astype(np.float32)
    actions = (anchor + np.cumsum(rng.normal(0, 0.5, (42, 6)), 0)).astype(np.float32)
    actions[:, 5] = rng.uniform(0, 100, 42)
    jd, g = to_deltas(actions, anchor)
    assert jd.shape == (42, 5) and g.shape == (42,)
    np.testing.assert_allclose(jd[0], actions[0, :5].astype(np.float64) - anchor[:5])
    np.testing.assert_allclose(integrate(jd, g, anchor), actions, atol=1e-9)

    stitcher = Stitcher(42, 6)
    assert stitcher.accept(actions, anchor, d=9) == 0  # nothing active: a hold, so index 0
    cmds, _ = run(stitcher, anchor.astype(np.float64), 42)
    np.testing.assert_allclose(cmds, actions, atol=1e-9)
    assert stitcher.next() is None


def test_planner_matches_the_chunk_with_float32_commands():
    anchor = np.array([10, -80, 70, 40, 5, 30], np.float32)
    actions = ramp(anchor, np.array([0.3, -0.2, 0.1, 0.05, -0.4]), gripper=np.linspace(30, 2, 42))
    planner = Planner("prefetch")
    assert planner.offer(actions, anchor, d=0) == 0
    last, cmds = anchor.copy(), []
    for _ in range(42):
        last = planner.step(last).astype(np.float32)
        cmds.append(last)
    np.testing.assert_allclose(np.array(cmds), actions, atol=1e-3)


def test_no_position_jump_at_a_seam():
    anchor = np.zeros(6)
    a = ramp(anchor, 0.5)
    stitcher = Stitcher(42, 6)
    stitcher.accept(a, anchor, 0)
    first, last = run(stitcher, anchor, 16)
    # the next plan believes the arm is 20 deg elsewhere and moves the other way
    b = ramp(anchor, -0.5, offset=20.0)
    assert stitcher.accept(b, anchor, d=12) == 12
    second, _ = run(stitcher, last, 30)
    cmds = np.vstack([first, second])
    steps = np.abs(np.diff(cmds[:, :5], axis=0))
    assert steps.max() <= 0.5 + 1e-9
    # the joint deltas blend from +0.5 to -0.5 over K ticks, then follow the new chunk
    seam = np.diff(np.vstack([first[-1:], second]), axis=0)[:, 0]
    w = crossfade_weights(6)
    np.testing.assert_allclose(seam[:6], (1 - w) * 0.5 + w * -0.5, atol=1e-9)
    np.testing.assert_allclose(seam[6:], -0.5, atol=1e-9)


def test_crossfade_weights():
    np.testing.assert_allclose(crossfade_weights(6), [1 / 7, 2 / 7, 3 / 7, 4 / 7, 5 / 7, 6 / 7])
    assert len(crossfade_weights(0)) == 0


def test_crossfade_stops_when_the_old_chunk_runs_out():
    anchor = np.zeros(6)
    stitcher = Stitcher(42, 6)
    stitcher.accept(ramp(anchor, 1.0), anchor, 0)
    run(stitcher, anchor, 39)  # 3 actions of the old chunk left
    stitcher.accept(ramp(anchor, 0.0), anchor, d=10)
    cmds, _ = run(stitcher, anchor, 6)
    deltas = np.diff(np.vstack([np.zeros(6), cmds]), axis=0)[:, 0]
    w = crossfade_weights(6)
    np.testing.assert_allclose(deltas[:3], (1 - w[:3]) * 1.0)
    np.testing.assert_allclose(deltas[3:], 0.0)


def test_drop_rule():
    anchor = np.zeros(6)
    stitcher = Stitcher(42, 6)
    stitcher.accept(ramp(anchor, 0.1), anchor, 0)
    run(stitcher, anchor, 5)
    assert stitcher.accept(ramp(anchor, 0.2), anchor, d=36) is None  # 36 > 42 - 1 - 6
    assert stitcher.index == 5  # the running chunk is untouched
    assert stitcher.accept(ramp(anchor, 0.2), anchor, d=35) == 35
    assert stitcher.remaining == 7
    stitcher.reset()
    assert stitcher.accept(ramp(anchor, 0.2), anchor, d=40) == 0  # after a hold any d starts at 0


def test_join_index_depends_on_a_hold_not_on_an_empty_chunk():
    anchor = np.zeros(6, np.float32)
    planner = Planner("prefetch")
    planner.offer(ramp(anchor, 0.1), anchor, d=0)
    last = anchor.astype(np.float64)
    for _ in range(42):
        last = planner.step(last)
    # the chunk is used up but no tick has held yet: a reply arriving now joins at d
    assert planner.offer(ramp(anchor, 0.1), anchor, d=10) == 10
    for _ in range(32):
        last = planner.step(last)
    assert planner.step(last) is None  # a hold
    assert planner.offer(ramp(anchor, 0.1), anchor, d=10) == 0  # after a hold: index 0


def test_gripper_stays_absolute():
    anchor = np.array([0, 0, 0, 0, 0, 90.0])
    g1 = np.linspace(90, 10, 42)
    g2 = np.linspace(5, 60, 42)
    stitcher = Stitcher(42, 6)
    stitcher.accept(ramp(anchor, 0.1, gripper=g1), anchor, 0)
    cmds, last = run(stitcher, np.array([0, 0, 0, 0, 0, 50.0]), 10)
    np.testing.assert_allclose(cmds[:, 5], g1[:10].astype(np.float32))
    stitcher.accept(ramp(anchor, 0.1, gripper=g2), anchor, d=8)
    cmds, _ = run(stitcher, last, 8)
    np.testing.assert_allclose(cmds[:, 5], g2[8:16].astype(np.float32))  # from the new chunk, also during the fade


def test_clamp():
    last = np.array([0, 0, 0, 0, 0, 50.0])
    out = clamp([10, -10, 3, 0, 0, 120], last, step_max=6.0)
    np.testing.assert_allclose(out, [6, -6, 3, 0, 0, 100])
    assert clamp([0, 0, 0, 0, 0, -5], np.array([0, 0, 0, 0, 0, 100.0]), step_max=6.0)[5] == 0.0  # gripper not rate-limited
    limits = [(-5, 5)] * 5
    np.testing.assert_allclose(clamp([20, -20, 1, 0, 0, 50], last, limits, 6.0)[:5], [5, -5, 1, 0, 0])
    # limits first, then the step: inside the limits it stays inside; outside it moves back at most step_max
    inside = clamp([20, 0, 0, 0, 0, 50], np.array([4.9, 0, 0, 0, 0, 50]), limits, 6.0)
    assert inside[0] == 5.0
    outside = clamp([20, 0, 0, 0, 0, 50], np.array([20.0, 0, 0, 0, 0, 50]), limits, 6.0)
    assert outside[0] == 14.0
    np.testing.assert_allclose(clamp([1, 2, 3, 4, 5, 6], last, None, None), [1, 2, 3, 4, 5, 6])


def test_to_deltas_rejects_bad_shapes():
    with pytest.raises(ValueError):
        to_deltas(np.zeros((42, 5)), np.zeros(6))
