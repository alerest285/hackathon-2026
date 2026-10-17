"""Engine interface and helpers shared by the engines. No torch here."""

import math
import os
import time

import numpy as np


class Engine:
    """One policy behind the server.

    ``infer(state (8, 6) f32, cmd_hist (8, 6) f32, frames {cam: uint8 (2, H, W, 3) RGB [i0, i7]}, task)``
    returns ``(42, 6)`` float32 absolute commands for ticks t..t+41. The server calls ``warmup`` and
    ``infer`` from one worker thread, one call at a time. ``meta`` is merged into the hello message.
    """

    name = "engine"

    def __init__(self):
        self.meta: dict = {}
        self.warm = False
        self.warm_tasks: list[str] = []  # captions encoded and cached by warmup()
        self.first_plan_ms: float | None = None  # plan 1 of the first warmup: compile, kernels and the caption

    def infer(self, state: np.ndarray, cmd_hist: np.ndarray, frames: dict, task: str) -> np.ndarray:
        raise NotImplementedError

    def warmup(self, task: str) -> None:
        """Two plans on a synthetic window: the first fills caches (caption, kernels), the second is steady state."""
        state, cmd_hist, frames = synthetic_window(self.meta)
        times = []
        for _ in range(2):
            t0 = time.perf_counter()
            actions = np.asarray(self.infer(state, cmd_hist, frames, task))
            times.append((time.perf_counter() - t0) * 1000)
            if actions.shape != (self.meta.get("chunk", 42), 6) or not np.isfinite(actions).all():
                raise RuntimeError(f"warmup produced invalid actions {actions.shape}")
        self.warm = True
        if self.first_plan_ms is None:
            self.first_plan_ms = times[0]
        if task not in self.warm_tasks:
            self.warm_tasks.append(task)
        print(f"warmup: {task!r} plan 1 {times[0]:.1f} ms, plan 2 {times[1]:.1f} ms", flush=True)


def snapshot_indices(n_obs: int, snapshots: int) -> list[int]:
    """Window slots that are VAE-encoded (flux-action processing/history.py snapshot_indices)."""
    if snapshots == 1:
        return [n_obs - 1]
    return [round(i * (n_obs - 1) / (snapshots - 1)) for i in range(snapshots)]


def window_order(n_obs: int, snapshots: list[int]) -> list[int]:
    """For each window slot, the index of the frame it shows, given one frame per snapshot slot.

    Slot j repeats the newest snapshot at or before j. Only the snapshot slots are encoded, so the
    repeats never reach the model: [0, 7] gives [0, 0, 0, 0, 0, 0, 0, 1].
    """
    order = []
    for j in range(n_obs):
        before = [i for i, s in enumerate(snapshots) if s <= j]
        order.append(before[-1] if before else 0)
    return order


def pin_revision(spec: str | None, revision: str | None, force: bool = False) -> tuple[str | None, str]:
    """Pin a Hub weight spec ``repo_id[:file][@revision]`` to ``revision``. Returns (spec, note).

    The checkpoint config's own pin wins, as in flux-action and LeRobot: a spec without a revision gets
    ``@revision``, a spec pinned to the same revision stays, and a spec pinned to a different one raises
    ValueError unless ``force``, which loads ``revision`` instead. Local paths, empty specs and an empty
    pin pass through unchanged.
    """
    if not spec or not revision or os.path.exists(spec):
        return spec, ""
    body, _, current = spec.partition("@")
    if current == revision:
        return spec, f"{body} pinned at @{revision[:8]} by the checkpoint config, as --base-revision says"
    if not current:
        return f"{body}@{revision}", f"{body} has no pin in the checkpoint config; pinned to @{revision[:8]}"
    if not force:
        raise ValueError(
            f"the checkpoint config pins {spec}, but --base-revision is @{revision[:8]}. Pass --base-revision '' "
            f"to load the checkpoint's own pin (what flux-action and LeRobot do), --base-revision {current} to "
            f"name it, or --force-base-revision to load @{revision[:8]} anyway"
        )
    return f"{body}@{revision}", f"WARNING: checkpoint config pins {spec}; --force-base-revision loads @{revision[:8]}"


def override_sampling(cfg, guidance=None, steps: int | None = None) -> list[str]:
    """Set guidance (video, action) and the step count on a policy config. Both engines read them at every
    plan, and guidance 1/1 skips the unconditional pass. Returns a WARNING line per value that differs
    from the checkpoint's (decision 06: such a change is its own decision)."""
    notes = []
    if guidance is not None:
        video, action = (float(g) for g in guidance)
        if not (math.isfinite(video) and math.isfinite(action)):
            raise ValueError(f"guidance must be finite, got {video}/{action}")
        action_before = cfg.guidance_scale if cfg.guidance_scale_action is None else cfg.guidance_scale_action
        before = (float(cfg.guidance_scale), float(action_before))
        cfg.guidance_scale, cfg.guidance_scale_action = video, action
        if before != (video, action):
            notes.append(f"WARNING: guidance {video:g}/{action:g} instead of the checkpoint's {before[0]:g}/{before[1]:g}")
    if steps is not None:
        if int(steps) < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        before_steps = cfg.num_inference_steps
        cfg.num_inference_steps = int(steps)
        if before_steps != int(steps):
            notes.append(f"WARNING: {int(steps)} sampling steps instead of the checkpoint's {before_steps}")
    return notes


def synthetic_frames(size: int = 256, seed: int = 0) -> np.ndarray:
    """Two deterministic RGB frames (2, size, size, 3) uint8: gradient, blocks and noise, so JPEG sizes are realistic."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    base = np.stack([200 * x + 30, 160 * y + 40, 120 * (1 - x) + 60], -1)
    blocks = ((np.floor(x * 8) + np.floor(y * 8)) % 2)[..., None] * 25
    frames = []
    for shift in (0, size // 16):
        img = np.roll(base + blocks, shift, axis=1) + rng.normal(0, 6, base.shape)
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    return np.stack(frames)


def synthetic_window(meta: dict, size: int = 256):
    """State, command history and frames for warmup and self-test: an episode start (8 copies, command = state).

    The state is the middle of the checkpoint's state range when the engine publishes it, else zeros.
    """
    n_obs = int(meta.get("n_obs", 8))
    lo, hi = meta.get("state_q01"), meta.get("state_q99")
    state = (np.asarray(lo) + np.asarray(hi)) / 2 if lo is not None and hi is not None else np.zeros(6)
    window = np.tile(np.asarray(state, dtype=np.float32), (n_obs, 1))
    frames = {cam: synthetic_frames(size, seed=i) for i, cam in enumerate(meta.get("cameras", ("scene", "wrist")))}
    return window, window.copy(), frames
