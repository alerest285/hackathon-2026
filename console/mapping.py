"""Per-joint linear map between OUR arm's joint values and the convention the checkpoint was trained in.

Why: BFL trained on ~1,000 community SO-100/SO-101 datasets, each normalized by its OWN joint range
(configs/so101/community_corpus.json, docs/so101-lora.md). The released flux-3-action-so101 normalizes with one fixed
range (its state_q01..state_q99, sent in the server's hello). An arm calibrated differently (LeRobot's default puts the
middle of each joint at 0 deg) must be mapped into that range, the same way training rescaled every dataset:

    model = m_lo + (ours - o_lo) * (m_hi - m_lo) / (o_hi - o_lo)        per joint, and the inverse for commands

o_lo/o_hi are OUR arm's 1st/99th percentiles over its workspace, measured by fit_mapping.py (hand-guided sweep).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
# flux-3-action-so101 @ c9e13b2a (decision 06 pin): state_q01 / state_q99 from the server hello, 17.10.2026.
MODEL_Q01 = [-92.021484375, 84.814453125, 61.435546875, -23.994140625, -124.541015625, 0.4595588445663452]
MODEL_Q99 = [54.140625, 184.21875, 178.857421875, 80.595703125, 60.205078125, 53.57535934448242]
DEFAULT_PATH = Path("~/hackathon-data/joint_mapping.json").expanduser()


class JointMap:
    def __init__(self, ours_lo, ours_hi, model_lo=MODEL_Q01, model_hi=MODEL_Q99, meta=None):
        self.o_lo, self.o_hi = np.asarray(ours_lo, float), np.asarray(ours_hi, float)
        self.m_lo, self.m_hi = np.asarray(model_lo, float), np.asarray(model_hi, float)
        span = self.o_hi - self.o_lo
        if np.any(np.abs(span) < 1e-3):
            raise ValueError(f"a joint did not move during the sweep: ours span {np.round(span, 2).tolist()}")
        self.k = (self.m_hi - self.m_lo) / span
        self.meta = meta or {}

    def to_model(self, x):
        return (self.m_lo + (np.asarray(x, float) - self.o_lo) * self.k).astype(np.float32)

    def to_ours(self, y):
        return (self.o_lo + (np.asarray(y, float) - self.m_lo) / self.k).astype(np.float32)

    def limits_to_model(self, limits):
        """5 x (lo, hi) in our degrees -> model space (order kept even if a slope is negative)."""
        if limits is None:
            return None
        out = []
        for j, (lo, hi) in enumerate(limits):
            a, b = self.m_lo[j] + (lo - self.o_lo[j]) * self.k[j], self.m_lo[j] + (hi - self.o_lo[j]) * self.k[j]
            out.append((float(min(a, b)), float(max(a, b))))
        return tuple(out)

    def to_json(self) -> dict:
        return {"joints": list(JOINTS), "ours_q01": self.o_lo.tolist(), "ours_q99": self.o_hi.tolist(),
                "model_q01": self.m_lo.tolist(), "model_q99": self.m_hi.tolist(), "meta": self.meta}

    def save(self, path: Path = DEFAULT_PATH):
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1))

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "JointMap":
        d = json.loads(Path(path).expanduser().read_text())
        return cls(d["ours_q01"], d["ours_q99"], d["model_q01"], d["model_q99"], d.get("meta"))


class MappedRobot:
    """Wraps the arm so the control loop, the server and the history all live in the checkpoint's convention.
    read_state() returns model-space values, write() takes model-space commands and sends our degrees."""

    def __init__(self, robot, jmap: JointMap):
        self._r, self.map = robot, jmap
        self.last_raw_state = None

    def __getattr__(self, name):  # connected, set_torque, calibration_path, ... pass through
        return getattr(self._r, name)

    def read_state(self):
        raw = self._r.read_state()
        self.last_raw_state = np.asarray(raw, float)
        return self.map.to_model(raw)

    def write(self, cmd):
        out = self.map.to_ours(cmd)
        out[5] = float(np.clip(out[5], 0.0, 100.0))  # the gripper stays in our 0..100
        return self._r.write(out)
