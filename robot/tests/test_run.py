"""run.py helpers without hardware."""

import json

import pytest
import run

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def test_load_limits_needs_lo_below_hi(tmp_path):
    good = {name: [-10, 10] for name in JOINTS}
    path = tmp_path / "limits.json"
    path.write_text(json.dumps({**good, "gripper_seen": [2.0, 61.0]}))
    assert run.load_limits(path) == ((-10.0, 10.0),) * 5
    path.write_text(json.dumps({**good, "elbow_flex": [10, 3]}))
    with pytest.raises(ValueError, match="elbow_flex"):
        run.load_limits(path)
    assert run.main(["--task", "t", "--fake-robot", "--limits", str(path)]) == 2
