"""Independent T23 review probes. Run with LeRobot's dataset dependencies installed.

python docs/research/check_lora_data_admission.py

Uses temporary files, no GPU, no robot, and no edits to recorded datasets. Admission
is tested with complete videos assumed; a separate probe uses real H.264 files.
"""

import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np


def main():
    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("review_prepare", repo / "train/prepare.py")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    with tempfile.TemporaryDirectory(prefix="lora-admission-review-") as temp:
        root = Path(temp)
        ds = SimpleNamespace(meta=SimpleNamespace(
            total_episodes=1, episodes=[{"length": 60, "tasks": ["pick"]}]))
        video_check = prepare.video_shortfalls
        prepare.video_shortfalls = lambda meta, path: {}
        cases = [
            ("live_success", {"run_mode": "live", "valid": True, "verdict": "success"}, False),
            ("dry_success", {"run_mode": "dry", "valid": False, "verdict": "success"}, True),
            ("fake_success", {"run_mode": "fake", "valid": False, "verdict": "success"}, True),
            ("encoder_loss", {"run_mode": "live", "valid": False, "verdict": "success",
                              "encoder_dropped": {"scene": 6}}, True),
            ("legacy_unknown_mode", {"verdict": "success"}, True),
            ("unknown_mode", {"run_mode": "unknown", "valid": True, "verdict": "success"}, True),
            ("null_valid", {"run_mode": "live", "valid": None, "verdict": "success"}, True),
            ("missing_episode", None, True),
        ]
        results = {}
        for label, side, must_drop in cases:
            (root / "console_episodes.jsonl").write_text(json.dumps({
                "episode_index": 1 if side is None else 0,
                "dropped_frames": 0, **(side or {}),
            }) + "\n")
            drop, reasons, kept = prepare.select_episodes(ds, root, "success", 50, {"pick"}, False, False)
            assert (0 in drop) == must_drop, (label, drop, reasons, kept)
            results[label] = "rejected" if must_drop else "accepted"
        prepare.video_shortfalls = video_check

        for camera, count in [("scene", 59), ("wrist", 60)]:
            with av.open(str(root / f"{camera}.mp4"), "w") as container:
                stream = container.add_stream("libx264", rate=30)
                stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
                for i in range(count):
                    frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i, np.uint8), format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
        episode = {"length": 60}
        for camera in prepare.CAMERAS:
            episode[f"videos/{camera}/from_timestamp"] = 0.0
            episode[f"videos/{camera}/to_timestamp"] = 2.0
        meta = SimpleNamespace(episodes=[episode], total_episodes=1, fps=30,
                               get_video_file_path=lambda i, camera: f"{camera.split('.')[-1]}.mp4")
        shortfalls = prepare.video_shortfalls(meta, root)
        assert 0 in shortfalls, "59-frame scene video was accepted for 60 rows"
        results["one_missing_video_frame"] = shortfalls[0]
        print(json.dumps({"status": "PASS", "checks": results}, indent=2))


if __name__ == "__main__":
    main()
