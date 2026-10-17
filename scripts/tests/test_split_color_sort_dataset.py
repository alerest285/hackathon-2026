from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("split_color_sort_dataset", ROOT / "scripts" / "split_color_sort_dataset.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SplitPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = MODULE.load_plan(ROOT / "scripts" / "color_sort_splits.json")

    def test_plan_covers_all_source_episodes(self):
        self.assertEqual(len(self.plan["lengths"]), 82)
        self.assertEqual(sum(self.plan["lengths"]), 51922)

    def test_ranges_conserve_frames_and_are_ordered_by_requested_task(self):
        counts = {color: 0 for color in MODULE.COLORS}
        frames = 0
        for episode, length in enumerate(self.plan["lengths"]):
            parts = MODULE.episode_parts(self.plan, episode)
            self.assertEqual([part[0] for part in parts], ["orange", "green"])
            covered = sorted((start, end) for _, start, end in parts)
            self.assertEqual(covered, [(0, covered[0][1]), (covered[0][1], length)])
            for color, start, end in parts:
                counts[color] += 1
                frames += end - start
        self.assertEqual(counts, {"orange": 82, "green": 82})
        self.assertEqual(frames, 51922)

    def test_reverse_order_source_episode_is_reordered_without_losing_frames(self):
        self.assertEqual(self.plan["first_task"][0], "green")
        self.assertEqual(MODULE.episode_parts(self.plan, 0), [("orange", 550, 1234), ("green", 0, 550)])

    def test_manually_verified_nonstandard_gripper_ranges(self):
        expected_release = {54: 240, 64: 270, 65: 330, 71: 360, 72: 360, 73: 390, 77: 396, 78: 300}
        for episode, release in expected_release.items():
            self.assertEqual(self.plan["release_frames"][episode], release)


if __name__ == "__main__":
    unittest.main()
