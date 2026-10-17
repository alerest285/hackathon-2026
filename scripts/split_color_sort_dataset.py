#!/usr/bin/env python3
"""Split two-move LeRobot color-sort demonstrations into one episode per color.

The split annotations are deliberately data-specific and checked into
``color_sort_splits.json``. They mark the first object's release; a short settling
tail stays with that action. Output episodes are always ordered orange then green,
even for source episode 0, whose demonstrations were recorded in the reverse order.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

DERIVED = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
COLORS = ("orange", "green")
DEFAULT_PLAN = Path(__file__).with_name("color_sort_splits.json")


def load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    lengths = plan["lengths"]
    releases = plan["release_frames"]
    first = plan["first_task"]
    if not lengths or not (len(lengths) == len(releases) == len(first)):
        raise ValueError("lengths, release_frames, and first_task must have the same non-zero length")
    if set(plan["tasks"]) != set(COLORS):
        raise ValueError(f"tasks must contain exactly {COLORS}")
    for episode, (length, release, first_color) in enumerate(zip(lengths, releases, first, strict=True)):
        split = release + plan["settle_frames"]
        if first_color not in COLORS:
            raise ValueError(f"episode {episode}: invalid first_task {first_color!r}")
        if not 0 < release < split < length:
            raise ValueError(f"episode {episode}: invalid length/release/settle {length}/{release}/{split}")
    return plan


def episode_parts(plan: dict, episode: int) -> list[tuple[str, int, int]]:
    """Return orange then green ranges as half-open source-frame intervals."""
    length = plan["lengths"][episode]
    split = plan["release_frames"][episode] + plan["settle_frames"]
    first = plan["first_task"][episode]
    chronological = {first: (0, split), COLORS[1 - COLORS.index(first)]: (split, length)}
    return [(color, *chronological[color]) for color in COLORS]


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return value


def _output_frame(item: dict, features: dict, video_keys: set[str], task: str) -> dict:
    frame = {}
    for key in features:
        value = _numpy(item[key])
        if key in video_keys:
            # LeRobot readers return RGB video as CHW; writers take HWC.
            if value.ndim != 3:
                raise ValueError(f"{key}: expected a 3-D image, got {value.shape}")
            if value.shape[0] in (1, 3, 4):
                value = value.transpose(1, 2, 0)
        frame[key] = value
    frame["task"] = task
    return frame


def convert(
    plan: dict,
    plan_path: Path,
    out_root: Path,
    repo_id: str,
    source_root: Path | None,
    video_backend: str,
    workers: int,
) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if out_root.exists():
        raise FileExistsError(f"{out_root} already exists; output is immutable, choose another path")
    source = LeRobotDataset(
        plan["source_repo_id"],
        root=source_root,
        revision=plan["source_revision"],
        video_backend=video_backend,
        return_uint8=True,
    )
    if source.meta.total_episodes != len(plan["lengths"]):
        raise ValueError(f"source has {source.meta.total_episodes} episodes, plan has {len(plan['lengths'])}")
    actual_lengths = [int(source.meta.episodes[i]["length"]) for i in range(source.meta.total_episodes)]
    if actual_lengths != plan["lengths"]:
        raise ValueError("source episode lengths differ from the pinned split plan")

    features = {k: copy.deepcopy(v) for k, v in source.meta.features.items() if k not in DERIVED}
    output = LeRobotDataset.create(
        repo_id,
        fps=source.meta.fps,
        features=features,
        root=out_root,
        robot_type=source.meta.robot_type,
        use_videos=True,
        image_writer_threads=workers,
        encoder_threads=workers,
        video_backend=video_backend,
    )
    video_keys = set(source.meta.video_keys)
    manifest_episodes = []
    try:
        for episode in range(source.meta.total_episodes):
            source_from = int(source.meta.episodes[episode]["dataset_from_index"])
            for color, start, end in episode_parts(plan, episode):
                for relative_index in range(start, end):
                    item = source[source_from + relative_index]
                    output.add_frame(_output_frame(item, features, video_keys, plan["tasks"][color]))
                # Keep the converter inside its explicit thread cap. LeRobot's default parallel
                # path starts one process per camera in addition to codec threads.
                output.save_episode(parallel_encoding=False)
                manifest_episodes.append(
                    {
                        "output_episode": len(manifest_episodes),
                        "source_episode": episode,
                        "color": color,
                        "source_frames": [start, end],
                        "length": end - start,
                    }
                )
                print(f"source {episode:02d} -> output {len(manifest_episodes)-1:03d} {color} [{start}:{end}]")
    finally:
        output.finalize()

    manifest = {
        "repo_id": repo_id,
        "source_repo_id": plan["source_repo_id"],
        "source_revision": plan["source_revision"],
        "split_plan": DEFAULT_PLAN.name,
        "fps": plan["fps"],
        "settle_frames": plan["settle_frames"],
        "episodes": manifest_episodes,
        "total_episodes": len(manifest_episodes),
        "total_frames": sum(x["length"] for x in manifest_episodes),
        "tasks": {color: {"text": plan["tasks"][color], "episodes": len(plan["lengths"])} for color in COLORS},
    }
    (out_root / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(plan_path, out_root / DEFAULT_PLAN.name)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-id", default="erh-hack-team-ale/so100_color_sort_split_20261017")
    parser.add_argument("--source-root", type=Path, help="optional existing local source dataset")
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--workers", type=int, default=8, help="image-writer and video-encoder thread cap")
    parser.add_argument("--push", action="store_true", help="publish the completed dataset to Hugging Face")
    args = parser.parse_args(argv)

    plan = load_plan(args.plan)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    manifest = convert(
        plan,
        args.plan,
        args.out.expanduser(),
        args.repo_id,
        args.source_root,
        args.video_backend,
        args.workers,
    )
    if manifest["total_episodes"] != 2 * len(plan["lengths"]):
        raise RuntimeError("output episode count does not match the split plan")
    if manifest["total_frames"] != sum(plan["lengths"]):
        raise RuntimeError("frame conservation check failed")
    if args.push:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        published = LeRobotDataset(args.repo_id, root=args.out.expanduser(), video_backend=args.video_backend)
        published.push_to_hub(
            tags=["color-sorting"],
            license="apache-2.0",
            private=False,
            upload_large_folder=True,
            dataset_description=(
                "Derived from `erh-hack-team-ale/so100_pick_cube_20261017_185713` at commit "
                f"`{plan['source_revision']}`. Each original two-move demonstration was split after the first "
                "object release into one orange-left episode and one green-right episode. The checked "
                "`color_sort_splits.json` and `split_manifest.json` files preserve the exact frame-level provenance."
            ),
            url=f"https://huggingface.co/datasets/{plan['source_repo_id']}",
        )
    print(json.dumps({k: manifest[k] for k in ("repo_id", "total_episodes", "total_frames", "tasks")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
