---
id: T28
title: Split color-sort demonstrations into per-color episodes
status: done
owner: joelleayoub
depends: 
paths:
  - scripts/split_color_sort_dataset.py
  - scripts/color_sort_splits.json
  - scripts/tests/test_split_color_sort_dataset.py
---

## Goal

Turn each two-move color-sorting demonstration into one orange-left episode and one green-right episode, without changing the source dataset. Preserve both cameras, state, action, and source provenance, then publish the result as a separate Hugging Face dataset.

## Done when

- [x] `python -m unittest scripts/tests/test_split_color_sort_dataset.py` passes.
- [x] The output loads as a LeRobot v3 dataset with 164 episodes, 82 per task, and the same total frame count as the 82 source episodes.
- [x] The published dataset records the pinned source revision and every verified split frame.

## Upstream

- docs/architecture.md

## Notes and handoffs (append, newest last)

- 2026-10-17 · joelleayoub: Published the verified public derivative at
  `alerest/so100_color_sort_split_20261017` because the available `alerest` token cannot create datasets in
  `erh-hack-team-ale`. Source remains pinned at `8dac3040f85d5fdbc666a22a85a48762ba03d3f2` and untouched.
  Validation: 164 episodes, 51,922 frames, 82 orange-left + 82 green-right, two 30 fps cameras; all frame indices
  contiguous and the manifest conserves every source frame. Persistent pod copy:
  `/workspace/datasets/so100_color_sort_split_20261017`.
- 2026-10-17T21:50Z · joelleayoub · done: published alerest/so100_color_sort_split_20261017: 164 episodes, 51,922 frames, 82 per task; public read-back verified
