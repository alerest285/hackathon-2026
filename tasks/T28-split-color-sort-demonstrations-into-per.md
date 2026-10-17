---
id: T28
title: Split color-sort demonstrations into per-color episodes
status: doing
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

- [ ] `python -m unittest scripts/tests/test_split_color_sort_dataset.py` passes.
- [ ] The output loads as a LeRobot v3 dataset with 164 episodes, 82 per task, and the same total frame count as the 82 source episodes.
- [ ] The published dataset records the pinned source revision and every verified split frame.

## Upstream

- docs/architecture.md

## Notes and handoffs (append, newest last)
