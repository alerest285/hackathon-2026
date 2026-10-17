# 03. Zero-shot runs on BFL's standalone flux-action, not LeRobot

**Date:** 2026-10-17 · **Status:** accepted · **Author:** alerest285

## Context

Two ways exist to run the released checkpoints. One is BFL's standalone repo, with `flux-action infer` / `serve-robolab` and Python
loaders for DROID and SO-101. The other is the LeRobot policy class.

## Decision

Use the standalone repo, pinned at `830f1ce` (2026-10-02). Its setup guide states that the released DROID and SO-101 policies
"need no LeRobot installation". It ships the WebSocket server we need for remote control, and it supports all six DROID packages,
including FP8r, which LeRobot cannot load. LeRobot stays the route for SO-101 LoRA, if we get to milestone 4.

## Consequences

The client speaks the OpenPI WebSocket protocol (`infra/client/ping.py`). Bumping the pin needs a new decision,
because `uv sync` and the NATTEN wheel must match it.
