---
id: T07
title: Episode runner, success logging, video capture
status: doing
owner: alerest285
depends: T01
paths:
  - eval/**
---

## Goal

Make the zero-shot result reportable: N episodes, each with a prompt, a seed, success yes/no,
video, and the actions sent. Results go in a table in `docs/status/`.

**After T01 (decision 05):** the Hanoi sequencer and episode runner. Solve the 3-disk puzzle, issue one
instruction per move, end each step on the operator's key, the automatic grasp-release-rest detector or a
30 s timeout, and log every attempt (prompt, times, end reason, success, interventions, action chunks,
both camera streams). Piece and place names come from a config file.

## Done when

- [x] `eval/` can run and record one episode end to end (sim or dry run is fine first)
- [x] the output layout is documented in `eval/README.md`
- [x] a full 7-step episode runs end to end against a fake robot and a fake policy, with no hardware, and writes that layout
- [x] unit tests for the solver, the state tracker and the step-end detector pass

## Upstream

- the T01 decision (docs/decisions/05) · docs/project.md milestone 3 · tasks/T06 (robot interface)

## Notes and handoffs (append, newest last)

- 2026-10-17T15:45Z claude-code-hanoi: all four checks pass. Evidence: `cd eval && uv run --python 3.13 pytest -q` gives 74 passed.
  That includes a 7-step strict episode over a real socket against `server.py --engine fake` (test_e2e_socket.py) and a
  mapped fake episode whose 7 LeRobot episodes are read back (test_dataset.py). The real arm runs behind console's
  MappedRobot and the NativeGuard (decision 10). `--live` refuses without limits, without the map, or with a map fitted
  for another arm. Not yet on hardware. The gripper thresholds in hanoi.toml are guesses until `robot/readout.py`. Lock
  kept for the first sanity runs; release it as done after them.
