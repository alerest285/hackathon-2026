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

- [ ] `eval/` can run and record one episode end to end (sim or dry run is fine first)
- [ ] the output layout is documented in `eval/README.md`
- [ ] a full 7-step episode runs end to end against a fake robot and a fake policy, with no hardware, and writes that layout
- [ ] unit tests for the solver, the state tracker and the step-end detector pass

## Upstream

- the T01 decision (docs/decisions/05) · docs/project.md milestone 3 · tasks/T06 (robot interface)

## Notes and handoffs (append, newest last)

