---
id: T06
title: Embodiment adapter: cameras and state in, actions out
status: doing
owner: alerest285
depends: T01, T10
paths:
  - robot/**
---

## Goal

Build the bridge for whatever T01 picks: grab synchronized frames and joint state, build the request
(docs/architecture.md), execute the first k actions, then replan. For SO-101 this means the standalone
`load_policy` loop at 30 Hz, which may need its own server on the box (add it under `infra/gpu/`
with a new task).

**After T01 (decision 05):** drive the SO-101 from the Mac. Read the scene and wrist cameras and the joint
state at 30 Hz, talk to the T10 server, execute the returned chunks, and fetch the next chunk before the
current one runs out, so the arm never stalls between chunks at the ~38 ms eu-north-1 round trip. The
instruction can change between steps (eval/ drives that).

## Done when

- [ ] a dry run (actions logged, not executed) at the control rate, with timing per tick
- [x] one supervised live episode, with the e-stop in reach
- [ ] the dry run against the live server shows zero ticks without a queued action, and the per-plan round trip is logged
- [ ] before the first live run, the composed scene|wrist view the model sees is saved and checked (a swapped camera cannot be fixed by renaming)

## Upstream

- the T01 decision (docs/decisions/05) · docs/architecture.md · tasks/T10 (server protocol)

## Notes and handoffs (append, newest last)

- 2026-10-17T15:45Z claude-code-hanoi: codex audit fixed and verified (3fb5132, f00bff8). A failed write never becomes
  history, a dry run keeps factual history, and the NativeGuard sits in SO101.write with latched stops (decision 10).
  Saad's live run at 15:16Z went through console/ on this loop. It counts as the supervised live episode: tracking p50
  0.7°, p95 3.8°; the plans failed, see docs/research/2026-10-17-saad-live-run-1.md. Still open:
  - a dry run with per-tick timing on the arm (run.py writes ticks.jsonl; console does not)
  - zero starved ticks against the compiled server
  - the composed-view check (camera_check.py)
