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
- [ ] one supervised live episode, with the e-stop in reach
- [ ] the dry run against the live server shows zero ticks without a queued action, and the per-plan round trip is logged
- [ ] before the first live run, the composed scene|wrist view the model sees is saved and checked (a swapped camera cannot be fixed by renaming)

## Upstream

- the T01 decision (docs/decisions/05) · docs/architecture.md · tasks/T10 (server protocol)

## Notes and handoffs (append, newest last)

