# Status 2026-10-17: repo bootstrapped, nothing has run on a GPU yet

**TL;DR.** The coordination system and the inference scripts exist. No AWS account exists yet,
so no GPU code has run. The next move is T02, and the quota request inside it is the slow part.

## Works (tested locally)

- Claims, overlap refusal, the race between two clones, release, the pre-commit block, the board. See the bootstrap test in alerest285's log.

## Written, not yet run

- `infra/aws/*`: launch/stop/ssh/budget/quota for one p5.4xlarge spot (decision 02)
- `infra/gpu/*`: BFL's setup + DROID smoke test + server, pinned to flux-action `830f1ce` (decision 03)
- `infra/client/ping.py`: OpenPI WebSocket client

## Next, ranked

1. T02: AWS account + **quota request today** + budget alarm. A human has to do this
2. T01: pick the embodiment and the task. A human call, in parallel with T02
3. T03 → T04 → T05 once the quota lands
