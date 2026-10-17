# 05. Milestone 3 is a 3-disk Tower of Hanoi on the SO-101, one move per instruction

**Date:** 2026-10-17 · **Status:** accepted · **Author:** alerest285 (T01, Ale's call)

## Context

The track (BFL, "Imagine, Then Act") scores longer autonomous chains higher, and FLUX must stay in the
control loop. Grading counts both the zero-shot result and the post-training result; we keep successful
attempts, total attempts and completed task steps. Each team gets an SO-101 with a leader arm and two
cameras. The released `black-forest-labs/flux-3-action-so101` takes one text instruction per run. Neither
BFL's repo nor LeRobot has an API for chaining instructions: the documented way is a new `task` string
plus `policy.reset()`.

## Decision

- **Embodiment:** the physical SO-101 follower at our station, a fixed scene camera and a wrist camera,
  checkpoint `black-forest-labs/flux-3-action-so101` unchanged. The leader arm is for resets now and
  demonstrations later (milestone 4).
- **Task:** Tower of Hanoi, 3 pieces, 3 places. Start: all pieces on the left place, largest at the bottom.
  Goal: the same tower on the right place. 7 moves.
- **Chaining:** our sequencer solves the puzzle and issues one instruction per move. FLUX runs closed-loop
  on that instruction until the step ends, the policy is reset, and the next instruction starts. FLUX makes
  every motion; the sequencer only chooses text. (Refined by decision 06: between steps the queued actions are
  dropped and the history is kept; the full reset is at the episode start and after an intervention.)
- **Physical set:** fixed at the venue: rings on pegs, or blocks on three marked spots. Piece and place names
  live in a config file, never in code. Default prompt: `pick up the {piece} and place it on the {place}`,
  e.g. "pick up the small red disk and place it on the right peg". The sanity runs below may change the wording.
- **End of a step**, whichever comes first: the operator's key; an automatic end (the gripper closed on a
  piece, reopened, then the arm rests for 1 s); a 30 s timeout. In evaluation the operator's key is the
  ground truth for success. The automatic end exists so the demo can run without a person.
- **Step success:** the named piece rests on top of the named place, no other piece moved, no larger piece on
  a smaller one, no human touch. **Episode success:** all 7 steps succeed in order with no intervention.
- **Protocol:**
  1. Sanity runs: every legal (from, to) single move at least twice, to settle the prompt wording.
  2. **N = 10 strict episodes** from the same start and camera placement. An episode stops at its first failed step.
  3. Optional recover runs: the operator restores the state before the failed step and that step is retried,
     at most twice. Logged as their own runs and never mixed into the strict numbers.
- **Logged per attempt:** episode id, step index, prompt, start and end time, end reason (key/auto/timeout),
  success, interventions, every action chunk received, both camera streams. **Per episode:** steps completed
  in a row from the start, and steps attempted.

## Consequences

- Milestones 1 and 2 move from DROID to SO-101 (`docs/project.md` updated in this commit). DROID tasks T04
  and T05 stay useful as a check that the GPU environment works, but they are off the critical path.
- BFL ships no SO-101 server, so the server on the GPU host is our component, under its own task.
  T06 (`robot/`) becomes the SO-101 client with chunk prefetch, and T07 (`eval/`) becomes the Hanoi
  sequencer and episode runner. The interfaces between them go into `docs/architecture.md` once its lock is free.
- Expected outcome: with per-step success p, full success is about p^7. A low p is still a valid zero-shot
  result and the baseline for LoRA. The same instruction strings become the task labels for demonstrations.
