# 10. SO-101 units and safety layers: joint map at the arm, native guard below it, factual history

**Date:** 2026-10-17 · **Status:** accepted · **Author:** alerest285 (claude-code-hanoi)

## Context

- flux-3-action-so101 was trained in the old SO-100 calibration, not LeRobot 0.6.1's range-midpoint degrees.
  Saad's arm at rest reads shoulder_lift −101.8°, while the checkpoint's state q01..q99 is +84.8..+184.2
  (docs/research/so101-calibration-audit-2026-10-17.md). console/ answered with a joint map: `console/mapping.py`
  `JointMap` + `MappedRobot` (T20). The reference-pose fit `console/fit_reference.py` (T21) writes
  `~/hackathon-data/joint_mapping.json` with `meta.robot_id` and `meta.calibration_sha256`, plus native
  `~/hackathon-data/limits.json`.
- Codex's audit of robot/ found three defects:
  - A failed write still advanced the history anchor.
  - A dry run re-integrated unsent commands. The anchor drifted to +654° in 13 s, the shape of Saad's +647° trace.
  - Every bound lived in the loop's units. Under a map with k = 0.48, a 6° step became 12.4 native degrees.
- console/, eval/, robot/ and train/ (T23, decision 09) each touch these values. Without one written contract,
  one of them will mix native and model units.

## Decision

**Two unit spaces, one boundary.** The arm wrapper is the boundary.

- *Model space*, the checkpoint's convention: the control loop, its 8-tick history, the so101.v1 server, every
  chunk.
- *Native*, LeRobot 0.6.1 degrees with the gripper 0..100: everything stored and every safety bound. That covers
  LeRobot datasets (console recordings, teleop demos, eval attempts), eval's `ticks.jsonl` and the limits files.

`MappedRobot(SO101, JointMap)` converts at the boundary:

- `console/app.py` and `eval/hanoi_eval/session.py` apply it.
- `robot/run.py` does not; it sends raw degrees and is for bring-up and timing only.
- eval keeps `chunks.npz` as the server sent it, in model space, and stores the map in `run.json`.

**The map must belong to the arm.** Before a map drives the arm, two checks apply:

- `meta.robot_id` must equal the robot id in use.
- `meta.calibration_sha256` must equal the sha256 of `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json`.

A live run refuses another id, a recalibrated arm or an unstamped map (the T20 sweep maps carry no hash). A dry
run warns. eval enforces this (`hanoi_eval/dataset.py: map_identity_problems`); console's owner matches it.

**Three safety layers, outermost first.**

1. **Loop clamp, model space.** The joint limits are mapped from native with `JointMap.limits_to_model`. At most
   `step_max` (6°) per tick per joint, and the gripper stays in 0..100. A joint that starts outside its limits may
   move back in at the planned pace but never further out.
2. **Native guard in `SO101.write`** (`robot/so101_client/safety.py`), below any map. It raises `SafetyStop` and
   writes nothing on any of these:
   - |goal − measured| > 25°
   - |goal − last written goal| > 15° in one write
   - a goal outside the native limits + 2°

   The gripper is exempt, since a grasp holds a steady error. `--live` requires native limits in robot/run.py,
   console/ and eval/.
3. **Latched stop.** These stop the loop for good (`ControlLoop.error`), and the motors keep the last goal that
   went out:
   - a `SafetyStop`, at once
   - 3 failed writes in a row
   - 30 failed reads in a row

**History is factual.** The ring only holds commands that reached the motors.

- A failed write leaves the last sent command in force, and the next tick anchors on it.
- A dry run records the held command and the measured state.
- The dry run's logged `cmd` is a preview: each accepted chunk is replayed from the held pose, restarting inside the
  limits. It shows one plan, not the live trajectory.

## Consequences

- **Training (T23, decision 09) converts native demos to model space with the same map artifact used at inference.**
  It records that artifact's hash. An adapter trained in one convention must never run behind another map.
- **The map stays experimental until physical poses validate it** (Codex: two correspondences plus a third pose
  check, with the gripper checked separately). Until then live runs are supervised, with the e-stop in reach.
- **The 25/15/2° thresholds are first guesses.** A `SafetyStop` during normal motion is a reason to look at the log
  before raising `NativeGuard(max_error_deg, max_step_deg)`, and any change gets a log line.
- **A dry run can no longer be read as a live trajectory, by construction.** Live behaviour is judged from
  supervised live runs.
- Codex's reproducer `docs/research/so101_calibration_audit.py` now fails its buggy-behaviour assertions: the
  anchor stays at −101.8 instead of +654.2. Robot tests 63, eval tests 74.
