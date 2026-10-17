# Zero-shot run 1 (17.10, Saad's SO-100): the arm moved "disoriented"

**TL;DR.** The pipeline worked end to end. The motion was wrong for reasons we can name, but not yet rank with numbers.
The leading suspect is the **joint convention**: the checkpoint reads joint values in one calibration convention, and our
arm reports another. Our map between them is a well-argued guess that has not been validated on the arm.
Second is **latency stalls**, now reduced. Third is **task and scene distance** from BFL's training data.
The next live run waits for the calibration check below.

## What ran

| | |
|---|---|
| when | 15:16:10–15:16:42Z, 48 plans (server log `so101-logs/so101-20261017T140022Z.jsonl`) |
| instruction | `put the blue cube in the box` |
| arm | SO-100 follower, `--robot-id my_follower_arm`, LeRobot 0.6.1 degrees ((raw−mid)·360/4095), drive_mode 0 on all motors |
| cameras | scene index 2 (640×480 MJPG 30 fps), wrist index 4 |
| map | T21 `console/fit_reference.py`: our degrees → the old manual SO-100 calibration convention, \|k\| = 1, signs [+1, −1, +1, +1, −1], offsets from the rest pose of `VoicAndrei/so100_kitchen` (47 episode starts) |
| safety | T21 `limits.json` (native), `--step-max 3`, `--live` |
| server | eager, 508 ms p50 per plan; turnaround p50 745 ms on Saad's link (network p50 179 ms); 18 starved ticks in the dry run before |

Forensics from Saad's agent (per-joint motion, map file, summary.json) will be appended below when they arrive.

## What we know (in the order it was found)

1. **No map:** at rest Saad's shoulder lift reads −101.8°, while the checkpoint's `state_q01..q99` is 84.8…184.2. A LeRobot 0.6.1
   calibration is symmetric about the middle (lift −102…+102, elbow −97…+97), so no recalibration can reach the checkpoint's range.
   It is a different convention. The dry-run command ran away to lift +647 within 13 s: accumulated dry history, fixed in `3fb5132`.
2. **T20 sweep map (mine):** scaled each joint by its swept range, giving k = 0.48–0.67. The arm would have moved 1.5–2.1× farther
   than commanded. Rejected before live.
3. **T21 reference map (Saad's agent):** \|k\| = 1, signs from upstream issue #2 plus rest-at-range-end geometry, offsets from a
   community SO-100 dataset. Hardened after the Codex audit (`1064c37`, `4fffb04`). Still marked **experimental**: the
   physical pose and gripper validation is not done (`docs/research/so101-calibration-audit-2026-10-17.md`).
4. **Offline A/B** (public SO-101 dataset with standard calibration, 16 frames): a range-normalizing map cut the error from 13.1° to 10.8°
   and raised direction agreement from 0.43 to 0.48. The map helps, but no map makes this model precise zero-shot.

## Causes, most likely first

1. **Joint convention not validated.** T21 assumes the checkpoint's convention equals the old manual SO-100 calibration of a
   community dataset. The checkpoint's fixed statistics come from BFL's own SO-101 fine-tune data, whose calibration is not
   published. One wrong sign or a 20–30° offset on lift or elbow is enough to make reaches look "disoriented".
2. **Stalls.** A 745 ms turnaround against a 533 ms stall-free budget meant brief holds plus chunks joined late. Since 15:30:53Z the
   server runs `--compile`: 351 ms p50, turnaround about 550–600 ms.
3. **Task and scene distance.** BFL's SO-101 policy was fine-tuned on about 200 teleop episodes of a few pick-and-place tasks
   ("place the box in the container"). Our wording, objects and camera placement differ. The base model saw about 1,000 community
   datasets, but only high-angle or top-down scene cameras (`community_corpus.json`).

## Calibration check before the next live run

`infra/gpu/calib_check.py` (T24) runs the model on **Saad's own teleop demos** through the map and reports, per joint, whether the
model moves with the operator. It has been tested on the public dataset; there the map raised the lift correlation from 0.10 to 0.63
and wrist flex from 0.03 to 0.72.

1. Saad records 3 short demos of the exact task with the leader arm (`console/teleop.sh record`, about 20 s each, natural motion
   from rest to cube to box and back). This needs the leader arm, but only for this check.
2. Copy the dataset and `joint_mapping.json` to the pod (`rsync … /workspace/datasets/`).
3. Run `calib_check.py`. Pass criteria per joint the operator moved: corr > 0.2 with the map, and the map not worse than raw.
   A negative corr means that joint's sign is reversed in the map. Fix it, rerun, and only then go live.
4. Live retry: same instruction, `--step-max 3`, one person at the power switch, and `--record` with the verdict saved.
