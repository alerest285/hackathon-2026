---
id: T21
title: Reference-pose joint mapping: old LeRobot SO-100 calibration (fit_reference.py)
status: doing
owner: saadl27
depends: 
paths:
  - console/fit_reference.py
---

## Goal

Map our arm's LeRobot 0.6.1 degrees into the convention flux-3-action-so101 was trained in: LeRobot's old manual SO-100
calibration (zero pose = 0, rotated pose = +90, gripper 0..100 closed..open), which VoicAndrei/so100_kitchen used and whose
q01/q99 the checkpoint ships. Per joint old = s * 4095/4096 * ours + b: signs from issue #2 and the rest-pose geometry, pan and
wrist_flex checked by hand, offsets from the dataset's rest pose. Writes the console's joint_mapping.json and a limits file.

## Done when

- [x] `console/fit_reference.py` on the real arm: pan and wrist_flex hand checks conclusive (pan +43 deg to the robot's right, straight
      wrist +4 old deg); the script refuses to write on an inconclusive check or invalid reading. The rest-pose match is by construction
      and validates nothing; lift/elbow/wrist_roll signs and their offsets are priors (rest geometry, issue #2): the map stays experimental
- [ ] console dry run with `--limits ~/hackathon-data/limits.json`: no out-of-range joint, chunks arrive, first commands move the
      way the prompt asks (supervised look, not a validation of the map)

## Upstream

- docs/research/so101-calibration-audit-2026-10-17.md (codex audit) · console/mapping.py (JointMap format)
- old LeRobot source: feetech_calibration.py run_arm_manual_calibration + feetech.py apply_calibration (@6d723c45, 5d24ce3160)
- https://github.com/black-forest-labs/flux-action/issues/2 · dataset VoicAndrei/so100_kitchen (meta, 47 episode starts)

## Notes and handoffs (append, newest last)

