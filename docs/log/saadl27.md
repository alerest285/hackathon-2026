# Log: saadl27

Append-only, newest at the bottom. One line per event.

- 2026-10-17T17:49Z saadl27 [T21] (claude-code) release (todo): fitter done + reviewed; next: demo-quantile candidate map after Saad's T24 demos (see handoff)
- 2026-10-17T19:07Z saadl27 [T23] (claude-code) stopped so101-serve on pod rmhs4cu2cejemv (Saad's call, idle 3.3 h, announced 19:04Z) to train a LoRA on Saad's teleop set (lora_so100_pick_cube_v1: 82 eps, 51,922 frames, dataset-quantile map with T21 signs, my_follower_arm new calibration); starting train.sh probe
- 2026-10-17T19:21Z saadl27 [T21] (saad-calib) (saad-calib) local calib check, no GPU (docs/research/2026-10-17-local-calib-check.md): arm recalibrated 16:48Z (sha d6d7cc), T21 joint_mapping.json stale (wrist_flex +86.2, pan +4.8, wroll +9.8 deg; dataset rest wflex 77.0 vs predicted 76.7). Data confirms lift -1 / elbow +1 (lift-elbow velocity corr kitchen +0.67, ours -0.64; flip lift/elbow -> NN dist 1.98/1.82 vs 0.88); wrist_roll sign undetermined by data; phys map puts 45% of lift frames outside ckpt q01..q99 (task reaches far), qmap keeps all inside: model needed to choose.
