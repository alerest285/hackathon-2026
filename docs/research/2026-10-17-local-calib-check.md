# Model-free calibration check of Saad's 82-episode dataset (no GPU)

2026-10-17 ~19:25Z, saadl27/saad-calib. Data: `erh-hack-team-ale/so100_pick_cube_20261017_185713` (82 episodes,
51,922 frames, native LeRobot 0.6.1 units, task "Put the green objects at the right of the black line…"). Reference:
300 of the 1,801 episodes of `VoicAndrei/so100_kitchen` (old manual SO-100 calibration, the source of the
checkpoint's q01/q99). Script: [`so100_local_calib_check.py`](so100_local_calib_check.py), CPU only:

```sh
uv run --no-project --with pandas --with pyarrow --with numpy --with scipy python -I \
    docs/research/so100_local_calib_check.py <ours_dataset_root> <kitchen_root> <qmap.json>
```

This does not replace the model-in-the-loop check (T24 `calib_check.py`). It tests what the joint data alone can
show.

## 1. The T21 map is stale: the arm was recalibrated

`robots/so_follower/my_follower_arm.json` was rewritten at 16:48Z (sha `d6d7cc…`). `~/hackathon-data/joint_mapping.json`
was fitted on the earlier calibration (sha `40e0f0…`). In LeRobot 0.6.1, degrees = (raw − homing − mid)·360/4095, so
the same physical pose now reads differently by a constant per joint (old homing/range from chat 14:40Z):

| | pan | lift | elbow | wrist_flex | wrist_roll | gripper |
|---|---:|---:|---:|---:|---:|---|
| shift, new − old (deg) | +4.8 | −0.2 | +0.2 | **+86.2** | +9.8 | ≈ 0.96·g + 1.2 |

The dataset confirms it: the median episode-start wrist_flex is 77.0, and the predicted value is 76.7 (old rest −9.54 + 86.2).
Do not drive with `joint_mapping.json`. Console and eval refuse it through the calibration hash (decision 10).
`maps/so100_pick_cube_qmap.json` (hackathon-2026-85, for the LoRA) carries the new hash.

## 2. Signs (T1: joint-velocity correlations, moving frames, 5-frame lag, 95% CI over episodes)

| pair | kitchen (old) | ours (native) |
|---|---:|---:|
| lift–elbow | +0.67 [+0.65, +0.69] | −0.64 [−0.66, −0.62] |
| lift–wflex | +0.68 [+0.66, +0.69] | +0.05 [+0.01, +0.09] |
| elbow–wflex | +0.28 [+0.25, +0.30] | −0.66 [−0.68, −0.63] |
| wflex–wroll | +0.19 [+0.18, +0.21] | −0.00 [−0.04, +0.03] |
| other pairs | \|r\| ≤ 0.09 | \|r\| ≤ 0.26 |

Agreement score change when one joint is flipped from the T21 signs [+1, −1, +1, +1, −1] (negative means T21 is better):
lift −0.81 [−0.86, −0.76], elbow −0.48 [−0.51, −0.45], wflex +0.44 [+0.38, +0.50], wroll −0.01 [−0.03, +0.01].

T2 below adds pose overlap: our poses mapped into the checkpoint's normalized space, compared with kitchen poses
(nearest-neighbour distance, 5 joints). Phys map (T21 signs, |k| = 4095/4096, rest-anchored): p50 0.88. With lift
flipped 1.98, elbow flipped 1.82, wflex flipped 1.00, wroll flipped 0.88.

- **lift −1, elbow +1: confirmed** by the strong, kinematic lift–elbow coupling and by pose overlap, on top of the
  rest-at-range-end geometry.
- **wrist_flex +1: keep it** (hand-checked in T21 with the straight wrist, and better pose overlap). The correlation
  test prefers −1, but wrist–elbow coupling depends on how the operator holds the gripper during the task. Kitchen
  operators moved the wrist with the shoulder, and Saad moves it with the elbow. That test is not reliable for wflex.
- **wrist_roll: not determined by data.** It has no coupling to the other joints in either dataset, and overlap is
  the same for both signs (phys 0.88/0.88, qmap 0.67/0.62). Its sign still rests on issue #2 alone.

## 3. Scale and offsets (T2)

Per-joint p1..p99 of our data in the checkpoint's normalized space (kitchen = [−1, 1]):

| map | pan | lift | elbow | wflex | wroll | NN dist p50/p90 |
|---|---|---|---|---|---|---|
| qmap (quantile scale) | −0.6..+0.9 | −0.8..+1.0 | −0.7..+1.0 | −0.3..+1.0 | −0.7..+0.7 | 0.67 / 0.88 |
| phys (\|k\| = 1, rest) | −0.5..+0.9 | −2.3..+1.0 | −1.7..+0.9 | +0.2..+1.4 | −1.3..0.0 | 0.88 / 1.44 |

With true degrees, this task leans the shoulder far past anything in the kitchen data. Lift reaches old ≈ 22°,
nearly horizontal, where kitchen's q01 is 85°, so 45% of our lift frames fall outside the checkpoint's range. The qmap
compresses lift ×0.545 and elbow ×0.63 to stay inside it, and wrist_roll's offset differs by ~72° between the two
maps. The data alone cannot say which the model reads better: in-range but physically rescaled, or physically true
but out of range. That needs the model (T24) or the LoRA result.

## What would settle the rest

1. **wrist_roll sign and offset, plus a held-out check of lift and elbow:** capture the old calibration's two
   reference poses, zero (arm stretched horizontally, old = 0) and rotated (old = +90 per joint), from LeRobot's photos
   `media/so100/follower_zero.webp` and `follower_rotated.webp` @ `6d723c45`. Read only, torque off. Two
   correspondences per joint give sign and offset directly.
2. **qmap vs phys scale:** `calib_check.py` on this dataset, with both maps, on a free GPU.

## 4. New physical map for the current calibration (~19:40Z)

[`so100_phys_map_from_dataset.py`](so100_phys_map_from_dataset.py) uses T21's method (signs [+1, −1, +1, +1, −1],
|k| = 4095/4096, gripper identity). The offsets match this dataset's median episode start to kitchen's, both teleop
starts. It wrote `~/hackathon-data/maps/so100_phys_map.json` (my_follower_arm, calibration `d6d7cc…`) and
`so100_phys_limits.json`: the range Saad's teleop reached ±5°, cut to the calibrated travel. Round trip max error 1.1e-5°.
It does not touch `joint_mapping.json`.

| | pan | lift | elbow | wrist_flex | wrist_roll |
|---|---:|---:|---:|---:|---:|
| offset b (old = s·k·ours + b) | −5.88 | 81.07 | 80.13 | 0.35 | −76.37 |
| T21 offset shifted to the new calibration | −4.66 | 80.01 | 73.80 | 0.62 | −83.45 |
| limits (ours) | −83.7..62.8 | −102.1..85.2 | −96.3..96.3 | −2.8..100.6 | −76.3..108.1 |

The two maps agree within about 1° on pan, lift and wrist_flex. Elbow differs by 6.3°: T21 anchored a hand-folded rest
at the mechanical stop, while the new map anchors teleop starts, which sit about 6° short of the stop, as kitchen's
do. wrist_roll differs by 7.1°: Saad's start roll varies (IQR 10.9°).

## 5. Reference-pose check on the arm (~20:40Z, `so100_ref_poses.py`)

Saad held the old calibration's zero and rotated poses (torque off). Readings, ours: zero `[8.1, 0.3, 6.9, 1.0, 94.5, 2.4]`,
rotated `[95.0, −6.3, 13.8, 84.0, 2.3, 85.7]`. Under the phys map (old degrees, target zero 0 / rotated +90):

| joint | zero | rotated | result |
|---|---:|---:|---|
| pan (base was turned too) | 2.2 | 89.1 | sign +1 and offset confirmed within ~2° |
| shoulder_lift | 80.7 | 87.3 | rotated pose confirms the map (−2.7°). The zero pose is unusable: lift and elbow barely moved (−6.6/+6.9), so the arm was in an L in both poses, not stretched out |
| elbow_flex | 87.0 | 93.9 | rotated pose confirms the map (+3.9°) |
| wrist_flex | 1.4 | 84.3 | sign +1 and offset confirmed (+2.2°) |
| wrist_roll | −170.8 | −78.7 | **sign −1 confirmed, but the offset was off by 170°**. The two poses agree on offset +93.4 (2.2° apart) |
| gripper | 2.4 closed | 85.7 fully open | old convention 0..100; identity kept |

wrist_roll is a free joint, so the rest-pose anchor (Saad's teleop start vs kitchen's) does not fix its physical
orientation. Saad's gripper rests about 170° rolled from kitchen's. The T21 map had the same rest anchor, so in live
run 1 the model probably saw the gripper roll roughly upside down. New map: `~/hackathon-data/maps/so100_refpose_map.json` (phys map +
wrist_roll offset 93.39; limits unchanged, `so100_phys_limits.json`). Under it, Saad's wrist_roll q01..q99 sits at
old 17.8..134.1, mostly above kitchen's q99 (60.2): his gripper is physically rolled differently from the training data.

## 6. Live run 2 (20:46:40–20:47:41Z, refpose map, `--step-max 3`, server sharing the GPU with the LoRA run)

From the server log (`/workspace/so101-logs/so101-20261017T203747Z.jsonl` on wog1l8ijpl3jp5): 52 plans in 69 s, infer p50
985 ms (shared GPU). Tracking was fine: |cmd − state| p95 ≤ 3.8° model. The dry run before it, at rest, gave 190 plans that
agreed with each other (consecutive-plan direction cosine p50 0.98): pan −40 (robot's left), shoulder forward 15,
elbow unfolding 37, wrist_roll −48 (into the trained range), gripper open to 35. Live:

- 0–30 s: pan went left to its limit (native −83), the gripper opened (to 60–97) and wrist_roll rolled about 100° to its
  limit (native 108). The shoulder came forward at most 59° from rest (native lift −102 → −43) and the elbow unfolded
  to native 30.
- 32–69 s: pan swung right to its other limit (native +62), wrist_flex reached its lower limit (native −2), and the
  gripper stayed open. No grasp.
- Live plans were much less consistent than the dry ones (cosine p50 0.57). The shoulder never went as far forward as
  Saad's own demos do (native lift up to +80).

Pan, wrist_flex and wrist_roll all ran into the limits file. The model keeps the upper arm behind vertical (model lift
133–183, where kitchen's q01 is 85), so it never reaches the table.

**Next candidate: the HF sim's normalization.** `multimodalart/flux-3-action-so101-sim@09edebb` dropped its sign/offset
conversion. It feeds LeRobot degrees directly and swaps the policy's state/action q01/q99 for median new-calibration SO-101
dataset stats, which is how the base was trained (per-part normalization). As a joint map:
`~/hackathon-data/maps/so100_mediannew_map.json`, with no sign flips. Saad's rest normalizes to [0.22, −1.03, 1.02, 0.49,
0.85, −0.89], all in range. The state normalization is exact. The action normalization is not: the server keeps the
checkpoint's action stats, so native step sizes run ×[1.04, 1.44, 1.36, 1.72, 0.88, 1.0] vs the sim. Exact replication
needs a server option that overrides `state_normalization`/`action_normalization`.

Frame recorder for the next run: [`so100_watch.py`](so100_watch.py) (`record` beside the console, then `sheet`).

## 7. What the zero-shot model imagines (2026-10-18, `so100_imagine.py`)

Zero-shot hfsim model (`/workspace/ckpt/flux-3-action-so101-hfsim-norm`), 16 moments from Saad's episodes 0/20/40/60
(live 8-tick windows, the episode's sorting caption). The video the sampler generates alongside the actions was decoded
with the policy's own VAE: 11 latent frames, 41 frames of scene|wrist, 0.6 s per moment on the H100. Outputs:
`/workspace/outputs/imagine/zeroshot-hfsim/`, copied to `~/hackathon-data/imagine/zeroshot-hfsim/` on Saad's laptop.

- **Perception is good.** The imagined frames keep the scene consistent (black line, tray, orange and green blocks),
  render the arm moving the way its predicted actions say, and keep a grasped block between the fingers (ep 20 t 187,
  ep 40 t 464).
- **Intent is weak.** Against what the operator did next, the predicted 42-step motion pointed the same way on 54% of
  moving joints (50 joint-moments; elbow 25%, wrist_flex 75%), with about half the real amplitude. The model often
  imagines holding still or lifting where the operator carried the block to its side of the line. 16 moments are a
  small sample, and a human demo is only one valid continuation.
