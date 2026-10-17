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
