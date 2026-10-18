# Robotpov LoRA live test (checkpoint 001000), 2026-10-18 09:29–09:34Z

saadl27/saad-calib. Console: `--robot-id my_follower_arm_e82` (Erfan's calibration = the motors' registers = the e82
recordings, sha256 4a9562e4…), stamped identity map, e82 limits, server `--engine lerobot --adapter
…robotpov…/001000` on the hfsim-norm base, eager, sharing the GPU with the robotpov training.

- **Calibration is consistent.** The arm's first state (6.3, −102.2, 97.1, −9.6, 5.6, 1.4) matches the e82
  episode-start median (3.6, −101.8, 91.1, −12.5, 2.2, 1.7). Map identity "ok", nothing out of range.
- **Live behaviour:** 22 plans at 1.7 s each (eager, shared GPU), turnaround 2.0 s. The arm held 67% of the ticks.
  Every plan was tiny (about 4° lift, −4° elbow, +5–9° wrist_flex per 42-tick chunk), so over 43 s the arm crept
  (lift −102 → −63) and never reached. The session ended in a latched fault: 30 failed state reads, "no status
  packet", i.e. the bus went silent (power cut or USB drop).
- **Offline, same checkpoint, real 30 Hz history from e82 episodes 10 (train) and 55/58 (held out):** from rest
  (t = 8) it also plans only ~4–5° per chunk. In demos the operator waits about 1 s at the start, so the model has
  learned to wait at rest. Mid-task it plans real motion (mean |pred| 17° vs 29.5° real over 42 ticks, 72% same
  direction; 001500: 16.5° / 64%). So the live creep is the start-of-episode "wait" behaviour, kept going by the
  stop-and-go loop: flat command history while holding. Calibration is not the cause.
- **Next:** retest after training ends, with `--compile` on an unshared GPU (≈0.35 s plans, no starvation), and
  with a later checkpoint. 001000/001500 are 250/375 of 2,500 updates.
