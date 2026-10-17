# Saad's calibration mismatch: independent audit

2026-10-17, alerest285/codex-arm-review. Read-only investigation of the arm/model code;
no hardware, cameras, network inference, or motor calibration writes. Claude owns the
console implementation. This note does not certify a numerical mapping for live use.

## Finding

There is evidence of a physical coordinate convention mismatch, not a degree/radian
conversion problem. Modern LeRobot uses range-midpoint zeros; historical LeRobot
used physical reference poses and explicit direction signs. Re-running ordinary
calibration cannot establish the checkpoint's reference convention.

Saad's reports are in [team chat](../chat/saadl27--claude-code.md): lift at rest
approximately -101.8 degrees, versus checkpoint state q01/q99 84.814453/184.218750.
At 14:41Z he supplied raw calibration limits: lift 946..3275, elbow 686..2895,
wrist flex and roll 0..4095. Full encoder range is not proof of a safe full-turn
mechanical workspace. No exact pose correspondence has yet been established.

The existing [upstream issue #2](https://github.com/black-forest-labs/flux-action/issues/2)
describes the same compatibility concern. It remains open; the visible comments do
not supply a maintainer-validated physical convention. Its experimental per-arm
adapter is not transferable: wrist behavior remained unresolved. The checkpoint's
statistics matching a historical dataset is evidence of normalization provenance,
not evidence of physical joint zeros. The local corpus manifest independently
matches Saad's reported shoulder quantiles to `VoicAndrei/so100_kitchen`.

The manufacturer's [calibration description](https://github.com/TheRobotStudio/SO-ARM100/blob/5f6d2b876a53a4872e405b991dd925556c9e38a4/Simulation/SO101/README.md#calibration-methods)
distinguishes the old horizontally extended zero from the new range-midpoint zero.
The [BFL adaptation guide](https://github.com/black-forest-labs/flux-action/blob/830f1ce178d7aea52f18fde31d8e8e36ce6bae21/docs/so101-lora.md#inputs-and-camera-layout)
explicitly treats its normalization as dataset-specific, not universal calibration.

## Verified driver behavior

Installed LeRobot 0.6.1, `motors/motors_bus.py:854` and `:883`, uses:

```text
mid = (range_min + range_max) / 2
native_degrees = (raw_position - mid) * 360 / 4095
raw_goal = int(native_degrees * 4095 / 360 + mid)
```

The degree branch ignores `drive_mode` and does not clamp the inverse to the
calibration bounds. `motors/feetech/feetech.py:281` defines present position as
actual position minus hardware homing offset. Shifting that offset and consistently
shifting/re-measuring both range endpoints leaves native degrees unchanged.

An offline execution of the exact degree methods extracted from source checked
24,576 normalization cases, 12,288 inverse cases, and 8,296,138 valid consistent
homing/range shifts. Maximum degree change under these shifts was 0.0. The ranges
in that exhaustive check were synthetic fixtures, not Saad's calibration.

The [historical motor implementation](https://github.com/huggingface/lerobot/blob/974028bd28c81868ed0fd0b5251cbd907cebf9ec/lerobot/common/robot_devices/motors/feetech.py#L418)
instead applied software sign and offset before conversion with 4096 ticks/turn.
Do not rewrite hardware homing or calibration limits to imitate that convention.

## The sweep-quantile proposal is unvalidated

Reviewed `console/mapping.py` subsequently committed as `29beb7f`: its default fit
maps a hand sweep's sorted percentiles to the checkpoint's sorted percentiles.
That cannot infer a reversed joint direction. Even with a manually selected flip,
matching distribution endpoints does not establish corresponding physical poses,
zero, or angular scale. The sweep's duration and workspace coverage change its fit.

Offline example using the actual `JointMap`, with illustrative lift bounds
[-102, 102]: slope = 0.487275965; a 6-degree model step becomes **12.313351 native
degrees**. Endpoint error was 0 and inverse round-trip error 0.000003815 degrees,
despite no physical correspondence being supplied. These checks establish algebraic
invertibility, not calibration. An in-range display is also expected by construction.

Safety limits and rate checks must be expressed/enforced in native physical units
after inverse conversion. Dataset storage should retain native LeRobot state/action
units, or explicitly identify and version a separate transformed dataset schema;
do not silently mix mapped policy episodes with native teleop demonstrations.

## Dry-run and write-failure confounds

[Offline reproducer](so101_calibration_audit.py), run with the robot Python environment:

```sh
robot/.venv/bin/python docs/research/so101_calibration_audit.py
```

The script uses the actual `ControlLoop._tick` with inert local stubs. It never calls
start/connect, opens a socket, or runs a model. Its assertions reproduce bugs in the
audited version; a pass is not a live-readiness check. It prints source hashes.

| Case | Ticks | Measured lift | Final command | Hardware writes |
|---|---:|---:|---:|---:|
| Zero-delta controls, sync and prefetch | 390 | -101.8 | -101.8 | 0 |
| Synthetic +2-degree delta, dry sync | 390 | -101.8 | +654.2 | 0 |
| Synthetic +2-degree delta, dry prefetch | 390 | -101.8 | +676.2 | 0 |
| Every synthetic write raises | 100 | -101.8 | +90.2 | 0 successful |

A following simulated-arm control has consistent measured/command history.
In real-hardware dry run, `loop.py:317-324` advances `_last_cmd` without sending it;
the next tick records it as past command while the arm remains still. This can
produce the scale of Saad's +647-degree trace independently of the coordinate
mismatch. It is not a prediction of the real arm's live trajectory.

The same lines advance history after a failed live write. Absolute joint limits
are optional, and the 6-degree slew bound references the previous target, not the
measured state. `robot_io.py` writes directly through the bus, bypassing the normal
`SOFollower.send_action` measured-relative guard. A fix needs mandatory validated
physical bounds, measured tracking-error checks, and a latched write-error stop.
Keep factual held-command history separate from hypothetical dry-run candidates.

No accidental double integration was found: BFL returns absolute chunks anchored
on the previous command; the client differences and re-integrates their joint
channels. Gripper remains absolute.

## Evidence needed for the next step

For an independently validated affine map `model = A * native + b`, transform both
measured state and issued-command history on requests, and inverse-transform every
absolute response before native motion checks. Degree-delta conversion uses `A`
only. Keep checkpoint normalization fixed during this validation.

Determine joint directions and zeros from identifiable physical poses, with at
least two correspondences and a separate third pose check; verify gripper open/closed
separately. Candidate historical SO-100 references are the pinned
[zero image](https://raw.githubusercontent.com/huggingface/lerobot/6d723c45a993d1e030dfda20d48a893a74a4fe11/media/so100/follower_zero.webp)
and [rotated image](https://raw.githubusercontent.com/huggingface/lerobot/6d723c45a993d1e030dfda20d48a893a74a4fe11/media/so100/follower_rotated.webp).
These were visually inspected; they are legacy references, not confirmation of
BFL's exact training pose. Confirm the actual arm variant and wrist orientation.
Do not run the historical auto-calibration routine, which issues motor commands.

Saad has offered read-only raw pose samples. Those and a confirmed training-frame
reference remain necessary before calling any map a validated calibration fix.
