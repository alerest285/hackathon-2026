# Saad's calibration mismatch: independent audit

2026-10-17, alerest285/codex-arm-review. Read-only investigation of the arm/model code;
no hardware, cameras, network inference, or motor calibration writes. Claude owns the
console implementation. This note does not certify a numerical mapping for live use.

**Follow-up:** the original history and write-failure findings below were fixed in
`3fb5132`. See [the parallel review](#parallel-review-after-claudes-fixes) for verified
fixes and remaining issues at `ca4ed38`, including console commit `8e303ef`.

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

## Parallel review after Claude's fixes

Reviewed on 2026-10-17 against `ca4ed38f9e99d584ac695721f9e8a404887079cc`, including
controller fix `3fb5132`, console/storage fix `8e303ef`, and reference-check fix
`ca4ed38`. This is an independent checker pass: no implementation edits, hardware
connections, motor operations, or GPU calls. Findings were sent to the owning
Claude agents through repository chat. Physical calibration remains experimental.

### Verified fixes

| Area | Evidence |
| --- | --- |
| Factual command history | 390 synthetic dry ticks retain the measured/held anchor at -101.8 degrees in both sync and prefetch modes; no motor writes. This supersedes the historical reproducer's dry-history finding. |
| Failed motor writes | Exactly three consecutive failures latch an error and stop; command history retains the last successful command. `SafetyStop` stops immediately. |
| Native motion guard | A 2-degree model step transformed into 20 native degrees is rejected before the fake bus writes. |
| Console limits | Real `--live` without `--limits` exits with code 2. Native limits reach the guard with and without mapping; outer limits are transformed separately. A reversed-scale case maps native lift [-20, 30] to model [85, 110]. |
| Disk usage default | Default console constructs no recorder. Recording requires `--record`. |
| Native LeRobot recording | With installed LeRobot 0.6.1, a 12-frame episode produced two playable 12-frame MP4s and zero PNGs; resume saved a second episode. |
| Reference checks | Zero pan movement and wrist residuals of 77.3 and 67.3 degrees now refuse to save, preserving both existing files. Valid normal/reversed wrist samples save appropriate check metadata. Explicit unchecked export retains its unchecked status. |
| Mapping algebra | 10,000 random state round trips had maximum error 1.14e-5 degrees, including negative slopes. This checks algebra, not physical correspondence. |

The controller review also ran 46 targeted robot tests successfully. The initial
`so101_calibration_audit.py` is a historical bug reproducer, not a current acceptance
test: its old bug assertions are expected to fail after the fix.

### Remaining findings

1. **Fitter output validation and identity.** `console/fit_reference.py` still marks a
   NaN wrist reference as confirmed and writes it. Consistent but implausible captures
   can produce reversed native wrist limits `[867.8, 175.0]`. The mapping is replaced
   before calibration is read and limits computed, so a calibration-read failure
   leaves a new map paired with old limits. `--from-saved` can relabel another robot's
   captures without checking robot ID or a calibration fingerprint. Validate finite
   six-channel captures and all generated intervals before replacing either output;
   preserve both previous files on failure and bind saved captures to their source.

2. **Console fault handling.** After a synthetic latched `SafetyStop`, `/status`
   returns top-level `error: null` (the error remains buried in `stats`). `/task`
   returns HTTP 200 and queues a task into the stopped loop because `started` remains
   true. Surface the latched error, reject task/connect requests with a restart-required
   response, and clear recording state. Do not automatically restart the motor loop.

3. **Dry preview bypasses its own clamps.** In `robot/so101_client/loop.py`, rebasing
   `_dry_cmd` onto an unbounded predicted action yields a 90-degree displayed jump to
   100 despite native/model limits of +/-10 and a step maximum of 1 in an identity-map
   test. This is a diagnostic defect: factual history stays correct and the live path
   still clamps. Apply equivalent candidate checks after rebasing.

4. **Streaming encoder backpressure is invisible to the recorder.** An isolated
   fault-injection test deliberately slowed only the scene encoder. It saved 12
   state/action rows, 6 scene video frames, and 12 wrist video frames, while both console
   status and the episode sidecar reported zero dropped frames and no error. LeRobot
   can drop frames inside its per-camera queue independently of the outer recorder
   queue. Propagate encoder drop counts and reject incomplete episodes, or use recorder
   thread backpressure that preserves frame alignment. This proves the failure path;
   it does not establish that Saad's laptop currently experiences it. Sustained
   two-camera throughput remains unmeasured on that laptop.

5. **Recording shutdown and provenance.** A simulated disk-full exception during
   `save_episode()` leaves the recorder thread alive and skips finalization. `close()`
   eventually returns after its join timeout, while the app unconditionally prints
   "dataset finalized". Propagate finalization failures. Explicit dry-run recording
   also writes hypothetical actions to the same dataset format without a mode marker;
   distinguish them from executed demonstrations. The recorder closes an episode on
   every task-text change, conflicting with the UI's "Next instruction continues the
   episode" description.

6. **Legacy nonidentity gripper maps can falsify command history.** For gripper scale
   0.5, model command 75 inverse-maps to native 150, which `MappedRobot.write()` clips
   to 100. The loop records 75, whereas the sent value corresponds to model 50. T21's
   identity gripper avoids this particular case; older sweep maps can still trigger it.

The remaining physical questions are independent of software guards: whether the
operator's rest matches the dataset's rest, lift/elbow/wrist-roll directions, gripper
aperture endpoints, and a held-out validation pose. Rest mapping within a tolerance is
not an independent check because the offset is fitted to make that equality hold.
The dataset's observed bounds and encoder ranges are candidate operating restrictions,
not proof of collision-free mechanical travel. Add gripper endpoints to the fitter's
explicitly unverified assumptions.

### Reproduction record

Local scratch artifacts from this review (not required by the application):

- `/private/tmp/so101_safety_fix_review.py`: factual history, write failures, native guard,
  old console-limit defects, dry preview, and legacy gripper clipping. Its old
  console-limit bug assertions are superseded by `8e303ef`.
- `/private/tmp/so101_console_limits_review.py`: current limits propagation and the
  remaining dry-preview/console-fault cases, without starting any component.
- `/private/tmp/recording_checker_probe.py`: actual LeRobot create/resume, encoder
  backpressure, and recorder shutdown faults. Temporary datasets are under
  `/private/tmp/recording-checker-opn6218i`.
- `/private/tmp/t21-ca4ed38-k4hx4lmj`: reference-check and invalid-output artifacts.

The scripts used
`/Users/ale/Developer/hackathon-2026-hanoi/robot/.venv/bin/python` and source from the
read-only main checkout. Scratch paths are session-local; the numerical observations
and fault setups above are the durable review record. No user recordings were deleted.
Before deleting old PNG directories, verify their videos were finalized or that the
recordings are explicitly disposable; unfinished episodes may contain their only
camera data in those PNGs.

### Subsequent changes reviewed at 15:25Z

- `d375dbc` added unconditional recursive deletion of directories named `images`
  under the data root. A read-only scratch selection test matched finished datasets,
  unfinished datasets, and unrelated reference-photo directories alike. After the
  checker escalated this, `60d4f16` removed automatic deletion; startup now reports
  sizes only. No user files were deleted by this review.
- The replacement `console/cleanup.sh` still overstates completion: one existing
  MP4 and a `meta/episodes` directory cannot establish that all PNGs have encoded
  equivalents. Saved episode 0 plus interrupted episode 1 is a counterexample.
  This was sent to the owner; no cleanup was executed.
- The new low-disk callback calls blocking `Recorder.save()` on the control thread.
  With its queue full at 600 entries and free space mocked to 1 GB, `_on_tick(k=300)`
  remained blocked for 255 ms and completed only after one queue slot was released.
  A failed writer can block it indefinitely. Disk checks and save coordination must
  not block the motor loop. Startup also checks `$HOME` rather than a custom
  `--record-root` filesystem; the runtime free-space method does use the dataset root.
  Reproducer: `/private/tmp/recording_lowdisk_probe.py`.
- Saad's `1064c37` adds finite-input, generated-limit, saved-identity and calibration
  fingerprint checks. Independent revalidation is in progress; the earlier fitter
  findings above describe `ca4ed38`, not a claim that every finding survives this fix.
- Claude acknowledged the remaining console fault and recording integrity cases
  and is implementing fixes. Their verification is not implied by this report.
