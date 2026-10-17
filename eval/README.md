# eval: the Hanoi sequencer and episode runner (T07)

`hanoi_eval` turns the 3-disk Tower of Hanoi (decision 05) into one instruction per move for the
robot client's `ControlLoop` (decision 06, `robot/so101_client`). It ends each step on the operator's key, the
grasp-release-rest detector or a timeout, logs every attempt with its action chunks and both camera
streams, and writes a results table. FLUX makes every motion; the runner only chooses the text.

## Run it

From the repo root:

```sh
uv sync --project eval --python 3.13                   # once; pulls ../robot (so101-client) editable

# the GPU server command that warms up every prompt of the config (run it on the box)
uv run --project eval python eval/run_hanoi.py --print-warmup

# no hardware: FakeRobot + FakeCameras + FakeNet(pickplace)
uv run --project eval python eval/run_hanoi.py --config eval/config/hanoi.toml --fake --episodes 1

# no arm, real server: FakeRobot + FakeCameras + NetClient (the GPU box through the tunnel, or a local fake engine)
uv run --project eval python eval/run_hanoi.py --config eval/config/hanoi.toml --fake-hardware --server ws://localhost:8000 --episodes 1 --label fakehw

# sanity runs on the arm, dry run first (commands logged, not written), then live
uv run --project eval python eval/run_hanoi.py --config eval/config/hanoi.toml --sanity 2 --port /dev/tty.usbmodem... --robot-id so101_hanoi
uv run --project eval python eval/run_hanoi.py --config eval/config/hanoi.toml --sanity 2 --live --port ... --robot-id ...

# the 10 strict episodes, then optional recover runs
uv run --project eval python eval/run_hanoi.py --config eval/config/hanoi.toml --live --port ... --robot-id ...
uv run --project eval python eval/run_hanoi.py --config eval/config/hanoi.toml --live --protocol recover --port ... --robot-id ...
```

The server is reached through `ssh -L 8000` (`infra/aws/ssh.sh`); `--server` overrides `server.url`. On connect
the runner compares the server's warmed captions (hello `warm_tasks`) with the config's prompts and prints a
WARNING naming the ones the first plan would have to encode; `--print-warmup` gives the server command that
warms them all (9 with the default config).
Without `--live` nothing reaches the motors. With `--fake` or `--fake-hardware` the commands go to the fake arm,
which follows them. `--limits limits.json` takes the joint limits written by `robot/readout.py --save-limits`.
`--unattended` (alias `--auto-confirm`) is test-only and needs `--fake` or `--fake-hardware`: it presses enter at
every prompt and answers a verdict with success after an auto end and fail after a timeout, for runs without a
terminal. Fake runs keep the configured label unless `--label` is given, so name them apart from real runs.

## The protocol (decision 05)

1. **Sanity:** `--sanity N` runs every legal single move N times, each as a one-step episode from its own
   setup, to settle the prompt wording. `sanity.moves = "all"` is every piece times every ordered pair of
   places (18 moves); `"solution"` is the 7 moves of the solution, each from its state in the solution.
   The runner prints the layout to set up before each trial. A/B prefetch against sync here (`--loop-mode`).
2. **Strict:** 10 episodes from the same start and camera placement. An episode stops at its first failed step.
   Episode success = all 7 steps succeed in order with no intervention.
3. **Recover (optional):** `--protocol recover`. After a failed step the operator restores the state before it
   and the step is retried, at most `max_retries` times. These runs get their own directory (`...-recover`)
   and their own summary; they never enter the strict numbers.

History: `set_task(prompt, reset_history=True)` at an episode start and after an intervention (a restore, or
resuming from a pause); between steps `reset_history=False` (queued actions dropped, history kept).
`between_steps = "reset_history"` exists for an A/B.

## Keys

| Key | During a step | At a prompt |
|---|---|---|
| `s` | end the step: success | verdict: success |
| `f` | end the step: fail | verdict: fail; at a restore prompt, end the episode |
| `r` | recover only: end the step, fail, ask for a retry (strict has no retries, so `r` does nothing there) | verdict: fail and retry (recover); at a restore prompt, retry |
| space | pause (hold position, the timer stops); again to resume, with the history reset (a pause is an intervention) | - |
| `q` | stop: the attempt is logged as aborted, files are finished, the run ends | quit |
| enter | - | start the episode; at a restore prompt, retry |

Keys pressed before a prompt appears are dropped, so a stray key never answers the wrong question.
Ctrl-C works like `q`. The terminal is restored on exit.

## How a step ends

Whichever comes first, then `loop.pause()`:

- **key:** the operator's `s`/`f`/`r` is the verdict.
- **auto:** the gripper closes past `closed_threshold` and, for `min_hold_s`, neither reopens past
  `open_threshold` nor closes past `empty_threshold` (closing that far means it closed on nothing: a miss),
  then reopens past `open_threshold`, then every joint moves slower than `rest_speed_deg_s` (measured between
  consecutive samples) for `rest_s`. A miss, or a piece that slips out, counts for nothing. A gripper that
  starts a step closed has to leave the closed band once first. Closing again after the release goes back to
  the grasp phase.
- **timeout:** `step_timeout_s` of robot time (pauses do not count).
- **error:** the control loop stopped (`loop.error`, e.g. 30 failed state reads after a USB drop) or the server
  connection was gone for 5 s outside a pause. That is not the policy's failure: the attempt gets
  `end_reason = "error"` and no verdict, the episode ends as `error` and stays out of the totals, and the run
  stops with the reason.

**eval mode:** after an auto end or a timeout the runner asks `s`/`f`; the operator is the ground truth.
**demo mode:** an auto end counts as success and the next step starts without a prompt; a timeout pauses
and asks. After a solved demo episode the next one carries the tower back, so the demo runs unattended.

**Tuning the auto end.** `closed_is` depends on the calibration (which end of the gripper's 0..100 range
is closed). Set `closed_threshold` above the gripper value while it holds a disk, `open_threshold`
below the value when it lets go, and `empty_threshold` between the fully closed value on air and the value
while it holds the thinnest piece (`robot/readout.py` shows both): every attempt in `attempts.jsonl` records
`auto_end.gripper_min` / `gripper_max`, and `ticks.jsonl` has the trace. In a dry run the measured arm does
not move, so use `signal = "cmd"` there: the gripper and the rest speed then come from the commands, and
there is no miss check (the model commands the same close for a piece as for air).

## Config (`config/hanoi.toml`)

| Field | Meaning |
|---|---|
| `pieces` | names, **largest first**; rename by colour at the venue |
| `places.left/middle/right` | the words used in prompts |
| `start`, `goal` | logical places (`left`, `middle`, `right`) |
| `prompt` | template with `{piece}`, `{place}` and optionally `{source}` |
| `step_timeout_s` | 30 |
| `mode` | `eval` or `demo` |
| `protocol`, `max_retries` | `strict` or `recover`, retries per step |
| `episodes`, `label` | episodes per run; the run directory suffix |
| `between_steps` | `keep_history` (decision 06) or `reset_history` |
| `[auto_end]` | `enabled`, `signal` (`state`/`cmd`), `closed_is` (`low`/`high`), `closed_threshold`, `open_threshold`, `empty_threshold` (optional; absent = no miss check), `min_hold_s`, `rest_s`, `rest_speed_deg_s` |
| `[loop]` | `LoopConfig`: `mode` (`prefetch`/`sync`), `fps`, `period`, `crossfade`, `exec_horizon`, `chunk`, `step_max_deg`, `spin_margin_ms`, `joint_limits` |
| `[robot]`, `[cameras]`, `[server]` | port and calibration id; camera sources, sizes, image format; server URL |
| `[video]` | `enabled`, `codec` (`mp4v`), `fps` |
| `[sanity]` | `moves` (`all`/`solution`), `repeats` |

Unknown keys and bad values are refused on load, all problems at once.

## Output

```
eval/runs/<UTC stamp>-<label>[-sanity|-recover][-demo]/     (gitignored)
  run.json        config, CLI, git commit and dirty flag, server hello, LoopConfig, loop stats at the end
  summary.md      the results table (below); summary.json the same as data
  epNN/
    episode.json    steps completed in a row, steps attempted, attempts, success, interventions, end,
                    UTC and perf_counter times, robot time, tick/chunk/video stats, loop stats
    attempts.jsonl  one line per step attempt: step, try, move, prompt, reset_history, times,
                    end_reason key|auto|timeout|quit|error, error, success, verdict_by, pauses, interventions,
                    auto-end phases and gripper range, chunk and tick counts, tower state before/after
    ticks.jsonl     every control tick: k, t, attempt, state, cmd, held, lateness, frame ages, remaining,
                    task, and the chunk handled on that tick (id, tick, d, accepted, reason, timings)
    chunks.npz      every chunk received: id, tick, d, k, accepted, start, reason, turnaround_ms,
                    server_ms, rtt_ms, attempt (the one running at the request tick), actions (N, 42, 6)
                    absolute. A step waits up to 3 s for the reply to its last request, so none is lost.
    scene.mp4, wrist.mp4             full resolution, mp4v, nominal 30 fps
    scene.times.csv, wrist.times.csv frame index -> perf_counter capture time (aligns with ticks.jsonl t)
```

Interventions are pauses plus restores before a retry. Aborted episodes (`q`) and episodes ended by an
infrastructure error stay in the table, marked, and are left out of the totals; `summary.md` prints the
control loop's error.

## Into docs/status

`summary.md` ends with a status row; paste it under the results table of the next `docs/status/` snapshot.
For several runs at once:

```sh
uv run --project eval python -m hanoi_eval.summary --status eval/runs/*-zero-shot*
uv run --project eval python -m hanoi_eval.summary eval/runs/<run>      # rewrite one run's summary
```

| Run | Kind | Episodes | Episode successes | Completed steps | Successful / total attempts | Mean steps in a row |
|---|---|---|---|---|---|---|

The grading numbers of the track are the last three columns: completed task steps, successful attempts, total attempts.

## Tests

```sh
cd eval && uv run --python 3.13 pytest -q
```

Unit tests cover the solver, the state tracker, prompts, the config checks and the step-end detector on
synthetic traces (including a miss on air, a slipped piece, a start between the thresholds and the commanded
signal). `test_runner_fake.py` runs full episodes through the real `ControlLoop` with `FakeRobot`,
`FakeCameras` and `FakeNet(pickplace)` and scripted keys (strict success, a failed step, recover, quit, sanity,
a dead control loop as an error episode, the history reset on resume, `r` ignored in strict mode, every chunk
logged with the attempt that requested it, `--print-warmup`).
`test_e2e_socket.py` starts `infra/gpu/so101/server.py --engine fake` (3 s pickplace cycle, 300 ms latency) as a
subprocess and runs one strict episode with fake hardware and the real `NetClient` over the socket, with the
unattended operator. It checks the output layout, zero holds after each step's first chunk, and that every recorded
chunk matches the server's request log (id, tick, actions). About 15 s.

## Known limits

- Between episodes the arm holds its last pose with torque on, so each episode starts from wherever the
  previous one ended. The loop API has no home command, and a torque call from this thread would race the
  control thread on the serial bus. A home pose or a torque-off hold between episodes belongs in robot/.
- `opencv-python-headless` instead of `opencv-python`: LeRobot 0.6.1 (through so101-client) requires the headless
  wheel, and two OpenCV wheels in one environment ship the same `cv2`. `VideoWriter` works in both.
