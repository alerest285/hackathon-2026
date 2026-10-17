# SO-101 policy server (T10)

Serves `black-forest-labs/flux-3-action-so101` from the GPU host to the Mac that drives the arm, over the
`so101.v1` WebSocket protocol in [docs/architecture.md](../../../docs/architecture.md) (decision 06). The server
is stateless: every request carries its own 8-tick window, and the Mac owns the history. One inference runs
at a time on a worker thread, and pings are answered while it runs. It binds 127.0.0.1:8000 only.

| File | What |
|---|---|
| `server.py` | CLI, WebSocket server, `--selftest` |
| `protocol.py` | `so101.v1` codec and message helpers. numpy + msgpack only, so the Mac client can import it |
| `engines/` | `flux-action` (default), `lerobot`, `fake`. torch is imported only when a GPU engine is built |
| `serve.sh` | picks the venv for the engine, reinstalls websockets/msgpack if a `uv sync` removed them, warns without an HF token |
| `bootstrap-lerobot.sh` | builds `~/lerobot-flux3` (LeRobot main `146d847`) for the `lerobot` engine |
| `tests/` | pytest with the fake engine over real localhost sockets |

## On the GPU host

1. `bash infra/gpu/bootstrap.sh` builds `~/flux-action` at `830f1ce` (T03). For the A/B engine, also run
   `bash infra/gpu/so101/bootstrap-lerobot.sh`.
2. Accept the licence of `black-forest-labs/flux-3-action-so101` and `black-forest-labs/flux-3-action-base`
   on the Hub, then `~/flux-action/.venv/bin/hf auth login`. The token stays in `~/.cache/huggingface/token`;
   never put it in the repo or a log line.
3. `bash infra/gpu/so101/serve.sh --selftest`. The first run downloads the policy and both shared encoders into
   the HF cache. It prints p50/p95/max ms per plan over 50 plans, peak GPU memory, and PASS or FAIL.
4. Serve inside tmux, with one `--warmup-task` per Hanoi instruction, so no caption is encoded for the first time
   mid-episode. On the Mac, `uv run --project eval python eval/run_hanoi.py --print-warmup` prints this command
   with every prompt the eval config can produce (9 with the default config):

   ```sh
   bash infra/gpu/so101/serve.sh --log-dir ~/so101-logs \
     --warmup-task 'pick up the small disk and place it on the right peg' --warmup-task '...'
   ```

   A plan whose caption was not warmed up prints a `WARNING: caption not warmed up` line, and hello lists the
   warmed captions in `warm_tasks`, which eval checks when it connects.
5. On the Mac: `infra/aws/ssh.sh` (it forwards 8000), or `ssh -N -L 8000:localhost:8000 ubuntu@<host>`. The
   client connects to `ws://localhost:8000`.

Pins (decision 06): checkpoint `c9e13b2aca6a0a472b3ea03fd90cff32d3e85849` (`--revision`), shared VAE and text
encoder `62878e2925e59b7a89ec14463ce89932624c490d` (`--base-revision`). The checkpoint config names the encoders
(`video_vae_id`, `text_encoder_id`, Hub specs of the form `repo[:file][@revision]`), and both upstream docs say
it references a pinned base revision. That config is gated and nobody has read it yet, so the config's own pin
wins, as in both upstream loaders: `--base-revision` pins only a spec without one, and a config that pins a
different commit stops the server before the weights load, with the commit in the message. Then either pass
`--base-revision ''` (the checkpoint's pin, what upstream does) or `--force-base-revision` (decision 06's pin), and
correct decision 06. The first lines of every start print the config's specs; hello and `--selftest` show them as
`config_video_vae_id`/`config_text_encoder_id` next to the specs in force.

Sampling: `--guidance VIDEO ACTION` and `--steps N` override the checkpoint's 3/3 and 4 and print a WARNING.
Guidance 1/1 runs one pass per step instead of two (about half the compute), and decision 06 keeps it as the
fallback if 3/3 misses the 533 ms budget; hello and the selftest line report the values in force.

Normalization: `--norm-stats stats.json` replaces the checkpoint's state and action q01/q99 for zero-shot experiments,
such as the new-calibration SO-101 stats of `multimodalart/flux-3-action-so101-sim@09edebb`, used with an arm in native
LeRobot degrees and no joint map. It does what a checkpoint copy with swapped normalizer files does, without the copy.
The file takes the hello's keys (`state_q01`, `state_q99`, `action_q01`, `action_q99`) or flux-action's nested form
(`{"state": {"q01", "q99"}, "action": {...}}`). Either stream may be left out, and each needs 6 finite values with
q99 > q01. Both engines apply it identically (bit-equal plans on the CPU fixture). With `--adapter` it only checks: an
adapter carries the normalization it was trained with in its processors, so other stats refuse to start. The hello
reports `norm_stats` (path, sha256, streams, mode) next to the effective `state_q01` ... `action_q99`.

## Engines

| Engine | Path | `--compile` |
|---|---|---|
| `flux-action` (default, decision 03) | `load_policy(repo, revision)`, `set_text_encoder_offload(True)` before `.to("cuda")`, `predict_action_chunk` on the explicit window | **experimental**: wraps the DiT forward in `torch.compile`, as LeRobot's `compile_model` does. flux-action itself compiles only the DROID geometry, and nobody has compiled the real SO-101 DiT this way. If the first warmup plan fails compiled, the engine goes back to eager, prints a WARNING and reports `compile: false` plus `compile_failed` in hello |
| `lerobot` | `Flux3Policy.from_pretrained` + the saved processors, stateless `observation.command_history` branch; `--adapter DIR` loads a PEFT/LoRA adapter (milestone 4) | sets `compile_model` before construction: DiT forward, VAE and Qwen layers |
| `fake` | no torch; `--fake-mode hold\|sine\|pickplace`, `--fake-latency-ms`, `--fake-jitter-ms`, `--fake-period-s` (pickplace cycle, default 9 s) | ignored |

The window's two frames sit at slots 0 and 7; slots 1-6 repeat frame 0 and are never encoded. The text encoder is
offloaded to the CPU by default (`--text-encoder-on-gpu` keeps it on the GPU). Sampling is the checkpoint's own
(Euler 4 steps, guidance 3/3, seed 42, BF16).

The A/B, one self-test each:

```sh
bash infra/gpu/so101/serve.sh --selftest                              # flux-action, eager
bash infra/gpu/so101/serve.sh --selftest --compile                    # flux-action, compiled DiT (experimental)
bash infra/gpu/so101/serve.sh --selftest --engine lerobot             # lerobot, eager
bash infra/gpu/so101/serve.sh --selftest --engine lerobot --compile   # lerobot, compile_model
bash infra/gpu/so101/serve.sh --selftest --guidance 1 1               # only if 3/3 misses the budget (decision 06)
```

The selftest line prints the first warmup plan (compile, kernels and the caption) next to the steady-state p50,
so a compiled run that graph-breaks into something slower than eager shows up there.

## What to log (AGENTS.md section 5)

After each self-test, one line with the numbers and settings, for example:

```sh
scripts/log.sh --task T10 "selftest flux-action eager, H100, BF16, Euler 4, guidance 3/3: p50 X ms p95 Y ms max Z ms; peak A GiB load+warmup, B GiB plans"
```

Then the round trip through the tunnel (milestone 2), and the decision record with protocol, settings and numbers
(T10 done-when). The GPU start and stop lines with the hourly price, as always.

## Protocol details

- Every message is a msgpack map with a `"type"` key: `hello`, `ping`, `pong`, `infer`, `chunk`, `error`.
  The fields are the ones in the architecture table.
- `hello` also carries the engine's settings: `sampler`, `seed`, `dtype`, `compile` (plus `compile_failed` after
  a fallback), `offload_text_encoder`, `device`, `gpu`, `torch`, the upstream commit, `video_vae_id` and
  `text_encoder_id` in force next to the checkpoint config's `config_video_vae_id`/`config_text_encoder_id`, the
  lerobot engine's `base`/`base_revision` (what it loaded, which an adapter can change), `warm_tasks` (the
  captions warmed up), and the checkpoint's normalization ranges `state_q01`/`state_q99`/`action_q01`/`action_q99`.
  A client can compare its measured state against `state_q01..q99` to catch a units or calibration mismatch
  before the arm moves. `guidance` is `[video, action]`.
- Frames are decoded with Pillow (OpenCV if Pillow is missing) to RGB. All four frames of a request must share
  one size; send 256x256, which the model's side-by-side layout uses as is.
- Validation errors come back at once with the request's `id`; decode and engine errors come back after the
  queue. The connection stays open. A ninth queued request on one connection gets a "server busy" error.
- `timing`: `queue_ms` is receipt to the worker starting (it includes waiting for an earlier plan),
  `decode_ms` the JPEG decode, `infer_ms` the engine including the GPU sync, `total_ms` receipt to the reply
  being sent. Network time is the client's round trip minus `total_ms`.

## Output

One stdout line per request (this one from the fake engine):

```
13:02:39.729 infer id=1 tick=101 queue 0.1 decode 0.9 infer 0.1 total 1.1 ms  256x256 jpeg 101 KB  127.0.0.1:55191  'pick up the small disk and place it on the right peg'
```

With `--log-dir DIR`, `DIR/so101-<UTC stamp>.jsonl` gets one line per event: `start` (hello, argv), `connect`,
`infer` (id, tick, task, timing, state, cmd_hist, actions, frame size, request bytes), `error`, `disconnect`.
No images.

## Troubleshooting

| Symptom | Fix |
|---|---|
| 401/403, `GatedRepoError` | accept both licences with the account whose token is logged in, then `hf auth login` |
| `the checkpoint config pins ... but --base-revision is ...` | the checkpoint names another encoder commit than decision 06: rerun with `--base-revision ''` (its own pin), log it, correct decision 06 |
| `No module named 'natten'`, VAE fails to build | a `uv sync` removed NATTEN; re-run `infra/gpu/bootstrap.sh` (or `bootstrap-lerobot.sh`). websockets and msgpack are reinstalled by `serve.sh` on start |
| CUDA out of memory | keep the default text-encoder offload; check `nvidia-smi` for another server on the GPU |
| `cannot listen on 127.0.0.1:8000` | another server holds the port (`ss -ltnp \| grep 8000`), or pass `--port` |
| the Mac cannot connect | is the tunnel up? The server listens on 127.0.0.1 only, by design |
| the first plan of a new instruction is slow, `WARNING: caption not warmed up` | the caption is being encoded (the offloaded Qwen visits the GPU once per caption) or the model is compiling; warm up every instruction with `--warmup-task` (`eval/run_hanoi.py --print-warmup`) |
| `lerobot` engine, new captions take seconds | its offloaded Qwen encodes on the CPU (LeRobot's documented low-VRAM mode). Warm up every caption, or `--text-encoder-on-gpu`. With `--compile` the Qwen layers compile on the CPU, which needs a C++ compiler |
| self-test prints FAIL | the traceback is printed above the FAIL line |

## Tests

From the repo root:

```sh
uv run --python 3.13 --with pytest --with websockets --with msgpack --with numpy --with pillow \
  pytest infra/gpu/so101/tests -q
```

20 tests, a few seconds, no torch: hello fields, ping/pong, infer in jpeg/png/raw (hold mode returns `cmd_hist[-1]`),
malformed requests on a connection that stays usable, pongs under 50 ms during a 300 ms plan, replies in
order, the busy limit, the fake motion modes and `--fake-period-s`, `--selftest`, the CLI server with
`--log-dir`, Ctrl-C and the unwarmed-caption warning, the encoder pin rule, `--guidance`/`--steps`, and the
flux-action fallback from a failed compiled warmup to eager. The Mac client is tested against this server in `eval/tests/test_e2e_socket.py`: the
server runs as a subprocess with the fake engine, and a full Hanoi episode goes through `robot/`'s `NetClient`.

## Verified on 17.10.2026 (Mac, no GPU)

- The test suite passes with websockets 17.2 and with 15.0.1 (the version in flux-action's lock).
- `serve.sh` picks the venv, installs missing websockets/msgpack/Pillow, warns without NATTEN or an HF token, and
  execs the server.
- Both GPU engines ran on CPU against a tiny SO-101 package built from the upstream test fixture (tiny DiT, fake
  VAE, mock text encoder), eager and compiled. The explicit start window gives exactly the actions of
  flux-action's fresh-episode padding and of LeRobot's synchronous first tick; frames in slots 1-6 change
  nothing; swapping i0 and i7 changes the output; the two engines return identical actions for the same window.

## Not verified yet

- Real weights on a GPU: download, load, memory, latency. That is the first `--selftest` on the box.
- Which encoder commit the gated checkpoint config pins: the first start prints it.
- `--compile` on the real SO-101 DiT (flux-action engine).
- `bootstrap-lerobot.sh` on the box, and loading a real LoRA adapter (`--adapter`).
