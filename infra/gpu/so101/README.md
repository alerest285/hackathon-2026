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
4. Serve inside tmux:

   ```sh
   bash infra/gpu/so101/serve.sh --log-dir ~/so101-logs \
     --warmup-task "pick up the small red disk and place it on the right peg" --warmup-task "..."
   ```

   Repeat `--warmup-task` for every Hanoi instruction, so no caption is encoded for the first time mid-episode.
5. On the Mac: `infra/aws/ssh.sh` (it forwards 8000), or `ssh -N -L 8000:localhost:8000 ubuntu@<host>`. The
   client connects to `ws://localhost:8000`.

Pins (decision 06): checkpoint `c9e13b2aca6a0a472b3ea03fd90cff32d3e85849` (`--revision`), shared VAE and text
encoder `62878e2925e59b7a89ec14463ce89932624c490d` (`--base-revision`). The checkpoint config names the encoders
(`video_vae_id`, `text_encoder_id`, Hub specs of the form `repo[:file][@revision]`), and both upstream docs say
it references a pinned base revision. The engines set `@<--base-revision>` on those specs, print which pin is in
force, and print a WARNING if the config pins a different one. `--base-revision ''` keeps what the config says.

## Engines

| Engine | Path | `--compile` |
|---|---|---|
| `flux-action` (default, decision 03) | `load_policy(repo, revision)`, `set_text_encoder_offload(True)` before `.to("cuda")`, `predict_action_chunk` on the explicit window | wraps the DiT forward in `torch.compile`, as LeRobot's `compile_model` does. flux-action itself compiles only the DROID geometry |
| `lerobot` | `Flux3Policy.from_pretrained` + the saved processors, stateless `observation.command_history` branch; `--adapter DIR` loads a PEFT/LoRA adapter (milestone 4) | sets `compile_model` before construction: DiT forward, VAE and Qwen layers |
| `fake` | no torch; `--fake-mode hold\|sine\|pickplace`, `--fake-latency-ms`, `--fake-jitter-ms`, `--fake-period-s` (pickplace cycle, default 9 s) | ignored |

The window's two frames sit at slots 0 and 7; slots 1-6 repeat frame 0 and are never encoded. The text encoder is
offloaded to the CPU by default (`--text-encoder-on-gpu` keeps it on the GPU). Sampling is the checkpoint's own
(Euler 4 steps, guidance 3/3, seed 42, BF16).

The A/B, one self-test each:

```sh
bash infra/gpu/so101/serve.sh --selftest                              # flux-action, eager
bash infra/gpu/so101/serve.sh --selftest --compile                    # flux-action, compiled DiT
bash infra/gpu/so101/serve.sh --selftest --engine lerobot             # lerobot, eager
bash infra/gpu/so101/serve.sh --selftest --engine lerobot --compile   # lerobot, compile_model
```

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
- `hello` also carries the engine's settings: `sampler`, `seed`, `dtype`, `compile`, `offload_text_encoder`,
  `device`, `gpu`, `torch`, the upstream commit, `video_vae_id`, `text_encoder_id`, and the checkpoint's
  normalization ranges `state_q01`/`state_q99`/`action_q01`/`action_q99`. A client can compare its measured
  state against `state_q01..q99` to catch a units or calibration mismatch before the arm moves.
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
13:02:39.729 infer id=1 tick=101 queue 0.1 decode 0.9 infer 0.1 total 1.1 ms  256x256 jpeg 101 KB  127.0.0.1:55191  'pick up the small red disk'
```

With `--log-dir DIR`, `DIR/so101-<UTC stamp>.jsonl` gets one line per event: `start` (hello, argv), `connect`,
`infer` (id, tick, task, timing, state, cmd_hist, actions, frame size, request bytes), `error`, `disconnect`.
No images.

## Troubleshooting

| Symptom | Fix |
|---|---|
| 401/403, `GatedRepoError` | accept both licences with the account whose token is logged in, then `hf auth login` |
| `No module named 'natten'`, VAE fails to build | a `uv sync` removed NATTEN; re-run `infra/gpu/bootstrap.sh` (or `bootstrap-lerobot.sh`). websockets and msgpack are reinstalled by `serve.sh` on start |
| CUDA out of memory | keep the default text-encoder offload; check `nvidia-smi` for another server on the GPU |
| `cannot listen on 127.0.0.1:8000` | another server holds the port (`ss -ltnp \| grep 8000`), or pass `--port` |
| the Mac cannot connect | is the tunnel up? The server listens on 127.0.0.1 only, by design |
| the first plan of a new instruction is slow | the caption is being encoded (the offloaded Qwen visits the GPU once per caption) or the model is compiling; warm up every instruction with `--warmup-task` |
| `lerobot` engine, new captions take seconds | its offloaded Qwen encodes on the CPU (LeRobot's documented low-VRAM mode). Warm up every caption, or `--text-encoder-on-gpu`. With `--compile` the Qwen layers compile on the CPU, which needs a C++ compiler |
| self-test prints FAIL | the traceback is printed above the FAIL line |

## Tests

From the repo root:

```sh
uv run --python 3.13 --with pytest --with websockets --with msgpack --with numpy --with pillow \
  pytest infra/gpu/so101/tests -q
```

17 tests, a few seconds, no torch: hello fields, ping/pong, infer in jpeg/png/raw (hold mode returns `cmd_hist[-1]`),
malformed requests on a connection that stays usable, pongs under 50 ms during a 300 ms plan, replies in
order, the busy limit, the fake motion modes and `--fake-period-s`, `--selftest`, and the CLI server with
`--log-dir` and Ctrl-C. The Mac client is tested against this server in `eval/tests/test_e2e_socket.py`: the
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
- `bootstrap-lerobot.sh` on the box, and loading a real LoRA adapter (`--adapter`).
