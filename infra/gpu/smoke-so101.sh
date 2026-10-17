#!/usr/bin/env bash
# Run ON the GPU host after bootstrap.sh. SO-101 milestone-1 check: the released SO-101 policy on a
# synthetic observation (no robot, no dataset). Uses BFL's examples/so101/inference.py unchanged.
# Prints load time, per-plan latency over N plans, peak GPU memory, and the action shape (1, 42, 6).
#   bash infra/gpu/smoke-so101.sh [N=5]
set -euo pipefail
cd "${FLUX_DIR:-$HOME/flux-action}"
N="${1:-5}"; OUT="outputs/so101/smoke-$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$OUT"

# Synthetic observation in the delivered units: two HWC uint8 RGB cameras, six joints
# (degrees; gripper in percent). Behaviour is meaningless; shape, load and timing are not.
uv run python - "$OUT/observation.npz" <<'PY'
import sys, numpy as np
rng = np.random.default_rng(0)
np.savez(sys.argv[1],
         **{"images.scene": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
            "images.wrist": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
            "state": np.array([0, -10, 10, 60, 0, 20], dtype=np.float32)})
PY

# Peak GPU memory sampler.
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) > "$OUT/mem.txt" 2>/dev/null &
sampler=$!; trap 'kill $sampler 2>/dev/null || true' EXIT

uv run --extra encoders python - "$OUT" "$N" <<'PY'
import sys, time, numpy as np, torch
from flux_action.inference.offline import load_observation
from flux_action.inference.so101 import load_policy
out, n = sys.argv[1], int(sys.argv[2])
t = time.perf_counter()
policy = load_policy("black-forest-labs/flux-3-action-so101").to("cuda")
print(f"load: {time.perf_counter() - t:.1f} s")
batch = load_observation(f"{out}/observation.npz", policy.config, "pick up the small ring and place it on the right peg", "cuda")
times = []
for i in range(n):
    torch.cuda.synchronize(); t = time.perf_counter()
    with torch.no_grad():
        a = policy.predict_action_chunk(batch).float().cpu().numpy()
    torch.cuda.synchronize(); times.append(time.perf_counter() - t)
assert np.isfinite(a).all(), "non-finite actions"
print("actions", a.shape, "first:", np.round(a[0, 0], 2))
print("plan latency s: first %.2f, rest p50 %.2f max %.2f" % (times[0], np.median(times[1:] or times), max(times[1:] or times)))
print("torch peak allocated GB: %.1f" % (torch.cuda.max_memory_allocated() / 1e9))
np.save(f"{out}/actions.npy", a)
PY
kill $sampler 2>/dev/null || true
echo "nvidia-smi peak MiB: $(sort -n "$OUT/mem.txt" | tail -1)"
echo "PASS: SO-101 smoke. Log these numbers: scripts/log.sh --task T13 \"...\""
