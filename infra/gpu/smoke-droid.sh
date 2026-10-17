#!/usr/bin/env bash
# Run ON the GPU host after bootstrap.sh. Milestone 1 check: released DROID policy, recorded
# observation, no robot. Commands are BFL's docs/setup.md + docs/prepare.md verbatim.
#   bash infra/gpu/smoke-droid.sh [subfolder]     e.g. variants/gd-fp8r (default: root BF16)
set -euo pipefail
cd "${FLUX_DIR:-$HOME/flux-action}"
SUB="${1:-}"; tag="${SUB:-root}"; tag="${tag//\//-}"
OUT="outputs/droid/run-$tag-$(date -u +%Y%m%dT%H%M%SZ)"

[ -d outputs/droid ] || uv run hf download black-forest-labs/flux-3-action-droid --exclude 'variants/*' --local-dir outputs/droid
if [ ! -f outputs/public-droid/observation.npz ]; then
  uv run hf download nvidia/Cosmos3-DROID --repo-type dataset \
    --revision 5c11a20accb11497270a5247a7f1e66ad04c956c \
    --local-dir outputs/public-droid/source \
    --include success/meta/info.json \
    --include success/meta/tasks.parquet \
    --include success/meta/episodes/chunk-000/file-000.parquet \
    --include success/data/chunk-000/file-000.parquet \
    --include success/videos/observation.image.wrist_image_left/chunk-000/file-000.mp4 \
    --include success/videos/observation.image.exterior_image_1_left/chunk-000/file-000.mp4 \
    --include success/videos/observation.image.exterior_image_2_left/chunk-000/file-000.mp4
  uv run hf download KarlP/droid keep_ranges_1_0_1.json --repo-type model \
    --revision bcb840c3b496533e0adf548a54b51f2f00057837 --local-dir outputs/public-droid/source
  uv run flux-action prepare-droid --source-root outputs/public-droid/source \
    --lock configs/droid/public_sample.lock.json --episode-index 0 --output outputs/public-droid/episode-000000
  uv run flux-action validate-data outputs/public-droid/episode-000000
  uv run python examples/droid/make_observation.py outputs/public-droid/episode-000000 \
    --seed 0 --output outputs/public-droid/observation.npz
fi
task_caption=$(uv run python -c 'import json; print(json.load(open("outputs/public-droid/observation.json"))["task"])')
echo "task: $task_caption"
if [ -n "$SUB" ]; then ckpt=(--checkpoint black-forest-labs/flux-3-action-droid --subfolder "$SUB"); else ckpt=(--checkpoint outputs/droid); fi
uv run flux-action infer "${ckpt[@]}" --observation outputs/public-droid/observation.npz --task "$task_caption" --output "$OUT"
uv run python - "$OUT" <<'PY'
import json, sys, numpy as np
out = sys.argv[1]
a = np.load(f"{out}/actions.npy"); print("actions", a.shape, a.dtype, "first:", np.round(a[0, 0], 3))
assert a.shape == (1, 32, 8), "unexpected shape"
r = json.load(open(f"{out}/report.json"))
keys = [k for k in r if any(s in k.lower() for s in ("tim", "mem", "gpu"))]
print(json.dumps({k: r[k] for k in keys}, indent=1)[:1500])
PY
echo "PASS: paste the numbers above into your log (scripts/log.sh --task T04 ...)."
