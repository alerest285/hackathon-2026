#!/usr/bin/env bash
# LoRA on flux-3-action-so101 with LeRobot (BFL docs/so101-lora.md recipe, train/lora.json). Run ON the GPU host
# after train/setup.sh, with train/ and infra/gpu/so101/ synced to the host and the dataset from train/prepare.py
# rsynced to /workspace/datasets/<name>.
#
#   bash train/train.sh probe  <name> [--batch 2|4|8]   PERFORMANCE ONLY, saves nothing: s/step, data/prep wait, VRAM, ETA
#   bash train/train.sh smoke  <name> [--batch B]       save -> resume -> load raw + EMA adapters in the server; required
#                                                        once per setup before a full run
#   bash train/train.sh run    <name> [--batch B] [--updates 2500] [--effective 8] [--tag v1]   full run in tmux
#   bash train/train.sh resume <run> [--workers N] [--wandb E/P]   restage the dataset from /workspace (manifest hash
#                                                        checked) and continue <run> from its last checkpoint, in tmux
#
# The optimizer always sees an effective batch of --effective (default 8, BFL's single-GPU preset): accumulation is
# effective / batch, and steps, save and eval scale with it, so --batch only changes speed, not the recipe.
# Checkpoints every 125 optimizer updates, raw and EMA. Every run has a sibling <run>.provenance/, written before
# training starts and never moved: the dataset's prepare_manifest.json, the joint maps it was converted with, and
# run.json (dataset, manifest sha256, stage path). Resume and serving read it there.
# Refuses to start while another process uses the GPU (live serving needs its latency) unless --share.
# Options: --workers N (loader processes; default 3 for probe/smoke, 6 for run: the old pod measured data_s 0.004 at
# 0.9 s/microstep, and every spawn worker re-imports the venv), --wandb ENTITY/PROJECT (needs WANDB_API_KEY in the env).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mode="${1:-}"; name="${2:-}"; shift 2 || true
batch=2 updates=2500 effective=8 tag="" share=0 workers="" wandb=""
while [ $# -gt 0 ]; do case "$1" in
  --batch) batch="$2"; shift 2 ;; --updates) updates="$2"; shift 2 ;; --effective) effective="$2"; shift 2 ;;
  --tag) tag="$2"; shift 2 ;; --share) share=1; shift ;;
  --workers) workers="$2"; shift 2 ;; --wandb) wandb="$2"; shift 2 ;;
  *) echo "unknown option $1" >&2; exit 2 ;; esac; done
usage() { sed -n 2,22p "$0"; exit 2; }
case "$mode" in probe|smoke|run|resume) ;; *) usage ;; esac
[ -n "$name" ] || usage
[ $((effective % batch)) -eq 0 ] || { echo "--effective $effective is not a multiple of --batch $batch" >&2; exit 2; }
accum=$((effective / batch))

export LEROBOT_DIR="${LEROBOT_DIR:-/workspace/lerobot-flux3}" HF_HOME="${HF_HOME:-/workspace/hf}"
[ -f /workspace/train-env.sh ] || { echo "run train/setup.sh first" >&2; exit 1; }
. /workspace/train-env.sh
PY="$LEROBOT_DIR/.venv/bin/python"
ENV_PREFIX="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false"
mkdir -p /workspace/train-logs

gpu_free_or_exit() {
  local others; others=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -v "^$" || true)
  if [ -n "$others" ] && [ "$share" = 0 ]; then
    echo "GPU busy (pid, MiB):"; echo "$others"
    echo "Training next to a live server breaks its latency budget. Stop it, use a training pod, or pass --share." >&2
    exit 1
  fi
}
sha() { sha256sum "$1" | cut -d' ' -f1; }
# Stage on RAM: every sample decodes 50 frames per camera, and /workspace is a network volume.
stage_dataset() {  # <dataset name> [<fixed stage dir>] -> prints the staged path
  local src="/workspace/datasets/$1" stage="${2:-}"
  if [ -z "$stage" ]; then
    local need_mb free_mb; need_mb=$(( $(du -sm "$src" | cut -f1) + 512 )); free_mb=$(df -m /dev/shm | awk 'NR==2{print $4}')
    stage=/dev/shm/datasets/$1; [ "$free_mb" -gt "$need_mb" ] || stage=/root/datasets/$1
  fi
  mkdir -p "$stage"; rsync -a --delete "$src/" "$stage/" >&2
  echo "$stage"
}
loader_workers() {  # <wanted>: capped at CPUs - 2
  local cpus quota; cpus=$(python3 -c 'import os;print(len(os.sched_getaffinity(0)))')
  quota=$(awk '{ if ($1=="max") print 0; else print int($1/$2) }' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 0)
  [ "$quota" -gt 0 ] && [ "$quota" -lt "$cpus" ] && cpus=$quota
  local cap=$(( cpus > 3 ? cpus - 2 : 1 ))
  echo $(( $1 < cap ? $1 : cap ))
}
# Spawn workers each re-import torch + lerobot; from the network volume that took ~30 s per worker (19:44Z). One
# parallel read of the venv puts it in the page cache first, so the imports read RAM.
warm_venv() {
  local t0; t0=$(date +%s)
  find "$LEROBOT_DIR/.venv" "$LEROBOT_DIR/src" -type f \( -name '*.py' -o -name '*.so*' -o -name '*.pyc' \) -print0 \
    | xargs -0 -P 16 -n 200 cat > /dev/null 2>&1 || true
  echo "venv warmed into page cache in $(( $(date +%s) - t0 )) s"
}
in_tmux() {  # <session> <cmd file> <log>: EXIT carries the trainer's own status, not tee's
  # The W&B key reaches the session by name (update-environment copies it from this shell), never as an argv value.
  # One tmux invocation: a fresh, empty server exits between separate calls and drops the option (codex 19:5xZ).
  local body="bash $2 2>&1 | tee $3; echo EXIT \${PIPESTATUS[0]} >> $3"
  if [ -n "${WANDB_API_KEY:-}" ]; then
    tmux start-server \; set-option -ga update-environment " WANDB_API_KEY" \; new-session -d -s "$1" "$body"
  else
    tmux new-session -d -s "$1" "$body"
  fi
  echo "tmux attach -t $1   (detach: Ctrl-b d).  Follow: tail -f $3 | grep -E 'step|loss|Error|EXIT'"
}

# ---------------------------------------------------------------- resume
if [ "$mode" = resume ]; then
  gpu_free_or_exit
  out="/workspace/outputs/lora/$name"; prov="$out.provenance"
  cfg="$out/checkpoints/last/pretrained_model/train_config.json"
  [ -f "$cfg" ] && [ -f "$prov/run.json" ] || { echo "no resumable run at $out (needs checkpoints/last and $prov/run.json)" >&2; exit 1; }
  read -r dname want stage < <(python3 -c "import json;r=json.load(open('$prov/run.json'));print(r['dataset'],r['manifest_sha256'],r['stage'])")
  have=$(sha "/workspace/datasets/$dname/prepare_manifest.json" 2>/dev/null || echo missing)
  [ "$have" = "$want" ] || { echo "dataset /workspace/datasets/$dname changed since $name started (manifest $have, run trained on $want): not resuming" >&2; exit 1; }
  for m in "$prov"/maps/*.json; do
    [ -e "$m" ] || { echo "run $name has no provenance maps: prepare.py always copies them, refusing to resume" >&2; exit 1; }
    [ "$(sha "$m")" = "$(basename "$m" .json)" ] || { echo "provenance map $m does not match its hash" >&2; exit 1; }
  done
  staged=$(stage_dataset "$dname" "$stage")
  echo "restaged $dname at $staged (manifest $want verified)"
  log="/workspace/train-logs/$name.resume-$(date -u +%m%d-%H%M).log"
  cmd="${log%.log}.cmd"
  extra=""   # explicit flags override the saved train_config.json
  [ -n "$workers" ] && extra+=" --num_workers=$(loader_workers "$workers")"
  if [ -n "$wandb" ]; then
    [ -n "${WANDB_API_KEY:-}" ] || grep -qs api.wandb.ai ~/.netrc || { echo "--wandb needs WANDB_API_KEY in the env (or wandb login)" >&2; exit 2; }
    extra+=" --wandb.enable=true --wandb.entity=${wandb%%/*} --wandb.project=${wandb#*/} --wandb.disable_artifact=true"
    # Same destination as the saved run: LeRobot continues its W&B run. New or changed destination: the pinned logger
    # would look up the old run id and fail, so start a fresh W&B run that allows resume (codex review 19:5xZ).
    same=$(python3 -c "import json;w=json.load(open('$cfg')).get('wandb') or {};print(int(bool(w.get('enable')) and w.get('entity')=='${wandb%%/*}' and w.get('project')=='${wandb#*/}' and bool(w.get('run_id'))))")
    [ "$same" = 1 ] || extra+=" --wandb.run_id=${name}-r$(date -u +%H%M%S) --wandb.resume=allow"
  fi
  [ -n "$workers" ] && warm_venv
  echo "cd \"$LEROBOT_DIR\" && $ENV_PREFIX $PY -m lerobot.scripts.lerobot_train --config_path=$cfg --resume=true$extra" > "$cmd"
  echo "resume overrides:${extra:- none (saved config)}"
  in_tmux "lora-$name" "$cmd" "$log"
  exit 0
fi

# ---------------------------------------------------------------- probe | smoke | run
src="/workspace/datasets/$name"
[ -f "$src/meta/info.json" ] || { echo "no dataset at $src (rsync train/prepare.py's --out there)" >&2; exit 1; }
[ -f "$src/prepare_manifest.json" ] || { echo "$src has no prepare_manifest.json: only train on train/prepare.py output (model space)" >&2; exit 1; }
gpu_free_or_exit
staged=$(stage_dataset "$name")
echo "dataset staged at $staged ($(du -sh "$staged" | cut -f1)); $(python3 -c "import json;m=json.load(open('$src/prepare_manifest.json'));print(m['episodes'],'episodes,',m['minutes'],'min')")"
[ -n "$workers" ] || { [ "$mode" = run ] && workers=6 || workers=3; }
workers=$(loader_workers "$workers")
warm_venv

stamp=$(date -u +%m%d-%H%M%S)-$(od -An -N2 -tx2 /dev/urandom | tr -d ' ')
run="${name}${tag:+-$tag}-b${batch}x${accum}-${stamp}"
case "$mode" in
  probe) out="/workspace/outputs/lora-probe/$run"; steps=$((40 / accum * accum)); save=$((steps * 10)); logf=$accum ;;
  smoke) out="/workspace/outputs/lora-smoke/$run"; steps=$((4 * accum)); save=$((2 * accum)); logf=$accum ;;
  run)   out="/workspace/outputs/lora/$run"; steps=$((updates * accum)); save=$((125 * accum)); logf=$((5 * accum)) ;;
esac
prov="$out.provenance"
[ -e "$out" ] || [ -e "$prov" ] && { echo "run id collision: $out or $prov exists; nothing touched" >&2; exit 1; }
mkdir -p "$(dirname "$out")"
log="/workspace/train-logs/$run.log"; cmd="/workspace/train-logs/$run.cmd"

args=(-m lerobot.scripts.lerobot_train --config_path="$ROOT/train/lora.json"
  --dataset.repo_id="alerest285/$name" --dataset.root="$staged"
  --policy.path="$POLICY_DIR" --policy.device=cuda
  --policy.video_vae_id="$ENCODERS_DIR/video_vae.safetensors" --policy.text_encoder_id="$ENCODERS_DIR/text_encoder"
  --batch_size="$batch" --accelerator.gradient_accumulation.steps="$accum"
  --steps="$steps" --save_freq="$save" --eval_steps="$save" --log_freq="$logf"
  --num_workers="$workers" --output_dir="$out")
[ "$mode" = probe ] && args+=(--save_checkpoint=false)
if [ -n "$wandb" ]; then
  [ -n "${WANDB_API_KEY:-}" ] || grep -qs api.wandb.ai ~/.netrc || { echo "--wandb needs WANDB_API_KEY in the env (or wandb login)" >&2; exit 2; }
  args+=(--wandb.enable=true --wandb.entity="${wandb%%/*}" --wandb.project="${wandb#*/}" --wandb.disable_artifact=true --job_name="$run")
fi
cat > "$cmd" <<EOF
# $(date -u +%FT%TZ) $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo no-git) mode=$mode batch=$batch accum=$accum effective=$effective workers=$workers
cd "$LEROBOT_DIR" && $ENV_PREFIX $PY ${args[*]}
EOF
echo "run $run ($mode): batch $batch x accum $accum = $effective, $steps microsteps ($((steps / accum)) updates), $workers loader workers"
echo "log $log"

write_provenance() {  # the exact data contract this adapter is tied to, before any training step
  mkdir -p "$prov/maps"
  cp "$src/prepare_manifest.json" "$prov/"
  cp "$src"/maps/*.json "$prov/maps/" 2>/dev/null || true
  python3 - "$prov/run.json" "$name" "$(sha "$src/prepare_manifest.json")" "$staged" "$mode" "$batch" "$accum" <<'PY'
import json, sys, time
p, name, msha, stage, mode, b, a = sys.argv[1:]
json.dump({"dataset": name, "manifest_sha256": msha, "stage": stage, "mode": mode, "batch": int(b), "accum": int(a),
           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(p, "w"), indent=1)
PY
}

[ "$mode" = probe ] || write_provenance
if [ "$mode" = run ]; then
  in_tmux "lora-$run" "$cmd" "$log"
  echo "Checkpoints: $out/checkpoints/<step>/pretrained_model{,_ema}.  Stop the training pod when it ends (bill)."
  exit 0
fi

( while sleep 2; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; done ) > "$log.mem" &
mem_pid=$!
t0=$(date +%s)
set +e; bash "$cmd" 2>&1 | tee "$log"; rc=${PIPESTATUS[0]}; set -e
kill $mem_pid 2>/dev/null || true
peak=$(sort -n "$log.mem" | tail -1)
echo; echo "=== $mode $run: exit $rc, wall $(( $(date +%s) - t0 )) s, peak GPU memory ${peak:-?} MiB of 81559"
[ "$rc" = 0 ] || { echo "$mode FAILED (trainer exit $rc): see $log" >&2; exit "$rc"; }

if [ "$mode" = probe ]; then
  echo "(performance only: nothing was saved; run smoke once before a full run)"
  grep -E "updt_s" "$log" | tail -3
  "$PY" - "$log" "$updates" "$accum" <<'PY'
import re, sys
log, updates, accum = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
def field(line, key):  # pinned trainer 146d847 prints "... data_s:X prep_s:Y updt_s:Z"; any order, any subset
    m = re.search(rf"\b{key}:([\d.]+)", line)
    return float(m.group(1)) if m else 0.0
lines = [l for l in open(log) if "updt_s:" in l]
if len(lines) < 2:
    sys.exit("no step timings parsed: read the log above")
rows = [(field(l, "updt_s"), field(l, "data_s"), field(l, "prep_s")) for l in lines[1:]]  # first line = warm-up
med = lambda i: sorted(r[i] for r in rows)[len(rows) // 2]
u, d, p = med(0), med(1), med(2)
hours = (u + d + p) * updates * accum / 3600
print(f"median per microstep: update {u:.2f} s + data {d:.3f} s + prep {p:.3f} s -> {updates} updates = {hours:.1f} h,"
      f" about ${hours * 3.99:.0f} at $3.99/h (plus startup)")
if d + p > 0.2 * u:
    print("loading/preprocessing is a bottleneck: --workers up, or a smaller dataset video (see README)")
PY
  exit 0
fi

# smoke: resume past the first save, then load raw and EMA adapters exactly as the server will
cfg="$out/checkpoints/last/pretrained_model/train_config.json"
[ -f "$cfg" ] || { echo "smoke FAILED: no checkpoint at $out/checkpoints/last" >&2; exit 1; }
echo "=== smoke: resume to $((6 * accum)) microsteps"
set +e; (cd "$LEROBOT_DIR" && env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false \
  "$PY" -m lerobot.scripts.lerobot_train --config_path="$cfg" --resume=true --steps=$((6 * accum))) 2>&1 | tee -a "$log"
rc=${PIPESTATUS[0]}; set -e
[ "$rc" = 0 ] || { echo "smoke FAILED: resume exit $rc" >&2; exit "$rc"; }
last=$(readlink -f "$out/checkpoints/last")
for variant in pretrained_model pretrained_model_ema; do
  [ -d "$last/$variant" ] || { echo "smoke FAILED: $last/$variant missing" >&2; exit 1; }
  echo "=== smoke: server selftest with --adapter $last/$variant"
  set +e; "$PY" "$ROOT/infra/gpu/so101/server.py" --engine lerobot --adapter "$last/$variant" --selftest --n 3 2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}; set -e
  [ "$rc" = 0 ] || { echo "smoke FAILED: server could not run $variant (exit $rc)" >&2; exit "$rc"; }
done
echo "=== smoke PASS: save, resume, raw + EMA adapters served. Full run: bash train/train.sh run $name --batch $batch"
