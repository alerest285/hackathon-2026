#!/usr/bin/env bash
# LoRA on flux-3-action-so101 with LeRobot (BFL docs/so101-lora.md recipe, train/lora.json). Run ON the GPU host
# after train/setup.sh, with the dataset from train/prepare.py rsynced to /workspace/datasets/<name>.
#
#   bash train/train.sh probe <name> [--batch 2|4|8]      40 microsteps: s/step, data wait, peak VRAM, full-run ETA
#   bash train/train.sh run   <name> [--batch B] [--updates 2500] [--effective 8] [--tag v1]   full run in tmux
#
# The optimizer always sees an effective batch of --effective (default 8, BFL's single-GPU preset): accumulation is
# effective / batch, and steps, save and eval scale with it, so --batch only changes speed, not the recipe.
# Checkpoints every 125 optimizer updates, raw and EMA. Picks: docs in train/README.md.
# Refuses to start while another process uses the GPU (live serving needs its latency) unless --share.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mode="${1:-}"; name="${2:-}"; shift 2 || true
batch=2 updates=2500 effective=8 tag="" share=0
while [ $# -gt 0 ]; do case "$1" in
  --batch) batch="$2"; shift 2 ;; --updates) updates="$2"; shift 2 ;; --effective) effective="$2"; shift 2 ;;
  --tag) tag="$2"; shift 2 ;; --share) share=1; shift ;;
  *) echo "unknown option $1" >&2; exit 2 ;; esac; done
case "$mode" in probe|run) ;; *) sed -n 2,12p "$0"; exit 2 ;; esac
[ -n "$name" ] || { sed -n 2,12p "$0"; exit 2; }
[ $((effective % batch)) -eq 0 ] || { echo "--effective $effective is not a multiple of --batch $batch" >&2; exit 2; }
accum=$((effective / batch))

export LEROBOT_DIR="${LEROBOT_DIR:-/workspace/lerobot-flux3}" HF_HOME="${HF_HOME:-/workspace/hf}"
[ -f /workspace/train-env.sh ] || { echo "run train/setup.sh first" >&2; exit 1; }
. /workspace/train-env.sh
PY="$LEROBOT_DIR/.venv/bin/python"
src="/workspace/datasets/$name"
[ -f "$src/meta/info.json" ] || { echo "no dataset at $src (rsync train/prepare.py's --out there)" >&2; exit 1; }
[ -f "$src/prepare_manifest.json" ] || { echo "$src has no prepare_manifest.json: only train on train/prepare.py output (model space)" >&2; exit 1; }

others=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -v "^$" || true)
if [ -n "$others" ] && [ "$share" = 0 ]; then
  echo "GPU busy (pid, MiB):"; echo "$others"
  echo "Training next to a live server breaks its latency budget. Stop it, use a training pod, or pass --share." >&2
  exit 1
fi

# Stage on RAM: every sample decodes 50 frames per camera, and /workspace is a network volume.
need_mb=$(( $(du -sm "$src" | cut -f1) + 512 ))
free_mb=$(df -m /dev/shm | awk 'NR==2{print $4}')
stage=/dev/shm/datasets; [ "$free_mb" -gt "$need_mb" ] || stage=/root/datasets
mkdir -p "$stage"; rsync -a --delete "$src/" "$stage/$name/"
echo "dataset staged at $stage/$name ($(du -sh "$stage/$name" | cut -f1)); $(python3 -c "import json;m=json.load(open('$src/prepare_manifest.json'));print(m['episodes'],'episodes,',m['minutes'],'min')")"

cpus=$(python3 -c 'import os;print(len(os.sched_getaffinity(0)))')
quota=$(awk '{ if ($1=="max") print 0; else print int($1/$2) }' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 0)
[ "$quota" -gt 0 ] && [ "$quota" -lt "$cpus" ] && cpus=$quota
workers=$(( cpus > 18 ? 16 : (cpus > 4 ? cpus - 2 : 2) ))

stamp=$(date -u +%m%d-%H%M)
run="${name}${tag:+-$tag}-b${batch}x${accum}-${stamp}"
out="/workspace/outputs/lora/$run"; [ "$mode" = probe ] && out="/workspace/outputs/lora-probe/$run"
mkdir -p "$(dirname "$out")" /workspace/train-logs
log="/workspace/train-logs/$run.log"

if [ "$mode" = probe ]; then
  steps=$((40 / accum * accum)); save=$((steps * 10)); evals=$save; logf=$accum
else
  steps=$((updates * accum)); save=$((125 * accum)); evals=$save; logf=$((5 * accum))
fi
args=(-m lerobot.scripts.lerobot_train --config_path="$ROOT/train/lora.json"
  --dataset.repo_id="alerest285/$name" --dataset.root="$stage/$name"
  --policy.path="$POLICY_DIR" --policy.device=cuda
  --policy.video_vae_id="$ENCODERS_DIR/video_vae.safetensors" --policy.text_encoder_id="$ENCODERS_DIR/text_encoder"
  --batch_size="$batch" --accelerator.gradient_accumulation.steps="$accum"
  --steps="$steps" --save_freq="$save" --eval_steps="$evals" --log_freq="$logf"
  --num_workers="$workers" --output_dir="$out")
[ "$mode" = probe ] && args+=(--save_checkpoint=false)

cat > "/workspace/train-logs/$run.cmd" <<EOF
# $(date -u +%FT%TZ) $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo no-git) mode=$mode batch=$batch accum=$accum effective=$effective workers=$workers
cd "$LEROBOT_DIR" && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false $PY ${args[*]}
EOF
echo "run $run: batch $batch x accum $accum = $effective, $steps microsteps ($((steps / accum)) updates), $workers loader workers"
echo "log $log"

if [ "$mode" = run ]; then
  tmux new-session -d -s "lora-$run" "bash /workspace/train-logs/$run.cmd 2>&1 | tee $log; echo EXIT \$? >> $log"
  echo "tmux attach -t lora-$run   (detach: Ctrl-b d).  Follow: tail -f $log | grep -E 'step|loss|Error'"
  echo "Checkpoints: $out/checkpoints/<step>/pretrained_model{,_ema}.  Stop the pod when it ends (bill)."
  exit 0
fi

# probe: run in the foreground, sample GPU memory, then summarize
( while sleep 2; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; done ) > "$log.mem" &
mem_pid=$!
t0=$(date +%s)
set +e; bash "/workspace/train-logs/$run.cmd" 2>&1 | tee "$log"; rc=${PIPESTATUS[0]}; set -e
kill $mem_pid 2>/dev/null || true
peak=$(sort -n "$log.mem" | tail -1)
echo; echo "=== probe $run: exit $rc, wall $(( $(date +%s) - t0 )) s, peak GPU memory ${peak:-?} MiB of 81559"
grep -E "updt_s|data_s" "$log" | tail -3
"$PY" - "$log" "$updates" "$accum" <<'PY'
import re, sys
log, updates, accum = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = [(float(u), float(d)) for u, d in re.findall(r"updt_s:([\d.]+).*?data_s:([\d.]+)", open(log).read())]
if len(rows) < 2:
    sys.exit("no step timings parsed: read the log above")
u = sorted(r[0] for r in rows[1:])[len(rows[1:]) // 2]   # median, first log line includes warm-up
d = sorted(r[1] for r in rows[1:])[len(rows[1:]) // 2]
per_step = u + d
hours = per_step * updates * accum / 3600
print(f"median update {u:.2f} s + data wait {d:.2f} s per microstep -> {updates} updates = {hours:.1f} h, about ${hours * 3.99:.0f} at $3.99/h")
if d > 0.2 * u:
    print("data loading is a bottleneck: more workers, or a smaller dataset video (see README)")
PY
