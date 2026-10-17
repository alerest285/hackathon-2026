#!/usr/bin/env bash
# One command from nothing to a measured FLUX inference on the H100, then the box stops again.
#   infra/aws/first-boot.sh [--droid] [--keep]
#     --droid  also run the DROID smoke test (extra ~15 GB download, ~10 min)
#     --keep   leave the box running at the end (default: stop it, the disk keeps everything)
# Needs: on-demand P quota >= 16, your HF token at ~/.cache/huggingface/token (run: uvx hf auth login,
# after accepting the licenses on the black-forest-labs/flux-3-action-* pages).
# Output is also saved to outputs/first-boot-<time>.log (gitignored): paste the numbers into your log.
set -eu; . "$(dirname "$0")/env.sh"; need_aws
here="$(cd "$(dirname "$0")" && pwd)"; droid=0; keep=0
for a in "$@"; do case "$a" in --droid) droid=1 ;; --keep) keep=1 ;; *) die "unknown option $a" ;; esac; done
tok="${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}"
[ -s "$tok" ] || die "no Hugging Face token at $tok. Accept the FLUX licenses, then: uvx hf auth login"
mkdir -p "$here/../../outputs"; log="$here/../../outputs/first-boot-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee "$log") 2>&1
t0=$(date +%s); stamp() { echo "[$(( $(date +%s) - t0 ))s] $*"; }

stamp "up"; up="$("$here/up.sh")"; echo "$up"
host="$(echo "$up" | sed -n 's/.* host: \([^ ]*\).*/\1/p')"; [ -n "$host" ] || die "could not read the host from up.sh"
S="ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=10 ubuntu@$host"
stop_box() { [ $keep = 1 ] && { stamp "kept running: infra/aws/down.sh when done"; return; }; stamp "stopping"; "$here/down.sh"; }
trap 'stamp "FAILED, see $log"; stop_box' ERR

stamp "waiting for ssh"; for i in $(seq 60); do $S true 2>/dev/null && break; sleep 10; done; $S true
stamp "waiting for first-boot setup (cloud-init)"; $S 'cloud-init status --wait >/dev/null; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader'
stamp "copying the HF token (over ssh, never into the repo)"
$S 'mkdir -p ~/.cache/huggingface && umask 077 && cat > ~/.cache/huggingface/token' < "$tok"
stamp "clone + bootstrap"
$S 'set -e; [ -d hackathon-2026 ] || git clone -q https://github.com/alerest285/hackathon-2026; git -C hackathon-2026 pull -q; bash hackathon-2026/infra/gpu/bootstrap.sh'
stamp "SO-101 smoke"; $S 'bash hackathon-2026/infra/gpu/smoke-so101.sh 5'
[ $droid = 1 ] && { stamp "DROID smoke"; $S 'bash hackathon-2026/infra/gpu/smoke-droid.sh'; }
trap - ERR; stamp "done: $(( ($(date +%s) - t0) / 60 )) min wall, about \$$(awk -v s=$(( $(date +%s) - t0 )) -v p=$(ondemand_price) 'BEGIN{printf "%.2f", s/3600*p}') of GPU"
stop_box
