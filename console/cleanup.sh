#!/usr/bin/env bash
# Explicit, targeted disk cleanup. Lists what it would delete with sizes and asks before deleting anything.
#   console/cleanup.sh                      list per-frame image folders of FINALIZED LeRobot datasets under ~/hackathon-data
#   console/cleanup.sh --delete             same list, then asks y/N per dataset
# Only a dataset's images/ folder is a candidate, and only when its meta/episodes/ exists and every camera has an mp4
# (that is, the frames were encoded into video). Unfinalized datasets are listed as KEEP: their PNGs may be the only copy.
set -euo pipefail
data="${HACK_DATA:-$HOME/hackathon-data}"; del=0; [ "${1:-}" = "--delete" ] && del=1
[ -d "$data" ] || { echo "nothing under $data"; exit 0; }
for img in $(find "$data" -type d -name images -prune 2>/dev/null); do
  ds="$(dirname "$img")"; mb=$(du -sm "$img" | cut -f1)
  if [ -d "$ds/meta/episodes" ] && [ -n "$(find "$ds/videos" -name '*.mp4' 2>/dev/null | head -1)" ]; then
    echo "CANDIDATE ${mb} MB  $img  (dataset finalized, videos present)"
    if [ $del = 1 ]; then printf '  delete it? [y/N] '; read -r ok; [ "$ok" = y ] && rm -rf "$img" && echo "  deleted"; fi
  else
    echo "KEEP      ${mb} MB  $img  (no finalized episodes/videos: these images may be the only copy)"
  fi
done
echo "Other space: du -sh $data/* ~/.cache/uv ~/.cache/huggingface ; 'uv cache prune' only drops package downloads."
