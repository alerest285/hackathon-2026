#!/usr/bin/env bash
# Explicit, targeted disk cleanup of LeRobot per-frame images. Lists everything with sizes; deletes only on request.
#   console/cleanup.sh             list
#   console/cleanup.sh --delete    list, then ask y/N for each CANDIDATE
# Decided PER EPISODE: images/<camera>/episode-NNNNNN is a CANDIDATE only if NNNNNN < total_episodes in meta/info.json
# (that episode was saved and encoded into the dataset's videos). Anything else is KEEP: an unsaved episode's PNGs can be
# its only camera data. Only a folder matching <data root>/**/images/<camera>/episode-NNNNNN can ever be deleted.
set -euo pipefail
data="${HACK_DATA:-$HOME/hackathon-data}"; del=0; [ "${1:-}" = "--delete" ] && del=1
[ -d "$data" ] || { echo "nothing under $data"; exit 0; }
data="$(cd "$data" && pwd)"
find "$data" -type d -path '*/images/*/episode-*' -prune 2>/dev/null | sort | while read -r ep; do
  [ -n "$ep" ] || continue
  case "$ep" in "$data"/*/images/*/episode-[0-9]*) ;; *) echo "SKIP      $ep (unexpected path)"; continue ;; esac
  ds="${ep%/images/*}"; idx=$((10#$(basename "$ep" | sed 's/episode-//'))); mb=$(du -sm "$ep" | cut -f1)
  total=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('total_episodes',0))" "$ds/meta/info.json" 2>/dev/null || echo 0)
  if [ "$idx" -lt "$total" ] && [ -n "$(find "$ds/videos" -name '*.mp4' 2>/dev/null | head -1)" ]; then
    echo "CANDIDATE ${mb} MB  $ep  (episode $idx saved; dataset has $total)"
    if [ $del = 1 ]; then
      printf '  delete it? [y/N] '; read -r ok </dev/tty
      [ "$ok" = y ] && rm -rf -- "${ep:?}" && echo "  deleted"
    fi
  else
    echo "KEEP      ${mb} MB  $ep  (episode $idx NOT saved: dataset has $total; these PNGs may be its only copy)"
  fi
done
echo "Other space: du -sh $data/* ~/.cache/uv ~/.cache/huggingface ; 'uv cache prune' only drops package downloads."
