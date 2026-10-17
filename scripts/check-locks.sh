#!/usr/bin/env bash
# Check that every changed file is either free or covered by a lock the owner holds.
#   check-locks.sh [--owner NAME] [--ref REF] [file ...]   (files also read from stdin)
# Used by: .githooks/pre-commit, scripts/guard-edit.sh, CI.
set -eu
. "$(dirname "$0")/lib.sh"

owner="" ref=""
while [ $# -gt 0 ]; do
  case "$1" in
    --owner) owner="$2"; shift 2 ;;
    --ref) ref="$2"; shift 2 ;;
    *) break ;;
  esac
done
[ -n "$owner" ] || owner="$(me)"
[ -n "$ref" ] || ref="$(state_ref)"
if [ $# -gt 0 ]; then files="$(printf '%s\n' "$@")"; else files="$(cat)"; fi
files="$(printf '%s\n' "$files" | sed '/^$/d' | sort -u)"
[ -n "$files" ] || exit 0

ids="$(lock_ids "$ref")"
bad=0
for f in $files; do
  is_free_path "$f" "$owner" && continue
  mine="" theirs=""
  for id in $ids; do
    for g in $(lock_paths "$ref" "$id"); do
      if path_matches "$f" "$g"; then
        o="$(lock_field "$ref" "$id" owner)"
        if [ "$o" = "$owner" ]; then mine="$id"; else theirs="$id ($o)"; fi
      fi
    done
  done
  if [ -n "$theirs" ]; then echo "  $f  -> claimed by $theirs"; bad=1
  elif [ -z "$mine" ]; then echo "  $f  -> unclaimed: run scripts/claim.sh <task>"; bad=1
  fi
done
if [ "$bad" = 1 ]; then
  echo "Lock check failed for '$owner' against $ref (see AGENTS.md section 3)." >&2
  exit 1
fi
