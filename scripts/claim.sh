#!/usr/bin/env bash
# Claim a task and the files it touches, atomically through git push.
#   scripts/claim.sh T03                 claim with the paths listed in the task file
#   scripts/claim.sh T03 'infra/x/**'    claim with explicit paths (overrides the task's list)
#   scripts/claim.sh T03                 run again later to renew your lock (+LOCK_HOURS)
#   scripts/claim.sh --steal T03         take over a lock whose expiry has passed
set -eu
. "$(dirname "$0")/lib.sh"

steal=0; [ "${1:-}" = "--steal" ] && { steal=1; shift; }
id="${1:-}"; [ -n "$id" ] || die "usage: scripts/claim.sh [--steal] <task-id> [path-glob ...]"; shift
who="$(me)"; agent="$(agent_name)"
on_main_or_die

attempt=1
while [ $attempt -le 4 ]; do
  pull_latest
  tf="$(task_file "$id")"; [ -n "$tf" ] || die "no task file for $id in tasks/. Create one: scripts/new-task.sh"
  if [ $# -gt 0 ]; then paths="$(printf '%s\n' "$@")"; else paths="$(task_paths "$tf")"; fi
  [ -n "$paths" ] || die "$id lists no paths. Add them to $tf or pass them as arguments."

  ref=HEAD; lf="locks/$id.md"; action=claim
  if [ -f "$lf" ]; then
    o="$(lock_field "$ref" "$id" owner)"
    oa="$(lock_field "$ref" "$id" agent)"
    if [ "$o" = "$who" ] && [ "$oa" = "$agent" ]; then action=renew
    elif [ "$o" = "$who" ]; then die "$id is held by your other agent '$oa'. Release it there, or set HACK_AGENT=$oa if this is that agent."
    elif [ $steal = 1 ] && lock_expired "$ref" "$id"; then action="steal from $o"
    elif lock_expired "$ref" "$id"; then die "$id is held by $o but expired. Ask them, or: scripts/claim.sh --steal $id"
    else die "$id is already claimed by $o (until $(lock_field "$ref" "$id" expires))."
    fi
  fi

  # No overlap with anyone else's live or expired lock.
  for other in $(lock_ids "$ref"); do
    [ "$other" = "$id" ] && continue
    o="$(lock_field "$ref" "$other" owner)"; oa="$(lock_field "$ref" "$other" agent)"
    # Only your own agent's locks may overlap: two agents of one member are still two holders.
    [ "$o" = "$who" ] && [ "$oa" = "$agent" ] && continue
    o="$o/$oa"
    for g in $(lock_paths "$ref" "$other"); do
      for p in $paths; do
        globs_overlap "$p" "$g" && die "'$p' overlaps '$g' held by $o under $other. Narrow your paths, wait, or talk to $o."
      done
    done
  done

  exp=$(( $(now_epoch) + LOCK_HOURS * 3600 ))
  {
    echo "task: $id"
    echo "owner: $who"
    echo "agent: $agent"
    echo "claimed: $(now_iso)"
    echo "expires: $(iso "$exp")"
    echo "expires_epoch: $exp"
    echo "paths:"
    for p in $paths; do echo "  - $p"; done
  } > "$lf"
  set_task_field "$tf" status doing
  set_task_field "$tf" owner "$who"

  if publish "lock: $action $id ($who)" "$lf" "$tf"; then
    echo "OK: $action $id until $(iso "$exp")."
    echo "Your paths:"; for p in $paths; do echo "  $p"; done
    echo "Upstream for this task is listed in $tf. Read it before editing."
    exit 0
  fi
  # Someone pushed first. Undo our commit and re-check against their state.
  undo_last_commit "$lf" "$tf"
  git checkout -q HEAD -- "$tf" 2>/dev/null || true
  git checkout -q HEAD -- "$lf" 2>/dev/null || rm -f "$lf"
  echo "push rejected (someone pushed first), retrying ($attempt)..." >&2
  attempt=$((attempt + 1)); sleep 1
done
die "could not claim $id after retries."
