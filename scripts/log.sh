#!/usr/bin/env bash
# Append one line to your own log, docs/log/<you>.md. Commit it with your work.
#   scripts/log.sh [--task T03] "what happened, why, result"
set -eu
. "$(dirname "$0")/lib.sh"
task="-"; [ "${1:-}" = "--task" ] && { task="$2"; shift 2; }
msg="$*"; [ -n "$msg" ] || die "usage: scripts/log.sh [--task T03] \"message\""
who="$(me)"; f="docs/log/$who.md"
[ -f "$f" ] || printf '# Log: %s\n\nAppend-only, newest at the bottom. One line per event.\n\n' "$who" > "$f"
line="- $(now_iso) $who [$task] ($(agent_name)) $msg"
echo "$line" >> "$f"
echo "$line"
