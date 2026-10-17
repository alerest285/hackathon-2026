#!/usr/bin/env bash
# Claude Code PreToolUse hook: refuse Edit/Write on files you have not claimed.
# Reads the hook JSON on stdin. Exit 2 blocks the tool call and shows stderr to the agent.
set -u
. "$(dirname "$0")/lib.sh"
input="$(cat)"
if command -v jq >/dev/null 2>&1; then
  fp="$(printf '%s' "$input" | jq -r '.tool_input.file_path // .tool_input.notebook_path // empty')"
else
  fp="$(printf '%s' "$input" | sed -nE 's/.*"(file_path|notebook_path)" *: *"([^"]*)".*/\2/p' | head -1)"
fi
[ -n "$fp" ] || exit 0
case "$fp" in "$ROOT"/*) rel="${fp#$ROOT/}" ;; /*) exit 0 ;; *) rel="$fp" ;; esac
who="$(git config --get hack.member || true)"
[ -n "$who" ] || { echo "Run scripts/setup.sh first (no member name set)." >&2; exit 2; }
fetch_throttled 60
out="$("$(dirname "$0")/check-locks.sh" --owner "$who" "$rel" 2>&1)" && exit 0
echo "$out" >&2
echo "Claim it first (scripts/claim.sh <task>), or pick a different file. See AGENTS.md section 3." >&2
exit 2
