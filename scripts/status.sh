#!/usr/bin/env bash
# Print the shared state: what changed upstream, who holds what, open tasks, latest logs.
#   scripts/status.sh            full board
#   scripts/status.sh --delta    only what changed on origin/main since you last looked (silent if nothing)
set -eu
. "$(dirname "$0")/lib.sh"
set +f   # this script globs files; it never iterates lock paths unquoted
mode="${1:-full}"
who="$(git config --get hack.member || echo '?')"
UPSTREAM="AGENTS.md docs/project.md docs/architecture.md docs/decisions"

if [ "$mode" = "--delta" ]; then
  fetch_throttled 60
  ref="$(state_ref)"; seen="$(git config --get hack.seen || true)"; head="$(git rev-parse "$ref")"
  [ -n "$seen" ] && [ "$seen" != "$head" ] || { git config hack.seen "$head"; exit 0; }
  git cat-file -e "$seen" 2>/dev/null || { git config hack.seen "$head"; exit 0; }
  echo "[team update] new on $BRANCH since your last look:"
  git log --format='  %h %an: %s' "$seen..$head" | head -15
  up="$(git diff --name-only "$seen" "$head" -- $UPSTREAM)"
  [ -z "$up" ] || { echo "  UPSTREAM DOCS CHANGED. Re-read before continuing:"; echo "$up" | sed 's/^/    /'; }
  [ "$(git rev-parse HEAD)" = "$head" ] || echo "  Your checkout is behind: run scripts/sync.sh"
  git config hack.seen "$head"
  exit 0
fi

ref="$(state_ref)"
echo "== hackathon-2026 · $(now_iso) · you: $who · state: $ref @ $(git rev-parse --short "$ref") =="
seen="$(git config --get hack.seen || true)"
if [ -n "$seen" ] && git cat-file -e "$seen" 2>/dev/null; then
  up="$(git diff --name-only "$seen" "$ref" -- $UPSTREAM)"
  [ -z "$up" ] || { echo; echo "UPSTREAM CHANGED SINCE YOUR LAST SESSION (read these first):"; echo "$up" | sed 's/^/  /'; }
fi
git config hack.seen "$(git rev-parse "$ref")"

echo; echo "READ ORDER: AGENTS.md -> docs/project.md -> docs/architecture.md -> your task file -> code"

echo; echo "ACTIVE LOCKS"
ids="$(lock_ids "$ref")"
[ -n "$ids" ] || echo "  (none)"
for id in $ids; do
  e="$(lock_field "$ref" "$id" expires_epoch)"; left=$(( (${e:-0} - $(now_epoch)) / 60 ))
  if [ "$left" -lt 0 ]; then t="EXPIRED ${left#-}m ago"; else t="$((left / 60))h$((left % 60))m left"; fi
  printf '  %-5s %-14s %-12s %-18s %s\n' "$id" "$(lock_field "$ref" "$id" owner)" "$(lock_field "$ref" "$id" agent)" "$t" "$(lock_paths "$ref" "$id" | tr '\n' ' ')"
done

echo; echo "TASKS"
for f in tasks/T*.md; do
  [ -f "$f" ] || continue
  printf '  %-5s %-8s %-12s %s   deps: %s\n' "$(task_field "$f" id)" "$(task_field "$f" status)" "$(task_field "$f" owner)" "$(task_field "$f" title)" "$(task_field "$f" depends)"
done

echo; echo "LATEST LOG (all members, newest last)"
cat docs/log/*.md 2>/dev/null | grep '^- [0-9]' | sort | tail -15 | sed 's/^- /  /' || true

echo; echo "DECISIONS (newest 5)"
ls docs/decisions/[0-9]*.md 2>/dev/null | sort | tail -5 | while read -r d; do echo "  $d: $(sed -n 's/^# //p' "$d" | head -1)"; done

echo; echo "LAST COMMITS"
git log --format='  %h %ar %an: %s' -8 "$ref"
