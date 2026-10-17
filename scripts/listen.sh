#!/usr/bin/env bash
# Stream the team's agent chat and board changes. Fetch-only: never touches your working tree.
#   scripts/listen.sh            run forever (poll every HACK_LISTEN_SECONDS, default 20)
#   scripts/listen.sh --once     print what is new since your last look, then exit
# Each new event is ONE line on stdout:
#   [chat] <from> -> <to>: <text>        messages from other agents (yours are skipped); "FOR YOU" when addressed to you
#   [board] <author>: <commit subject>   claims, releases, new tasks, decisions, upstream doc changes
# Claude Code: start it with the Monitor tool (timeout max, re-arm on expiry) so every line arrives as a notification.
set -u
. "$(dirname "$0")/lib.sh"
set +f
once=0; [ "${1:-}" = "--once" ] && once=1
every="${HACK_LISTEN_SECONDS:-20}"
who="$(me)"; ag="$(agent_name | tr -d '\n' | tr -c 'A-Za-z0-9._-' '-')"
key="hack.listen-$ag"
has_remote || die "no remote"
git fetch -q "$REMOTE" "$BRANCH" 2>/dev/null || true
seen="$(git config --get "$key" || git rev-parse "$REMOTE/$BRANCH")"
git cat-file -e "$seen" 2>/dev/null || seen="$(git rev-parse "$REMOTE/$BRANCH")"
git config "$key" "$seen"
while true; do
  git fetch -q "$REMOTE" "$BRANCH" 2>/dev/null || true
  head="$(git rev-parse "$REMOTE/$BRANCH")"
  if [ "$head" != "$seen" ]; then
    git diff "$seen" "$head" -- docs/chat/ 2>/dev/null | sed -n 's/^+- //p' | while IFS= read -r l; do
      from="$(printf '%s' "$l" | awk '{print $2}')"
      [ "$from" = "$who/$ag" ] && continue
      to="$(printf '%s' "$l" | sed -n 's/^[^ ]* [^ ]* -> \(@[^:]*\):.*/\1/p')"
      tag=""; case "$to" in "@$who"|"@$who/$ag") tag=" FOR YOU" ;; esac
      echo "[chat$tag] $(printf '%s' "$l" | cut -d' ' -f2-)"
    done
    git log --reverse --format='%an|%s' "$seen..$head" 2>/dev/null | grep -vE '\|chat: ' | \
      grep -E '\|(lock: |task: |decision: )|\|.*(AGENTS|architecture|project)' | sed 's/^\([^|]*\)|/[board] \1: /' || true
    up="$(git diff --name-only "$seen" "$head" -- AGENTS.md docs/project.md docs/architecture.md docs/decisions 2>/dev/null)"
    [ -z "$up" ] || echo "[board] UPSTREAM DOCS CHANGED, re-read: $(echo $up)"
    seen="$head"; git config "$key" "$seen"
  fi
  [ $once = 1 ] && exit 0
  sleep "$every"
done
