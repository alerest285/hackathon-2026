#!/usr/bin/env bash
# Post a one-line message to the team's agent chat; it is pushed immediately.
#   scripts/say.sh "message to everyone"
#   scripts/say.sh @saadl27 "message for Saad's agents"        (@login or @login/agent)
# Each agent appends only to its own file docs/chat/<login>--<agent>.md, so writes never conflict.
# Read with scripts/listen.sh (Claude Code: run it under the Monitor tool to get every message live).
set -eu
. "$(dirname "$0")/lib.sh"
to="@all"; case "${1:-}" in @*) to="$1"; shift ;; esac
msg="$(printf '%s' "$*" | tr '\n' ' ')"; [ -n "$msg" ] || die 'usage: scripts/say.sh [@login] "message"'
who="$(me)"; ag="$(agent_name | tr -d '\n' | tr -c 'A-Za-z0-9._-' '-')"
f="docs/chat/$who--$ag.md"
on_main_or_die
pull_latest
mkdir -p docs/chat
[ -f "$f" ] || printf '# Chat: %s / %s\n\nAppend-only. One line per message. Written by scripts/say.sh.\n\n' "$who" "$ag" > "$f"
line="- $(now_iso) $who/$ag -> $to: $msg"
echo "$line" >> "$f"
git add "$f"
git commit -q -m "chat: $who/$ag -> $to: $(printf '%s' "$msg" | cut -c1-80)" -- "$f"
push_retry
echo "sent: $line"
