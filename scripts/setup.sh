#!/usr/bin/env bash
# One-time setup per clone. Sets your member name (= GitHub login), agent name and git hooks.
#   scripts/setup.sh [agent-name]      e.g. scripts/setup.sh claude-code | codex | cursor | gemini
set -eu
. "$(dirname "$0")/lib.sh"
login="$(gh api user --jq .login 2>/dev/null || true)"
[ -n "$login" ] || { printf 'GitHub login: '; read -r login; }
git config hack.member "$login"
git config hack.agent "${1:-$(git config --get hack.agent || echo claude-code)}"
git config core.hooksPath .githooks
git config pull.rebase true
git config rebase.autoStash true
chmod +x scripts/*.sh .githooks/* 2>/dev/null || true
echo "OK: member=$(git config hack.member) agent=$(git config hack.agent), hooks on."
echo "Next: scripts/sync.sh"
