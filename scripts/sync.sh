#!/usr/bin/env bash
# Pull the latest main, then print the board. Run at the start of every session.
set -eu
. "$(dirname "$0")/lib.sh"
b="$(git rev-parse --abbrev-ref HEAD)"
if [ "$b" = "$BRANCH" ]; then pull_latest; else warn "on '$b', not $BRANCH: fetched only"; fetch_throttled 0; fi
exec "$(dirname "$0")/status.sh"
