#!/usr/bin/env bash
# Create the next task file and publish it right away, so ids never collide.
#   scripts/new-task.sh "short title" 'path/glob/**' ['another/path' ...]
set -eu
. "$(dirname "$0")/lib.sh"
title="${1:-}"; [ -n "$title" ] && [ $# -ge 2 ] || die "usage: scripts/new-task.sh \"title\" path-glob [...]"; shift
on_main_or_die
for i in 1 2 3 4; do
  pull_latest
  n="T$(next_id T tasks HEAD)"
  slug="$(echo "$title" | tr 'A-Z' 'a-z' | sed 's/[^a-z0-9]\{1,\}/-/g; s/^-//; s/-$//' | cut -c1-40)"
  f="tasks/$n-$slug.md"
  {
    echo "---"; echo "id: $n"; echo "title: $title"; echo "status: todo"; echo "owner: "; echo "depends: "
    echo "paths:"; for p in "$@"; do echo "  - $p"; done; echo "---"; echo
    echo "## Goal"; echo; echo "TODO: one or two sentences."; echo
    echo "## Done when"; echo; echo "- [ ] a check someone else can run"; echo
    echo "## Upstream"; echo; echo "- docs/architecture.md"; echo
    echo "## Notes and handoffs (append, newest last)"; echo
  } > "$f"
  publish "task: add $n $title" "$f" && { echo "OK: $f"; exit 0; }
  undo_last_commit "$f"; rm -f "$f"
done
die "could not publish the task."
