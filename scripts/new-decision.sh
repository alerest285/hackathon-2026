#!/usr/bin/env bash
# Create the next numbered decision record and publish it right away (numbers never collide).
#   scripts/new-decision.sh "use fp8r gd package for serving"
# Then fill Context / Decision / Consequences and commit. If it changes an upstream doc,
# update that doc in the same commit (you need a lock on it).
set -eu
. "$(dirname "$0")/lib.sh"
title="${1:-}"; [ -n "$title" ] || die "usage: scripts/new-decision.sh \"title\""
on_main_or_die
for i in 1 2 3 4; do
  pull_latest
  n="$(next_id '' docs/decisions HEAD)"
  slug="$(echo "$title" | tr 'A-Z' 'a-z' | sed 's/[^a-z0-9]\{1,\}/-/g; s/^-//; s/-$//' | cut -c1-50)"
  f="docs/decisions/$n-$slug.md"
  printf '# %s. %s\n\n**Date:** %s · **Status:** pending · **Author:** %s\n\n## Context\n\n## Decision\n\n## Consequences\n' \
    "$n" "$title" "$(date -u +%Y-%m-%d)" "$(me)" > "$f"
  publish "decision: $n $title (draft)" "$f" && { echo "OK: $f"; exit 0; }
  undo_last_commit "$f"; rm -f "$f"
done
die "could not publish the decision."
