#!/usr/bin/env bash
# Release your lock when the work is committed.
#   scripts/release.sh T03 done   "what shipped"      task -> done
#   scripts/release.sh T03 todo   "where I stopped"   task back to the pool (handoff note required)
#   scripts/release.sh T03 blocked "why"              task -> blocked
set -eu
. "$(dirname "$0")/lib.sh"

id="${1:-}"; state="${2:-done}"; note="${3:-}"
[ -n "$id" ] || die "usage: scripts/release.sh <task-id> done|todo|blocked \"note\""
case "$state" in done|todo|blocked) ;; *) die "state must be done, todo or blocked" ;; esac
[ -n "$note" ] || die "a one-line note is required: it is the handoff for whoever reads this next."
who="$(me)"
on_main_or_die
pull_latest

lf="locks/$id.md"; tf="$(task_file "$id")"
[ -f "$lf" ] || die "$id is not locked."
o="$(lock_field HEAD "$id" owner)"; [ "$o" = "$who" ] || die "$id is held by $o, not you."

dirty="$(git status --porcelain -- $(lock_paths HEAD "$id") 2>/dev/null || true)"
[ -z "$dirty" ] || die "uncommitted changes inside your locked paths. Commit them first:
$dirty"

git rm -q "$lf"
set_task_field "$tf" status "$state"
[ "$state" = done ] || set_task_field "$tf" owner ""
printf -- '- %s · %s · %s: %s\n' "$(now_iso)" "$who" "$state" "$note" >> "$tf"
"$(dirname "$0")/log.sh" --task "$id" "release ($state): $note" >/dev/null

log="docs/log/$who.md"
git add -- "$tf" "$log"
git commit -q -m "lock: release $id ($who, $state)" -- "$lf" "$tf" "$log"
push_retry
echo "OK: released $id ($state)."
