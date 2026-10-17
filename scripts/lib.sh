# Shared helpers for the coordination scripts. Source it, don't run it.
# Bash 3.2 compatible (macOS default shell), only needs git.

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "error: not inside the repo" >&2; exit 1; }
cd "$ROOT"
REMOTE="${HACK_REMOTE:-origin}"
BRANCH="${HACK_BRANCH:-main}"
LOCK_HOURS="${HACK_LOCK_HOURS:-3}"
# Path globs travel through unquoted variables; never let the shell expand them.
set -f

die()  { echo "error: $*" >&2; exit 1; }
warn() { echo "warning: $*" >&2; }

now_epoch() { date -u +%s; }
iso() { date -u -r "$1" +%Y-%m-%dT%H:%MZ 2>/dev/null || date -u -d "@$1" +%Y-%m-%dT%H:%MZ; }
now_iso() { iso "$(now_epoch)"; }

me() {
  local m; m="$(git config --get hack.member || true)"
  [ -n "$m" ] || die "no member name set. Run: scripts/setup.sh"
  echo "$m"
}
agent_name() { echo "${HACK_AGENT:-$(git config --get hack.agent || echo unknown)}"; }

has_remote() { git remote get-url "$REMOTE" >/dev/null 2>&1; }

# Fetch at most once every $1 seconds (default 0 = always). Quiet, never fatal.
fetch_throttled() {
  has_remote || return 0
  local every="${1:-0}" stamp="$ROOT/.git/FETCH_HEAD" age=999999
  if [ -f "$stamp" ]; then age=$(( $(now_epoch) - $(stat -f %m "$stamp" 2>/dev/null || stat -c %Y "$stamp") )); fi
  [ "$age" -ge "$every" ] && git fetch -q "$REMOTE" "$BRANCH" 2>/dev/null || true
}

# The freshest shared state we know of: origin/main if it exists, else HEAD.
state_ref() {
  if git rev-parse -q --verify "$REMOTE/$BRANCH" >/dev/null 2>&1; then echo "$REMOTE/$BRANCH"; else echo HEAD; fi
}

on_main_or_die() {
  local b; b="$(git rev-parse --abbrev-ref HEAD)"
  [ "$b" = "$BRANCH" ] || die "you are on '$b'. This repo is trunk-based: work on $BRANCH (see docs/decisions/01)."
}

pull_latest() {
  has_remote || return 0
  git pull -q --rebase --autostash "$REMOTE" "$BRANCH" 2>/dev/null || {
    git rebase --abort 2>/dev/null || true
    die "git pull --rebase failed (conflict). Resolve it by hand, then rerun."
  }
}

# ---- locks -----------------------------------------------------------------
# A lock is locks/<TASK>.md with "key: value" lines and a trailing "paths:" list.

lock_ids()    { git ls-tree --name-only "$1" locks/ 2>/dev/null | sed -n 's#^locks/\(.*\)\.md$#\1#p'; }
lock_field()  { git show "$1:locks/$2.md" 2>/dev/null | sed -n "s/^$3: *//p" | head -1; }
lock_paths()  { git show "$1:locks/$2.md" 2>/dev/null | sed -n '/^paths:/,$p' | sed -n 's/^ *- *//p'; }
lock_expired(){ local e; e="$(lock_field "$1" "$2" expires_epoch)"; [ -n "$e" ] && [ "$(now_epoch)" -gt "$e" ]; }

# Does file $1 match glob $2? (* and ** both cross directories: conservative on purpose)
path_matches() { case "$1" in $2) return 0 ;; esac; return 1; }

has_wild() { case "$1" in *[\*\?\[]*) return 0 ;; esac; return 1; }
glob_prefix() { echo "$1" | sed 's/[*?[].*$//'; }

# Can two globs ever touch the same file? Errs on the side of "yes".
globs_overlap() {
  local a="$1" b="$2"
  if ! has_wild "$a" && ! has_wild "$b"; then [ "$a" = "$b" ]; return; fi
  if ! has_wild "$a"; then path_matches "$a" "$b"; return; fi
  if ! has_wild "$b"; then path_matches "$b" "$a"; return; fi
  local pa pb; pa="$(glob_prefix "$a")"; pb="$(glob_prefix "$b")"
  case "$pa" in "$pb"*) return 0 ;; esac
  case "$pb" in "$pa"*) return 0 ;; esac
  return 1
}

# Paths anyone may touch without a lock (coordination files, own log, new records).
is_free_path() {
  local f="$1" who="$2"
  case "$f" in
    locks/*|tasks/*|docs/decisions/*|docs/status/*|docs/research/*) return 0 ;;
    "docs/log/$who.md") return 0 ;;
  esac
  return 1
}

# ---- tasks -----------------------------------------------------------------
task_file() { (set +f; ls tasks/"$1"-*.md tasks/"$1".md 2>/dev/null | head -1); }
frontmatter() { awk '/^---$/{n++; next} n==1' "$1" 2>/dev/null; }
task_field() { frontmatter "$1" | sed -n "s/^$2: *//p" | sed 's/ *#.*$//' | head -1; }
task_paths() { frontmatter "$1" | sed -n '/^paths:/,$p' | sed -n 's/^ *- *//p'; }
set_task_field() { # file field value
  local tmp; tmp="$(mktemp)"
  awk -v k="$2" -v v="$3" 'BEGIN{n=0} /^---$/{n++} n==1 && $0 ~ "^"k":" {print k": "v; next} {print}' "$1" > "$tmp" && mv "$tmp" "$1"
}
next_id() { # prefix dir ref  -> next free number, from both working tree and ref
  local max=0 n
  for n in $( (set +f; ls "$2" 2>/dev/null; git ls-tree --name-only "$3" "$2/" 2>/dev/null | sed 's#.*/##') | sed -n "s/^$1\([0-9][0-9]*\).*/\1/p"); do
    n=$((10#$n)); [ "$n" -gt "$max" ] && max=$n
  done
  printf "%02d" $((max + 1))
}

# Commit exactly the given paths and push. Returns 1 if the push was rejected.
publish() {
  local msg="$1"; shift
  git add -- "$@"
  git commit -q -m "$msg" -- "$@" >/dev/null
  has_remote || return 0
  git push -q "$REMOTE" "HEAD:$BRANCH" 2>/dev/null
}
undo_last_commit() { git reset -q --soft HEAD~1; git restore -q --staged -- "$@" 2>/dev/null || true; }

# Push current HEAD, rebasing on rejection. Never force.
push_retry() {
  has_remote || return 0
  local i=1
  while ! git push -q "$REMOTE" "HEAD:$BRANCH" 2>/dev/null; do
    [ $i -ge 4 ] && die "push kept failing. Run: git pull --rebase && git push"
    pull_latest; i=$((i + 1))
  done
}
