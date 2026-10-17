# AGENTS.md: the contract for every agent in this repo

Several people run coding agents (Claude Code, Codex, Cursor, Gemini CLI and others) on this
repo at the same time, each on their own machine. This file is how they avoid each other's
files and stay on the same picture. Every agent reads it at the start of every session.
Claude Code loads it through `CLAUDE.md`, Gemini CLI through `.gemini/settings.json`, Cursor
through `.cursor/rules/agents.mdc`, and Codex reads it natively.

The project is zero-shot inference with FLUX 3 Action. See `docs/project.md`.

## 1. Session start (always, before anything else)

```sh
scripts/sync.sh
```

This pulls `main` and prints the board: what changed upstream since your last session, who
holds which files, the open tasks, the latest log lines from every member, the newest
decisions and the recent commits. If it prints **UPSTREAM CHANGED**, read those files before
you do anything else. First time in a clone, run `scripts/setup.sh <agent-name>` before this.

## 2. Docs flow downstream: read from the source

Each layer may only rely on the layers above it. Read them top-down, and when the work shows
that an upper layer is wrong, fix that layer first. Never let code quietly contradict it.

| # | Layer | Answers | Changed by |
|---|---|---|---|
| 1 | `AGENTS.md` | how we work | lock + decision record |
| 2 | `docs/project.md` | what we are building, the milestone, scope | lock + decision record |
| 3 | `docs/architecture.md` | components, interfaces, units, which directory owns what | lock + decision record |
| 4 | `docs/decisions/NN-*.md` | why a choice was made | anyone, via `scripts/new-decision.sh` |
| 5 | `tasks/TNN-*.md` | one unit of work: goal, done-when, paths, handoff notes | anyone (new tasks), owner (edits) |
| 6 | code under `infra/`, `src/`, ... | the work | owner of the lock |

Rules that follow from the table:

- Before you start a task, read its **Upstream** section and every file listed there.
- If you need an interface that `docs/architecture.md` does not describe, add it there first,
  in the same commit as the code. You need a lock on `docs/architecture.md` for that.
- A record in `docs/decisions/` outranks older text anywhere else. If they disagree, fix the older text.
- Facts about FLUX 3 Action come from `docs/research/flux-3-action.md`, which cites BFL's docs.
  Do not trust model memory about it: the model is newer than most training data.

## 3. Claim before you edit: the lock protocol

A lock is a file `locks/<TASK>.md` that names the owner and the path globs. Pushing it to
`main` is the claim. Git allows only one push to land first, so two agents cannot hold the
same task. A rejected push means someone else got there first; the script re-checks and retries.

```sh
scripts/claim.sh T03                 # claims the paths listed in tasks/T03-*.md (default 3 h)
scripts/claim.sh T03                 # run again to renew before it expires
scripts/release.sh T03 done "what shipped"       # or: todo "where I stopped" / blocked "why"
scripts/claim.sh --steal T03         # only if the lock has expired; say so in your log
```

- **Never edit a file outside your locks.** These paths are free for everyone: `locks/`,
  `tasks/`, `docs/decisions/`, `docs/status/`, `docs/research/`, and your own `docs/log/<you>.md`.
- Claims refuse to overlap. If `claim.sh` reports an overlap, narrow your globs, wait, or
  leave a note in that person's task file. Do not take a broader glob than you need.
- Shared hot files (`pyproject.toml`, lockfiles, root configs) are claimed like any other
  path. Hold them for one commit, then release.
- Need a file mid-task that is not in your lock? Claim it under a new small task
  (`scripts/new-task.sh "bump torch" pyproject.toml`), not by widening someone else's.
- Enforcement: `.githooks/pre-commit` blocks commits on unclaimed files for every agent.
  For Claude Code a hook also blocks Edit/Write on them. CI re-checks every pushed commit.
  `HACK_SKIP_LOCKS=1` / `[skip-locks]` exist for humans fixing the system, never for agents.

## 4. Git rhythm: trunk-based, small, often

- Everyone works on `main`. Locks keep the file sets disjoint, so `git pull --rebase` stays
  clean. Branches are allowed for experiments but are not merged without the owner's lock.
- Commit and push at least every 30–60 minutes, and always before releasing a lock.
- Run `scripts/sync.sh` before each push and whenever the prompt hook prints a team update.
- Never force-push. Never rewrite history on `main`. Never commit weights, datasets,
  `outputs/`, tokens or `.env` (see `.gitignore`).

## 5. Write as you go: the rolling docs

| What | Where | Rule |
|---|---|---|
| Event log | `docs/log/<you>.md` via `scripts/log.sh --task T03 "what, why, result"` | Append-only, one line per event, written as it happens. One file per member, so appends never conflict. `status.sh` merges all logs by time. |
| Handoff | the task file, "Notes and handoffs" section | Before releasing as `todo` or `blocked`, write what you tried, the current state and the next step, so a cold agent can continue. `release.sh` adds the closing line. |
| Decision | `scripts/new-decision.sh "title"` then fill it in | One per non-obvious choice: Context, Decision (exact flags, versions, files), Consequences. Measured numbers beat adjectives. Numbers never collide because the script publishes immediately. |
| Status snapshot | `docs/status/YYYY-MM-DD-HHMM-slug.md` | At each milestone: TL;DR, what works, what is broken, the next three things. The newest file is the current truth. |

Log lines that matter most: commands with side effects (GPU pod created or stopped, with
id and cost per hour), measured numbers (latency, VRAM, success counts), dead ends, and
anything a teammate would otherwise rediscover.

## 6. Picking work

1. In `scripts/status.sh`, find the TASKS rows with status `todo` whose `depends` are all `done`.
2. Prefer the lowest id. If your human assigned you something, do that instead.
3. No suitable task? Create one with `scripts/new-task.sh` that has a narrow path glob, then claim it.
4. Each task file says **Done when**. Do not release as `done` until that check passes, and
   put the evidence (command and output) in your log.

## 7. Session end

1. Commit and push your work.
2. `scripts/release.sh <task> done|todo|blocked "note"` for every lock you hold, or renew it
   if you will be back within the expiry.
3. If you reached a milestone, write a status snapshot.
4. If you started a paid GPU, stop it and log the stop.

## 8. Things that are never OK

- Editing, reformatting or "cleaning up" files in someone else's lock.
- Deleting another member's log lines, task notes or decisions. Correct them with a new line instead.
- Pushing secrets, model weights or datasets. The repo is **public**.
- Running a paid GPU without a log line that records the start, the cost per hour and the stop.
