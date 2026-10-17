# 01. Coordination: git lock files, per-member logs, trunk-based main

**Date:** 2026-10-17 · **Status:** accepted · **Author:** alerest285

## Context

Four or so people each run their own coding agent (mixed tools) on one repo for a hackathon.
We need three things. First, no two agents edit the same file at once. Second, every agent starts from the
latest shared state. Third, agents read the canonical docs before acting and write back afterwards.
The methods we surveyed are in `docs/research/coordination-methods.md`.

## Decision

- **One instructions file.** `AGENTS.md`. `CLAUDE.md` imports it with `@AGENTS.md`, Gemini CLI reads it via `.gemini/settings.json`,
  Cursor via `.cursor/rules/agents.mdc`, and Codex natively.
- **Claims are lock files pushed to `main`** (`locks/<task>.md`, holding the owner, agent, expiry and path globs).
  This is the pattern Anthropic used for its parallel-Claude C compiler. Git accepts only one push first, which makes the claim atomic.
  `scripts/claim.sh` refuses overlapping globs, retries on a rejected push, and expires a lock after 3 h unless renewed.
- **Enforcement in three layers.** A `pre-commit` hook (every agent) blocks commits on unclaimed paths. A Claude Code
  `PreToolUse` hook blocks Edit/Write. CI re-checks each pushed commit against the locks in that commit's tree.
- **Rolling docs that cannot conflict.** Each member has their own append-only log (`docs/log/<login>.md`,
  also `merge=union`). Handoffs live inside task files. Numbered decisions and task ids are published
  the moment they are created, so numbers never collide. The board (`scripts/status.sh`) is generated, never committed.
- **Fresh context.** A Claude `SessionStart` hook runs `scripts/sync.sh`, and `UserPromptSubmit` prints new commits
  and flags changed upstream docs. Other agents run `scripts/sync.sh` by rule (AGENTS.md section 1).
- **Trunk-based on `main`, no required PRs.** Disjoint locks keep rebases clean. Branch protection that requires PRs
  would block the direct lock pushes.

## Consequences

- No installs beyond git, bash and gh. Works on macOS (bash 3.2), Linux, and Windows via Git Bash.
- Locks are advisory for agents without hooks until commit time. The pre-commit hook is the real gate,
  so every clone must run `scripts/setup.sh`.
- CI identifies the committer by the GitHub actor (the pusher), so `hack.member` must equal the GitHub login. `setup.sh` sets that.
- Revisit if contention gets high, meaning frequent overlap refusals: then move to per-task branches + PRs, or Beads (`bd`).
