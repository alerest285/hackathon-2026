# Coordinating parallel coding agents: methods surveyed (2026-10-17)

Why the repo works the way it does (decision 01). Every claim was checked against the linked
page unless it is marked unverified.

| Method | What it does | For our team of ~4 |
|---|---|---|
| [AGENTS.md](https://agents.md) | One Markdown instructions file read by Codex, Cursor, Gemini CLI, Copilot, Aider and others. The nearest file wins | **Used.** Claude Code needs `@AGENTS.md` inside CLAUDE.md ([docs](https://code.claude.com/docs/en/memory)) |
| Lock files pushed to git ([Anthropic, parallel Claudes building a C compiler](https://www.anthropic.com/engineering/building-c-compiler)) | An agent writes `current_tasks/<task>.txt` and pushes. If the push fails, the task is taken | **Used**, extended with path globs, expiry and overlap checks |
| [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) | A progress file, a JSON feature list where agents may only flip `passes`, `init.sh`, a commit per change, and a fixed start-of-session routine | **Used** in spirit: sync at start, logs, "Done when" checks in tasks |
| GitHub Issues assignment | `gh issue edit N --add-assignee @me` | Visible to humans but not atomic. Optional mirror |
| [MCP Agent Mail](https://github.com/Dicklesworthstone/mcp_agent_mail) | Inboxes plus advisory file leases with expiry, and an optional pre-commit guard | Needs a server; sharing it across machines is not documented. Skipped |
| [Beads `bd`](https://github.com/steveyegge/beads) | A Dolt-backed issue graph with atomic `--claim` and `bd ready` | Good, but every machine must install it. The upgrade path if contention grows |
| [claude-squad](https://github.com/smtg-ai/claude-squad) | tmux plus a worktree per agent | Single machine (inferred). Does not fit |
| [Vibe Kanban](https://github.com/BloopAI/vibe-kanban) | A board with a workspace per task | README says it is sunsetting. Avoid |
| [Backlog.md](https://github.com/MrLesk/Backlog.md), [Task Master](https://github.com/eyaltoledano/claude-task-master) | Markdown or PRD-derived task lists | No verified claim/lock feature. Extra installs |
| [Cline Memory Bank](https://docs.cline.bot/prompting/cline-memory-bank) | Six context files, read every task | Too heavy for 48 h. Its `progress` + `activeContext` became our log + status |
| [spec-kit](https://github.com/github/spec-kit) | Constitution → spec → plan → tasks | Too much ceremony for a hackathon. Our docs-flow table is the light version |
| CODEOWNERS + branch protection | Directory ownership enforced on PR review | Requires PRs, which conflicts with direct lock pushes |
| [Claude Code hooks](https://code.claude.com/docs/en/hooks) | SessionStart stdout goes into context; PreToolUse exit 2 blocks a tool call | **Used** for sync-at-start, team deltas and the edit guard |
