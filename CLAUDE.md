@AGENTS.md

## Claude Code specifics

The project hooks in `.claude/settings.json` do three things for you:

- At session start they run `scripts/sync.sh`, so the board is already in your context.
- Before each prompt they print team changes, if there are any.
- They block Edit and Write on files you have not claimed.

If a hook blocks you, claim the file or pick another one. Do not work around the hook.
