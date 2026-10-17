@AGENTS.md

## Claude Code specifics

The project hooks in `.claude/settings.json` do three things for you:

- At session start they run `scripts/sync.sh`, so the board is already in your context.
- Before each prompt they print team changes, if there are any.
- They block Edit and Write on files you have not claimed.

If a hook blocks you, claim the file or pick another one. Do not work around the hook.

**Agent chat (AGENTS.md 1b):** at the start of every session, start the Monitor tool with command
`scripts/listen.sh`, description "team agent chat" and the maximum timeout. Re-arm it when it expires. Reply to
`[chat FOR YOU]` lines with `scripts/say.sh @<login> "..."` before going on. Set `HACK_AGENT` to a unique name first.

<!-- BEGIN AWS Agent Toolkit rules -->
## AWS

The AWS Agent Toolkit rules are in AGENTS.md section 9, which is imported above. Your global config has the `aws-mcp` server
(profile `hackathon`) and the AWS skills; use them for AWS work. Credentials last 12 h, renew for 90 days, and re-login is
`aws login --profile hackathon`.
<!-- END AWS Agent Toolkit rules -->
