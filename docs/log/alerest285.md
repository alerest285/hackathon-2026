# Log: alerest285

Append-only, newest at the bottom. One line per event.

- 2026-10-17T11:23Z alerest285 [-] (claude-code) bootstrap: repo created; lock test with two local clones passed: claim, refusal, overlap refusal, simultaneous-claim race (loser retried and backed off), pre-commit block (foreign + unclaimed), edit guard exit 2, release refuses dirty paths, parallel new-task ids T08/T09 distinct, CI per-commit check flags the foreign commit
- 2026-10-17T12:16Z alerest285 [T02] (claude-code) aws: profile hackathon signed in (project, eu-north-1, FREE plan, $100 credits); agent toolkit installed (24 skills, aws-mcp for Claude/Codex/Gemini, AWS_MCP_PROXY_PROFILES=hackathon); quota requests P spot 16 + P on-demand 16 PENDING; dry run: on-demand p5.4xlarge allowed, spot denied by free-plan SCP; spot $3.16/h, on-demand $7.36/h
- 2026-10-17T12:16Z alerest285 [T03] (claude-code) scripts: region eu-north-1 + profile hackathon defaults, HACK_MARKET spot|ondemand, dry-run preflight in up.sh; AMI ami-0b00ba21bb3194a5a /dev/sda1 verified
- 2026-10-17T12:17Z alerest285 [T08] (claude-code) release (done): AWS toolkit rules in AGENTS.md section 9 (+CLAUDE.md pointer, .cursor/rules/aws.mdc), docs for eu-north-1; help_level alerest285=MEDIUM
- 2026-10-17T12:18Z alerest285 [T01] (claude-code-hanoi) decision 05 accepted (Ale): physical SO-101 + 3-disk Tower of Hanoi, one move per instruction via our sequencer; N=10 strict episodes, operator key = ground truth, auto end = grasp+release+1s rest, 30s timeout; project.md milestones 1-3 moved from DROID to SO-101
- 2026-10-17T12:18Z alerest285 [T01] (claude-code-hanoi) release (done): decision 05; project.md open question 1 points to it
- 2026-10-17T12:20Z alerest285 [T09] (claude-code) fix: locks are per member+agent (two agents of one member can no longer overlap; CI uses --agent '*'); pull_latest uses a named stash and never drops edits (an autostash had orphaned my edits once, recovered from d9d1c29); scripts re-exec from a temp copy when scripts/ is dirty. Tested in 3 scratch clones.
