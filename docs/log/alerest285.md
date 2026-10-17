# Log: alerest285

Append-only, newest at the bottom. One line per event.

- 2026-10-17T11:23Z alerest285 [-] (claude-code) bootstrap: repo created; lock test with two local clones passed: claim, refusal, overlap refusal, simultaneous-claim race (loser retried and backed off), pre-commit block (foreign + unclaimed), edit guard exit 2, release refuses dirty paths, parallel new-task ids T08/T09 distinct, CI per-commit check flags the foreign commit
- 2026-10-17T12:16Z alerest285 [T02] (claude-code) aws: profile hackathon signed in (project, eu-north-1, FREE plan, $100 credits); agent toolkit installed (24 skills, aws-mcp for Claude/Codex/Gemini, AWS_MCP_PROXY_PROFILES=hackathon); quota requests P spot 16 + P on-demand 16 PENDING; dry run: on-demand p5.4xlarge allowed, spot denied by free-plan SCP; spot $3.16/h, on-demand $7.36/h
- 2026-10-17T12:16Z alerest285 [T03] (claude-code) scripts: region eu-north-1 + profile hackathon defaults, HACK_MARKET spot|ondemand, dry-run preflight in up.sh; AMI ami-0b00ba21bb3194a5a /dev/sda1 verified
- 2026-10-17T12:17Z alerest285 [T08] (claude-code) release (done): AWS toolkit rules in AGENTS.md section 9 (+CLAUDE.md pointer, .cursor/rules/aws.mdc), docs for eu-north-1; help_level alerest285=MEDIUM
- 2026-10-17T12:18Z alerest285 [T01] (claude-code-hanoi) decision 05 accepted (Ale): physical SO-101 + 3-disk Tower of Hanoi, one move per instruction via our sequencer; N=10 strict episodes, operator key = ground truth, auto end = grasp+release+1s rest, 30s timeout; project.md milestones 1-3 moved from DROID to SO-101
- 2026-10-17T12:18Z alerest285 [T01] (claude-code-hanoi) release (done): decision 05; project.md open question 1 points to it
- 2026-10-17T12:20Z alerest285 [T09] (claude-code) fix: locks are per member+agent (two agents of one member can no longer overlap; CI uses --agent '*'); pull_latest uses a named stash and never drops edits (an autostash had orphaned my edits once, recovered from d9d1c29); scripts re-exec from a temp copy when scripts/ is dirty. Tested in 3 scratch clones.
- 2026-10-17T12:20Z alerest285 [T09] (claude-code) release (done): per-agent locks + no-loss pull + safe re-exec, tested in scratch clones
- 2026-10-17T13:06Z alerest285 [T03] (claude-code) up.sh: imports ~/.ssh/id_ed25519.pub as EC2 key hack-2026-alerest285 (alerest285 has no GitHub keys), disk 200 GB; bootstrap checks gated HF access; real up.sh run on FREE plan stopped at dry-run with the spot message, 0 instances created
- 2026-10-17T13:06Z alerest285 [T02] (claude-code) budget hackathon-gpu $150/month created, alerts 50/80/100% + forecast to alerest285 mail; quota cases P spot 16 + P on-demand 16 = CASE_OPENED; plan still FREE (upgrade pending, owner)
- 2026-10-17T13:06Z alerest285 [T02] (claude-code) release (blocked): waiting on AWS quota review (2 cases open) + paid-plan upgrade by alerest285; budget done
- 2026-10-17T13:06Z alerest285 [T03] (claude-code) release (todo): scripts final + dry-run verified on the live project; first boot waits on T02. See handoff.
- 2026-10-17T13:09Z alerest285 [T03] (claude-code) plan upgraded to PAID by alerest285 (credits $100 kept); spot still denied: paid-plan SCP has DenySpotMarketType. Ale chose on-demand (decision 07). HACK_MARKET default ondemand; full launch dry run (AMI, 200 GB gp3, 1a, IMDSv2) = DryRunOperation; on-demand P quota still 0, case open
- 2026-10-17T13:09Z alerest285 [T02] (claude-code) release (blocked): only the on-demand P quota (case open) is left; plan PAID, budget live
- 2026-10-17T13:09Z alerest285 [T03] (claude-code) release (todo): on-demand default, full launch dry run passes; start with infra/aws/up.sh once quota >= 16
- 2026-10-17T13:09Z alerest285 [T12] (claude-code) release (done): project.md says on-demand, decision 07
- 2026-10-17T13:11Z alerest285 [T13] (claude-code) first-boot.sh + smoke-so101.sh written (syntax-checked); waiting on on-demand P quota (background watcher) and Ale's HF token
- 2026-10-17T13:12Z alerest285 [T02] (claude-code) spend limit $150 set by alerest285 on project 'Just Ship It' (AWS Settings, screenshot 17.10): $0 spent, $100 credits available
- 2026-10-17T13:14Z alerest285 [T02] (claude-code) teammates saadl27 + joelleayoub: invited as repo collaborators (write), added to members.txt; Runpod pod create refused: account balance too low (Ale to top up)
- 2026-10-17T13:39Z alerest285 [T13] (claude-code) FIRST INFERENCE PASS on Runpod pod rmhs4cu2cejemv (H100 80GB, EU-FR-1, $3.99/h, ssh root@<pod-host> -p <ssh-port>): flux-3-action-so101 @0713a7a, synthetic obs, eager: load 123 s, first plan 1.94 s, then p50 0.58 s / max 0.61 s per 42x6 chunk, peak GPU 25.9 GB. Weights in /workspace/hf (so101 13 GB + base encoders).
- 2026-10-17T13:39Z alerest285 [T13] (claude-code) release (done): SO-101 smoke PASS on Runpod H100: 0.58 s/chunk eager, 25.9 GB, load 123 s
- 2026-10-17T13:40Z alerest285 [T03] (claude-code) release (todo): compute is a Runpod H100 for now (status 2026-10-17-1545); AWS launch still waits on quota
- 2026-10-17T13:41Z alerest285 [T14] (claude-code) repo made private; README rewritten for the team (pod access, status, join steps); AGENTS.md + project.md updated; decision 08 Runpod
- 2026-10-17T13:42Z alerest285 [T14] (claude-code) release (done): README + private repo + decision 08
- 2026-10-17T13:50Z alerest285 [T15] (claude-code) real-frame zero-shot (hungho77/so101_left_arm_pick_red_cube_into_pot ep0, top->scene, wrist->wrist, resized to 480x640; units are degrees, ranges exceed +-100): f60 operator idle, model starts the reach (lift -102 -> -36, grip opens); f150 mid-reach, model follows direction, mean |err| pan 3.9 lift 5.2 elbow 11.8 wflex 17.4 wroll 5.4 grip 2.7; f300 diverges (lift 35, wroll 40). ~0.96 s per single plan. Open-loop, no history, someone else's rig: a sanity check, not a score.
- 2026-10-17T13:50Z alerest285 [T15] (claude-code) release (done): real-frame inference works; f150 tracks the teleop reach, f300 diverges; outputs /workspace/outputs/real-so101*
- 2026-10-17T13:52Z alerest285 [T16] (claude-code) README: team table (GitHub/SSH/Runpod per member), zero-shot runbook for the pod, GPU sharing rules, parallel-agent rules; ErfanRzt invited (write), no GitHub SSH keys yet
- 2026-10-17T13:52Z alerest285 [T16] (claude-code) release (done): README runbook + agent rules
