# hackathon-2026: zero-shot FLUX 3 Action

We run Black Forest Labs' open 7B world action model, [FLUX 3 Action](https://huggingface.co/blog/black-forest-labs/flux-3-action),
zero-shot on one robot task. Compute is one AWS H100 on spot, with a $150 budget. Several people
work in parallel, each with their own coding agent, and this repo stops those agents from
editing the same files.

## Join in 3 commands

```sh
gh repo clone alerest285/hackathon-2026 && cd hackathon-2026
scripts/setup.sh claude-code      # or codex | cursor | gemini. Sets your name = GitHub login, turns hooks on
scripts/sync.sh                   # pull + the board: locks, tasks, latest logs, decisions
```

Then tell your agent: **"Read AGENTS.md and pick a task."** The rest is in that file.

## How it fits together

| Read in this order | Why |
|---|---|
| [`AGENTS.md`](AGENTS.md) | the contract: claim before editing, write as you go, docs flow downstream |
| [`docs/project.md`](docs/project.md) | goal, milestones, open questions, the $150 constraint |
| [`docs/architecture.md`](docs/architecture.md) | components, the exact request/response formats, which directory belongs to which task |
| [`docs/decisions/`](docs/decisions) | why each choice was made |
| [`tasks/`](tasks) | units of work with "Done when" checks |

| Command | Does |
|---|---|
| `scripts/claim.sh T04` | locks a task's files for you (3 h, renew by running it again) |
| `scripts/release.sh T04 done "note"` | frees them and records the handoff |
| `scripts/log.sh --task T04 "..."` | one line in your own log, `docs/log/<you>.md` |
| `scripts/new-task.sh "title" 'path/**'` | a new task with a collision-free id |
| `scripts/new-decision.sh "title"` | a new numbered decision record |
| `infra/aws/up.sh` · `down.sh` · `ssh.sh` · `cost.sh` | the shared GPU ([infra/aws/README.md](infra/aws/README.md)) |

Guards: a pre-commit hook blocks commits on files you have not claimed. Claude Code is also
blocked from editing them. CI re-checks every push. The full design and the alternatives we
looked at are in decision 01 and `docs/research/coordination-methods.md`.

**This repo is public.** Never commit keys, `.env`, the GPU's hostname, weights or datasets.

## Licenses

Our code is MIT. FLUX 3 Action code is Apache-2.0. The weights are under the
[FLUX Kommunity License](https://huggingface.co/black-forest-labs/flux-3-action-base/blob/main/LICENSE.md).
They are downloaded on the GPU and never stored here.
