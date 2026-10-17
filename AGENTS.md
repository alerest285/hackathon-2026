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

<!-- BEGIN AWS Agent Toolkit rules -->
## 9. AWS (only when work touches AWS: `infra/aws/`, the GPU box, billing)

This repo's sections 1–8 come first. Where they conflict with the AWS rules below, this repo wins.
The project is **alerest285's AWS project in eu-north-1**, with CLI profile `hackathon` (`aws login --profile hackathon`).
Every agent that runs AWS commands must follow these rules. Agents that never touch AWS can skip this section.
Before AWS work, load the matching AWS skill (e.g. `aws-compute`, `aws-billing-and-cost-management`) if your tool has it.

### AWS Guidance for the new AWS experience

This user has signed up for the new AWS experience. This experience lets you sign into AWS using a social provider and requires the following additional context.

Where this guidance conflicts with the project's own instructions, the project's instructions take precedence.

## Context

#### Terminology:

- Say "project" instead of "account" — a project contains an AWS account and settings for sharing with other collaborators
- Say "team member" instead of "IAM user" — users are invited by email, not created or federated in IAM
- Say "AWS Settings" when referring to management tasks at [settings.aws.com](https://settings.aws.com/) (project management, billing, team members, spend limits). Users view their actual AWS resources in the AWS Management Console.
- Say "selected Region" when referring to the user's Region — not "home Region"
- The user has a managed IAM experience. This includes a managed service control policies (SCP) and resource control policies (RCP) that govern the use of AWS. They will still need to use IAM to create policies to let services work with each other. If there are questions about the SCPs or RCPs, go to the documentation at https://docs.aws.amazon.com/accounts/latest/reference/scps-and-rcps-for-projects.html

#### Constraints:

- All projects share a single AWS Region determined by the user's contact address. Resources cannot be created in other Regions
- When developing:
  - MUST create all Regional resources in the project's assigned Region
  - You CAN create AWS WAF and Cloudwatch Logs resources in us-east-1 when there are global resources (like a global WAF instance) that require a connection to dependencies in us-east-1. You should not use these for any other reason, because resources in the selected Region will provide lower cost (due to no cross-Region traffic), increased availability (due to no cross-Region traffic), and easier manageability (due to not needing to look in another Region). When you need to do an inventory of resources, you need to look in both the selected Region and us-east-1 for Cloudwatch Logs or WAF resources.
  - MUST NOT attempt to create Lambda, API Gateway, or other Regional resources in any other Region
  - MUST direct users to confirm their Region in AWS Settings > View all projects > Overview > Additional Info > Region. If the user cannot confirm their Region, check in ~/.aws/config
  - MUST NOT use Lambda@Edge — excluded from both Lambda and CloudFront
  - MUST NOT use CloudFormation StackSets — no multi-account or multi-Region deployments
  - MUST NOT attempt cross-Region actions — no cross-Region replication for DynamoDB/S3/RDS, no multi-Region KMS keys
  - MUST NOT use Route 53 cross-Region routing — geolocation, latency-based, and failover routing policies are not available
  - CloudFront is a global service and its actions ARE allowed in `us-east-1`. A user can create a CloudFront distribution pointing to their project-region Lambda function URL or API Gateway. However, Lambda and API Gateway themselves MUST NOT be created in `us-east-1` — they must be in the project Region.
  - Reduced availability in `eu-north-1` specifically: Amazon Rekognition, Amazon Textract, Amazon Personalize, AWS App Runner are not available in that Region.
- IAM permissions for human access are managed by AWS. Don't assign roles to team members unless absolutely necessary
- The user may have a spend limit if they are on the paid plan. The limit that pauses their project if it's exceeded. If resources suddenly become inaccessible, ask if they have a spend limit configured. Only project owners can modify a spend limit.
- When developing:
  - MUST ask about spend limit status if the user reports sudden "Access Denied" errors on operations that previously worked
  - MUST direct users to check spend status in AWS Settings > Billing
  - MUST check if a user has upgraded their account to the paid plan
  - MUST ask the user if they want to clean up the successfully created resources or keep them to reduce cost
- The user sets up billing, creates spend limits, and retrieves and pays invoices in AWS Settings. The user creates budgets and optimizes their costs in the AWS Billing and Cost Management console
- Not all AWS services are available. If a service isn't working, do the following:
  1. Run the command `aws freetier get-account-plan-state`
  2. If accountPlanType": "FREE", check the [Free Tier supported services list](https://docs.aws.amazon.com/accounts/latest/reference/supported-services-sign-up-new.html#supported-services-free-tier) next,
  3. If accountPlanType": "PAID", check the [Paid Tier supported services list](https://docs.aws.amazon.com/accounts/latest/reference/supported-services-sign-up-new.html#supported-services-paid-plan).
  4. If neither list shows the service, check the [Not supported for this experience list](https://docs.aws.amazon.com/accounts/latest/reference/supported-services-sign-up-new.html#unsupported-services). The user will need to activate advanced features to access this service.
- Users can activate advanced AWS services and capabilities for their account.
- Before starting a task, check whether a relevant AWS skill is available. Load the skill with retrieve_skill and prefer its guidance over general knowledge.

#### Help level

- help_level (required): LOW, MEDIUM, or HIGH. While a user is building, you MUST ask the user: "How much guidance would you like from me? Low (I only flag security risks), medium (I ask a couple of clarifying questions if something seems off), or high (I explain what I'm doing, suggest alternatives, and flag best practices)."

You CAN update this rule file to save a user's help_level.

Saved help_level: **alerest285 = MEDIUM** (17.10.2026). Other team members: ask yours once, then add a line here (lock AGENTS.md first).

Constraints for each level:

**LOW:**

- MUST follow all constraints in this context file
- MUST execute the user’s request without modification
- MUST NOT ask clarifying questions unless the action would create a security vulnerability
- MUST NOT suggest alternatives or improvements

**MEDIUM:**

- MUST execute the user's request
- MAY ask up to two clarifying questions per task if the request has an ambiguity or a potential issue
- MUST NOT repeat a question or suggestion the user has already dismissed
- MUST NOT explain trade-offs or alternatives unless the user asks

**HIGH:**

- MUST explain what each step does and why before executing it
- MUST suggest alternatives when a better approach exists
- MUST flag best practices and explain trade-offs
- MUST still execute the user's choice if they disagree with a suggestion
<!-- END AWS Agent Toolkit rules -->
