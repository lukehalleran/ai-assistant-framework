# Agent Branch Portfolio — Milestones 1–2

The supervisor harness that runs isolated workers, evaluates their diffs in a
trusted environment the workers never control, kills the bad ones cheaply, and
ranks the survivors — **without** the GUI, ProposalStore, push, or real data
(those are M3+).

- **M1 — prove isolation (N=1):** one worker provably cannot escape the sandbox.
- **M2 — prove the reaper (N=3 + a saboteur):** run strategy-diverse workers
  against one objective, kill the known-bad branch at the gate, externally
  evaluate survivors, and emit one human-scannable ranked Trial Report. Merging
  is **not** a goal — *"prove the reaper before scaling the garden."*

## Why this shape

The May 18 safety guards (`utils/*_guard.py`, `scripts/safe_*.sh`) are
**cooperative / in-process** — an adversarial worker bypasses them (`/usr/bin/rm`,
`ctypes`, or a plain `open(path,"w")`). They remain useful as defense-in-depth +
audit, but they cannot be the boundary. So:

- **Container, not worktree.** A git worktree shares the parent `.git` object
  store; only an OS boundary holds. Workers run in **rootless podman** with
  `--network=none`, `--cap-drop=ALL`, `--security-opt=no-new-privileges`, no host
  writable mount, no real secrets, no real stores.
- **Evaluation runs outside the worker.** The supervisor **computes the diff
  itself** (never a worker-authored patch), gates it statically *before any
  execution*, then runs the trusted tests in a **separate disposable container** —
  never on the host. The worker controls its clone's `.git`, and git executes
  repo-local config (clean filters, `core.fsmonitor`, external diff) and honours
  replace-refs — so git is **never** run against the worker's `.git`. The diff is
  computed by overlaying the worker's *worktree* (sans `.git`, symlinks not
  followed) onto a fresh **trusted** checkout of the baseline and diffing that
  with hardened flags + a sanitised git env.
- **The proof assumes the inner guards are OFF.** The container/namespace alone
  enforces every constraint.

## Components

| File | Role |
|------|------|
| `manifest.py` | Frozen, SHA-256-verified `BranchManifest` (the worker's contract) |
| `provisioning.py` | Rootless-podman clone/provision/run/teardown; supervisor-computed diff |
| `llm_proxy.py` | UDS reverse-proxy: allowlist (not an open relay) + tamper-proof token meter; the API key lives here, never in the worker |
| `eval_static.py` | **Layer 1a** — static kill decisions, no code execution (runs first, gates everything; M2 adds config/threshold-tamper detection) |
| `eval_sandboxed.py` | **Layer 1b** — apply diff + run trusted PROOF tests in a disposable container; worker-added tests run separately as EVIDENCE |
| `supervisor.py` | Single-branch orchestrator: issue → run → diff → gate → report |
| `portfolio.py` (M2) | Run N workers (bounded concurrency, isolated each), rank, emit one Trial Report |
| `scoring.py` (M2) | Mechanical survivor ranking (objective → diff economy → cost) + pairwise divergence |
| `report.py` (M2) | The human-scannable Agent Branch Trial Report (markdown + JSON) |
| `eval_layer2.py` | **Stub (M3)** — LLM intent judge |
| `workers/attacker.py` | Scripted adversary (M1 isolation gate) |
| `workers/strategy_worker.py`, `saboteur.py` (M2) | Deterministic strategy diffs + the known-bad worker (reaper proof) |
| `workers/benign_llm.py`, `llm_strategy_worker.py` | Real-LLM workers over the UDS proxy (manual demos) |
| `fixtures/` | Fake `.env` + synthetic store (never the real ones) |

## Network model

- **Attacker** → `--network=none` (zero egress; the gate). Full SELinux confinement.
- **LLM worker** → `--network=none` **+** one bind-mounted Unix-domain socket to
  the supervisor's allowlist proxy. No IP egress; a single mediated channel.

## Run it

```bash
# automated proof (skips cleanly if rootless podman is unavailable)
pytest tests/agent_branch/ -q                 # 81 tests (M1 + M2 + M2.5 + hardening)
pytest tests/agent_branch/ -m "not slow" -q   # pure logic only, no containers

# manual demos
python -m agent_branch.supervisor --demo attacker   # M1: scripted attacker is contained and killed
python -m agent_branch.portfolio  --demo            # M2: 3 strategies survive + ranked, saboteur killed → Trial Report
```

Requires rootless **podman** and the `python:3.11-slim` base image present
offline (the minimal `agentbranch-worker` image is built on first run).

## The acceptance gate (red-team)

`tests/agent_branch/test_isolation_redteam.py` proves: no writes reach the host,
protected stores are absent, only the fake secret exists, no network egress, the
manifest mount is EROFS, tampered manifests are detected, test/conftest/payload
edits are killed before execution, budgets are enforced, the one happy-path LLM
worker survives both gates — **and all of it holds with `DISABLE_FS_GUARD=1`.**

`tests/agent_branch/test_diff_integrity.py` + `test_redteam_hardening.py` harden
the trusted-side plumbing the scripted adversaries never aimed at:

- **The worker's `.git` cannot execute host code.** `compute_trusted_diff` never
  runs git against the worker clone's `.git`; a clean-filter payload that fires
  under a naive `git add` (proven by a control test) does **not** fire here, and
  worker symlinks are diffed as symlinks, never dereferenced into the diff.
- **The diff parser can't be fooled by content lines.** An added line whose text
  begins with `++ ` (rendered `+++ ...`) is counted as real work, not misread as
  a file header — so it can't bypass `max_diff_lines`, dodge scope checks, or
  deflate the branch's own diff-economy score.
- **Every pytest-config surface is watched.** `pyproject.toml`, `setup.cfg`,
  `tox.ini`, and `.gitattributes` join `pytest.ini` as high-risk CONFIG_TAMPER —
  closing the "forge a passing proof via `addopts`/`-p`" and "hide diff content
  via `*.py -diff`" holes.
- **The token meter can't be starved.** The prompt is metered on every call —
  including streaming, usage-less, and failed responses — and request bodies are
  size-capped so a worker can't OOM the supervisor.
- **The proxy is narrow on every axis.** Beyond the host allowlist, the request
  *path* is pinned (chat-completions/messages only) so the injected key can't
  reach key-management/upload routes, and the socket's directory is `0700` so no
  other host process can reach the UDS.
- **No unprovable free passes.** A diff that clears the static gate but has no
  supervisor proof test fails closed (killed) instead of trivially passing, and
  the default proof command runs trusted checks as stdlib `python` scripts (the
  bare image has no pytest) so a default manifest dies on merit, not on import.

## The M2 reaper proof

`tests/agent_branch/test_portfolio.py` proves: the saboteur is killed at the gate
(config/threshold tamper), the 3 strategy workers survive and rank (smallest
correct change first), strategies provably diverged, a worker-added test is
recorded as *evidence* but never substitutes for the supervisor's *proof*, all
evaluation is external (host unchanged), isolation holds at N (distinct
clones/containers), and the Trial Report is readable without opening a diff.

**M2.5 — catches sincere failure, not just sabotage.** The reaper must kill the
*common* case: a real model that genuinely tries and is subtly wrong, not just a
cheat. The proof is deepened (subtract pinned across order/sign/zero) so an
overfit hardcode (`return 3`) or a sincere mistake (`return abs(a-b)`) both fail
it — even though their diffs are tiny. Two scripted failure workers demonstrate
this, and a gap-and-closure test shows the sincere `abs` error **passes** a
one-example proof but **dies** on the deep one. Ranking enforces
correctness-before-economy: a thin-but-wrong diff can never reach the ranking,
because a wrong branch can't be a survivor. (No finite proof is complete — the
human-in-the-loop review of the #1 survivor covers the residue.)

## Deferred to M3 (do NOT build here)

LLM intent judge (`eval_layer2`); reuse of the full 20-tool agentic controller +
`LLM_BASE_URL` override in `models/model_manager.py`; converting survivors to
`ProposalSource.AGENT_BRANCH` proposals in `ProposalStore`; the GUI Proposals tab;
push; auto-merge; moving-main merge-conflict scoring; scaling beyond N=3. (The
schema — `ProposalSource.AGENT_BRANCH`, `ProposalOutcome.merge_branch` — already
anticipates the M3 integration.)
