# The Proposal System — Daemon's Self-Improvement Pipeline

> Status: living document. Last substantive update 2026-06-14.
> Scope: the end-to-end "project proposer" — how Daemon proposes changes to its
> own codebase, classifies their risk, stores and surfaces them, tracks whether
> they shipped, and (experimentally) executes them in isolation. Covers both the
> **live proposer** (in production) and the **agent_branch harness** (the
> isolated-execution research track).

---

## 1. The big idea

Daemon is a system that can **propose changes to itself**, under a strict
**human-in-the-loop** contract. The defining principle:

> The machine may *propose* and *evaluate*; only a human may *merge*.

Nothing in the live system commits, pushes, or merges code. Proposals are
*formal artifacts* — structured, risk-classified records — that a human reviews
in the GUI and acts on by hand. The proposal is the contract between an
automated generator and the human gatekeeper.

There are **two tracks**, deliberately separated:

| Track | Status | What it does | Autonomy |
|-------|--------|--------------|----------|
| **A. Live proposer** | Production | Generates text plans (+ optional staged code), classifies risk, stores in ChromaDB, surfaces in the GUI and in chat | Human approves; no code is executed or merged |
| **B. agent_branch** | Research (M1/M2/M2.5 built; M3 bridge landed) | Runs *isolated* workers that actually write code in a podman sandbox, computes their diffs trust-side, kills bad ones, ranks survivors, and (opt-in) ingests survivors into Track A's store as PENDING proposals | No merge — proves the *isolation + evaluation* substrate; survivors now surface in the GUI for human review but are never auto-applied |

The two are connected in **one** direction (M3 bridge): Track B can ingest its
ranked survivors into Track A's store as PENDING proposals, so a
supervisor-verified diff surfaces in the GUI for human review. The *reverse* — a
Track A proposal being *executed* by an isolated agent and its real diff gated —
is still future work. Track A is what a user interacts with today; Track B is the
safety substrate, and nothing it produces is ever auto-applied.

---

## 2. Track A — the live proposer

### 2.1 Pipeline at a glance

```
                ┌──────────────────────────────────────────────────────────┐
   triggers     │                                                          │
 ┌───────────┐  │   GoalDirectedGenerator (knowledge/proposal_generator.py) │
 │ GUI button│──▶│   • gather_context(): skeleton, GOALS, CLAUDE.md,        │
 │ "Generate"│  │     QUICK_REFERENCE, recent commits, recent convos        │
 └───────────┘  │   • LLM → JSON proposals                                  │
 ┌───────────┐  │   • _parse_proposal(): validate + CLASSIFY risk           │
 │ shutdown  │──▶│   • _annotate_conflicts(): live-registry depends_on       │
 │ hook(≥3)  │  └──────────────────────────────┬───────────────────────────┘
 └───────────┘                                 │ List[CodeProposal]
                                               ▼
                        classify_proposal()  (memory/proposal_risk.py)
                        touches_core_system + risk_level
                                               │
                                               ▼
                          ProposalStore (memory/proposal_store.py)
                          • dedup (semantic + title-overlap)
                          • ChromaDB "proposals" collection
                                               │
                ┌──────────────────────────────┼───────────────────────────┐
                ▼                               ▼                           ▼
   GUI Proposals tab            [PROPOSED FEATURES] in chat     ImplementationDetector
   (gui/tabs/proposals.py)      (core/prompt/proposal_filter)   (4-stage "did it ship?")
   • risk/CORE badges           surfaced to the assistant        confirmed/likely/uncertain
   • acknowledge-before-approve
   • generate staged code  ─────▶ data/proposal_code/<id>/
   • human marks status
```

### 2.2 The data model — `CodeProposal`

`memory/code_proposal.py`. A Pydantic `BaseModel` (per project convention).
Stored flat in ChromaDB metadata via `to_metadata()` / `from_metadata()`; full
JSON via `to_dict()` / `from_dict()`.

Core fields: `id`, `title`, `proposal_type` (`feature|refactor|bugfix|test|docs|infra`),
`status`, `source`, `priority` (1–10), `reasoning`, `description`,
`implementation_steps` (list of `ImplementationStep`: order/description/file_path/action/code_snippet),
`affected_files`, `tags`, `estimated_complexity`, `requires_tests`.

Implementation-tracking fields: `implementation_confidence`,
`implementation_status`, `implementation_evidence`, `last_tracked_at`,
`commit_hash`, `rollback_available`.

**Supervision fields** (the branch-gating contract):

| Field | Meaning |
|-------|---------|
| `risk_level` | `RiskLevel` — `low / medium / high / critical` |
| `touches_core_system` | bool — modifies orchestration, memory, safety, or supervision code |
| `depends_on` | proposal/feature IDs this collides with or needs first (advisory) |
| `test_files` | actual test paths that validate it |
| `outcome` | `ProposalOutcome` — accepted, notes, merged_at, merge_branch, reviewed_by |

**Lifecycle** (`ProposalStatus`):

```
PENDING ──approve──▶ APPROVED ──build──▶ COMPLETED
   │                                         ▲
   ├──reject──▶ REJECTED                     │
   └──(execution fails)──▶ FAILED            └── record_outcome(accepted=True)
```

`ProposalSource`: `GOAL_DIRECTED`, `SESSION_INSIGHT`, `USER_REQUEST`,
`SHUTDOWN_ANALYSIS`, `AGENT_BRANCH` (the last reserved for Track B / M3).

### 2.3 Generation — `GoalDirectedGenerator`

`knowledge/proposal_generator.py`.

**Triggers (two, both in Track A):**
1. **GUI "Generate Proposals Now"** — `gui/tabs/proposals.py:_generate_proposals_now()`.
2. **Session shutdown hook** — `memory/shutdown_processor.py:_generate_proposals()`
   (line ~981), gated on `CODE_PROPOSALS_ENABLED`, fires only when the session
   had **≥3** conversation turns. Prefers `generate_proposals_with_context()`
   (rich pipeline context) and falls back to `generate_proposals()` (cold file
   reads).

> Note: generation is **not** an autonomous loop. It is user- or
> session-end-triggered, produces *text plans* (and optional staged code on
> explicit request), and never touches git.

**Context gathering** (`gather_context`): `docs/PROJECT_SKELETON.md` (with the
~112K-char "Core Components" section filtered out as redundant), `docs/GOALS.md`,
`CLAUDE.md`, `docs/QUICK_REFERENCE.md`, and the last 10 git commits. The
shutdown path additionally folds in recent conversation excerpts.

**Prompt**: instructs the model to act as a "senior software architect / product
visionary," emit 3–5 diverse JSON-line proposals, name concrete files/APIs, and
avoid duplicating an injected "Existing Proposals" list.

**Parsing** (`_parse_proposal`): tolerant of code fences / arrays / line-delimited
JSON; validates type and complexity; then **computes supervision fields** — it
collects every path the proposal would touch (`affected_files` + step
`file_path`s) plus all code/text it would introduce (step `code_snippet`s +
description + reasoning) and calls `classify_proposal(...)`. This is why a
proposal can never silently land at the default risk.

**Conflict annotation** (`_annotate_conflicts`): for each proposal, queries the
**live** feature registry for shipped features whose files overlap, and records
them (plus their transitive deps) in `depends_on`. Advisory only — it informs
the reviewer; it is not a merge gate and does not auto-rebase.

**Code generation** (`generate_code_for_proposal`): on explicit GUI request,
reads the current source for `modify` steps, asks the LLM for complete file
contents, and writes them to a **staging directory** `data/proposal_code/<id>/`
alongside a `_manifest.json`. **This writes only to the staging dir — never to
the live tree, never to git.**

### 2.4 Supervision classification — `memory/proposal_risk.py`

The single source of truth for `touches_core_system` + `risk_level`. Replaced an
older exact-string heuristic that lived only in a migration script.

Three path groups (matched by **directory prefix**, not exact string):

- **`CORE_SYSTEM_PATHS`** → at least **HIGH**: orchestrator, context_pipeline,
  best_of_handler, response_generator, intent_classifier, the entire
  `core/prompt/` pipeline, the memory spine (coordinator/storage/scorer/
  retriever), `processing/gate_system.py`.
- **`SAFETY_PATHS`** → **CRITICAL**: `utils/*_guard.py`, `fs_snapshot`,
  `safe_*.sh`, `scripts/bin/`, `escalation_tracker`, `action_claim_guard`,
  `core/actions/`, `tone_detector`.
- **`SUPERVISION_PATHS`** → **CRITICAL**: `code_proposal.py`, `proposal_store.py`,
  `proposal_risk.py` (itself), `proposal_generator.py`, `feature_registry.*`,
  `agent_branch/`.

Two detection modes, so a change can't hide:
1. **Path-prefix match** on every touched file.
2. **Import-based detection** — a brand-new file on a "clean" path that
   `import`s or `from`-imports a core/safety/supervision module still trips
   (`_imports_any` over dotted module prefixes). This closes the
   refactor-and-move / re-export gap.

Safety and supervision touches are **unconditionally CRITICAL** — no exceptions,
including transitive/import touches. Keyword fallbacks (`safety|guard|crisis|...`
→ HIGH, `data.loss|rm.-rf|...` → CRITICAL) catch risky titles even on clean
paths. Docs/test proposals with no risky touch fall to LOW.

**The acknowledge policy** — `requires_human_ack(risk_level, touches_core_system)`
returns True for HIGH/CRITICAL **or** any core touch. It lives here (not in the
GUI) so the review gate can never drift from the classifier that produced the
fields. The GUI gates Approve / Mark-Built on it; Reject is never gated.

### 2.5 Storage — `ProposalStore`

`memory/proposal_store.py`, ChromaDB `proposals` collection.

- `store_proposal()` — embeds `title + reasoning` (match on *problem*, not
  solution), stores full data in metadata.
- **Dedup** (`check_similarity`): semantic cosine ≥ `CODE_PROPOSALS_DEDUP_THRESHOLD`
  (0.70) **or** title-word Jaccard ≥ 0.60. Returns the existing ID, so callers
  skip near-duplicates.
- `update_status()` — ChromaDB has no native update, so this is delete-then-re-add
  with the mutated proposal.
- `update_tracking_metadata()` — merges implementation-detection fields.
- `get_for_dedup()` — a compact recent-proposals text block fed back into the
  generation prompt to suppress repeats.

### 2.6 Implementation tracking — `ImplementationDetector`

`knowledge/implementation_detector.py`. Answers "did this proposal actually ship?"
via a 4-stage pipeline, with a per-proposal cooldown (`IMPL_TRACKING_COOLDOWN`,
default 24h):

1. **File existence** — fraction of `affected_files` (or step targets) that exist.
2. **Code content** — identifiers parsed from title/steps (class/def/CONSTANT
   names) grepped in those files.
3. **Git history** — recent commits scored by file overlap (0.7) + title-keyword
   overlap (0.3).
4. **LLM judgment** — only for borderline composites (0.30–0.84), batched ≤5 per
   call, temperature 0.

Composite = files·0.35 + code·0.35 + git·0.30 → status
`confirmed (≥0.85) / likely (≥0.60) / uncertain (≥0.30) / not_implemented`.
`lightweight=True` (used by shutdown) runs Stage 1 only. The detector never
mutates proposal status — the caller (GUI / store) decides.

### 2.7 Feature registry — `config/feature_registry.{py,yaml}`

A **retrospective catalog of shipped features** (the things already merged), used
for conflict/dependency awareness. `feature_registry.yaml` holds `FeatureEntry`
records (proposal_id, title, risk_level, touches_core_system, depends_on,
implemented_files, test_files, outcome). The loader (`feature_registry.py`)
caches once per process and degrades to empty on a missing/malformed file. API:
`get_feature`, `get_dependencies` (transitive BFS), `get_core_features`,
`get_implemented_files`, `check_conflicts(proposed_files)` (file-overlap → list of
colliding features). `_annotate_conflicts` and the re-classify script both read it
**live**, so a feature added after a proposal was generated is still seen.

### 2.8 The GUI review workflow — `gui/tabs/proposals.py`

The tab where a human acts on supervision metadata.

- **Cards** (`_render_proposal_card`): each shows status, a **risk badge**
  (CRITICAL red `#dc2626` / HIGH orange `#f97316` / MEDIUM / LOW), a purple
  **CORE-SYSTEM** badge when applicable, the implementation-tracking badge, and
  an **"Overlaps / depends on"** line from `depends_on`.
- **Manage dropdown**: prefixes `⛔` (critical) / `⚠` (high or core) so risk is
  visible before selection.
- **Acknowledge-before-approve**: an acknowledge checkbox gates **Approve** and
  **Mark Built** via `_advance_proposal()` → `requires_human_ack`. A HIGH/CRITICAL
  or core proposal refuses to advance until the box is ticked. **Reject is never
  gated** (rejecting is always safe).
- **Generate Code**: stages full files to `data/proposal_code/<id>/` and renders
  them; the human still copies them into the tree by hand.
- **Check Implementation**: runs `ImplementationDetector` and updates badges.

### 2.9 Surfacing into chat

Generated proposals are also retrieved into the conversation prompt as a
`[PROPOSED FEATURES]` section (`core/prompt/proposal_filter.py`, gated by
`CODE_PROPOSALS_PROMPT_ENABLED` / `CODE_PROPOSALS_PROMPT_MAX`), so the assistant
is aware of its own pending self-improvement ideas.

### 2.10 Config reference (`config/app_config.py`, YAML section `code_proposals`)

| Constant | Default | Purpose |
|----------|---------|---------|
| `CODE_PROPOSALS_ENABLED` | True | Master switch (also env `CODE_PROPOSALS_ENABLED`) |
| `CODE_PROPOSALS_COLLECTION` | `proposals` | ChromaDB collection |
| `CODE_PROPOSALS_DEDUP_THRESHOLD` | 0.70 | Semantic dedup cosine |
| `CODE_PROPOSALS_KEYWORD_DEDUP_TAG_THRESHOLD` | 0.60 | Title-overlap dedup |
| `CODE_PROPOSALS_MAX_PER_SESSION` | 5 | Cap per generation run |
| `CODE_PROPOSALS_REQUIRE_TESTS` | True | Default `requires_tests` |
| `CODE_PROPOSALS_PROMPT_ENABLED` / `_PROMPT_MAX` | True / 3 | `[PROPOSED FEATURES]` surfacing |
| `CODE_PROPOSALS_LLM_RANKING` (+ model/weights) | False | Optional LLM re-rank of generated proposals |
| `IMPL_TRACKING_COOLDOWN` / `_GIT_DEPTH` / `_CONFIDENCE_*` | 86400 / 50 / 0.85,0.60 | Implementation detector |

### 2.11 Maintenance scripts

- `scripts/migrate_proposals_supervision.py` — the *original* one-time backfill;
  only filled **missing** supervision fields, using exact-string core-path
  matching. Superseded for re-scoring purposes.
- `scripts/reclassify_proposals.py` — re-runs the **current** classifier over
  **all** stored proposals (prefix + import detection), unions `depends_on` from
  the live registry. Non-destructive (`update_metadata` merge, never deletes),
  **dry-run by default**, `--apply` to write. Idempotent. (2026-06-14 run:
  275 proposals → 34 upgraded, 0 downgraded.)

---

## 3. Track B — the `agent_branch` isolation harness

`agent_branch/` (self-documented in `agent_branch/README.md`). This is the
substrate for *executing* self-modification safely. It is largely **separate**
from Track A — no real data, **no merge** — its whole purpose being to prove that
an untrusted worker can write code and have its diff evaluated **without the
evaluation ever becoming an escape hatch**. The one deliberate connection is the
opt-in `proposal_bridge` (§3.6): ranked survivors can be ingested into Track A's
store as PENDING proposals for human review — but nothing is ever auto-applied.

### 3.1 Trust model

> A worker may do anything inside its sandbox. The supervisor trusts **nothing**
> the worker produced — not its patch, not its `.git`, not its self-report, not
> its token count.

- **Isolation:** rootless **podman**, base `python:3.11-slim`, worker runs
  `--network=none --cap-drop=ALL --security-opt=no-new-privileges` with
  pids/memory caps. The repo is a clone `podman cp`'d in (writable only inside),
  never a live host mount. The manifest + fixtures are read-only mounts (writes →
  EROFS at the mount layer).
- **One deliberate hole (LLM workers):** still `--network=none`, plus a single
  bind-mounted Unix-domain socket to the supervisor's allowlist proxy.

### 3.2 The manifest — `manifest.py`

`BranchManifest` is the **immutable, SHA-256-verified contract** issued to one
worker: `allowed_paths`, `forbidden_paths` (defaults include `data/`, `.env`,
`tests/`, `config/`, every pytest-config file, `.gitattributes`, the safety
guards, and `agent_branch/` itself), `protected_stores`, and hard caps
(`max_diff_lines`, `max_commits`, `max_cycles`, `token_budget`,
`wallclock_seconds`). `issue()` stamps a canonical hash; `assert_intact()`
recomputes and detects any tamper of the supervisor-side copy.

### 3.3 The LLM proxy — `llm_proxy.py`

The single mediated egress for an LLM worker. Reverse-proxies to **one** fixed
upstream (default-deny on any other host → 403; **not** an open relay). The
**request path is pinned** (`allowed_request_paths`, chat/messages only) so the
injected key can't reach key-management/upload routes. The **real API key lives
only here**; the worker never holds it. **Token metering is tamper-proof**: the
prompt is metered on *every* call — failed, streaming, and usage-less ones alike
— and upstream `total_tokens` is trusted when present; over budget → 429 +
`budget_exceeded`. Request bodies are size-capped; the socket directory is `0700`.

### 3.4 Two-layer evaluation

**Layer 1a — the static gate** (`eval_static.py`, runs first, **never executes
worker code**). Parses the **supervisor-computed** diff with a hunk state machine
(so a content line rendered `+++ x` can't masquerade as a header and dodge the
size counter or pollute scope), then collects *all* violations: forbidden/safety
path touches, config/threshold tamper (a "fake-success" surface), out-of-scope
edits, protected-store touches, oversize diffs, vendored-noise dominance, and
destructive-op/secrets markers in untrusted logs. Any hit → `killed`; safety,
supervision, config-tamper, protected-store, and destructive/secrets hits also
set `high_risk`.

**Layer 1b — the sandboxed proof** (`eval_sandboxed.py`). Clones the **trusted**
repo, `git apply`s the diff there (apply executes nothing), then runs the
**supervisor-owned** proof tests inside a disposable `--network=none` container.
Non-zero exit → killed. Worker-added tests run **separately as evidence**, never
as proof (D3). A branch with **no** proof tests is **unprovable → killed**
(fail-closed); a custom `test_cmd` can't rescue it.

### 3.5 Trusted-side diff computation — `provisioning.py`

**Security-critical.** The worker owns its `.git`, and git executes code from
repo-local config (clean/smudge filters, `core.fsmonitor`, `diff.external`) and
honours replace-refs. So the supervisor **never runs git against the worker's
`.git`**. `compute_trusted_diff()` builds a fresh trusted checkout at `base_sha`,
**overlays only the worker's worktree** (excluding `.git` at *every* depth,
symlinks recreated but never followed), and diffs that with hardened flags
(`_GIT_HARDENED`) and a sanitised env (`_git_safe_env`).

> **2026-06-14 finding & fix:** a worker could hide a payload by dropping a *new
> untracked directory* containing real files **plus a nested `.git`** — the
> trusted `git add -A` would record only a submodule gitlink (`mode 160000`,
> "+Subproject commit …"), making the content invisible to the static gate
> (`parse_diff` saw 1 line for a 50-line payload). Fixed by stripping `.git` at
> all depths in `_overlay_worktree`.

### 3.6 Orchestration

- **`supervisor.py` (M1, N=1):** `run_branch()` = provision → run worker →
  collect trusted diff → static gate → (if not killed) sandbox proof → resolve
  outcome → write `branch_report.json`. **Outcome = killed** if the static gate
  fired, the wallclock timed out, or the sandbox proof failed; **passed**
  otherwise. The `BranchReport` carries the run outcome, run stats (incl.
  proxy-metered tokens), both gate results, a diff excerpt, and collected
  artifacts.
- **`portfolio.py` (M2, "the reaper"):** runs N strategy-diverse workers (+ a
  saboteur) against **one shared objective**, each fully isolated, over a bounded
  `ThreadPoolExecutor`. Issues a per-branch manifest from a shared template,
  collects all `BranchReport`s, ranks them, and emits one Trial Report.
- **`scoring.py`:** **mechanical, not an LLM judge.** A branch is *ranked only
  if it both passed the gates and met the objective* (proof passed) — correctness
  before economy. Survivors sort by diff economy (lines, then files) then cost
  (tokens, then wallclock). Pairwise Jaccard file-overlap measures whether
  strategies actually diverged. **Never auto-merges.**
- **`report.py`:** the human-scannable Trial Report (markdown + JSON): ranking
  table, per-branch status/proof/evidence/flags/cost, divergence, and a
  recommendation that explicitly states the harness does not merge.
- **`proposal_bridge.py` (the Track B → Track A hand-off):** converts each ranked
  **survivor** into a `CodeProposal` (`source=AGENT_BRANCH`, `status=PENDING`,
  priority from rank so rank-1 sorts to the top, risk via `classify_proposal` over
  the touched paths + the added diff lines, the supervisor diff + proof in the
  description) and stores it via `ProposalStore` — so a supervisor-verified diff
  appears in the GUI Proposals tab and is reviewed under the same
  acknowledge-before-approve gate as a self-generated proposal. Only ranked
  survivors are ingested (killed/rejected branches never become proposals). Opt-in:
  `run_portfolio(..., proposal_store=...)` or `python -m agent_branch.portfolio
  --demo --ingest`. Still **never merges** — the proposal is a PENDING review
  artifact. The Proposals tab tags these with a `🤖 AGENT-BRANCH ✓` provenance
  badge.

### 3.7 Workers (`agent_branch/workers/`, stdlib-only, bare image)

| Worker | Demonstrates |
|--------|--------------|
| `attacker.py` | M1 isolation: scripted escape attempts (RO-mount writes, secret reads, network egress, forbidden edits) |
| `benign_llm.py` | M1/M2 happy path: one bounded in-scope edit via the UDS proxy |
| `strategy_worker.py` | M2 divergence: deterministic surgical/refactor/testfirst diffs that all pass the same proof |
| `saboteur.py` | M2 reaper: deliberate config-tamper cheat, must be killed at the static gate |
| `failure_worker.py` | M2.5 sincere failure: overfit (`return 3`) / honest-wrong (`abs(a-b)`) that clear the gate but fail the deep proof |
| `llm_strategy_worker.py` | manual real-LLM demo of strategy directives |

### 3.8 Milestones

- **M1 — prove isolation (N=1):** a worker provably cannot escape. ✅
- **M2 — prove the reaper (N=3 + saboteur):** run diverse workers, kill the bad
  one at the gate, rank survivors, emit a Trial Report. Merging is **not** a
  goal. ✅
- **M2.5 — catch sincere failure:** a deepened proof test catches honest
  wrongness, not just sabotage (the reaper is only as strong as the proof's
  discriminating power). ✅
- **M3 — partially landed:** the **Track B → Track A bridge is done** —
  `proposal_bridge.py` turns ranked survivors into `ProposalSource.AGENT_BRANCH`
  proposals that flow into the GUI Proposals tab for human review (see §3.6).
  Still **deferred:** LLM intent judge (`eval_layer2` is a stub), reuse of the
  20-tool agentic controller, a real deps image, and an actual (still human-gated)
  merge path that applies an approved proposal's diff to main.

### 3.9 How to run

```bash
pytest tests/agent_branch/ -q                 # full suite (91 tests; slow ones need podman)
pytest tests/agent_branch/ -m "not slow" -q   # logic only, no containers
python -m agent_branch.supervisor --demo attacker   # M1 isolation demo
python -m agent_branch.portfolio  --demo            # M2 reaper demo
```

---

## 4. Consolidated threat model

| Threat | Defense | Where |
|--------|---------|-------|
| A generated proposal silently looks safe | Risk computed on every parse; safety/supervision → CRITICAL; import-based detection | `proposal_risk.py`, `_parse_proposal` |
| A reviewer approves a dangerous change on a glance | Risk/CORE badges + acknowledge-before-approve gate | `gui/tabs/proposals.py`, `requires_human_ack` |
| Worker patch forges the diff | Supervisor computes the diff itself from a trusted checkout | `compute_trusted_diff` |
| Worker `.git` executes host code | Git never runs against worker `.git`; hardened flags + sanitised env | `provisioning.py` |
| Nested `.git` masks content as a gitlink | `.git` stripped at all depths in the overlay | `_overlay_worktree` |
| Worker edits a test/guard/config to fake success | Static gate kills forbidden/safety/config-tamper touches before execution | `eval_static.py` |
| Worker runs zero proof tests for a free pass | No proof tests → unprovable → killed | `eval_sandboxed.py` |
| Worker exhausts/escapes the LLM budget or key | Tamper-proof proxy metering, path/host allowlist, key held supervisor-side | `llm_proxy.py` |
| Manifest tamper | Frozen + SHA-256 verified every pass | `manifest.py` |
| Auto-merge / push | There is no merge or push path anywhere | by construction |

---

## 5. Test inventory

- `tests/unit/test_code_proposal.py` (48) — model, supervision fields, registry.
- `tests/unit/test_proposal_risk.py` — classifier (prefix/import/CRITICAL) +
  `requires_human_ack` policy.
- `tests/unit/test_proposals_tab.py` (8) — GUI card rendering (risk/CORE badges,
  depends_on, AGENT-BRANCH provenance badge, safe defaults).
- `tests/unit/test_reclassify_proposals.py` (5) — re-classify extraction adapter.
- `tests/agent_branch/test_proposal_bridge.py` (10) — survivor→proposal mapping,
  risk classification, ranked-survivors-only ingest, dedup.
- `tests/agent_branch/` — isolation red-team (slow/podman), diff integrity,
  red-team hardening, eval gate, manifest, proxy, portfolio, proposal bridge.

---

## 6. What this system is *not* (current limitations)

- **Not autonomous.** Generation is user/session-triggered; nothing merges,
  commits, or pushes. Track A produces text plans + optional staged code only.
- **Connection is one-way and review-only.** Track B can ingest its ranked
  survivors into Track A's store (the `proposal_bridge`), so they surface in the
  GUI for human review — but nothing is auto-applied, and the reverse (a Track A
  proposal *executed* by agent_branch) is not wired.
- **`depends_on` is advisory**, not an enforced merge gate (no merge path exists
  to gate).
- **`eval_layer2` (LLM intent judge) is a stub** — it never kills today.
- **No finite proof test is complete.** The reaper elevates the cheapest *correct*
  survivor; human review of the #1 survivor covers the residue.
```
