# Claim-support trace for the 2026-09-02 repository assessment

## Purpose

This is a claim-level provenance record for selected statements in the repository
assessment immediately preceding this file. It is meant to distinguish:

- what was directly observed in source or local data;
- what was computed from those observations;
- what was inferred from incomplete evidence;
- what was an evaluative judgment rather than a factual result; and
- what the inspection could not establish.

The commands below were run from `/home/lukeh/Daemon_v1` on 2026-09-02. The
original repository assessment did not inspect personal-memory contents; the
later current-session addendum narrowly inspected the specific records named by
the owner so their provenance could be traced. Secret/identifier values are not
reproduced. Runtime results describe this checkout and machine at that time;
they are not universal claims about every installation.

## Evidence labels

| Label | Meaning |
|---|---|
| `DIRECT-SOURCE` | Read directly from a tracked source/configuration file. |
| `DIRECT-RUNTIME` | Observed by executing a local command or test. |
| `DERIVED` | Computed mechanically from direct observations. |
| `INFERENCE` | Best explanation of direct evidence, but not itself directly observed. |
| `JUDGMENT` | An appraisal requiring human criteria; not a repository fact. |
| `NOT-ESTABLISHED` | The available evidence cannot support the broader claim. |

## Summary matrix

| ID | Claim under examination | Evidence class | Verdict |
|---|---|---|---|
| C01 | The repository is a substantial implemented system, not an empty scaffold. | `DIRECT-SOURCE`, `DERIVED` | Supported. Size alone does not establish quality. |
| C02 | Daemon is a working single-user application. | `DIRECT-SOURCE`, `DIRECT-RUNTIME`, `INFERENCE` | Implemented and populated; not fully smoke-tested in this audit. “Working” is only partially verified. |
| C03 | It has been genuinely used against real personal data. | `DIRECT-RUNTIME`, `INFERENCE` | Strongly suggested by populated stores and dated logs, but organic human use was not independently authenticated. |
| C04 | The current checkout has 8,035 collected tests, but this audit did not prove the full suite passes. | `DIRECT-RUNTIME` | Supported exactly. |
| C05 | The current Docker packaging is stale/broken. | `DIRECT-SOURCE`, `DIRECT-RUNTIME`, `DERIVED` | Supported for this checkout. |
| C06 | There is no current desktop distribution artifact in this checkout. | `DIRECT-RUNTIME`, `DIRECT-SOURCE` | Supported locally; does not disprove a historical or remote release artifact. |
| C07 | The application is local-first but not fully local/offline for answer generation. | `DIRECT-SOURCE`, `DIRECT-RUNTIME` | Supported for the active configuration. No live provider request was traced. |
| C08 | Tests are not completely isolated from live runtime data. | `DIRECT-RUNTIME`, `INFERENCE` | Supported as a defect; precise causation is high-confidence temporal inference. |
| C09 | There is no outside adoption or independent validation. | `NOT-ESTABLISHED` | Too broad. Only “none was evidenced inside the inspected checkout” is supported. |
| C10 | The project demonstrates strong-junior or some mid-level engineering instincts. | `JUDGMENT` | Opinion based on selected artifacts, not a measurable repository fact. |
| C11 | The checkout is clean, two commits ahead of its GitHub-tracking ref, and ordinary `git push origin` is redirected locally. | `DIRECT-RUNTIME` | Supported at inspection time. |
| C12 | The full Wikipedia retrieval assets expected by the active default path were unavailable here. | `DIRECT-SOURCE`, `DIRECT-RUNTIME` | Supported for this machine at inspection time. |
| C13 | The repository's licensing signals conflict. | `DIRECT-SOURCE` | Supported. |

## Current-session integrity addendum (20:17 session)

This addendum traces four failures observed in the later Fable/Claude relay
session. It deliberately omits the owner's GTID, phone number, and other
unnecessary personal values.

### T01 — Full-prompt export crosses the privacy boundary without redaction

**Observed claim**

> The exported debug prompt contains the owner's GTID and phone number in clear
> text because recent conversation turns are copied into the prompt.

**Verdict: `DIRECT-SOURCE`, `DIRECT-RUNTIME` — supported.**

- [`api/routes/debug.py`](../api/routes/debug.py#L52) returns the stored query and
  full context prompt verbatim in a downloadable text response.
- [`web/src/components/debug/DebugPage.tsx`](../web/src/components/debug/DebugPage.tsx#L88)
  likewise copies the raw query, prompt, system prompt, and response.
- The legacy path in [`gui/launch.py`](../gui/launch.py#L1462) writes the same
  material to `/tmp/daemon_prompts` without redaction.
- The recent-conversation source contains both identifiers. Their values are not
  reproduced here.
- `daemon_debug.log` also contains raw provider request payloads, so the same
  prompt material exists in a local debug log independently of manual export.

The Git history scrub and pre-commit privacy hook address source-control
exposure. They do not sanitize the runtime conversation corpus, assembled
prompts, debug records, logs, clipboard copies, or downloaded prompt exports.
This is therefore a separate outbound-data-loss-prevention boundary, not a
failure of `git filter-repo`.

### T02 — The response planner did not receive the answering model's context

**Observed claim**

> “The floor is yours, communicate with Fable” was planned as an invitation for
> the user to describe their Fable experience.

**Verdict: `DIRECT-SOURCE`, `DIRECT-RUNTIME` — supported.**

The recorded plan said to encourage the user to share insights from the Fable
session and offer help with Fable. That reverses the requested direction of
communication.

The planner and answering model did not see the same inputs:

- [`core/response_planner.py`](../core/response_planner.py#L135) builds its input
  from the literal query, classified intent/tone/topics, thread depth, and at
  most 400 characters from each side of the preceding exchange.
- [`core/orchestrator.py`](../core/orchestrator.py#L1350) starts that planner in
  parallel with full prompt gathering. The planner therefore cannot inspect the
  retrieved conversations, profile, graph, notes, emails, or temporal narrative
  being assembled for the answer.
- Runtime evidence shows the planner model was `openai/gpt-4o-mini`. Kimi-3 was
  the main answering model. Kimi-3 received the erroneous plan plus the full
  assembled prompt and nevertheless addressed Fable directly.
- [`core/orchestrator.py`](../core/orchestrator.py#L1405) retains only plan count
  and tone in ordinary turn telemetry; it does not preserve the plan's actual
  points. The exact bad plan was recoverable only because raw debug logging
  captured a later provider request containing the injected system prompt.

This particular error was not caused solely by missing retrieval context: the
literal query itself was sufficient to infer the imperative. It is a planner
instruction-following failure made more dangerous by a divergent context path
and weak audit persistence.

### T03 — Profile/graph provenance contains several distinct contamination modes

**Observed claims and verdicts**

| Item | Verdict | Trace |
|---|---|---|
| `lived_in=Atlanta` | Fact supported; displayed evidence misleading | The same long user turn ends by saying the user moved to Atlanta. The stored excerpt shows unrelated opening lyrics because the pipeline truncates the selected whole message from character zero. |
| `relationship=Sarah` | Partially supported; relation is broader than the exact wording | The same turn says the move was “with Sarah” and that “she left.” The displayed lyric excerpt still does not show that support. |
| `laundry_done` under career | Misclassification confirmed | [`data/category_cache.json`](../data/category_cache.json) persistently maps `laundry_done`, `laundry_status`, and `laundry_time_left` to `career`; [`data/user_profile.json`](../data/user_profile.json#L4780) follows that cached mapping. |
| Mochi/Waffles as dogs | Unsupported relation confirmed; stale contamination | The source turn only says the user played with them. The shutdown LLM invented `has_dog`; older graph metadata identifies them as cats. Current source and graph guards were added later, but the stale facts/edges remain injectable. |
| WHOOP | Quoted-model contamination, not established user testimony | Its source is an earlier user message quoting another model's profile comparison (“you use WHOOP”). The extractor treated text contained in a user turn as an assertion by the user. No source excerpt was retained with the fact. A prior junk-candidate scan flagged both the boolean and canonical versions, but the canonical fact remains in live Chroma and its graph edge still surfaces. |
| D&D and running | Synthetic test contamination confirmed | Both facts have Chroma source `test_calibration`. [`scripts/generate_test_facts.py`](../scripts/generate_test_facts.py#L3) intentionally creates synthetic personal facts and, at lines 137-147, imports the active `CHROMA_PATH` and graph paths. The live facts collection currently contains 48 records carrying that source. |

The Atlanta/Sarah evidence mismatch follows directly from
[`memory/llm_fact_extractor.py`](../memory/llm_fact_extractor.py#L606): it picks a
whole conversation pair by keyword overlap, combines user and Daemon text, then
stores `best_msg[:200]`. [`memory/user_profile.py`](../memory/user_profile.py#L245)
truncates the excerpt to 200 characters again. It stores neither a source turn
ID nor a span surrounding the supporting words.

`fact_verification=ON` does not mean evidence verification. The verifier's fast
path stores a fact when no existing candidate exists
([`memory/fact_verification.py`](../memory/fact_verification.py#L123)), and its
candidate filter only compares identical subject and predicate
([`memory/fact_verification.py`](../memory/fact_verification.py#L342)). It does
not test whether the source entails the triple, distinguish quoted third-party
text from user assertion, or recognize cross-predicate contradictions such as
`has_cat` versus `has_dog`. The truth scorer is likewise a source/confirmation/
decay score, not an independent entailment check.

### T04 — Temporal synthesis preserved one duration and lost the competing one

**Observed claim**

> The final prompt contains “six days stable” in temporal grounding while a
> recent Goldsman email says “two weeks” functional.

**Verdict: `DIRECT-RUNTIME` — supported, with a semantic qualification.**

- The user directly described the “last 6 days” as evidence of recovery in the
  session transcript.
- A later email draft in that same session described being consistently
  functional for two weeks.
- The September 2 daily note generated at 18:02 retained only “six consecutive
  stable days” / “Day 6 stable.”
- [`data/narrative_context.txt`](../data/narrative_context.txt#L5), generated one
  minute later, synthesized that daily note into “six days stable.”
- [`memory/memory_consolidator.py`](../memory/memory_consolidator.py#L522) builds
  temporal grounding from monthly, weekly, and daily notes, not from the raw
  current-session transcript. Once the daily note omitted the two-week statement,
  the downstream narrative had no opportunity to reconcile it.

“Stable” and “functional” are not necessarily identical predicates, so the two
durations can coexist. The actual defect is that the pipeline silently flattened
the distinction and injected both claims into one prompt without marking or
reconciling the discrepancy.

### Addendum conclusion

These incidents show that the system currently maintains source *labels* more
reliably than source *boundaries*. It can name a component as profile, graph,
planner, or temporal grounding while still allowing synthetic fixtures, quoted
model text, stale derived claims, misleading excerpts, and raw PII to cross
those boundaries. The continuity is therefore not yet auditable enough to treat
an `ON` feature flag or confident narrative as verification.

### Remediation status (2026-09-02, later session — code fixes up to the human-gated store cleanup)

All four addendum failures have code fixes on disk (uncommitted; the running
daemon started 20:17 pre-fix and needs a restart to pick them up). Nothing in
the live Chroma/profile/graph stores was modified; every store change below is
a dry-run candidate awaiting the owner.

| Item | Fix | Evidence |
|---|---|---|
| T01 privacy boundary | `utils/privacy_redaction.py` (structured PII + credentials, deterministic, idempotent) applied at every share surface: `/api/debug`, `/api/provenance`, prompt TXT export (API + Gradio), the SSE `complete` debug payload, SPA clipboard copy (`web/src/utils/privacy.ts`). Transport loggers (`openai`/`httpx`/`httpcore`/`urllib3`) get a WARNING floor + handler filter so the DEBUG file sink no longer persists provider request bodies; raw bodies need BOTH `DAEMON_MODE=dev` and `DAEMON_ALLOW_SENSITIVE_HTTP_LOGS=1`. | `tests/unit/test_privacy_redaction.py`, `test_api_debug_settings.py`, `test_api_chat.py`, `test_test_mode_isolation.py`. **Still open (owner):** pre-fix `daemon_debug*.log` archives on disk contain raw request bodies (one such line surfaced during this session's log grep). |
| T02 planner alignment | Gather first, then plan: `build_full_prompt` hands the planner a bounded digest of the exact `prompt_ctx` (`ResponsePlanner.build_context_digest`); explicit direct-communication commands are deterministically locked (`directive_locked`, no LLM call); the exact operative plan + digest hash + section names persist in the debug record, provenance, and telemetry (`response_plan`); the parallel 35s-wait/grace machinery is gone. | `tests/unit/test_response_planner.py`, `test_process_user_query.py`, `test_turn_telemetry.py`, `test_stream_artifacts.py`. |
| T03 fact provenance | `memory/fact_source.py`: Daemon responses excluded from the extraction prompt; every triple must join to a USER-authored span (quoted/reported text, transcripts, blockquotes, fences never count; user-scoped facts need a user-owned span; relation names that make a claim need a cue — pet ownership needs the possessive species phrase; no last-message fallback); `source_excerpt` is the claim-bearing sentence on both paths, with `source_role/source_turn_id/source_turn_index/source_support/source_anchor` forwarded to Chroma. Household chores are ephemeral and never CAREER (`_HOUSEHOLD_ACTIVITY_RE` ahead of the poisoned cache). `generate_test_facts.py` is sandbox-only. | `tests/unit/test_fact_source.py` (23), `test_source_excerpt_pipeline.py`, `test_llm_fact_extractor_comprehensive.py`, `test_chore_relation_hygiene.py`, `test_calibration_data_isolation.py`. **Human-gated store cleanup (dry-runs written, Daemon must be DOWN):** `scripts/purge_calibration_facts.py --apply` (48 `test_calibration` facts + 35 fully-synthetic graph edges; 24 would-be-orphan nodes listed for `graph_junk_cleanup.py`, never auto-removed — some synthetic edges hang off REAL entities, e.g. invented attributes on the brother and cat nodes); `scripts/quarantine_facts.py --from-file data/audit_fact_quarantine_candidates_20260902.jsonl --apply` (WHOOP + the two `has_dog` facts, reversible); `scripts/purge_profile_facts.py --from-file data/profile_audit_candidates_20260902.txt --apply` (the two `has_dog` quick-profile facts). The `laundry_done`-class profile facts need no data action: they are now ephemeral at read time and re-categorize deterministically. |
| T04 temporal claims | `daily_notes_generator.build_temporal_claim_audit` (mechanical status+duration extraction, draft/conditional labeling, PII-redacted) feeds the daily-note prompt; daily + narrative prompts must keep each predicate's own duration and surface discrepancies. | `tests/unit/test_temporal_claim_audit.py`. |

**Rehearsal of the three human-gated apply paths (`DIRECT-RUNTIME`, 2026-09-02 22:11-22:12, sandbox only):**
a sandbox store was built with `generate_test_facts.py --sandbox-dir` (57 synthetic facts, 47 nodes, 44 edges,
paths verified to resolve under the scratch directory), then each script's `--apply` ran against it with the
project interpreter: `quarantine_facts.py` flipped one fact (`is_quarantined` True with reason) and `--undo`
restored it; `purge_calibration_facts.py` dry-run selected 57/44/0-mixed, apply deleted 57 facts and removed 44
edges, nodes untouched (47), graph reloaded cleanly, pre-image backups written; `purge_profile_facts.py` on a copy
of the live profile matched 2/2 candidate ids, apply removed exactly 2 of 2565 facts, the copy reloaded, and the
live file's hash was identical before and after. Ops gotcha found on the way: outside the repo directory pyenv
falls back to system Python 3.13 with ChromaDB 0.6.3, which cannot read a 1.0.7 store (`KeyError: '_type'`) —
run scripts from the repo root or with the explicit 3.11.8 interpreter.

**Applied (`DIRECT-RUNTIME`, 2026-09-02 22:21, owner-authorized for this batch, Daemon down, no other
process holding the stores, live dry-runs re-examined first):** `purge_calibration_facts.py --apply` deleted
48 facts (3462→3414, zero `test_calibration` remaining) and removed 35 synthetic-only edges (1016→981; 767 nodes
and the alias file unchanged by hash; graph reloads via `GraphMemory`); `quarantine_facts.py --apply` flagged the
three audit-named facts with reason "provenance audit 2026-09-02"; `purge_profile_facts.py --apply` removed the
two `has_dog` quick-profile facts (2565→2563, reloads via the deployed loader). Pre-images:
`data/backups/purge_calibration_facts_preimage_20260902_222119.jsonl`,
`purge_calibration_graph_preimage_20260902_222119.json`, `quarantine_facts_preimage_20260902_222139.jsonl`,
`user_profile_preimage_20260902_222147.json`. Still open for the owner: the 24 orphan-node candidates
(`data/calibration_orphan_node_candidates_20260902_222029.txt` → `graph_junk_cleanup.py`), the pre-fix
`daemon_debug_20260902_*.log` archives holding raw prompts, restart, commit.

Not addressed (by design or out of scope): entailment verification inside
`fact_verification` (the provenance layer runs at extraction, before
verification); re-verification of historical facts whose stored excerpts are
the old head-of-turn windows (no source turn ids exist for them); the
`categories` filing of existing profile facts.

## Original detailed traces

### C01 — Substantial implementation rather than empty scaffolding

**Statement assessed**

> “You have a real, unusually substantial solo engineering/research prototype.”

**Direct/derived support**

```text
$ git ls-files | wc -l
948

$ git ls-files '*.py' | rg -v '^tests/' | wc -l
404

$ git ls-files '*.py' | rg -v '^tests/' | xargs wc -l | tail -1
151712 total

$ git ls-files 'tests/**/*.py' 'tests/*.py' | sort -u | xargs wc -l | tail -1
101309 total

$ git rev-list --count HEAD
277
```

Representative implemented surfaces include:

- [`main.py`](../main.py): application construction and entry points;
- [`api/`](../api/): FastAPI application and routes;
- [`web/src/`](../web/src/): React/TypeScript frontend;
- [`core/agentic/`](../core/agentic/): tool-selection and execution loop;
- [`core/prompt/`](../core/prompt/): parallel context retrieval and prompt assembly;
- [`memory/`](../memory/): storage, scoring, truth, graph, curation, and retrieval;
- [`eval/`](../eval/): prompt snapshot, variant, judging, and objective-check harnesses.

**Boundary**

File and line counts establish scope, not correctness, originality, maintainability,
or how much was written with assistance. The adjective “substantial” is a reasonable
size/scope judgment; it is not a quality metric.

### C02 — “Working single-user application”

**Statement assessed**

> “A working single-user personal AI application—not merely prompts or a chatbot wrapper.”

**Support**

- `pyproject.toml` describes a single-user desktop application.
- [`main.py`](../main.py) contains CLI, wizard, legacy GUI, and FastAPI launch paths.
- [`api/routes/chat.py`](../api/routes/chat.py) exposes chat/session operations.
- [`web/src/api/useChatStream.ts`](../web/src/api/useChatStream.ts) implements the
  browser streaming path.
- The persistent stores counted under C03 are populated.
- Focused tests and frontend type-checking passed under C04.

**Boundary**

The audit did **not** launch the full application, issue a chat turn, verify its health
endpoint, or complete an end-to-end response through a live LLM provider. Therefore:

- “implemented application with evidence of prior operation” is directly supported;
- “the current checkout starts cleanly and completes a live turn today” is
  `NOT-ESTABLISHED` by this audit.

This is an important tightening of the wording in the original assessment.

### C03 — Populated/used system

**Statement assessed**

> “This has also clearly been operated against real data.”

**Observed counts (content was not printed)**

```text
data/corpus_v4.json:                         2,001 entries
data/chroma_db_v4/chroma.sqlite3:           14 collections
data/chroma_db_v4/chroma.sqlite3:           48,691 embedding rows
data/knowledge_graph.json:                  767 nodes, 1,016 edges
conversation_logs/:                         1,411 files
data/backups/:                              166 files after the test run
```

Reproduction commands:

```bash
jq 'length' data/corpus_v4.json
sqlite3 data/chroma_db_v4/chroma.sqlite3 'select count(*) from collections;'
sqlite3 data/chroma_db_v4/chroma.sqlite3 'select count(*) from embeddings;'
jq '.nodes|length, .edges|length' data/knowledge_graph.json
find conversation_logs -type f | wc -l
```

**Boundary**

These observations prove that the local state is populated and longitudinally dated.
They do not independently prove who generated every record, whether all records came
from organic conversations, or whether every stored item is correct. “Genuinely used”
is a strong inference, not cryptographic proof.

### C04 — Tests and verification status

**Statement assessed**

> “Exactly 8,035 collected test cases,” while not claiming that every test passed.

**Direct results**

```text
$ HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest --collect-only -q
8035 tests collected in 8.91s

$ python -m ruff check .
All checks passed!

$ cd web && npm run typecheck
tsc --noEmit
exit 0
```

Focused Python results:

```text
tests/unit/test_sep02_checkin_fixes.py       20 passed
tests/unit/test_file_access_manager.py       46 passed
tests/unit/test_git_stats.py                 56 passed
tests/agent_branch/test_llm_proxy.py          7 passed
```

The proxy tests initially failed because the analysis sandbox prohibited binding a
localhost socket. Rerunning that file with socket permission produced `7 passed`.

For `tests/test_prompt_internal_methods.py`, forcing offline model access produced:

```text
16 passed, 2 skipped, then execution remained in test_gather_context
until the external 90-second process timeout; the final test was not reached.
```

The CI configuration itself states that its fast job omits 17 heavyweight files; see
[`tests.yml`](../.github/workflows/tests.yml). No `.coverage`, `htmlcov/`, or
`coverage.json` artifact was present.

**Boundary**

- Test *collection* is not test *success*.
- Focused successes do not imply full-suite success.
- A timeout is not automatically an assertion failure, but an unmarked unit test that
  cannot complete within 90 seconds is a developer-harness problem.
- Ruff uses the intentionally narrow rule set in [`pyproject.toml`](../pyproject.toml),
  so “lint passes” must not be generalized to “the code passes comprehensive static
  analysis.”

### C05 — Docker packaging

**Statement assessed**

> “The present Dockerfile is stale/broken.”

**Direct support**

[`Dockerfile`](../Dockerfile) contains:

```dockerfile
COPY --chown=daemon:daemon personality/ /app/personality/
```

but this checkout has no `personality/` path:

```text
$ test -e personality; echo $?
1
```

The Dockerfile copies `config`, `core`, `memory`, `models`, `utils`, `gui`,
`processing`, `knowledge`, and `integrations`, but not the current `api/` directory or
the built `web/dist/` frontend. It exposes and probes port 7860. By contrast,
[`main.py`](../main.py) routes the default non-legacy GUI through FastAPI/uvicorn, whose
documented/default port is 8000.

**Derivation**

A Docker build using this context should fail at the missing `COPY personality/` before
runtime. If that line were removed, the current default FastAPI/frontend path would
still be incomplete in the image, and its port/health configuration would remain stale.

**Boundary**

Docker itself was not installed on this machine, so a literal `docker build` was not
run. The missing build-context source is sufficient to establish the immediate build
failure under ordinary Docker semantics.

### C06 — Desktop build/release status

**Statement assessed**

> “No current desktop executable or installer is present; the recorded old build needs rebuilding.”

**Direct support**

```text
$ find . -maxdepth 3 -type f \( -iname '*.exe' -o -iname 'Daemon' -o -iname '*.AppImage' \)
<no output>

$ git tag --sort=-creatordate
v1.0.1
v1.0.0
```

[`docs/GOALS.md`](GOALS.md) records the Windows executable as previously shipped and
says its spec needs updates and a clean end-to-end rebuild. [`installer/README_INSTALLER.md`](../installer/README_INSTALLER.md)
documents a build procedure but is not a built artifact.

**Boundary**

This proves only that the current local checkout lacks the artifact. It does not prove
that no executable exists in a private GitHub release, another directory, an archive,
or a recipient's machine.

### C07 — Local-first, not fully local/offline

**Statement assessed**

> “Daemon stores memory locally but substantially relies on outside APIs for answer generation and tools.”

**Direct support**

- [`requirements.txt`](../requirements.txt) declares the OpenAI-compatible client,
  Tavily, E2B, and Google authentication dependencies.
- [`config/config.yaml`](../config/config.yaml) selects an API-served active model
  (`kimi-3`) and enables web/agentic functionality.
- `.env` contained non-empty keys for multiple model/tool providers. Only key names and
  `SET`/`EMPTY` status were inspected; values were not printed.
- Local Hugging Face caches contained BGE, MiniLM, cross-encoder, CLIP, and several
  generation-model directories.

**Boundary**

The audit did not trace a live answer request or network packet. It therefore establishes
configured dependencies and local model assets, not which provider handled the most
recent real conversation. “Not fully offline under the active configuration” is
supported; “every response leaves the machine” would be too broad.

### C08 — Test isolation defect and audit side effects

**Statement assessed**

> “Tests are not completely isolated from live runtime data.”

**Direct observations during the test run**

- `data/backups/20260902_200727/manifest.json` appeared with reason `shutdown`.
- `data/corpus_v4.json` and `data/curation_queue.json` received new modification times,
  although they remained byte-for-byte identical to the new backup copies.
- `logs/curation_audit.jsonl` received a new modification time and contained 176 lines.
- `data/knowledge_graph.json` changed byte serialization. Comparison found 63 changed
  scalar positions, all inside node `aliases` arrays. Sorting every alias array made the
  current graph identical to the backup, so no semantic node/edge change was detected.
- No Daemon/pytest Python process remained after the timed commands.

The prompt test fixture uses temporary corpus and Chroma paths, but its builder also
initialized the default profile path (`data/user_profile.json`) and a broad production-like
context pipeline. See [`tests/test_prompt_internal_methods.py`](../tests/test_prompt_internal_methods.py).

**Boundary**

The timing, the `shutdown` manifest, and the absence of another Daemon process make the
test run the high-confidence cause. This was not proven with syscall-level tracing, so
the causal statement remains `INFERENCE`, while the filesystem modifications themselves
are `DIRECT-RUNTIME`.

No cleanup was performed because these are ignored user-data paths and deleting or
rolling them back without explicit authorization would be unsafe.

**Correction (later session, same day — `DIRECT-RUNTIME`):** the archived daemon
log `daemon_debug_20260902_200727.log` contains the backup manager's own
`[Backup] Wrote data/backups/20260902_200727` line at 20:07:27, i.e. the backup,
the curation scan, and the store re-saves were the DAEMON's shutdown (the owner
restarted it; the replacement instance started 20:17:07). No test file imports
`main.py` or calls `run_shutdown_backup`. The inference above is therefore
refuted for this event. The isolation guards were still added as defence in
depth: `run_shutdown_backup` is skipped under `DAEMON_TEST_MODE`, and the
curation journal/queue defaults redirect to test-only files when a test process
would otherwise write the prod paths (`tests/unit/test_test_mode_store_guards.py`).

### C09 — Outside adoption and independent validation

**Statement assessed**

> “You do not have strong outside validation.”

**What was actually observed**

- The user stated that the GitHub repository is private.
- The checkout contains no contributor guide or in-repository record of external users.
- Git shortlog showed three identities that appear to be aliases of the same named
  contributor.
- Benchmark documentation and data are maintained inside the same repository.

**Boundary / correction**

The checkout cannot establish that no external users, reviewers, private collaborators,
downloads, or independent evaluations exist. The supportable wording is:

> “This inspection found no in-repository evidence of outside adoption or independent
> validation.”

The broader negative claim is `NOT-ESTABLISHED` and would require evidence from the
private hosting account, release distribution records, user interviews, or independent
reports.

### C10 — Professional-level appraisal

**Statement assessed**

> “Strong evidence for junior backend/AI-application engineering ability, with some
> mid-level systems instincts, but not evidence of senior engineering.”

**Inputs to the judgment**

- architectural decomposition and multiple runnable surfaces;
- regression tests tied to recorded incidents;
- safety, approval, backup, and audit mechanisms;
- measured retrieval and evaluation infrastructure;
- packaging drift, incomplete test isolation, configuration sprawl, and retained
  experimental systems;
- no inspected evidence of team delivery or external production ownership.

**Boundary**

This is explicitly a `JUDGMENT`, not a tool-derived fact. Repository inspection cannot
measure the author's unaided understanding, teamwork, communication, or performance in
an employment setting. A defensible evaluation would also require a code walkthrough and
a live scoped change performed by the author.

### C11 — Git state and backup risk

**Statement assessed**

> “The tree is clean, but master is two commits ahead and ordinary origin pushes are redirected locally.”

**Direct results**

```text
$ git status --short
<no output>

$ git branch -vv
* master 1fcac11 [origin/master: ahead 2] ...

$ git log --oneline origin/master..HEAD
1fcac11 Check-in-turn fixes: ...
64b21cf Repo-audit grounding: ...

$ git remote -v
origin  https://github.com/lukehalleran/ai-assistant-framework.git (fetch)
origin  /home/lukeh/daemon_cleanup/PUSH_DISABLED_PHASE6_STAGING (push)
```

**Boundary**

This establishes local Git configuration and refs at inspection time. It does not prove
that the GitHub server lacks the objects through another ref or remote, because no fetch
or hosting-service query was performed.

### C12 — Wikipedia assets and active path

**Statement assessed**

> “The full Wikipedia index expected by the active path is currently unavailable here.”

**Direct support**

[`knowledge/semantic_search.py`](../knowledge/semantic_search.py) defaults to:

```text
/run/media/lukeh/T9/wiki_data/vector_index_ivf.faiss
/run/media/lukeh/T9/wiki_data/metadata.parquet
```

Both files were absent during inspection. The workspace did contain:

```text
data/wiki/enwiki-latest-pages-articles.xml    108,971,089,504 bytes
data/vector_index_ivf.faiss                     817,963,475 bytes
```

No matching local Wikipedia metadata parquet was found beneath `data/`. Test diagnostics
also recorded `index_exists=False, meta_exists=False` for the configured external paths.

**Boundary**

The raw XML dump is source material, not by itself a queryable vector index. The local
FAISS file may be useful with its matching metadata elsewhere, but that complete pair was
not available through the inspected configuration. This does not mean Wikipedia fallback
or web search is absent.

### C13 — Licensing conflict

**Statement assessed**

> “Distribution rights are not documented consistently.”

**Direct support**

- [`pyproject.toml`](../pyproject.toml) declares `license = { text = "Proprietary" }`.
- [`installer/LICENSE.txt`](../installer/LICENSE.txt) contains the MIT License.
- No root `LICENSE` or `LICENSE.md` was present.

**Boundary**

This establishes inconsistent repository metadata, not a legal conclusion about
ownership or the enforceability of either notice. Resolving that requires the owner to
choose the intended license and review third-party distribution obligations.

## Overall epistemic result

The prior assessment was mostly grounded in reproducible local evidence, but several
sentences compressed evidence and interpretation too aggressively:

1. “Working application” should have been narrowed to “implemented, populated application
   with focused tests passing; full live-turn startup was not verified.”
2. “Genuinely used” is a strong and reasonable inference from longitudinal populated data,
   not direct authentication of the provenance of every record.
3. “No outside validation” cannot be concluded from a private local checkout; only absence
   of evidence in the inspected sources was established.
4. Career-level descriptions were judgments and should always be presented as such.

Those qualifications are not cosmetic. They are the line between a provenance trace and
a persuasive narrative that merely sounds continuous.
