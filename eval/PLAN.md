# Daemon Prompt Section Ablation & Eval System — Full Plan

> **Guiding Principle:** The measurement infrastructure is the product first. The eval
> system produces reports and recommendations. It must not auto-patch production prompt
> behavior. Production policy changes come later, after human review and shadow-mode
> validation.

---

## Current Status (2026-05-09)

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Instrumentation & Snapshot Replay | **COMPLETE** | 89 tests passing, live capture verified |
| Phase 2: Variant Generation + Query Corpus + Utilization | **COMPLETE** | 75 tests passing, 4 strategies, 27-query seed corpus |
| Phase 3: *(merged into Phase 2)* | — | Query corpus and utilization bundled with variants |
| Phase 4: Generation Harness | **COMPLETE** | 24 tests passing, dry-run verified against 13 snapshots |
| Phase 5: Pairwise Judge Harness | **COMPLETE** | 25 tests passing, position randomization, blind judging |
| Phase 6: Objective Checks | **COMPLETE** | 33 tests passing, 5 automated checks, run against 1849 responses |
| Phase 7: Aggregation & Reporting | **PARTIAL** | Aggregation logic integrated into Phase 5-6 runners; no dedicated module |
| Phase 8: Prompt Policy & Shadow-Mode | **PARTIAL** | Intent-conditioned gating wired, shadow-mode deferred |

### Phase 1 Deliverables (Implemented)

```
eval/
  __init__.py              # Package init
  PLAN.md                  # This file
  README.md                # Phase 1 usage guide
  schema.py                # Pure data models (PromptSnapshot, SectionSnapshot, etc.)
  section_registry.py      # 28-entry canonical section registry
  snapshots.py             # SnapshotCapture, SnapshotReplay, save/load
  no_store_generation.py   # EvalGenerator + EvalGenerationConfig
  persistence_guard.py     # PersistenceGuard + file/collection fingerprinting
  snapshots/               # Saved snapshot JSON files (.gitignored)
    .gitignore

tests/test_eval/
  __init__.py
  test_section_registry.py   # 29 tests
  test_snapshots.py          # 13 tests
  test_replay_hash.py        # 14 tests
  test_no_store_generation.py # 17 tests
  test_persistence_guard.py  # 16 tests
```

### Phase 2 Deliverables (Implemented)

```
eval/
  variants.py              # VariantGenerator: LOO, AOI, bundle, reorder strategies
  corpus.py                # CorpusManager + 27-query seed corpus (3 per IntentType)
  utilization.py           # UtilizationAnalyzer: section presence/token stats
  run_phase2.py            # CLI runner: snapshot dir -> variant manifest + utilization report

tests/test_eval/
  test_variants.py           # 36 tests
  test_corpus.py             # 24 tests
  test_utilization.py        # 15 tests
```

**No production code modified.** All Phase 2 code reads from Phase 1 snapshots.

**Production code modified:**
- `core/prompt/builder.py` — added `_maybe_capture_eval_snapshot()` gated hook
  (lines 132-229). Disabled by default (`DAEMON_EVAL_CAPTURE=0`). Zero overhead
  when disabled.

**Key design decisions made:**
- System prompt is tracked via hash in provenance (separate from context prompt)
- `_assemble_prompt()` `directives` param is dead code — system prompt composed in orchestrator
- STM marked ablatable (prompt valid without it)
- `time_context` marked structurally required (temporal grounding always needed)
- Builder hook captures post_hygiene layer only (raw_retrieval requires pre-hygiene hook which would be more invasive)
- Uses `dataclass` (not Pydantic) for eval schema — decoupled from Daemon internals
- Replay uses stored `formatted_text` directly — never re-runs formatter logic

---

## Table of Contents

- [Overview](#overview)
- [Daemon Prompt Architecture (Reference)](#daemon-prompt-architecture-reference)
- [Phase 1: Instrumentation & Snapshot Replay](#phase-1-instrumentation--snapshot-replay)
- [Phase 2: Variant Generation](#phase-2-variant-generation)
- [Phase 3: Query Corpus](#phase-3-query-corpus)
- [Phase 4: Generation Harness](#phase-4-generation-harness)
- [Phase 5: Pairwise Judge Harness](#phase-5-pairwise-judge-harness)
- [Phase 6: Objective Checks](#phase-6-objective-checks)
- [Phase 7: Aggregation & Reporting](#phase-7-aggregation--reporting)
- [Phase 8: Prompt Policy & Shadow-Mode Rollout](#phase-8-prompt-policy--shadow-mode-rollout)

---

## Overview

Daemon's prompt is assembled from 27 optional context sections, a separately-composed
system prompt, and the user's query. The eval system answers: **which sections actually
help, which hurt, and under what query conditions?**

The system works in phases:

```
Phase 1: Capture snapshots of real prompts → replay them deterministically
Phase 2: Generate ablation variants (drop/replace/reorder sections)
Phase 3: Build a diverse query corpus covering all intent types
Phase 4: Run variants × queries through side-effect-free LLM generation
Phase 5: Judge response pairs (baseline vs variant) for quality
Phase 6: Run objective checks (hallucination, citation accuracy, etc.)
Phase 7: Aggregate results into section-level scorecards
Phase 8: Human-reviewed prompt policy changes, shadow-mode validated
```

Each phase depends on the previous. No phase auto-modifies production behavior.

---

## Daemon Prompt Architecture (Reference)

This section documents the actual prompt structure as of 2026-05-04, derived from
reading `core/prompt/formatter.py:_assemble_prompt()` (moved from builder.py during Phase 2/3 god-object decomposition).

### System Prompt (Separate from Context)

The system prompt is **not** part of the assembled context string. It is composed in
`core/orchestrator.py:build_full_prompt()` and passed as a separate `system_prompt`
parameter to the LLM API.

**Composition:**
1. `config/prompts/default_personality.txt` (or `custom_personality.txt` if exists)
2. `config/prompts/operating_principles.txt`
3. Identity placeholder substitution (`{USER_NAME}`, `{PRONOUN_*}`)
4. Citation instructions (if enabled)
5. Topic context injection
6. Thread surfacing instructions (first message only)
7. Response mode instructions (best-of, agentic, etc.)
8. Escalation strategy instructions (if crisis state)
9. Session header instructions (first message only)
10. Thinking instructions (if model lacks native reasoning)
11. Response plan injection (if planner enabled)

**Note:** The `directives` parameter in `_assemble_prompt()` is accepted but unused.
The formatter's `_load_directives()` (which reads `core/system_prompt.txt`) is legacy.

### Context Prompt Assembly Order

These are the exact sections emitted by `_assemble_prompt()`, in order. Each section
is conditional — only emitted when its context field is non-empty.

| Order | Header | Context Field | Category |
|-------|--------|---------------|----------|
| 1 | `[RECENT CONVERSATION]` | `recent_conversations` | retrieved |
| 2 | `[RELEVANT MEMORIES]` | `memories` | retrieved |
| 3 | `[RECENT SUMMARIES]` | `recent_summaries` | retrieved |
| 4 | `[SEMANTIC SUMMARIES]` | `semantic_summaries` | retrieved |
| 5 | `[RECENT REFLECTIONS]` | `recent_reflections` | retrieved |
| 6 | `[SEMANTIC REFLECTIONS]` | `semantic_reflections` | retrieved |
| 7 | `[BACKGROUND KNOWLEDGE]` | `wiki` | retrieved |
| 8 | `[WEB SEARCH RESULTS]` | `web_search_results` | retrieved |
| 9 | `[RELEVANT INFORMATION]` | `semantic_chunks` | retrieved |
| 10 | `[DREAMS]` | `dreams` | retrieved |
| 11 | `[USER'S PERSONAL NOTES]` | `personal_notes` | retrieved |
| 12 | `[USER UPLOADED ITEMS]` | `user_uploads` | retrieved |
| 13 | `[DAEMON DOCUMENTATION]` | `reference_docs` | retrieved |
| 14 | `[PROJECT COMMIT HISTORY]` | `git_commits` | retrieved |
| 15 | `[ADAPTIVE WORKFLOWS]` | `procedural_skills` | retrieved |
| 16 | `[PROPOSED FEATURES]` | `proposed_features` | retrieved |
| 17 | `[KNOWLEDGE GRAPH]` | `graph_context` | generated_context |
| 18 | `[UNRESOLVED THREADS]` | `unresolved_threads` | generated_context |
| 19 | `[PROACTIVE INSIGHTS]` | `proactive_insights` | generated_context |
| 20 | `[USER PROFILE]` | `user_profile` | metadata |
| 21 | `[ACTIVE FEATURES]` | (computed) | metadata |
| 22 | `[CODEBASE CHANGES SINCE LAST SESSION]` | `codebase_changes` | metadata |
| 23 | `[TIME CONTEXT]` | (computed) | structural |
| 24 | `[TEMPORAL GROUNDING]` | `narrative_state` | generated_context |
| 25 | `[SHORT-TERM CONTEXT SUMMARY]` | `stm_summary` | generated_context |
| 26 | `[CURRENT USER QUERY]` | `user_input` | structural |
| 27 | `[VISUAL MEMORIES]` | `visual_memories` | retrieved |

Sections are joined with `"\n\n"`. Each section starts with its header and `n=` count.

### Key Architectural Facts

- **System prompt and context prompt are separate strings** passed to the LLM API as
  distinct messages (system role vs user role).
- **`_assemble_prompt()`** only builds the context/user prompt. The system prompt is
  built entirely in the orchestrator.
- **All sections are conditional** — empty context fields produce no section.
- **`[CURRENT USER QUERY]`** includes `[LAST EXCHANGE FOR CONTEXT]` (most recent Q/A)
  for coherence in the high-attention area.
- **Token budget trimming** happens BEFORE assembly, in `token_manager._manage_token_budget()`.
  The assembly step just formats whatever survived the budget.
- **Images** (from personal notes and uploads) are collected during assembly and stored
  in `context["note_images"]` for multimodal API calls — side effect of assembly.

### Token Budget Priority Order

From `token_manager.py` (higher priority = trimmed last):

| Priority | Sections |
|----------|----------|
| 10 | stm_summary (never trimmed) |
| 9 | user_profile |
| 8 | narrative_state, web_search_results |
| 7 | recent_conversations, graph_context, unresolved_threads |
| 6 | semantic_chunks, personal_notes, user_uploads |
| 5 | reference_docs, memories |
| 4 | procedural_skills, facts |
| 3 | summaries, proposed_features, git_commits, proactive_insights |
| 2 | reflections, dreams, codebase_changes |
| 1 | wiki |

### Side-Effect-Free Generation Path

`ModelManager.generate_once()` is a pure LLM call with no side effects:
```python
async def generate_once(self, prompt, model_name=None, system_prompt="...",
                        max_tokens=256, temperature=None, top_p=None) -> str
```
This is the correct path for eval generation. It does not call the orchestrator,
does not store interactions, does not extract facts, and does not update the graph.

### Post-Generation Side Effects (to avoid in eval)

These happen in `orchestrator.process_user_query()` and `gui/handlers.py` after
generation. Eval generation must bypass ALL of these:

1. `memory_coordinator.store_interaction()` — persists query + response to ChromaDB
2. `correction_detector.detect_corrections()` — updates truth scores
3. `correction_detector.detect_confirmations()` — updates truth scores
4. `claim_index.cascade_staleness()` — updates staleness metadata
5. `_cascade_entity_resolution()` — updates entity resolution notes
6. `escalation_tracker.record_response()` — updates crisis state
7. Fact extraction (shutdown) — extracts and stores new facts
8. Graph ingestion (shutdown) — adds nodes/edges to knowledge graph
9. Thread extraction/resolution (shutdown) — updates thread store
10. Summary generation (shutdown) — creates compressed summaries

---

## Phase 1: Instrumentation & Snapshot Replay [COMPLETE]

### Purpose

Build the measurement foundation: capture what the prompt builder actually produces,
save it, replay it deterministically, verify hashes match, and generate LLM responses
without any persistence side effects.

### Inputs

- A live Daemon query (captured via gated hook in builder.py)
- OR a saved snapshot JSON file (for offline replay)

### Outputs

- `eval/snapshots/*.json` — saved prompt snapshots
- `PromptSnapshot` objects with two layers: `raw_retrieval` and `post_hygiene`
- `EvalGenerationResult` — LLM response from frozen prompt (no side effects)
- `PersistenceSnapshot` diffs — proof that eval didn't mutate state

### Dependencies

- None (this is the foundation)

### Files

```
eval/
  __init__.py
  PLAN.md                    # This file
  README.md                  # Phase 1 usage guide
  schema.py                  # Pure data models (no Daemon imports)
  section_registry.py        # Canonical section definitions
  snapshots.py               # Capture, save/load, replay, hash verification
  no_store_generation.py     # Side-effect-free LLM generation
  persistence_guard.py       # State fingerprinting and diff

tests/test_eval/
  __init__.py
  test_section_registry.py
  test_snapshots.py
  test_replay_hash.py
  test_no_store_generation.py
  test_persistence_guard.py
```

Production code changes (minimal, gated):
- `core/prompt/builder.py` — read-only eval snapshot hook, disabled by default

### What Phase 1 Does

1. **Section Registry** — canonical definition of all 27 prompt sections with:
   - internal key, header text, source field, category
   - whether ablatable, structurally required, assembly order
   - validation that registry matches actual builder output

2. **Snapshot Capture** — two-layer capture:
   - `raw_retrieval`: structured context dict from `build_prompt()` (before assembly)
   - `post_hygiene`: formatted section text + assembled prompt string (after assembly)
   - Provenance: git hash, model, system prompt hash, config hash, timestamps

3. **Snapshot Replay** — reconstruct the prompt from saved formatted text:
   - Orders sections by `assembly_order`
   - Uses stored `formatted_text` (not re-running formatter)
   - Verifies exact hash and normalized hash match

4. **No-Store Generation** — call `ModelManager.generate_once()` directly:
   - Bypasses orchestrator entirely
   - `EvalGenerationConfig` rejects any persistence flag set to True
   - Returns response text + metadata only

5. **Persistence Guard** — fingerprint all mutable state before/after eval:
   - ChromaDB collection counts
   - JSON file hashes (graph, aliases, claims, surfacing history, category cache)
   - `before.assert_same_as(after)` fails if anything changed

### What Phase 1 Does NOT Do

- Generate ablation variants (Phase 2)
- Build a query corpus (Phase 3)
- Run batch eval generation (Phase 4)
- Judge response quality (Phase 5)
- Check for hallucination/citation accuracy (Phase 6)
- Aggregate results or produce scorecards (Phase 7)
- Modify production prompt behavior (Phase 8)

### Exit Criteria

1. Section registry covers all 27 sections from actual builder output
2. Snapshot capture saves both layers with structured + formatted content
3. Snapshot stores system/config/model provenance
4. Replay from post_hygiene produces matching exact hash (stable formatting)
5. Replay from post_hygiene produces matching normalized hash (whitespace variance)
6. Replay does not trigger retrieval, web search, tools, or memory access
7. EvalGenerator calls `generate_once()` directly, not the orchestrator
8. EvalGenerationConfig rejects any persistence flag set to True
9. PersistenceGuard detects intentional writes
10. PersistenceGuard passes when no persistence changes occur
11. All tests in `tests/test_eval/` pass
12. `DAEMON_EVAL_CAPTURE=0` (default) means zero behavior change

### Implementation Order

1. `eval/schema.py` — pure data models
2. `eval/section_registry.py` — canonical section definitions
3. Unit tests for schema and registry
4. `eval/snapshots.py` — capture, save/load, replay, hash verification
5. Snapshot tests (roundtrip, replay, hash)
6. Builder hook in `core/prompt/builder.py` (gated, read-only)
7. `eval/persistence_guard.py` — state fingerprinting
8. Persistence guard tests
9. `eval/no_store_generation.py` — side-effect-free generation
10. No-store generation tests (fake model first, then real path)
11. `eval/README.md` — usage guide
12. Full test run
13. Live snapshot capture with `DAEMON_EVAL_CAPTURE=1`
14. Load and replay; verify normalized hash match

### Completion Notes (2026-05-04)

All 12 exit criteria met. 89 tests passing. Live capture → replay → exact hash match
verified. No regression in existing prompt builder tests (18 passed, 4 skipped).

The builder hook captures post_hygiene only. For Phase 2, if raw_retrieval layer is
needed (pre-hygiene context), a second hook would need to be added in `build_prompt()`
before `ContentHygiene._hygiene_and_caps()` is called (now in `hygiene.py`; call site in builder.py). This was intentionally
deferred to keep Phase 1 production changes minimal.

---

## Phase 2: Variant Generation + Query Corpus + Utilization [COMPLETE]

### Purpose

Generate ablation variants from captured snapshots, build a labeled query corpus
covering all 9 intent types and 4 tone levels, and analyze section utilization
across the corpus.

### Inputs

- `PromptSnapshot` objects from Phase 1
- Section registry (ablatable sections)
- `IntentType` enum from `core/intent_classifier.py` (9 types)
- Tone levels from `utils/tone_detector.py` (HIGH, MEDIUM, CONCERN, CONVERSATIONAL)

### Outputs

- `PromptVariant` objects with metadata about what changed
- Variant manifest (JSON) mapping variant IDs to ablation descriptions
- `eval/corpus.json` — labeled query sets with coverage metadata
- Utilization report (JSON + text) — per-section presence/token stats by intent

### Dependencies

- Phase 1 (snapshot capture + replay)

### Files

```
eval/
  variants.py              # VariantGenerator: LOO, AOI, bundle, reorder
  corpus.py                # CorpusManager, 27-query seed corpus, section key validation
  utilization.py           # UtilizationAnalyzer, SectionUtilization, UtilizationReport
  run_phase2.py            # CLI runner

tests/test_eval/
  test_variants.py           # 36 tests
  test_corpus.py             # 24 tests
  test_utilization.py        # 15 tests
```

### What Phase 2 Does

1. **Variant Generation** — four ablation strategies:
   - **Leave-one-out (LOO)**: Remove one ablatable section at a time
   - **Add-one-in (AOI)**: Start from structural skeleton, add one section back
   - **Bundles**: Remove pre-defined groups of related sections (7 default bundles
     including `all_retrieved` auto-populated from registry)
   - **Reorder**: Move a section to a different assembly_order position (tests
     attention-position effects). `generate_reorder_to_high_attention()` moves
     each section to right before the query.
   - All strategies correctly access `SectionSnapshot.token_count` and
     `SectionSnapshot.formatted_text` (not nonexistent `section_token_counts` dict).
   - `PromptVariant` has full `to_dict()`/`from_dict()` serialization.

2. **Query Corpus** — 27 seed queries, 3 per `IntentType`:
   - Reuses the real `IntentType` enum from `core/intent_classifier.py` (not a
     parallel taxonomy). All 9 types covered: FACTUAL_RECALL, TEMPORAL_RECALL,
     EMOTIONAL_SUPPORT, CASUAL_SOCIAL, TECHNICAL_HELP, CREATIVE_EXPLORATION,
     META_CONVERSATIONAL, PROJECT_WORK, GENERAL.
   - Each query tagged with tone level. 2 HIGH, 2 CONCERN, rest CONVERSATIONAL.
   - `ExpectedBehavior.expected_sections_used` validated against `SECTION_REGISTRY`
     keys on add. Invalid keys raise `ValueError`.
   - `CorpusManager` tracks intent + tone coverage gaps.
   - Full `to_dict()`/`from_dict()` roundtrip on `CorpusQuery`.

3. **Utilization Analysis** — section presence stats:
   - Per-section: presence rate, nonempty rate, avg tokens when present
   - Per-intent breakdown of which sections appear
   - Classifications: always_empty, always_present, high_variance (20-80%),
     intent_specific (only appears for one intent)
   - Human-readable text report + JSON serialization

### Deferred to Later Phases

- **Replace variants**: Substituting real content with synthetic/degraded content
  requires content generation, which is Phase 4 territory.
- **Arbitrary N-combinations**: Combinatorial explosion. Bundles cover the useful
  categorical cases. If needed, can add later.

### Key Design Decisions

- **Merged Phase 3 into Phase 2**: Query corpus is small infrastructure that's
  needed immediately for utilization analysis. Keeping them separate would mean
  building variants with nothing to run against.
- **Uses `IntentType` from classifier, not custom enum**: Avoids mapping headaches
  and ensures intent-conditioned analysis in Phase 7 uses real system labels.
- **Tone labeling added to corpus**: PLAN.md Phase 3 required "at least 2 queries
  per tone level." Tone drives escalation tracker and system prompt injection.
- **`all_retrieved` bundle computed from registry**: Not a placeholder — populated
  at import time from `SectionCategory.RETRIEVED`.

### Completion Notes (2026-05-05)

75 tests passing. No production code changes. Phase 1 tests (89) still passing
(164 total). Runner script tested with `python -m eval.run_phase2` (requires
snapshots from `DAEMON_EVAL_CAPTURE=1`).

Utilization analysis was not in the original 8-phase plan but fills a critical gap:
it provides data-driven evidence for which sections to optimize, bridging "we have
snapshots" to "we know what to test." The utilization report is the primary input
for prioritizing Phase 4 generation runs.

---

## Phase 3: Query Corpus [MERGED INTO PHASE 2]

Phase 3 was merged into Phase 2. See Phase 2 section above for query corpus details.

The seed corpus has 27 queries (3 per intent type). The original plan called for
45+ (5 per type). The corpus is designed to grow through manual curation during
live snapshot capture sessions. Use `CorpusManager.add_query()` to expand.

Coverage gaps to fill in future sessions:
- MEDIUM tone (only 0 queries, need 2+)
- Deictic queries ("this", "that", "the thing we discussed")
- Multi-turn context-dependent queries
- Queries that trigger web search, agentic search, uncertainty fallback

---

## Phase 4: Generation Harness [COMPLETE]

### Purpose

Run all variant x query combinations through the side-effect-free generation path.
Produce a matrix of (variant, query, response) triples for downstream judging.

### Inputs

- Variants from Phase 2
- Snapshots from Phase 1
- `EvalGenerator` from Phase 1

### Outputs

- `eval/runs/<run_id>/results/*.json` — per-(snapshot, variant) generation results
- `eval/runs/<run_id>/manifest.json` — run metadata, progress, cost tracking
- `EvalGenerationResult` per (snapshot, variant) pair

### Dependencies

- Phase 1 (EvalGenerator + PersistenceGuard)
- Phase 2 (variants, query corpus, utilization report)

### Files

```
eval/
  harness.py               # GenerationHarness, HarnessConfig, RunManifest
  run_phase4.py            # CLI runner with dry-run, resume, model config

tests/test_eval/
  test_harness.py            # 24 tests
```

### What Phase 4 Does

1. **Pair Planning** — for each snapshot, generates one baseline (full prompt) plus
   all LOO, AOI, bundle variants. Optionally reorder variants via `--include-reorder`.

2. **Batch Generation** — replays each variant's prompt via `SnapshotReplay`, passes
   through `EvalGenerator.generate()`, saves result JSON per pair. Rate-limited via
   configurable RPM (default 30).

3. **Resume Capability** — reads existing manifest to find completed pairs, skips them.
   Manifest saved after each pair for crash recovery. Use `--run-id <id>` to resume.

4. **Persistence Guard** — optional wrapping with `PersistenceGuard.capture()` before
   and after run. Raises if any ChromaDB collection or JSON file changed.

5. **Cost Tracking** — manifest records total prompt/response tokens, generation time,
   completed/failed/skipped counts.

### CLI Usage

```bash
# Dry run — show pair count and cost estimate
python -m eval.run_phase4 --dry-run

# Run with default settings (gpt-4o-mini, temp 0.3, 30 rpm)
python -m eval.run_phase4

# Run with specific model
python -m eval.run_phase4 --model sonnet-4.5

# Resume a previous run
python -m eval.run_phase4 --run-id 20260505_131500

# Include reorder variants
python -m eval.run_phase4 --include-reorder
```

### Deferred

- **Multi-model comparison**: Run same variant against different models in parallel.
  HarnessConfig supports `model` field but the runner takes one model per run.
  Multiple runs with different `--model` flags achieves the same result.
- **Parallel generation across machines**: Not needed at current scale (450 pairs).

### Completion Notes (2026-05-05)

24 tests passing. Dry run verified against 13 captured snapshots: 450 pairs planned
(13 baselines + 177 LOO + 177 AOI + 83 bundles), ~5.4M estimated input tokens.
No production code changes. Phase 1-2 tests (164) still passing (188 total).

---

## Phase 5: Pairwise Judge Harness [COMPLETE]

### Purpose

For each query, compare the baseline response (full prompt) against each variant
response. Use LLM-as-judge with structured rubrics to determine which is better.

### Inputs

- Generation results from Phase 4 (eval/runs/<run_id>/results/)
- Judge model configuration

### Outputs

- `JudgeVerdict` per (baseline, variant) pair with per-criterion scores
- Aggregate win/loss/tie counts per section
- Per-criterion score deltas (accuracy, helpfulness, conciseness, tone, grounding)
- Section ablation report (JSON + text)

### Dependencies

- Phase 4 (generation results)

### Files

```
eval/
  judge.py                 # PairwiseJudge, BatchJudge, aggregation, report formatting
  run_phase5.py            # CLI runner with dry-run, resume, report-only modes

tests/test_eval/
  test_judge.py              # 25 tests
```

### What Phase 5 Does

1. **Pairwise Judging** — for each (baseline, variant) pair:
   - Position randomization: baseline randomly assigned to A or B (deterministic
     per pair via MD5 hash). Prevents position bias.
   - Blind judging: judge prompt does not reveal which is baseline vs variant.
   - 5-criterion rubric: accuracy, helpfulness, conciseness, tone, grounding.
     Each scored 1-5 for both A and B.
   - Structured verdict: winner (A/B/tie) + confidence + explanation.
   - Unblinding: after verdict, resolves winner_is_baseline from position.

2. **Batch Runner** — loads all Phase 4 results, pairs each variant with its
   snapshot's baseline, runs judge on each pair. Rate-limited. Resume via existing
   verdict files. Saves per-verdict JSON to eval/runs/<run_id>/judgments/.

3. **Aggregation** — `aggregate_by_section()` groups LOO verdicts by removed section:
   - Baseline win rate (removing section hurt quality = section is valuable)
   - Variant win rate (removing section helped = section is harmful)
   - Per-criterion score deltas (positive = section helps that criterion)

4. **Reporting** — `format_section_report()` produces human-readable table showing
   which sections are most/least valuable, sorted by baseline win rate.

### CLI Usage

```bash
# Dry run — show pair count
python -m eval.run_phase5 --gen-run eval/runs/20260506_100331 --dry-run

# Run judging
python -m eval.run_phase5 --gen-run eval/runs/20260506_100331

# Use stronger judge
python -m eval.run_phase5 --gen-run eval/runs/20260506_100331 --judge-model gpt-4o

# Report only (analyze existing verdicts)
python -m eval.run_phase5 --gen-run eval/runs/20260506_100331 --report-only
```

### Judge Rubric

The system prompt instructs the judge to penalize filler/padding and reward
appropriate brevity. Criteria:
- **accuracy**: Factual correctness given available context
- **helpfulness**: How well the response addresses user intent
- **conciseness**: Appropriate length — penalize padding, not brevity
- **tone**: Natural conversational register matching user energy
- **grounding**: References real context, no confabulation

### Deferred

- **Multi-judge agreement**: Run same pairs with different judge models and compute
  inter-judge kappa. Infrastructure supports it (just run twice with different
  `--judge-model`), but agreement scoring is Phase 7.
- **Human judging interface**: GUI for manual A/B verdicts. Future work.

### Completion Notes (2026-05-06)

25 tests passing. No production code changes. Full eval suite (213 tests) passing.
Dry-run verified against Phase 4 run (858 judgment pairs from 24 snapshots).

**Run 1a — gpt-4o-mini judge, 24 snapshots** (`20260506_133955`):
858/858 completed, 0 failures. BL wins 347 (40%), Var wins 204 (24%), Ties 307 (36%).

**Run 1b — haiku-4.5 judge, 24 snapshots** (`20260506_144714`):
858/858 completed, 0 failures. BL wins 458 (53%), Var wins 296 (35%), Ties 104 (12%).

**Run 2a — gpt-4o-mini judge, 50 snapshots** (`20260507_gpt`):
1799/1799 completed, 0 failures. BL wins 782 (43%), Var wins 504 (28%), Ties 513 (29%).

**Run 2b — haiku-4.5 judge, 50 snapshots** (`20260507_143207`):
1799/1799 completed, 0 failures. BL wins 554 (43%), Var wins 546 (42%), Ties 193 (15%).
(Note: 506 skipped from partial earlier run, counted in BL/Var/Tie totals above.)

**Inter-judge agreement (50 snapshots)**: 24% on section-level verdict direction (5/21
sections). Down from 50% at 24 snapshots — more diverse queries exposed intent-dependent
behavior that judges weight differently. gpt-4o-mini remains conciseness-biased (29%
ties); Haiku remains grounding-biased (15% ties, nearly even BL/Var split).

**Both judges agree (high confidence findings, 50 snapshots):**
- VALUABLE: `personal_notes` (47%/50%), `proposed_features` (100%/75%)
- HARMFUL: `recent_reflections` (20%/37%), `memories` (38%/25% — GPT mixed, Haiku clear harmful with -0.58 conciseness)
- MIXED: `unresolved_threads` (37%/41%)
- LOW-IMPACT: `proactive_insights` (29%/14%)

**Both reflection types underperform:**
- `recent_reflections`: 20% GPT / 37% Haiku — both say harmful
- `semantic_reflections`: 31% GPT / 35% Haiku — both in mixed-to-harmful territory, -0.22 conciseness (Haiku)
- Meta-insights from session-end reflections add tokens without improving accuracy or grounding

**Key shift from 24 → 50 snapshots:**
- `recent_conversation` dropped from "clearly valuable" (46%/62%) to "mixed" (40%/42%)
- `user_profile` dropped from "valuable" (39%/61%) to "mixed/valuable" (36%/48%)
- Adding technical/project/emotional queries diluted the casual-dominated signal
- Sections aren't universally good or bad — they're intent-dependent

**Judges disagree (needs intent-conditioned analysis):**
- `reference_docs` (37%/51%), `semantic_summaries` (24%/40%), `memories` (38%/25%),
  `narrative_state` (39%/32%), `procedural_skills` (33%/38%), `stm_summary` (36%/38%),
  `git_commits` (38%/31%), `user_uploads` (38%/46%), `user_profile` (36%/48%)

See Phase 7 section for intent-conditioned breakdown and Phase 8 for draft gating rules.

---

## Phase 6: Objective Checks [COMPLETE]

### Purpose

Run automated checks that don't require subjective judgment: response length,
thinking leak detection, profile grounding, citation validity, filler detection.

### Inputs

- Phase 4 generation results (eval/runs/<run_id>/results/)

### Outputs

- Per-response check results (pass/fail/score per check)
- Aggregate check deltas per section (LOO comparison: baseline vs variant)
- JSON + text reports

### Dependencies

- Phase 4 (generation results)

### Files

```
eval/
  checks.py                # 5 checks + batch runner + aggregation + report
  run_phase6.py            # CLI runner

tests/test_eval/
  test_checks.py             # 33 tests
```

### What Phase 6 Does

Five automated checks, no LLM calls:

1. **Response length** — truncated (<20 chars), verbose casual (>500 for greetings),
   excessive (>5000). Casual detection via regex.
2. **Thinking leak** — detects `<thinking>` tags, meta-reasoning patterns ("the user
   wants", "I should check", "based on my context"). Requires 2+ pattern hits.
3. **Profile grounding** — extracts proper names from response, checks if each appears
   anywhere in the prompt. Catches confabulated names not in context.
4. **Citation validity** — validates `[MEM_N]`, `[WEB_N]`, `[GRAPH_N]` markers against
   prompt text. Reports valid/invalid ratio.
5. **Filler detection** — catches "feel free to ask", "I hope this helps", "let me know
   if you need more" etc. Penalizes tail filler more heavily.

Aggregation compares baseline vs LOO variant scores per section to detect which
sections cause or prevent each check failure.

### CLI Usage

```bash
python -m eval.run_phase6 --gen-run eval/runs/20260506_100331
```

### Completion Notes (2026-05-08)

33 tests passing. Run against 1849 responses from Phase 4. No production code changes.
Full eval suite (246 tests) passing.

**Overall pass rates:**
- citation_validity: 100% (no invalid citations in any response)
- response_length: 100% (6 failures, all truncated variants)
- thinking_leak: 98% (32 failures)
- filler: 88% (224 failures — 12% of responses contain padding phrases)
- profile_grounding: 57% (799 failures — names in responses not found in prompt)

**Key findings by check:**

*Filler:* `memories` (+0.042 delta) — removing memories REDUCES filler. Memories cause
the model to pad with "feel free to ask" more. `stm_summary` (-0.044), `narrative_state`
(-0.034), `recent_conversation` (-0.034) — removing these INCREASES filler. They help
constrain the model.

*Profile grounding:* `memories` (-0.070 delta) — removing memories causes more
ungrounded names (model confabulates when it doesn't have memory context to reference).
`semantic_chunks` (-0.087) same pattern. These sections prevent hallucination.

*Thinking leak:* `memories` (-0.042) — removing memories slightly increases thinking
leaks. The model is more likely to show meta-reasoning when it lacks context.

**Tension with Phase 5:** Phase 5 judges said `memories` hurts quality overall (25-38%
baseline win rate). Phase 6 shows `memories` prevents filler and prevents name
confabulation. The section adds noise but also adds guardrails. The solution is
intent-gating, not removal — keep memories for factual/emotional queries where grounding
matters, drop for casual where filler is the bigger problem.

---

## Phase 7: Aggregation & Reporting

### Purpose

Aggregate judge verdicts and objective check results into actionable section-level
scorecards. Answer: "Is section X worth its token cost?"

### Inputs

- Judge verdicts from Phase 5
- Objective check results from Phase 6
- Token costs per section from Phase 1 snapshots
- Query intent/tone labels from Phase 3

### Outputs

- Section scorecards: impact score, token cost, net value
- Intent-conditioned analysis (section X helps for FACTUAL_RECALL but hurts CASUAL_SOCIAL)
- Confidence intervals and statistical significance
- Recommendations (keep/drop/conditional per intent)

### Dependencies

- Phase 5 (judge verdicts)
- Phase 6 (objective checks)

### Files

No dedicated aggregation module — functionality absorbed into Phase 5-6 runners:
- `judge.py`: `aggregate_by_section()`, `format_section_report()` — per-section LOO win rates
- `checks.py`: LOO check aggregation — per-section objective metric deltas
- Output written to `eval/runs/<run_id>/judgments/` and `eval/runs/<run_id>/checks/`

### Scope

- Per-section win rate (when dropped, does quality decrease?)
- Per-section hallucination delta (when dropped, do hallucinations increase?)
- Token efficiency score (quality impact / token cost)
- Intent-conditioned breakdowns
- Confidence intervals via bootstrap resampling
- Markdown + JSON report output

### Findings (2026-05-07, 50 snapshots, 1799 pairs, 2 judges)

Snapshot distribution: casual=20, technical=6, knowledge=5, project=4, factual=4,
creative=3, meta=3, temporal=3, emotional=2. Format: GPT BL% / Haiku BL%.

**Intent-conditioned section analysis** — sections that flip value by query type:

| Section | casual | technical | project | emotional | knowledge | meta | factual | temporal |
|---------|--------|-----------|---------|-----------|-----------|------|---------|----------|
| `memories` | DROP? (45/20) | SPLIT (60/0) | KEEP? (25/50) | KEEP (50/100) | KEEP? (50/25) | DROP? (33/33) | meh (25/25) | DROP (0/33) |
| `reference_docs` | KEEP? (36/64) | SPLIT (17/67) | KEEP (75/50) | SPLIT (50/0) | KEEP? (25/50) | KEEP? (100/33) | KEEP? (0/50) | KEEP? (33/67) |
| `recent_conversation` | KEEP (55/55) | KEEP? (33/50) | DROP? (25/25) | SPLIT (0/50) | DROP? (20/0) | meh (33/33) | KEEP (50/75) | DROP? (33/33) |
| `user_profile` | DROP? (27/40) | KEEP (67/50) | KEEP (75/75) | KEEP (100/100) | SPLIT (0/60) | KEEP? (100/33) | meh (0/25) | DROP (0/0) |
| `recent_reflections` | DROP (21/37) | DROP (17/33) | SPLIT (50/25) | KEEP? (0/50) | KEEP? (20/60) | DROP (0/0) | KEEP? (0/50) | KEEP? (33/67) |
| `semantic_reflections` | DROP? (26/21) | DROP (33/0) | KEEP (50/75) | KEEP (50/100) | meh (20/40) | meh (33/33) | KEEP? (25/50) | DROP? (33/33) |
| `narrative_state` | DROP? (27/33) | DROP? (33/17) | KEEP (75/50) | KEEP (100/50) | DROP? (40/20) | KEEP? (67/33) | meh (25/25) | KEEP (50/100) |
| `personal_notes` | KEEP (46/54) | SPLIT (0/50) | KEEP (50/50) | KEEP (100/50) | KEEP (75/75) | KEEP (100/50) | DROP? (0/0) | KEEP? (0/50) |
| `user_uploads` | DROP? (25/40) | KEEP (50/50) | KEEP (50/100) | KEEP (50/100) | KEEP? (80/40) | SPLIT (67/0) | DROP? (0/0) | SPLIT (0/67) |
| `procedural_skills` | DROP (31/44) | DROP? (40/40) | SPLIT (67/33) | KEEP (100/100) | meh (25/25) | KEEP? (100/33) | meh (0/25) | KEEP? (0/67) |
| `unresolved_threads` | DROP? (37/37) | KEEP (50/67) | KEEP (100/50) | KEEP? (0/100) | DROP? (40/40) | DROP? (33/0) | DROP? (0/0) | meh (33/33) |

**High-confidence findings (both judges agree across 50 snapshots):**

1. `personal_notes` — KEEP for casual (46/54), emotional (100/50), knowledge (75/75),
   meta (100/50), project (50/50). Most consistently valuable section across intents.
2. `memories` — HARMFUL for casual (45/20 GPT mixed, Haiku clear DROP). KEEP for
   emotional (50/100). Conciseness delta: -0.04 GPT / -0.58 Haiku.
3. `recent_reflections` — DROP for casual (21/37), technical (17/33), meta (0/0).
   Neither reflection type adds value for most query types.
4. `semantic_reflections` — DROP for casual (26/21), technical (33/0). Similar to
   recent_reflections. KEEP for emotional (50/100) and project (50/75).
5. `user_profile` — KEEP for emotional (100/100), project (75/75), technical (67/50).
   DROP for casual (27/40). Strong intent dependency.
6. `recent_conversation` — KEEP for casual (55/55), factual (50/75). DROP for
   knowledge (20/0), project (25/25). Chat context helps continuity but dilutes focus.
7. `unresolved_threads` — KEEP for technical (50/67) and project (100/50). DROP? for
   casual (37/37). Cheap section (82 avg tokens) with intent-specific value.

**Conciseness remains the dominant criterion.** Both judges agree that `memories`
hurts conciseness (-0.04/-0.58) and `user_profile` helps it (+0.16/+0.43). Sections
that constrain response length tend to win on quality — the model performs better with
less context when the context isn't query-relevant.

**Caveats**: emotional=2, creative=3, temporal=3 snapshots — small sample. Casual and
technical have enough data for confidence. `dreams` 0% presence (disabled). `wiki` and
`semantic_chunks` now have some data (7 and 4 snapshots) but sample too small for
intent-conditioned analysis.

### Out of Scope

- Interactive dashboards
- Real-time monitoring
- Automated decision-making (that's Phase 8)

---

## Phase 8: Prompt Policy & Shadow-Mode Rollout

### Purpose

Translate Phase 7 recommendations into concrete prompt policy changes. Validate via
shadow-mode (run old and new prompts in parallel, compare live) before production rollout.

### Inputs

- Phase 7 recommendations
- Human review and approval
- Shadow-mode comparison results

### Outputs

- Updated section inclusion rules (conditional on intent/tone)
- Updated token budget allocations
- Updated section ordering
- Shadow-mode comparison reports
- Production rollout changelog

### Dependencies

- Phase 7 (recommendations)
- Human review (non-automatable gate)

### Files (Likely)

```
eval/
  policy.py                # Prompt policy rules engine
  shadow_mode.py           # Dual-prompt comparison in live chat

tests/test_eval/
  test_policy.py
  test_shadow_mode.py
```

### Scope

- Rule engine: "include section X only when intent is Y and tone is Z"
- Shadow mode: run both old and new prompt, log both responses, compare
- A/B rollout: gradually shift traffic from old to new
- Rollback capability: revert to previous policy instantly

### Out of Scope

- Automated policy changes without human review (NEVER)
- Changes to the system prompt core personality/principles
- Changes to the retrieval pipeline itself (only section inclusion/ordering)

### Draft Gating Rules (updated 2026-05-07, 50 snapshots, 1799 pairs)

Based on 50 snapshots, 1799 pairs, two judges (gpt-4o-mini + haiku-4.5). Higher
confidence than 27-snapshot draft but emotional/creative/temporal cells still thin.

```
CASUAL_SOCIAL:
  DROP: memories, recent_reflections, semantic_reflections, procedural_skills
  TRIM: user_profile (reduce to top-5 facts), narrative_state
  KEEP: recent_conversation, personal_notes, user_uploads (borderline)
  NEUTRAL: unresolved_threads (cheap, 82 tokens, ~37% BL both judges)

TECHNICAL_HELP:
  KEEP: user_profile, unresolved_threads, user_uploads
  SPLIT: memories (GPT 60% KEEP, Haiku 0% — judges disagree strongly)
  SPLIT: reference_docs (GPT 17%, Haiku 67% — judges disagree strongly)
  DROP: recent_reflections, semantic_reflections

PROJECT_WORK:
  KEEP: user_profile, narrative_state, unresolved_threads, reference_docs,
        semantic_reflections, user_uploads
  DROP: recent_conversation (both agree 25%/25% — chat context dilutes focus)

EMOTIONAL_SUPPORT:
  KEEP: memories, user_profile, semantic_reflections, narrative_state,
        user_uploads, procedural_skills, unresolved_threads
  NOTE: Both judges agree 100%/100% on user_profile, 50%/100% on memories.
        Emotional queries benefit from maximum personal context.

FACTUAL_RECALL:
  KEEP: recent_conversation
  NEUTRAL: most sections show meh (0%/25% range, high tie rates)
  DROP: user_uploads, unresolved_threads

META_CONVERSATIONAL:
  KEEP: reference_docs (GPT 100%, Haiku 33% — GPT strong, Haiku mixed)
  KEEP: procedural_skills (GPT 100%, Haiku 33%)
  DROP: memories, recent_reflections, unresolved_threads

KNOWLEDGE (web/wiki):
  KEEP: personal_notes (75%/75% — both agree strongly)
  KEEP: user_uploads (80%/40% — GPT strong)
  DROP: recent_conversation (20%/0%)

ALL INTENTS:
  ALWAYS KEEP: time_context, current_query, personal_notes
  ALWAYS DROP: recent_reflections (20%/37% aggregate — worst performer)
```

**Token savings estimate (revised)**: Gating `memories` (8353 avg tokens) for
CASUAL_SOCIAL saves ~8K tokens on ~40% of queries. Dropping `recent_reflections`
(1970 avg tokens) and `semantic_reflections` (2023 avg tokens) across casual +
technical saves ~4K more. Trimming `user_profile` for casual saves ~400 tokens.
Conservative estimate: ~6-8K tokens saved per casual query, ~4K per technical.

**Key change from 27→50 snapshot draft**: `recent_conversation` is no longer
"always keep" — it DROPS for project and knowledge queries. `user_uploads` flipped
from "harmful" to "mixed/valuable" with more diverse data. Both reflection types
confirmed as weak performers across most intents.

### Critical Constraint

**No production prompt changes may be applied automatically.** Phase 8 changes require:
1. Phase 7 report reviewed by a human
2. Proposed policy changes documented and approved
3. Shadow-mode validation shows no regression
4. Explicit human approval before production rollout

### Implementation Notes (2026-05-08)

**Implemented:** Intent-conditioned section gating via existing `_PROFILES` mechanism.
No new infrastructure — the `IntentResult.retrieval_overrides` → `eff_max_*` pipeline
was already wired. Changes:

1. `core/intent_classifier.py` — Updated `_PROFILES` retrieval counts based on eval data.
   New keys: `max_reference_docs`, `max_narrative`, `max_user_uploads`, `max_proactive`.
   CASUAL_SOCIAL gates 4 additional sections to 0. TECHNICAL_HELP drops reflections.
   PROJECT_WORK reduces recent_conversation. META_CONVERSATIONAL reduces memories 20→5.
   Config flag `PROMPT_SECTION_GATING_ENABLED` strips Phase 8 keys when disabled.

2. `core/prompt/builder.py` — Added 4 new `eff_max_*` variables plus `max_narrative`
   gate. Parallel task launches use `> 0` guards so `max_X: 0` skips entirely.

3. `config/app_config.py` + `config/config.yaml` — `PROMPT_SECTION_GATING_ENABLED`
   flag (default True), env var override.

**Deferred:** Shadow-mode (dual-prompt comparison in live chat), A/B rollout,
formal policy rules engine. These are needed for production validation but the
gating mechanism is live and can be tuned via `_PROFILES` directly.

74 intent classifier tests + 246 eval tests passing, zero regressions.

---

## Cross-Phase Architecture

### Data Flow

```
Phase 1: PromptSnapshot ─────────────────────────────────────────┐
                                                                  │
Phase 2: PromptSnapshot → PromptVariant[] + QueryCorpus ──────┐  │
              │                → UtilizationReport             │  │
              │                                                │  │
Phase 4: QueryCorpus × PromptVariant[] → EvalGenResult[]       │  │
              │                                                │  │
              ├──→ Phase 5: Pairwise Judge → JudgeVerdict[]    │  │
              │                                                │  │
              └──→ Phase 6: Objective Checks → CheckResult[]   │  │
                        │                                      │  │
Phase 7: JudgeVerdict[] + CheckResult[] → SectionScorecard[]   │  │
                        │                                      │  │
Phase 8: SectionScorecard[] → PromptPolicy (human-gated)       │  │
                                                                │  │
PersistenceGuard: wraps Phase 4 generation ←────────────────────┘  │
EvalGenerator: used by Phase 4 ←───────────────────────────────────┘
```

### Invariants Across All Phases

1. **No eval path writes to memory, graph, threads, summaries, or reflections.**
2. **No eval path calls `process_user_query()` or `store_interaction()`.**
3. **No eval path triggers web search, agentic search, or tool execution.**
4. **PersistenceGuard wraps every generation run.**
5. **All eval data is stored under `eval/` — never in `data/` or ChromaDB.**
6. **Production behavior is unchanged when eval is disabled.**

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DAEMON_EVAL_CAPTURE` | `"0"` | Enable snapshot capture in builder hook |
| `DAEMON_EVAL_CAPTURE_STRICT` | `"0"` | Raise on capture/replay errors (vs warn) |
| `DAEMON_EVAL_SNAPSHOT_DIR` | `"eval/snapshots"` | Snapshot save directory |

---

## Appendix: Section Registry Quick Reference

Sections marked `structurally_required=True` are never ablatable.
Sections marked `eligible_for_ablation=True` can be dropped in Phase 2 variants.

| Key | Structurally Required | Ablatable | Notes |
|-----|----------------------|-----------|-------|
| `current_query` | Yes | No | Core — prompt is meaningless without it |
| `time_context` | Yes | No | Temporal grounding for all responses |
| `recent_conversation` | No | Yes | Session continuity |
| `memories` | No | Yes | Semantic memory retrieval |
| `recent_summaries` | No | Yes | Compressed history (recent) |
| `semantic_summaries` | No | Yes | Compressed history (query-relevant) |
| `recent_reflections` | No | Yes | Meta-insights (recent) |
| `semantic_reflections` | No | Yes | Meta-insights (query-relevant) |
| `wiki` | No | Yes | Background knowledge |
| `web_search_results` | No | Yes | Real-time web content |
| `semantic_chunks` | No | Yes | Project-specific chunks |
| `dreams` | No | Yes | Dream memories |
| `personal_notes` | No | Yes | Obsidian vault content |
| `user_uploads` | No | Yes | User-uploaded files |
| `visual_memories` | No | Yes | CLIP-matched image metadata |
| `reference_docs` | No | Yes | System/project docs |
| `git_commits` | No | Yes | Procedural memory |
| `procedural_skills` | No | Yes | Adaptive workflows |
| `proposed_features` | No | Yes | Code proposals |
| `graph_context` | No | Yes | Knowledge graph relationships |
| `unresolved_threads` | No | Yes | Open thread surfacing |
| `proactive_insights` | No | Yes | Cross-domain insights |
| `user_profile` | No | Yes | Stored user facts |
| `active_features` | No | Yes | Feature inventory |
| `codebase_changes` | No | Yes | Session-start diff |
| `narrative_state` | No | Yes | Temporal grounding narrative |
| `stm_summary` | No | Yes | Short-term context summary |

**System prompt** is structurally required and never ablatable. Future work may
separate optional personality/style layers for ablation testing, but the core
operating instructions are always required.
