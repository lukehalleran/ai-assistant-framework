# Pattern Analysis Subsystem

**Status:** v2 generic deliberation path (2026-08-30), on-demand only
**Core:** `memory/pattern_engine.py` · **Planner:** `core/insight/deliberation.py` ·
**Coordinator:** `core/insight/coordinator.py` · **Rolling stage:** `core/insight/temporal.py`
**Config:** `pattern_analysis:` (config.yaml) → `PATTERN_*` constants
**Tests:** `tests/unit/test_pattern_engine.py`, `tests/unit/test_insight_pattern_facet.py`

## Why this exists

The best-received cross-session observation Daemon ever made — "two days
running now you've found songs that take you into the heaviest material" —
was **pure model synthesis over `[RECENT CONVERSATION]`**: both song turns
happened to sit in one prompt (recent conversations span sessions). That
mechanism is structurally blind beyond the ~20-turn/2-day window. The memory
gate scores documents one at a time against a cosine threshold and cannot
assemble "N low-pairwise-similarity events that matter collectively" — the
same limitation that motivated insight mode, now with a temporal axis.

Four Daemon-generated proposals asked for exactly this capability (proposals
collection): *Emotional Arc Tracker* (`d0db9576`), *Cross-Session Pattern
Detector* (`b85bc622`), *Sleep Pattern Tracker* (`d17bf634`), *Conversation
Rhythm Intelligence* (`ed7d47df`) — all citing Goals #5 (temporal awareness)
and #8 (proactive intelligence).

## Doctrine

1. **Deterministic core, narrating model.** The engine COUNTS (pure Python,
   no LLM); the synthesizer restates the computed numbers and is explicitly
   forbidden to recount, extrapolate, or invent counts (`_PATTERN_TAIL`).
2. **Never fabricates.** Every exemplar is a real store excerpt with its real
   timestamp (and doc id where the store has one).
3. **Denominator, always.** Every result carries turns-per-bucket over the
   same window: "mentioned X more" must be distinguishable from "talked
   more". The corpus over-samples days the user chose to talk — often hard
   days — so record-frequency ≠ life-frequency, and synthesis must say so
   with the actual numbers.
4. **ON-DEMAND ONLY** (owner decision 2026-08-29). Nothing here is injected
   into prompts uninvited — no `[OBSERVED PATTERNS]` section, no shutdown
   notes, no proactive surfacing. Anti-amplification doctrine: mood-pattern
   observations arriving during a low moment are exactly the mood-congruent
   amplification the 07-21 batch guards against.
5. **Organic signals only.** The tone dimension excludes
   `distress_sticky_floor` turns (carried tone, not fresh evidence — the
   escalation-tracker doctrine) and notes how many were excluded. First live
   run: 410 of 542 "elevated" turns in a 30-day window were floored
   carry-over from the latch-bug era, not mood data.
6. **Generic, not fixture-driven.** Medication cessation is one regression
   fixture. Production planning has no medication, symptom, or outcome
   recognizer: an LLM proposes a typed evidence contract for arbitrary English
   before any retrieval, and deterministic validation fails closed if it
   cannot identify the outcomes, claims, sources, phases, or source queries.
7. **The evidence set is the primary safety boundary.** Each claim declares
   its required channels and evidence standard. Outcome phrases, directional
   phrases, observable confounders, phase bounds, counterevidence facets, and
   source-specific queries are frozen before tools run. A requested-but-failed
   source stays visible as failed/partial/unavailable and cannot be silently
   replaced by a different source class.
8. **Assessment is allowed; unsupported authority is not.** Historical,
   research, causal, and forward-looking claims may be assessed independently
   with calibrated status and cited source IDs. Diagnosis, prescribing, dose
   selection, and claims that qualified professional review is unnecessary
   remain `outside_authority`; this status does not erase assessable upstream
   claims.

## Engine API (`memory/pattern_engine.py`)

`run_pattern_query(PatternQuery, *, corpus_manager, user_profile,
telemetry_path) -> PatternResult` — synchronous, read-only, never raises
(failures degrade to a noted empty result). `PatternQuery.now` is injectable
(test determinism).

| Dimension | Source | Notes |
|---|---|---|
| `topic_keyword` | `corpus_manager.search_keyword` (word-boundary, time-bounded, speaker-attributed) | **User-side hits only by default** — the assistant echoes the user's vocabulary constantly; `speaker="both"` opts in |
| `tone` | `logs/turn_records.jsonl` stream-parse | `test_env` rows and `distress_sticky_floor` turns excluded; coverage note when telemetry starts inside the window |
| `relation` | `user_profile.get_fact_history` | value timeline; appraisal-stance values labeled (stance doctrine) |
| `session_rhythm` | corpus timestamps | messages + sessions per bucket (>30 min gap = new session), median first-message hour |
| `content_type` | deployed `content_type_detector` over corpus entries at query time | no backfill needed; new turns also persist the tag in chroma metadata |
| `daily_notes` | Obsidian daily-note frontmatter + Emotional State section (`read_daily_note`) | per-day `usage_intensity`/`active_hours` series + emotional-state exemplars; missing days noted ("absence ≠ nothing happened"); auto-added as a mood-theme overlay |

**Counted vs. evidence — where each store participates.** The engine COUNTS
only independent series: conversational turns (corpus), telemetry, profile
fact history, and daily-note frontmatter. **Summaries and reflections are
deliberately never counted** — they compress the same conversations the
corpus dimensions already count, so counting a mention in both a turn and
its summary double-counts one event. They still participate fully as
EVIDENCE: the insight sweep quotes summaries, reflections, Obsidian notes,
threads, and facts beside the computed numbers. Daily/weekly notes are the
exception that IS counted (via `daily_notes`) because their frontmatter
metrics are an independent per-day observation, not a compression.

`PatternResult`: buckets (empty buckets INCLUDED — a gap is information),
total, active_days, longest_streak/gap (day granularity), first/last_seen,
trend (first-half vs second-half, deliberately coarse: increasing /
decreasing / stable / insufficient), denominator_total, notes.
`render_table()` produces the compact block the synthesizer consumes.

Buckets auto-size: window ≤14d → day, ≤120d → week, else month.

## Generic deliberation integration

- `core/insight/detector.py` recognizes common English longitudinal shapes
  (frequency/trend, before/after, changed-since, variables tracking together,
  theory-versus-record, explicit pattern tool). The unified LLM routing
  fallback owns unusual personal longitudinal requests, including mixed
  requests that also name web/PubMed/Wikipedia/tools, before ordinary web
  routing can consume the turn.
- `core/insight/deliberation.py` asks the model only to propose a strict JSON
  plan. Validation bounds every field, rejects missing outcomes and invalid or
  cyclic claim graphs, preserves explicitly requested tools, and requires an
  optimized query for every external source. Raw prose is never converted into
  outcome keywords by a fallback heuristic.
- `core/insight/coordinator.py` owns the frozen plan. It resolves explicit,
  relative, or record-located anchors without silently choosing among
  ambiguous dates; executes source adapters concurrently with timeouts;
  distinguishes success, partial failure, failure, and unavailability; runs
  exact phase assignment; and assesses each dependency-aware claim against
  only admissible source IDs.
- Live adapters currently cover full user corpus, Obsidian notes, extracted
  fact locators, uploaded/reference files, cached Wikipedia, Tavily web,
  PubMed, arXiv, Stack Exchange, and Wolfram Alpha when configured. Public API
  parsers preserve stable IDs, titles, abstracts/text, dates, authors, and
  canonical URLs instead of flattening results to untraceable prose.
- `memory/pattern_engine.py` separates phase outcome events from all admissible
  observations (the denominator) and from phase-aligned frozen
  covariate/confounder events. Literal matching uses word boundaries; one
  canonical source lineage is counted once; assistant-derived material is
  excluded from outcome evidence; increase/decrease is descriptive and never
  automatically mapped to good/bad.
- `core/insight/temporal.py` remains the rolling frequency/trend adapter for
  `time_series` plans only. Exact event/period comparisons use the resolved
  phase scan, avoiding a second default window with a different evidence set.
- `core/insight/synthesizer.py`: `_PATTERN_TAIL` (restate-never-recount,
  trend confidence calibrated to n, bounded causal assessment only when
  supported by explicit phases and counterevidence, denominator with
  numbers) plus a structurally bounded JSON deliberation manifest. Claim
  statuses remain separate; an outside-authority downstream claim never
  flattens the historical or research findings.
- `gui/handlers._run_insight_mode` runs planning and requested adapters before
  any raw-theme sweep. If planning fails, retrieval against conversational
  framing words is skipped. Frozen phase/research evidence is ordered before
  supplemental sweep evidence so prompt caps cannot crowd it out.

Detector hardening shipped with v1 (`core/content_type_detector.py`): lyric
section headers (`[Verse 1]`, `[Chorus]`) are a max-precision lyrics marker
(0.85) beating the structural heuristic that page chrome defeats; a long
message framed by song-listening narration ("this song is making me think…",
≥1200 chars) is a lyrics share (0.70) even with newlines collapsed — both
live 2026-08-28/29 paste shapes now detect.

## Current and future surfaces

Deliberate non-goals for v1, each pre-designed as an adapter:
- **`pattern_scan` agentic tool** — available for bounded deterministic scans;
  longitudinal deliberation additionally freezes phases and reports channel
  coverage. Rich semantic behavioral coding remains future work.
  (DISPATCH_TABLE + protocols + parity test wiring).
- **SPA 📈 Patterns view** — `/api/patterns` route + CurationPage-style view.
- **Shutdown pattern notes / `[OBSERVED PATTERNS]` prompt section /
  15th-collection proactive store** — **considered and DECLINED by the owner
  (2026-08-29)**. The deciding argument: semantic retrieval would surface a
  tone/mood pattern doc precisely when the user is talking about feeling bad
  — the mood-congruent-amplification mechanism the 07-21 anti-amplification
  batch exists to prevent (same class as the refdocs distress leak). If ever
  revisited, the agreed path is SHADOW MODE first (compute + store at
  shutdown, inject nothing, review what would have surfaced), with mood/tone
  dimensions permanently on-demand-only, distress-gated injection, 1-2 item
  cap, and dated/superseding docs. Do not build proactive surfacing without
  a fresh owner decision.

## Verification expectations

The focused regression suite must contain multiple domains (health is only
one), uncommon English routing shapes, explicit and relative phases,
inclusive calendar boundaries, non-hit denominators, cross-store lineage
deduplication, generic direction coding, phase-aligned confounders, planner and
adapter failures, required-channel enforcement, source-ID citation rejection,
structured PubMed/arXiv/Stack Exchange parser fixtures, and bounded synthesis
manifests. Run related suites in memory-bounded batches per `CLAUDE.md`.
