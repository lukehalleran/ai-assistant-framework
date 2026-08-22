# GENERALIZATION_AUDIT.md — "Me-Shaped" Code Inventory & De-Personalization Plan

*2026-08-21. Three-agent sweep of the codebase for owner-specific assumptions,
motivated by two questions: (1) why did the "program is online" fact need
manual injection + a hardcoded quick-key, and (2) how owner-shaped are the
behavioral gates (vent-shape etc.)? Target population: any American-English
user, fresh profile or base-seeded.*

## The taxonomy

Every finding falls into one of three shapes — they need different remedies:

1. **MECHANISM-GENERAL** — the logic works for anyone (vent-shape's structure:
   first-person + non-info-seeking after epistemic stripping; truth scoring;
   TTL tiers; dedup). *No action.*
2. **OWNER-TUNED** — general mechanism, but constants/seeds/lists were
   calibrated on the owner's data or accreted from his incidents. *Remedy:
   per-user adaptation or self-calibration, not better constants.*
3. **OWNER-SPECIFIC** — literally the owner in code (his name, test strings,
   hobbies, vault path). *Remedy: externalize to config/data.*

## The key insight: the codebase already contains its own answer, three times

Three general patterns exist and work; generalization = applying them uniformly:

- **Seeds + per-user learned store** (`utils/adaptive_exemplars.py`): code
  ships neutral seeds; confirmed live classifications personalize per user
  with poisoning guards. *Gap found: several seeds are contaminated with the
  owner's idiolect/life (HOI4 exemplar, "which gym do I usually go to").*
- **Auto-promotion from observed data** (`memory/learned_relations.py`):
  vocabulary grows from what survives gates on ≥3 distinct days. *Gap: not
  applied to CORE_RELATIONS trimming, quick-profile identity facts, or
  hot-topic retrieval pools.*
- **Calibrate-on-own-data probes** (`scripts/probe_gate_embedding_mismatch.py`,
  `probe_tone_backstop.py`, `budget_experiment.py`): thresholds derived from
  the corpus, not guessed. *Gap: one-shot owner-run tools, not first-run /
  periodic self-calibration.*

Plus the externalization pattern (`config.local.yaml`) for literal PII —
already established, incompletely applied.

## P0 — breaks or misbehaves for ANY other user immediately

| Item | Where | Problem | Fix |
|---|---|---|---|
| "Luke" hardcoded in notes prompts | `utils/daily_notes_generator.py:132+`, `monthly_notes_generator.py:64+` | prompts literally address Luke | thread `user_name` from profile |
| Vault path `~/Documents/Luke Notes` | `daily_notes_generator.py:221`, `monthly_notes_generator.py:156` | broken path for anyone else | config key (obsidian section exists) |
| Box-test battery exact strings | `memory/utils.py:29-35` (`_TRIVIAL_TEST_QUERIES`) | "should I add prompt caching?" is a REAL question other users ask — legit turns junk-filtered | move to per-install junk list file (data/, seeded empty); keep mechanism |
| Quick-key curation | `memory/user_profile.py:_update_quick_profile` | school/program added by hand for the owner; other users' identity-grade facts (partner, disability, night-shift job) never promote | **identity auto-promotion** (below) |

## P1 — wrong-for-others behavior baked into seeds/special-cases

| Item | Where | Fix |
|---|---|---|
| HOI4 / strategy-game crisis exemplar | `tone_detector.py:466-468` | rephrase neutral ("I crushed everyone in my game last night" is fine; drop owner-comment coupling); exemplar-store learned channel handles per-user gaming idiom |
| "which gym do I usually go to" intent exemplar | `intent_classifier.py:505` | neutral habit phrasing |
| project_work exemplars naming Daemon internals | `intent_classifier.py:523-527` | generic project phrasings |
| Care-team relations in CORE_RELATIONS | `llm_fact_extractor.py:157-166` | shrink CORE to universal relations; pre-seed owner's extras into `data/learned_relations.json` as promoted (the store supports it) |
| `is_gym_health_query` keyword special-case | `memory_retriever.py:1296-1309` | replace fixed list with top-N hot topics derived from `logs/turn_records.jsonl` topic distribution (telemetry already records topics) |
| STM substance rule framed around meds | `stm_analyzer.py:24-28` | generalize to ALL entity naming: current-message mention always wins over context-dominant entity |

## P2 — calibration constants that silently assume the owner's distribution

| Item | Where | Fresh-user story | Fix |
|---|---|---|---|
| Gate thresholds 0.60/0.61 | `app_config` / gate_system | quantile-matched to owner's corpus; meaningless for an empty store | first-run defaults + periodic self-calibration job (re-run the probe quantile logic once corpus > N docs) |
| Tone thresholds 0.58/0.50/0.43 + 0.37 backstop | `tone_detector.py:81-101` | probe-calibrated on owner's misses | env-overridable today; add probe-as-maintenance-job; NEVER auto-adjust without the probe-set regression gate (knife-edge lesson 08-14) |
| escalation_threshold=3 / distress_threshold=5 | `escalation_tracker.py:122-125` | tuned to owner's terse-spiral cadence | expose in YAML; consider per-user message-cadence normalization later |
| PROMPT_TOKEN_BUDGET_DEFAULT=10000 | config | validated on the owner's 27-query corpus | the experiment harness (`budget_experiment.py`) is reusable per user; keep 10K as default |
| CONTINUATION_MAX_WORDS=6, valence saturation 5.0, caps | various | plausible AmE defaults | leave; document as defaults not truths |

## P3 — AmE-general lists that grew via owner incidents (coverage bias, not wrongness)

Epistemic markers, info-seeking cues, casual starters, heavy-topic keywords,
MEDIUM_CRISIS_KEYWORDS additions. The entries are general American English;
the RISK is what's absent — holes were discovered through ONE user's idiolect,
so another user's phrasing patterns have undiscovered holes. The remedy is not
longer lists; it is the two feedback mechanisms that detect misses for any
user:

1. **Adaptive learning loops** (already shipped for tone/need/intent/web) —
   confirmed misses become that user's exemplars.
2. **Visible failure + user-as-oracle** (the 2026-08-21 tone-deferral clarify
   loop) — when a gate blocks a request, the user is asked, and the override
   is immediate signal. Silent failure is what made holes expensive to find;
   visibility makes every user their own hole-detector.

## The direct answer to "how do we never need to inject a fact again"

The program fact's lifecycle shows the failure was **prominence, not data**:
extraction captured it (07-27), retrieval even surfaced it in the failing
prompt — the model steamrolled one flat fact line with a world-prior. The
manual steps (curated re-add + hardcoding quick keys) each patched prominence
by hand.

General mechanism — **identity-grade fact auto-promotion**: promote facts
into the always-rendered Quick Profile block by SIGNALS, not by a relation
whitelist. Every fact already carries what's needed (no schema change):
`category`, `confidence`, `confirmation_count`, `last_confirmed_at`,
`truth_score`, `is_current`. Sketch:

```
promote if: is_current AND truth_score ≥ 0.7 AND (
    category ∈ {identity, education, work, relationships, health-durable}
    AND (confirmation_count ≥ 2 OR age ≥ 30d without contradiction))
cap: ~10 entries, ranked by (category priority, confirmation_count, truth);
demote on supersession; quick_keys floor (name/location/…) retained.
```

A user whose durable fact is "works night shifts" or "is a wheelchair user"
gets the same always-visible treatment the owner's "fully online program" now
gets — with nobody editing a list. NOT built yet — parked here for a
dedicated session (churn/demotion semantics deserve care + tests).

## Vent-shape specifically (the question that prompted this)

Verdict: **structure general, discovery biased**. First-person + non-info-
seeking after epistemic stripping is sound AmE linguistics and ports to any
user. The marker/cue/verb lists are general American English (idk/imo/afaik
are internet-English, not Luke-English). What IS owner-shaped: (a) which
holes got patched — each was found via one of the owner's incidents, so
coverage follows his idiolect; (b) the tone calibration that feeds the veto
(owner-probe-derived thresholds). Both are addressed by P2 self-calibration +
P3 feedback loops rather than by list edits.

## Status

Audit only — no generalization changes applied yet (2026-08-21). The
inventory above is the worklist; P0 items are small and safe to do first.
