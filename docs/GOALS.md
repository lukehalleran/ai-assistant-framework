# Daemon Project Goals

**Last Updated**: 2026-02-15

---

## Mission

Daemon is a personal cognitive assistant with deep, persistent memory. It should feel like talking to someone who genuinely knows you — your projects, preferences, history, and context — not a stateless chatbot that forgets everything between sessions.

---

## Active Goals (Current Sprint)

### 1. Complete Modular Refactor (`refactor/prompt-modular`)
- **Status**: In progress (mostly done)
- Orchestrator should be a thin coordinator, not a god object
- All query analysis flows through `ContextPipeline` — **done**
- All prompt building flows through `UnifiedPromptBuilder` — **done**
- Memory system delegates to `MemoryRetriever`, `MemoryStorage`, `MemoryScorer`, `ShutdownProcessor` — **done**
- Intent-driven retrieval and scoring via `IntentClassifier` (Stage 4.5) — **done 2026-02-15**
- Evidence-based truth scoring via `TruthScorer` + `CorrectionDetector` — **done 2026-02-15**
- Eliminate remaining calls to deprecated `prepare_prompt`
- Target: orchestrator.py under 400 lines (currently still large — truth/escalation/correction logic added)

### 2. Code Proposals System (Goal-Directed)
- **Status**: In progress (Phases 1-3 complete, GUI + shutdown integration done)
- LLM-generated proposals for codebase improvements, stored in ChromaDB
- Proposals generated at shutdown from session context and on-demand from GUI
- Deduplication via semantic similarity
- Next: improve proposal quality with richer project context, CLAUDE.md injection

### 3. Agentic Search Reliability
- **Status**: Stabilizing
- ReAct loop with Tavily + Wolfram + E2B sandbox
- LLM-first trigger (not keyword heuristic) decides when to search
- Query relaxation for failed searches
- Next: improve loop termination, reduce unnecessary search triggers

---

## Medium-Term Goals (Next 1-3 Months)

### 4. Memory Quality Over Quantity
- Summaries, facts, and skills should be high-signal, not noisy
- Improve fact extraction precision (reduce false triples)
- Cross-collection deduplication (duplicates + fact contradictions) — **done 2026-02-13**
  - `CrossCollectionDeduplicator`: cosine >= 0.92 for duplicates, subject+predicate match for contradictions
  - Dry-run only on shutdown; live deletions require explicit GUI action
  - Contradiction resolution feeds truth penalties to TruthScorer
- Evidence-based truth scoring — **done 2026-02-15**
  - `TruthScorer`: initial scores by source, confirmation/correction/contradiction adjustments, time decay
  - `CorrectionDetector`: pattern-based detection of user corrections and confirmations in conversation
  - Replaces old access-count echo-chamber system (`update_truth_scores_on_access()` → no-op)
  - `MemoryScorer.rank_memories()` uses `TruthScorer.compute_effective_truth()` for dynamic scoring
  - User profile `add_fact()` sets truth metadata on all three add paths (new/confirm/correct)
  - Orchestrator runs correction/confirmation detection after `store_interaction()`
- Intent-aware retrieval tuning per query type — **done 2026-02-15**
  - `IntentClassifier`: 9 intent types with per-intent weight/retrieval/gate profiles
  - Retrieval overrides: CASUAL_SOCIAL minimizes retrieval, FACTUAL_RECALL maximizes facts, TEMPORAL_RECALL boosts summaries
  - Weight overrides: EMOTIONAL_SUPPORT boosts continuity + lowers truth, FACTUAL_RECALL boosts truth
- Prune low-value entries over time (decay + consolidation)

### 5. Temporal Awareness
- Narrative context system provides rolling "life state" grounding — **done**
- Daily + weekly notes auto-generated from conversations — **done**
- Goal: Daemon should know what you were working on last week without being told

### 6. Multi-Provider LLM Resilience
- Duel mode (2 models + judge) for high-stakes responses — **done**
- Best-of-N with diverse temperatures — **done**
- Graceful fallback chains: primary API -> secondary API -> local model — **done**
- Token budget management across providers with different context limits

### 7. Personal Knowledge Integration
- Obsidian vault with multimodal image support — **done**
- Reference document uploads (technical docs, project outlines) — **done**
- Git commit history as procedural knowledge — **done**
- Procedural skills (adaptive WHEN/THEN workflows) — **done**
- Goal: all personal knowledge surfaces when relevant, not just chat history

---

## Long-Term Vision

### 8. Proactive Intelligence
- Daemon should surface relevant information before being asked
- Code proposals are the first step — extend to life reminders, project nudges
- Pattern recognition across sessions (recurring themes, unresolved questions)

### 9. Production Readiness
- Desktop executable via PyInstaller (daemon.spec exists) — **done**
- Docker deployment (Dockerfile + docker-compose exist) — **done**
- Health check endpoint for monitoring
- Graceful shutdown with full memory persistence — **done**

### 10. Privacy and Trust
- All data stays local (ChromaDB on disk, JSON corpus) — **done**
- API calls only for LLM generation (no telemetry, no cloud storage) — **done**
- User profile and identity managed locally — **done**
- Crisis-aware tone detection with appropriate response escalation — **done**
  - Escalation Tracker for crisis cooldown (VALIDATE → GROUNDING → QUIET → GENTLE)

---

## Recent Completions

### Query Intent Classifier (2026-02-15)
- `core/intent_classifier.py` — Regex-first classification into 9 intent types (no LLM calls)
- `IntentType`: FACTUAL_RECALL, TEMPORAL_RECALL, EMOTIONAL_SUPPORT, CASUAL_SOCIAL, TECHNICAL_HELP, CREATIVE_EXPLORATION, META_CONVERSATIONAL, PROJECT_WORK, GENERAL
- `IntentResult` dataclass: intent, confidence, source, weight_overrides, retrieval_overrides, gate_threshold_override
- Per-intent `_PROFILES` dict: weight overrides (SCORE_WEIGHTS), retrieval count overrides (PROMPT_MAX_*), gate threshold overrides
- Tone bias: HIGH/MEDIUM tone_level → EMOTIONAL_SUPPORT for ambiguous queries
- STM refinement: low-confidence results (< 0.50) upgraded via keyword-matching STM's free-text intent
- Integrated at Stage 4.5 of ContextPipeline, refined by STM at Stage 6b
- `MemoryScorer.rank_memories(weight_overrides=...)` uses per-intent weights; `_intent_weight_overrides` instance attribute for deep call chains
- `UnifiedPromptBuilder.build_prompt(retrieval_overrides=..., weight_overrides=...)` applies overrides to all parallel retrieval tasks
- Config: `INTENT_ENABLED`, `INTENT_STM_REFINEMENT_THRESHOLD` (app_config.py + config.yaml)
- 74 unit tests in `tests/unit/test_intent_classifier.py`, all passing

### Evidence-Based Truth Scoring (2026-02-15)
- `memory/truth_scorer.py` — Stateless TruthScorer class
  - `calculate_initial_score(source)`: user_stated=0.80, corrected=0.75, llm_extracted=0.60, inferred=0.50
  - `apply_confirmation(score)` / `apply_correction(score)` / `apply_contradiction(score)`: evidence-based adjustments
  - `compute_effective_truth(metadata)`: dynamic truth with time decay at read time
- `core/correction_detector.py` — CorrectionDetector + CorrectionEvent (Pydantic)
  - Pattern-based detection of user corrections ("actually it's X", "no, my Y is Z")
  - Pattern-based detection of user confirmations ("yes", "that's right", "correct")
  - Returns CorrectionEvent with fact_id, event_type, relation, old_value, new_value
- Integrated changes:
  - `memory_scorer.py`: `rank_memories()` uses `TruthScorer.compute_effective_truth(md)` instead of access-count boosting; `update_truth_scores_on_access()` is now a no-op
  - `user_profile.py`: `add_fact()` sets truth metadata (truth_score, truth_source, last_confirmed_at, confirmation_count) on all three cases (new/confirm/correct)
  - `cross_deduplicator.py`: `_apply_contradiction_truth_penalties()` feeds contradiction penalties to TruthScorer when resolving fact contradictions
  - `orchestrator.py`: runs correction/confirmation detection after `store_interaction()`, applies truth events to profile facts
- Config: `TRUTH_SCORER_ENABLED`, `TRUTH_SCORER_CORRECTION_DETECTION`, `TRUTH_SCORER_TIME_DECAY_*` (app_config.py)
- 25 unit tests in `test_truth_scorer.py` + 16 in `test_correction_detector.py`, all passing

### Cross-Collection Deduplicator (2026-02-13)
- `memory/cross_deduplicator.py` — CrossCollectionDeduplicator: unified dedup across ChromaDB collections
- `memory/dedup_models.py` — Pydantic models: DedupPlan, DuplicatePair, ContradictionCluster
- Two modes: cross-duplicate detection (cosine >= 0.92) and fact contradiction detection (same subj+pred, diff obj)
- Ephemeral relation skip: predicates in PROFILE_EPHEMERAL_RELATIONS excluded from dedup
- Dry-run only on shutdown; live deletions require explicit GUI action (Preview/Run buttons)
- Contradiction resolution now feeds truth penalties via TruthScorer (2026-02-15 enhancement)
- 47 unit tests in `tests/unit/test_cross_deduplicator.py`, all passing

### Escalation Tracker (2026-02-13)
- `core/escalation_tracker.py` — EscalationTracker class, ResponseStrategy enum
- Tracks session-level emotional momentum: consecutive elevated count, engagement detection, escalation velocity
- Strategy progression: VALIDATE_AND_SUGGEST → GROUNDING_PRESENCE → QUIET_COMPANIONSHIP → GENTLE_REENGAGEMENT
- De-escalation nuance: PERSPECTIVE need_type = intensity shift (→ VALIDATE), not genuine calming (→ GENTLE)
- Integrated into orchestrator: updated after build_context(), strategy instructions appended to system prompt, token budget override
- 51 unit tests in `tests/unit/test_escalation_tracker.py`, all passing

### Code Proposals System (2026-02-09)
- `memory/code_proposal.py` — Pydantic data models (CodeProposal, ImplementationStep, enums)
- `knowledge/proposal_generator.py` — GoalDirectedGenerator (LLM-based proposal generation with pipeline-enriched context)
- `memory/proposal_store.py` — ChromaDB-backed storage with dedup, query, status lifecycle
- GUI management tab + shutdown integration
- 71 unit tests across 3 test files, all passing

### Procedural Skills System (2026-01-27)
- `memory/procedural_skill.py` — ProceduralSkill dataclass + SkillCategory enum
- Adaptive WHEN/THEN workflows stored in `procedural_skills` ChromaDB collection
- Extracted at shutdown from session conversations

### Git Memory System (2026-01-27)
- `knowledge/git_memory.py` — Git commit history extraction with metadata + conventional commit tagging
- `knowledge/git_memory_loader.py` — Backfill/incremental sync of git commits to PROCEDURAL ChromaDB collection

---

## Non-Goals (Explicit Exclusions)
- **Auto-execution of code proposals** — proposals are advisory only, human reviews and implements
- **Auto-deletion of user data** — all destructive operations default to dry_run=True; live deletions require explicit GUI action
- **Real-time collaboration** — single-user system, not multi-tenant
- **Mobile/web deployment** — desktop-first (Gradio GUI or CLI)
- **Fine-tuning or training** — uses commercial APIs and pre-trained local models only
