# Daemon Project Goals

**Last Updated**: 2026-03-15

---

## Mission

Daemon has two purposes that reinforce each other:

1. **Personal cognitive assistant** — A companion with deep, persistent memory that genuinely knows you. Your projects, preferences, history, emotional state, and context. Not a stateless chatbot. This is both valuable on its own and serves as a live testbed for the retrieval and scoring infrastructure.

2. **Knowledge synthesis engine** — The long-term goal. Use random walks (Markov chains) across large embedded knowledge bases (Wikipedia, arXiv, PubMed) to generate candidate cross-domain connections, then filter them through a multi-stage pipeline to surface genuinely novel, non-obvious links between ideas. Most output will be junk. The entire value is in the filter.

Everything built so far — the multi-stage gating, composite scoring, cross-encoder reranking, truth scoring, deduplication, intent classification — is filter infrastructure tested daily against real retrieval problems.

---

## Active Goals (Current Sprint)

### 1. Infrastructure Consolidation
- **Status**: Starting — retrieval benchmarks now provide safety net for subtractive changes
- **Why**: Codebase at 94K lines approaching solo maintainability ceiling (~100-120K). Must be lean and reliable before scaling to batch synthesis workloads.
- Use retrieval benchmark suite to ablate prompt sections — measure recall impact of removing each
- Audit ChromaDB collections: do all 10 justify separate indexes? Can any be merged without retrieval regression?
- Reduce prompt context sections (currently 14+) to minimum set that maintains quality
- Clean up mutable state side-channels in scorer/builder pipeline (e.g., `_intent_weight_overrides` on MemoryScorer — replace with scoped context object or context manager)
- Resolve JSON corpus + ChromaDB dual-write: define source of truth, add reconciliation or eliminate one path
- Target: reduce codebase to ~85K lines without measurable retrieval quality loss
- Target: orchestrator.py under 400 lines (extract remaining truth/escalation/correction logic)
- Eliminate remaining calls to deprecated `prepare_prompt`

### 2. Proposal System Quality
- **Status**: Functional, needs filtering improvements
- Hundreds of unread proposals — retrieval bias surfaces additive proposals matching current work context
- Add type-weighted retrieval: boost `refactor` and `deletion` proposals in default views
- Add dedicated consolidation query mode to surface subtractive proposals
- Periodic proposal pruning: expire or archive low-scored proposals older than N days
- Improve generation prompt to explicitly produce subtractive proposals (merges, deletions, simplifications)

### 3. Agentic Search Reliability
- **Status**: Stabilizing
- Improve loop termination, reduce unnecessary search triggers
- Query relaxation for failed searches

---

## Medium-Term Goals (Next 1-3 Months)

### 4. Knowledge Synthesis Pipeline (Core Vision)
- **Status**: Infrastructure built, first real run pending
- **This is the primary goal. Everything else supports this.**

**Data Ingestion**:
- Wikipedia embedded in `wiki_knowledge` collection — **done** (currently query-based retrieval; will need different access pattern for random-walk synthesis)
- arXiv paper abstracts/full text — planned
- PubMed abstracts — planned
- Unified embedding space across all sources

**Connection Generation**:
- Markov chain random walks across embedded knowledge graph
- Walk results passed to LLM with synthesis instructions to articulate connections
- High volume — most output will be nonsense or trivially true

**Multi-Stage Filtering** (the hard problem):
- Stage 1: Cheap bulk filter — cosine similarity against baseline of known good connections, kill ~80-90% of incoherent noise
- Stage 2: LLM coherence judge (smaller model) — "is this connection logically sound and interesting?" (not "is this true")
- Stage 3: LLM pairwise reranking (stronger model) — relative novelty comparison across candidates
- Stage 4: Novelty detection — embed winners, search against source literature. High similarity = known connection. Low similarity + high coherence = signal. (Inverted retrieval: confirm nothing similar exists)

**Open Questions**:
- How to distinguish "interesting but known" from "actually novel"
- Optimal Markov walk parameters (step count, restart probability, domain crossing frequency)
- Whether coherence and novelty should be separate stages or combined
- Calibration: what does the score distribution look like for known good cross-domain papers?

### 5. Dreaming Engine (Batch Generation Infrastructure)
- **Status**: Pending — blocked on consolidation (Goal 1) and first manual synthesis run
- Background thread/async task runs during idle periods
- Iterates over knowledge bases with Markov walks, generates connection candidates in batches
- Full pipeline scoring: cosine → cross-encoder → LLM pairwise → novelty check
- Coverage tracking: which domains/topics explored, avoid redundant generation
- Scheduler: activates after N minutes inactivity, pauses on user input
- **Prerequisite**: Infrastructure must be lean and hardened first — batch workloads will stress every pipeline component at scale

### 6. Memory Quality Maintenance
- Prune low-value entries over time (decay + consolidation)
- Improve fact extraction precision (reduce false triples)
- Monitor retrieval benchmark scores for regression after any scoring/weight changes

---

## Foundational Infrastructure (Built, Maintaining)

These systems are complete and working. Listed here for context, not as active work items. Changes should be consolidation-oriented (simplify, merge, delete) rather than additive.

- **Memory system**: 5 types (episodic, semantic, procedural, summary, meta), modular components with Protocol contracts, ~498 line thin coordinator
- **Multi-stage gating**: FAISS → Cosine → Cross-Encoder reranking
- **Intent classification**: 9 types, regex-first, per-intent weight/retrieval/gate overrides, STM refinement
- **Truth scoring**: Evidence-based (TruthScorer + CorrectionDetector), replaces access-count system
- **Cross-collection dedup**: Cosine duplicates + fact contradiction detection, dry-run default
- **Escalation tracker**: 4-state emotional momentum (VALIDATE → GROUNDING → QUIET → GENTLE)
- **User profile**: Append-only with temporal history, 12 categories, hybrid retrieval
- **Temporal awareness**: Narrative context (3-tier: monthly/weekly/daily), temporal-aware recency decay
- **Knowledge integration**: Obsidian vault (multimodal, mtime-based re-embedding), reference docs, git commits, procedural skills, Wikipedia
- **Fast Mode**: Reduced retrieval for mobile/slow connections with progress keepalives
- **PDF support**: Full pipeline from upload through pdfplumber text extraction
- **Multi-provider LLM**: Duel mode, best-of-N, ensemble, fallback chains
- **Code proposals**: LLM-generated, ChromaDB-stored, GUI management, shutdown integration
- **Agentic search**: ReAct loop with Tavily + Wolfram Alpha + E2B sandbox
- **Production**: PyInstaller desktop build, Docker deployment, graceful shutdown
- **Privacy**: All data local, API calls only for LLM generation, no telemetry
- **Knowledge graph**: Queryable fact graph with connectivity-ranked query expansion, junk node prevention at ingestion, graph-boosted memory scoring
- **Testing**: 1,670+ tests, retrieval quality benchmarks with 30 seed memories and 19 test cases

---

## Recent Completions

### Broadened Agentic Gate for Memory Search (2026-03-15)
- 3-tier agentic trigger in `gui/handlers.py`: keyword heuristic → knowledge graph entity match → LLM fallback
- Memory keyword list (20 phrases): "do you remember", "my notes", "search your memory", etc.
- Entity match via `extract_graph_entities()`: queries mentioning known entities (Flapjack, Auggie, etc.) auto-route to agentic memory search
- LLM `needs_memory_search` field added to `WebSearchDecision` — piggybacks on existing web search trigger call (zero extra cost)
- Fixed casual skip filter: `'no'` prefix was matching `'now'` (startswith bug); entity/keyword checks now run before skip filter
- `skip_initial_search=True` for memory queries — skips web search, ReAct loop goes straight to `search_memory` tool

### Smarter Query Expansion + Junk Node Prevention (2026-03-15)
- `rank_expansion_candidates()` in `graph_utils.py` — ranks expansion terms by lateral connectivity (non-hub edge count) instead of name length. Real entities (Flapjack=8 edges) now rank above junk phrases ("coffee"=0 edges).
- `_is_expansion_junk()` — filters temporal ("2 years"), measurements ("5'11\""), verb phrases ("stopped being religious"), 4+ word names from expansion results
- Tightened `_is_graph_worthy_object()` in `memory_storage.py` — prevents new junk nodes at ingestion: temporal/duration, measurements, verb-phrase objects now stored as subject-node metadata instead of creating graph nodes
- 50 graph integration tests (was 37), 70 knowledge graph tests — all passing

### Monthly Notes + 3-Tier Hierarchy (2026-03-10)
- 3-level folder structure: `<Month YYYY>/<Week N Mon YYYY>/<filename>` with legacy path compatibility
- Monthly summary generator (`utils/monthly_notes_generator.py`), GUI startup catch-up, CLI commands
- Narrative context upgraded to 3-tier synthesis (monthly for arc, weekly for threads, daily for recent shifts)

### Model Migration to sonnet-4.5 (2026-03-10)
- Default lightweight model switched from `gpt-4o-mini` to `sonnet-4.5` across all generators (daily/weekly/monthly notes, tags, narrative synthesis)
- `sonnet-4.6` added to model registry

### PDF Upload Support (2026-03-10)
- `pdfplumber`-based text extraction in `file_processor.py` with page headers for multi-page PDFs
- `.pdf` added to `FILE_UPLOAD_ALLOWED_EXTENSIONS`

### Fast Mode (2026-03-10)
- Mobile-friendly reduced retrieval: fewer candidates, fewer context items, progress keepalive messages
- Toggled via GUI checkbox, reduces retrieval limits in context_gatherer + hybrid_retriever

### Obsidian Mtime-Based Re-Embedding (2026-03-10)
- `obsidian_manager.py` now compares file mtime vs stored mtime metadata for change detection
- Changed files: old chunks deleted, then re-embedded (instead of skip-if-exists)
- `EmbedResult.updated_files` tracks re-embedded files separately

### Temporal-Aware Recency Decay (2026-02-17)
- `memory/memory_scorer.py` — `rank_memories()` reshapes recency decay for TEMPORAL_RECALL queries
- `core/intent_classifier.py` — `classify()` extracts temporal anchor via `extract_temporal_window()`
- 4 new unit tests in `tests/unit/test_memory_scorer.py`

### Retrieval Quality Benchmarks (2026-02-17)
- `tests/benchmarks/` — End-to-end suite with real embeddings (all-MiniLM-L6-v2)
- 30 seed memories + 19 test cases across all 9 intent types, recall@K and MRR

### Wizard Test Env Fix (2026-02-17)
- `tests/test_wizard.py` — Fixed env var poisoning via try/finally save/restore

### Query Intent Classifier (2026-02-15)
- 9 intent types, regex-first, per-intent weight/retrieval/gate profiles, STM refinement
- 74 unit tests

### Evidence-Based Truth Scoring (2026-02-15)
- TruthScorer + CorrectionDetector, replaces access-count echo-chamber
- 25 + 16 unit tests

### Cross-Collection Deduplicator (2026-02-13)
- Cosine duplicates + fact contradiction detection, dry-run default, GUI preview/run
- 47 unit tests

### Escalation Tracker (2026-02-13)
- 4-state crisis cooldown (VALIDATE → GROUNDING → QUIET → GENTLE)
- 51 unit tests

### Code Proposals System (2026-02-09)
- LLM-generated proposals, ChromaDB storage, GUI management, shutdown integration
- 71 unit tests

### Procedural Skills + Git Memory (2026-01-27)
- Adaptive WHEN/THEN workflows, git commit history as procedural knowledge

---

## Non-Goals (Explicit Exclusions)

- **Auto-execution of code proposals** — proposals are advisory only, human reviews and implements
- **Auto-deletion of user data** — all destructive operations default to dry_run=True; live deletions require explicit GUI action
- **Real-time collaboration** — single-user system, not multi-tenant
- **Mobile/web deployment** — desktop-first (Gradio GUI or CLI)
- **Fine-tuning or training** — uses commercial APIs and pre-trained local models only
- **Adding infrastructure complexity before consolidation** — new systems must justify their existence against the maintainability budget. Default answer to "should I add this?" is "not yet."

---

## Principles

1. **The filter is the product.** Connection generation is cheap. Identifying which connections are novel and meaningful is the entire value proposition. Every infrastructure decision should be evaluated by whether it improves filtering quality.

2. **Subtractive work is higher priority than additive work.** The codebase must get leaner before it gets bigger. Removing a system that doesn't pull its weight is more valuable than adding a new one.

3. **The personal assistant earns its keep.** It's not scaffolding — it's a daily-use product AND the live testbed. Both purposes matter.

4. **Measure before changing, measure after changing.** The retrieval benchmark suite exists for a reason. No scoring/weight/retrieval changes without before-and-after benchmark runs.

5. **Batch scale reveals infrastructure debt.** Every quirk that's tolerable at conversational volume becomes a critical bug at synthesis volume. Consolidate now while the feedback loop is fast.
