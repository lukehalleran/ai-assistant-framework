# Daemon Project Goals

**Last Updated**: 2026-03-31

---

## Mission

Daemon has two purposes that reinforce each other:

1. **Personal cognitive assistant** — A companion with deep, persistent memory that genuinely knows you. Your projects, preferences, history, emotional state, and context. Not a stateless chatbot. This is both valuable on its own and serves as a live testbed for the retrieval and scoring infrastructure.

2. **Knowledge synthesis engine** — The long-term goal. Use random walks (Markov chains) across large embedded knowledge bases (Wikipedia, arXiv, PubMed) to generate candidate cross-domain connections, then filter them through a multi-stage pipeline to surface genuinely novel, non-obvious links between ideas. Most output will be junk. The entire value is in the filter.

Everything built so far — the multi-stage gating, composite scoring, cross-encoder reranking, truth scoring, deduplication, intent classification — is filter infrastructure tested daily against real retrieval problems.

---

## Phase: Feature-Complete — Consolidate, Populate, Ship

All major feature systems are built. The focus now shifts to: making what exists robust and shippable, populating the knowledge graph with real data, rebuilding the desktop executable, and splitting the onboarding wizard into user vs dev channels.

---

## Active Goals (Current Sprint)

### 1. Consolidation & Stabilization
- **Status**: Primary focus — all features built, now harden
- **Why**: Codebase at ~116K lines (incl. tests). Must be lean and reliable before scaling to batch synthesis workloads or shipping to users.
- Use retrieval benchmark suite to ablate prompt sections — measure recall impact of removing each
- Audit ChromaDB collections: do all 12 justify separate indexes? Can any be merged without retrieval regression?
- Reduce prompt context sections (currently 14+) to minimum set that maintains quality
- Clean up mutable state side-channels in scorer/builder pipeline (e.g., `_intent_weight_overrides` on MemoryScorer — replace with scoped context object or context manager)
- Resolve JSON corpus + ChromaDB dual-write: define source of truth, add reconciliation or eliminate one path
- Target: reduce codebase to ~85K lines without measurable retrieval quality loss
- Target: orchestrator.py under 400 lines (extract remaining truth/escalation/correction logic)
- Eliminate remaining calls to deprecated `prepare_prompt`

### 2. Rebuild Desktop Executable
- **Status**: Stale — many features added since last working build
- **Why**: PyInstaller spec (`daemon.spec`) hasn't been updated for new modules (wiki_enrichment, wiki_tracker, wikidata_models, synthesis_filter, claim_tracker, context_surfacer, implementation_detector, etc.). Hidden imports, data files, and bundled assets likely out of date.
- Audit `daemon.spec` for missing hidden imports and data files
- Test frozen build end-to-end: startup → wizard → chat → shutdown
- Verify FAISS index + parquet files are handled correctly (large external data, not bundled)
- Ensure `orjson` optional import fallback works in frozen context
- Target: clean `pyinstaller daemon.spec --clean --noconfirm` producing a working `dist/Daemon/Daemon`

### 3. Wizard Dual-Channel (User vs Dev)
- **Status**: Not started
- **Why**: Current wizard (`gui/wizard.py`) is a single onboarding flow. Users and developers need different setup paths — users care about API keys, data paths, and persona; devs need repo config, model selection, debug toggles, and feature flags.
- Split wizard into two channels selectable at launch: "I'm a user" vs "I'm a developer"
- **User channel**: API key setup, data directory, persona/name, privacy overview, optional Obsidian vault path
- **Dev channel**: All user steps plus: model provider selection, feature flag toggles (synthesis, graph walks, wiki enrichment, etc.), debug/logging level, benchmark suite intro, config.yaml deep-edit
- Shared: both channels write to the same `config.yaml` — dev channel just exposes more knobs
- Consider making dev channel accessible from GUI settings tab post-wizard (not just first-run)

### 4. Knowledge Graph Data Population
- **Status**: Graph infrastructure complete, data sparse
- **Why**: Graph-boosted scoring, query expansion, synthesis walks, and proactive surfacing all scale with graph density. Current graph has real structure but needs volume.
- Run `scripts/migrate_facts_to_graph.py` against full ChromaDB fact corpus to backfill
- Run wiki enrichment across broader article set (not just session-tracked articles)
- Evaluate batch Wikidata subgraph import (`scripts/import_wikidata_to_graph.py`) for entity linking
- Monitor `count_by_source()` and `count_bridge_edges()` to track density milestones
- Target: 500+ nodes, 200+ bridge edges before enabling graph walk synthesis

---

## Medium-Term Goals (Next 1-3 Months)

### 5. Knowledge Synthesis Pipeline (Core Vision)
- **Status**: Infrastructure built, calibration in progress, first real run pending
- **This is the primary goal. Everything else supports this.**
- Blocked on: graph density (Goal 4) and consolidation (Goal 1)

**Data Ingestion**:
- Wikipedia embedded via FAISS IVFPQ index (40M vectors, ~2 GB RAM) — **done**
- arXiv paper abstracts/full text — planned
- PubMed abstracts — planned

**Connection Generation**:
- Markov chain random walks across embedded knowledge graph
- Walk results passed to LLM with synthesis instructions to articulate connections
- High volume — most output will be nonsense or trivially true

**Multi-Stage Filtering** (the hard problem):
- 8-stage filter pipeline built (`synthesis_filter.py`), calibration fixture with 72 labeled candidates
- IVFPQ threshold recalibration done (novelty, co-occurrence, composite thresholds adjusted for quantized distances)
- Remaining: end-to-end validation at scale, pairwise reranking stage

**Open Questions**:
- How to distinguish "interesting but known" from "actually novel"
- Optimal Markov walk parameters (step count, restart probability, domain crossing frequency)
- Calibration: what does the score distribution look like for known good cross-domain papers?

### 6. Dreaming Engine (Batch Generation Infrastructure)
- **Status**: Pending — blocked on consolidation (Goal 1) and graph density (Goal 4)
- Background thread/async task runs during idle periods
- Iterates over knowledge bases with Markov walks, generates connection candidates in batches
- Full pipeline scoring: cosine → cross-encoder → LLM pairwise → novelty check
- Coverage tracking: which domains/topics explored, avoid redundant generation
- Scheduler: activates after N minutes inactivity, pauses on user input
- **Prerequisite**: Infrastructure must be lean and hardened first — batch workloads will stress every pipeline component at scale

### 7. Memory Quality Maintenance
- Prune low-value entries over time (decay + consolidation)
- Improve fact extraction precision (reduce false triples)
- Monitor retrieval benchmark scores for regression after any scoring/weight changes

---

## Foundational Infrastructure (Built, Maintaining)

These systems are complete and working. Listed here for context, not as active work items. Changes should be consolidation-oriented (simplify, merge, delete) rather than additive.

- **Memory system**: 6 types (episodic, semantic, procedural, summary, meta, fact), 12 ChromaDB collections, modular components with Protocol contracts, thin coordinator
- **Multi-stage gating**: FAISS → Cosine → Cross-Encoder reranking
- **Intent classification**: 9 types, regex-first, per-intent weight/retrieval/gate overrides, STM refinement
- **Truth scoring**: Evidence-based (TruthScorer + CorrectionDetector), replaces access-count system
- **Fact verification**: 4-stage gate (ephemeral → candidate → confirmation → LLM adjudication), no auto-deletion
- **Memory staleness**: Claim tracker with reverse index, cascade staleness scoring, prompt prefixes for outdated items
- **Cross-collection dedup**: Cosine duplicates + fact contradiction detection, dry-run default
- **Escalation tracker**: 4-state emotional momentum (VALIDATE → GROUNDING → QUIET → GENTLE)
- **User profile**: Append-only with temporal history, 12 categories, hybrid retrieval
- **Temporal awareness**: Narrative context (3-tier: monthly/weekly/daily), temporal-aware recency decay
- **Knowledge integration**: Obsidian vault (multimodal, mtime-based re-embedding), reference docs, git commits, procedural skills, Wikipedia (FAISS IVFPQ 40M vectors)
- **Knowledge graph**: Queryable fact graph with connectivity-ranked query expansion, junk node prevention at ingestion, graph-boosted memory scoring, wiki enrichment at shutdown
- **Proactive surfacing**: Cross-domain insight generation from knowledge graph, session-cached LLM calls, novelty-filtered
- **Synthesis pipeline**: Cross-store candidate generation, 8-stage filter, LLM bridge articulation, convergence tracking
- **Implementation tracking**: 4-stage proposal detection (file → grep → git → LLM), cooldown-gated
- **Fast Mode**: Reduced retrieval for mobile/slow connections with progress keepalives
- **PDF/DOCX support**: Full pipeline with table extraction (pdfplumber + python-docx), chunking with header detection
- **Multi-provider LLM**: Duel mode, best-of-N, ensemble, fallback chains
- **Code proposals**: LLM-generated, ChromaDB-stored, GUI management, shutdown integration
- **Agentic search**: ReAct loop with Tavily + Wolfram Alpha + E2B sandbox, context inventory, collection diversity tracking, memory expansion tool, full-document retrieval tool, git stats tool
- **Thread surfacing**: Proactive open-thread detection with resolution tracking
- **Session awareness**: Codebase diff + active features inventory at session start
- **Production**: PyInstaller desktop build (needs rebuild — Goal 2), Docker deployment, graceful shutdown
- **Privacy**: All data local, API calls only for LLM generation, no telemetry
- **Testing**: 2,900+ tests across 148 test files, retrieval quality benchmarks with 30 seed memories and 19 test cases

---

## Recent Completions

### Wiki-to-Graph Enrichment (2026-03-31)
- Session-level Wikipedia article tracking → graph nodes at shutdown
- `wiki_tracker.py` (thread-safe singleton), `wiki_enrichment.py` (shutdown step 6.9)
- `mentioned_alongside` edges at weight 0.5, `source="wiki_retrieved"` provenance on nodes
- `wikidata_models.py` Pydantic models for future Wikidata subgraph import
- 28 tests across 2 files

### Synthesis Calibration + IVFPQ Threshold Tuning (2026-03-31)
- Calibration fixture expanded 54 → 72 candidates (7 tiers)
- IVFPQ-aware threshold recalibration (novelty, co-occurrence, composite)
- Verification scripts: `verify_synthesis_pipeline.py` (26 pairs), `calibrate_coherence_live.py`
- End-to-end test infrastructure: `test_end_to_end_synthesis.py`, `generate_test_facts.py`

### FAISS IVFPQ + Zero-Copy Semantic Search + Knowledge Routing (2026-03-31)
- `build_faiss_index.py`: IVFFlat+OnDiskInvertedLists → IVFPQ (Product Quantization). 48 subquantizers × 8 bits = 48 bytes/vector (~32x compression). Full 41M-vector index fits in ~2 GB RAM, no ondisk inverted lists needed.
- `semantic_search.py`: Zero-copy parquet metadata — no DataFrame loaded into RAM. Row-group offset index built at load time, metadata read on-demand for just the ~8 result rows per query. Total footprint: FAISS index (~2.2 GB) + embedder (~0.4 GB).
- `controller.py`: FAISS Wikipedia for all wiki queries — agentic `search_memory(wiki_knowledge)`, prompt retrieval, and synthesis pipeline all route through FAISS (40M vectors). ChromaDB `wiki_knowledge` retained as fallback only.
- `handlers.py` + `web_search_trigger.py`: Knowledge search intent routing — keyword-based (`explain in depth`, `how does`, `consult wikipedia`, etc.) and LLM-based `needs_knowledge_search` field. 3-way routing: web search vs memory search vs knowledge search.

### Git Stats Agentic Tool (2026-03-29)
- New `git_stats` tool for agentic loop — answers temporal git questions (commit counts, files changed, contributors)
- Keyword-based intent parsing with temporal phrase extraction, no LLM calls
- Read-only safety: only allowlisted git subcommands
- 51 unit tests

### Thread Surfacing System (2026-03-23)
- Proactive open-thread detection: LLM extracts unresolved commitments, deadlines, unanswered questions, and unfinished topics at shutdown
- `ThreadStore` (ChromaDB-backed) stores threads with urgency scoring and deadline tracking
- `ThreadExtractor` detects resolutions of existing threads in new session conversations
- `[UNRESOLVED THREADS]` prompt section added, surfaced at session start with conversational weaving instructions
- Configurable: `THREAD_SURFACING_ENABLED`, `THREAD_MAX_OPEN` (cap 50), `THREAD_STALE_DAYS` (14), `THREAD_MAX_SURFACED` (3)
- Integrated into MemoryCoordinator, ShutdownProcessor, ContextGatherer, UnifiedPromptBuilder, and Orchestrator

### Agentic Search: Context Inventory + Diversity Tracking (2026-03-23)
- Context inventory: `_compute_context_inventory()` summarizes what RAG pipeline already gathered, injected into iteration prompt to prevent redundant re-searches
- Collection diversity: `memory_search_counts` tracks per-collection search frequency; hints steer LLM to under-explored collections after 2+ searches of the same collection
- Improved `search_memory` tool description with per-collection guidance and diversity instructions
- Updated routing table for profile/biographical queries to prefer summaries over facts

### Personal Notes Filtering + Web Search Robustness (2026-03-23)
- Personal notes: stricter post-gate relevance threshold (`PERSONAL_NOTES_GATE_THRESHOLD=0.30` vs general 0.18) prevents topically-similar but contextually-irrelevant Obsidian notes from leaking into responses
- Web search: query length truncation at 400 chars to prevent Tavily 400 Bad Request errors
- Web search: long-paste prefilter (>500 chars without explicit search phrases) skips LLM trigger to save time
- Web search: LLM-optimized search terms now always use first term as primary query with `auto_decompose=False`

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

- **New feature systems** — feature-complete. No new subsystems until consolidation targets are met.
- **Auto-execution of code proposals** — proposals are advisory only, human reviews and implements
- **Auto-deletion of user data** — all destructive operations default to dry_run=True; live deletions require explicit GUI action
- **Real-time collaboration** — single-user system, not multi-tenant
- **Mobile/web deployment** — desktop-first (Gradio GUI or CLI)
- **Fine-tuning or training** — uses commercial APIs and pre-trained local models only

---

## Principles

1. **The filter is the product.** Connection generation is cheap. Identifying which connections are novel and meaningful is the entire value proposition. Every infrastructure decision should be evaluated by whether it improves filtering quality.

2. **No new features until consolidation targets are met.** The codebase is feature-complete. Every addition now is net-negative until the existing ~116K lines are trimmed and hardened. Removing a system that doesn't pull its weight is more valuable than adding a new one.

3. **Data over code.** The limiting factor is now graph density, not missing features. Effort spent populating the knowledge graph with real data has higher ROI than writing new code.

4. **The personal assistant earns its keep.** It's not scaffolding — it's a daily-use product AND the live testbed. Both purposes matter.

5. **Measure before changing, measure after changing.** The retrieval benchmark suite exists for a reason. No scoring/weight/retrieval changes without before-and-after benchmark runs.

6. **Ship what's built.** A working desktop executable and a smooth onboarding wizard are prerequisite to anyone else using this. Polish and packaging matter now.
