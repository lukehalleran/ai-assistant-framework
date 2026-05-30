# Daemon Project Goals

**Last Updated**: 2026-05-27

---

## Mission

Daemon has two purposes that reinforce each other:

1. **Personal cognitive assistant** — A companion with deep, persistent memory that genuinely knows you. Your projects, preferences, history, emotional state, and context. Not a stateless chatbot. This is both valuable on its own and serves as a live testbed for the retrieval and scoring infrastructure.

2. **Knowledge synthesis engine** — The long-term goal. Use random walks (Markov chains) across large embedded knowledge bases (Wikipedia, arXiv, PubMed) to generate candidate cross-domain connections, then filter them through a multi-stage pipeline to surface genuinely novel, non-obvious links between ideas. Most output will be junk. The entire value is in the filter.

Everything built so far — the multi-stage gating, composite scoring, cross-encoder reranking, truth scoring, deduplication, intent classification — is filter infrastructure tested daily against real retrieval problems.

---

## Phase: Generation — Re-enable Synthesis, Enrich Proposals, Grow the Graph

Consolidation is largely complete (3,800+ tests, 0 failures, clean module extractions). The focus now shifts to: turning the frozen synthesis pipeline back on, improving the proposals system with wiki-seeded idea generation, and growing graph bridge density to unlock the walk generator.

---

## Active Goals (Current Sprint)

### 1. Synthesis Pipeline Re-enablement
- **Status**: All 3 generators DISABLED in `config.yaml` since 2026-04-01 pending grading validation
- **Why**: The synthesis pipeline is the core vision of this project. Everything else supports it. It's been frozen while the grading protocol (`docs/grading_plan.md`) validates filter quality. Time to close that loop.
- Complete grading on existing audit queue candidates (two-layer: 3 binary screening Qs + 1-5 gut-feel slider)
- Validate FP rate is below `SYNTHESIS_AUDIT_FP_HALT_THRESHOLD` (0.50) — this gates auto-halt
- Re-enable Tier 0 (RetrievalSynthesisGenerator) first — highest signal, named mechanisms
- Monitor composite/novelty/coherence rejection rates after re-enablement
- Decide whether to loosen composite threshold (currently tight) once live data flows
- Tier 1 (GraphWalkGenerator) blocked on bridge count (6 < 40) — see Goal 3
- Tier 2 (SynthesisGenerator) lowest priority — lower insight quality than Tier 0

### 2. Proposals Pipeline — Wiki-Enriched Idea Generation
- **Status**: Extracted to standalone branch. Core modules (`knowledge/proposal_generator.py`, `memory/proposal_store.py`, `memory/code_proposal.py`) still on master pending cleanup.
- **Why**: Current proposals are generated from Daemon's own context (goals, retrieval, code state). Adding wiki-seeded generation would surface ideas the user wouldn't arrive at independently — cross-domain techniques, algorithms, architectural patterns from unrelated fields.
- Design wiki→proposal bridge: use FAISS wiki index (40M vectors) to retrieve domain-adjacent concepts given current project state / goals
- Generate proposals that apply techniques from unrelated domains to Daemon's architecture (e.g., biological memory consolidation → summary compression, information-theoretic coding → dedup thresholds)
- Quality bar: proposals must be actionable and specific (file paths, method signatures), not vague metaphor
- Remove dead imports/references from master once standalone extraction is confirmed complete
- Consider whether standalone proposal tool feeds back into Daemon's `proposals` ChromaDB collection or stays external

### 3. Knowledge Graph Bridge Growth
- **Status**: 6 quality bridges (need ≥40 for walk generator activation)
- **Why**: Graph walks (Tier 1 synthesis) are blocked by insufficient personal↔wikidata bridges. The bridge feedback loop (accepted synthesis insights → provisional edges) is the designed growth mechanism, but requires Tier 0 running to produce candidates.
- Re-enable Tier 0 first (Goal 1) — accepted insights automatically create bridge edges
- Monitor `count_bridge_edges()` after synthesis re-enablement
- Consider additional bridge sources: Obsidian notes → wikidata entity linking, fact extraction → entity mapper
- WikidataEntityMapper threshold (0.80) may need revisiting once bridge quality is validated at scale
- Target: 40+ quality bridges → walk generator auto-activates

---

## Medium-Term Goals (Next 1-3 Months)

### 4. Dreaming Engine (Batch Generation Infrastructure)
- **Status**: Shutdown dreaming infrastructure built but generators currently disabled. Idle-time dreaming pending.
- Shutdown dreaming wired to run generators in parallel at session end (Step 6.8 in shutdown_processor.py)
- Idle-time background thread: activates after N minutes inactivity, pauses on user input — not yet implemented
- Coverage tracking: which domains/topics explored, avoid redundant generation
- Scheduler design pending
- **Prerequisite**: Synthesis re-enablement (Goal 1) must land first

### 5. Additional Data Ingestion
- Wikipedia embedded via FAISS IVFPQ index (40M vectors, ~2 GB RAM) — **done**
- arXiv paper abstracts/full text — planned
- PubMed abstracts — planned
- These expand the source pool for both synthesis candidates and wiki-enriched proposals

### 6. Rebuild Desktop Executable
- **Status**: v1.0.0 shipped 2026-04-08 — spec needs updates for modules added since
- Audit `daemon.spec` for missing hidden imports and data files
- Test frozen build end-to-end: startup → wizard → chat → shutdown
- Target: clean `pyinstaller daemon.spec --clean --noconfirm` producing a working `dist/Daemon/Daemon`

### 7. Memory Quality Maintenance
- Prune low-value entries over time (decay + consolidation)
- Improve fact extraction precision (reduce false triples)
- Monitor retrieval benchmark scores for regression after any scoring/weight changes
- **Current benchmark baseline** (2026-05-17): Combined MRR=0.8823, R@1=0.8143, R@3=0.9196, R@topK=0.9750

---

## Foundational Infrastructure (Built, Maintaining)

These systems are complete and working. Listed here for context, not as active work items.

- **Consolidation (2026-05)**: Codebase hardened — 3,800+ tests (0 failures), module extractions (agentic gate, handler helpers, prompt formatter, context gatherer), retrieval benchmarks stable (MRR 0.88), orchestrator decomposed

- **Memory system**: 6 types (episodic, semantic, procedural, summary, meta, fact), 14 ChromaDB collections, modular components with Protocol contracts, thin coordinator
- **Multi-stage gating**: FAISS → Cosine → Cross-Encoder reranking
- **Intent classification**: 9 types, regex-first, per-intent weight/retrieval/gate overrides, STM refinement, intent-conditioned section gating (eval-driven)
- **Truth scoring**: Evidence-based (TruthScorer + CorrectionDetector), replaces access-count system
- **Fact verification**: 4-stage gate (ephemeral → candidate → confirmation → LLM adjudication), no auto-deletion
- **Fact extraction**: Source excerpt pipeline with anti-confabulation guardrails, user message snippets attached to extracted triples
- **Memory staleness**: Claim tracker with reverse index, cascade staleness scoring, prompt prefixes for outdated items
- **Cross-collection dedup**: Cosine duplicates + fact contradiction detection, dry-run default
- **Escalation tracker**: 4-state emotional momentum (VALIDATE → GROUNDING → QUIET → GENTLE)
- **User profile**: Append-only with temporal history, 12 categories, hybrid retrieval, canonicalized relation namespace (safe aliases, ephemeral relations, 5-layer categorization)
- **Temporal awareness**: Narrative context (3-tier: monthly/weekly/daily), temporal-aware recency decay
- **Knowledge integration**: Obsidian vault (multimodal, mtime-based re-embedding), reference docs, git commits, procedural skills, Wikipedia (FAISS IVFPQ 40M vectors)
- **Knowledge graph**: Queryable fact graph with connectivity-ranked query expansion, junk node prevention at ingestion, graph-boosted memory scoring, wiki enrichment at shutdown
- **Proactive surfacing**: Cross-domain insight generation from knowledge graph, session-cached LLM calls, novelty-filtered
- **Synthesis pipeline**: Cross-store candidate generation, 7-stage filter (`claude-opus-4.8` coherence judge), LLM bridge articulation, convergence tracking, human audit queue with two-layer grading and auto-halt. All three generators currently disabled pending grading validation (see `docs/grading_plan.md`)
- **Implementation tracking**: 4-stage proposal detection (file → grep → git → LLM), cooldown-gated
- **Fast Mode**: Reduced retrieval for mobile/slow connections with progress keepalives
- **PDF/DOCX support**: Full pipeline with table extraction (pdfplumber + python-docx), chunking with header detection
- **Multi-provider LLM**: Duel mode, best-of-N, ensemble, fallback chains
- **Code proposals**: LLM-generated, ChromaDB-stored, GUI management, shutdown integration
- **Agentic search**: ReAct loop with 20 tools (Tavily, Wolfram Alpha, E2B sandbox, memory search/expand, file read/grep/list, git stats, GitHub API, full document, recall image, generate document, propose_action, lookup_contact, signal done), context inventory, collection diversity tracking, uncertainty fallback (auto-retry via agentic search when standard response indicates "I don't know")
- **Response planning**: Pre-answer plan + post-answer review gate with silent agentic retry on low-confidence reviews
- **Thread surfacing**: Proactive open-thread detection with resolution tracking
- **Session awareness**: Codebase diff + active features inventory at session start
- **Web search**: Tavily integration with WEB_N citation markers, broad news decomposition, semantic similarity trigger layer
- **Prompt eval system**: Snapshot/replay infrastructure, variant generation (LOO/AOI/bundle/reorder), pairwise judge, objective checks, 31-entry section registry, 246 tests (`eval/`)
- **Production**: PyInstaller desktop build (v1.0.0 shipped 2026-04-08), Docker deployment, graceful shutdown
- **Privacy**: All data local, API calls only for LLM generation, no telemetry
- **Testing**: 3,559 tests across 173 test files, retrieval quality benchmarks with 30 seed memories and 19 test cases

---

## Recent Completions (April-May 2026)

### Google OAuth2 + Calendar + Gmail Integration (2026-05)
- **Google OAuth2**: `core/actions/google_auth.py` — OAuth2 flow with 6 scopes (`gmail.send`, `gmail.readonly`, `calendar.readonly`, `calendar.events`, `contacts.readonly`, `contacts.other.readonly`). Token persistence at `data/google_token.json`, automatic refresh, scope-upgrade detection (re-auth when new scopes needed). Contact resolution via `google_contacts.py` + `gmail_search.py`.
- **Google Calendar read**: `core/actions/google_calendar.py` — real-time calendar events fetched via Calendar API (read-only), 5-min in-memory cache, minimal fields (title, start, end, location). Injected as `[GOOGLE CALENDAR]` prompt section via `gatherer_knowledge.py`.
- **Google Calendar write**: `core/actions/google_calendar_create.py` — event creation via `propose_action` agentic tool (human-in-the-loop confirmation through GUI approve/reject). Routed via `ActionExecutorRegistry`.
- **Gmail API send**: `core/actions/email.py` — Gmail API (`gmail.send` scope) as primary email transport, SMTP as fallback. Eliminates need for app passwords when Google OAuth2 is configured.
- Config: `GOOGLE_CALENDAR_ENABLED`, `GOOGLE_CALENDAR_MAX_EVENTS`, `GOOGLE_CALENDAR_LOOKAHEAD_DAYS` in `internet_actions` YAML section

### Ephemeral Fact Filtering (2026-05)
- Dual-layer filtering for transient-state predicates (current_feeling, is, has, thinks, etc.)
- **Extraction-time blocking**: `fact_extractor.py` drops ephemeral predicates during `extract_facts()`, preventing storage entirely
- **Retrieval-time TTL expiry**: `memory_retriever.py` filters stored ephemeral facts older than `PROFILE_EPHEMERAL_TTL_HOURS` (default 24h)
- LLM fact extractor (`llm_fact_extractor.py`) also blocks ephemeral relations
- Same `PROFILE_EPHEMERAL_RELATIONS` set governs both layers

### B2 Encrypted Backup (2026-05)
- `scripts/backup_data.sh` — encrypted backup upload to Backblaze B2 via rclone (`b2-daemon-crypt:backups` remote)
- Automatic remote pruning of backups older than configurable retention period
- Designed for cron scheduling

### Internet Actions System (2026-05)
- `core/actions/` subsystem: propose→confirm→execute flow for internet write actions
- 6 action types: send_telegram, send_discord, send_email, github_create_issue, github_comment_pr, calendar_create_event
- `propose_action` agentic tool (19th tool), GUI approve/reject buttons, JSONL audit log
- Working executors: Telegram (Bot API), Discord (webhook), Email (Gmail API primary, SMTP fallback), Calendar event creation (Google Calendar API). GitHub stubs.
- Config: `internet_actions` YAML section, `INTERNET_ACTIONS_*` app_config constants

### Agentic Gate Extraction (2026-05)
- Extracted 4-tier agentic gate from `gui/handlers.py` → `core/agentic/gate.py`
- `AgenticDecision` dataclass + `evaluate_agentic_gate()` async function
- Handler helpers refactored: `handle_submit` 1541→1200 LOC

### Schedule Extraction (2026-05)
- Structured calendar event extraction — 5 patterns (`work_schedule`, `class_schedule`, `exam_date`, `shift_pattern`, `day_off`), temporal normalization, supersession, gated `[UPCOMING SCHEDULE]` prompt section. 64 tests.

### God Object Decomposition — Phases 2-3 (2026-05-08)
- Extracted `PromptFormatter`, `ContentHygiene`, `ContextGatherer` mixins from UnifiedPromptBuilder
- Extracted `AgenticFormatter` + `ToolExecutor` from agentic controller
- Behavioral pattern extraction (cross-turn habit detection at shutdown)

### Phase 8 Intent-Conditioned Section Gating (2026-05-06)
- Eval-driven prompt section filtering per query intent type
- Sections gated based on ablation results from Phase 5/6 judge + objective checks

### Prompt Eval System — Phases 1-2, 4-6 (2026-05-04)
- Snapshot capture/replay, variant generation (LOO/AOI/bundle/reorder), pairwise judge harness, objective checks
- 246 tests in `tests/test_eval/`, CLI runners for each phase
- Infrastructure for validating synthesis generator re-enablement

### Response Planning (2026-04-20)
- `ResponsePlanner`: pre-answer plan (lightweight LLM ~200 tokens) + post-answer review gate
- Silent agentic retry when review fails with confidence >= 0.90
- Skips for CASUAL_SOCIAL intent and short queries to reduce LLM calls

### Uncertainty Fallback (2026-04-18)
- `UncertaintyDetector`: keyword regex + semantic embedding similarity detection of "I don't know" responses
- Auto-retry via agentic search when standard response indicates uncertainty, 33 tests

### Web Search Citations (2026-04-15)
- `NumberedWebSource` with stable WEB_N IDs, citation marker validation/stripping in orchestrator
- Broad news decomposition with source-aware facet queries, semantic similarity trigger layer

### Profile Namespace Canonicalization (2026-04-12)
- Safe relation aliases, ephemeral relations, snapshot relations, 5-layer categorization cascade
- `canonicalize_profile_relation()` with value-aware job disambiguation, 26 tests

### Source Excerpt Pipeline (2026-04-10)
- Anti-confabulation guardrails: user message snippets attached to extracted facts
- Profile context injection shows `(said: "...")` for grounded facts
- System prompt instruction: never add specific names/apps/platforms not in stored facts

### v1.0.0 Desktop Build + Wizard (2026-04-08)
- PyInstaller desktop executable shipped
- Wizard dual-channel (user vs dev) included in release

---

## Earlier Completions (January-March 2026)

### Synthesis Audit Queue (2026-04-01)
- Human-in-the-loop blind grading of synthesis pipeline output
- SynthesisResult fields: `human_grade`, `graded_at`, `grade_notes`
- SynthesisMemory audit methods: `grade_result()`, `get_ungraded()`, `get_graded()`, `get_audit_stats()`, `store_rejected_for_audit()`
- `SynthesisFilter.process_batch()` stores composite-rejected candidates for FN review
- GUI "Synthesis" tab with two-layer grading UI: 3 binary screening toggles (changes thinking? / real mechanism? / heard before?) + 1-5 gut-feel slider. See `docs/grading_plan.md` for full grading protocol.
- Auto-halt in `shutdown_processor.py`: skips synthesis if FP rate > `SYNTHESIS_AUDIT_FP_HALT_THRESHOLD` (0.50)
- Coherence judge model: `claude-opus-4.8` (set in config.yaml `synthesis_filter.coherence_model`)
- Config: `SYNTHESIS_AUDIT_ENABLED`, `SYNTHESIS_AUDIT_FP_HALT_THRESHOLD`, `SYNTHESIS_AUDIT_MIN_GRADED`; YAML section `synthesis_audit`
- 34 tests in `tests/unit/test_synthesis_audit.py`

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
- LLM `needs_memory_search` field added to `WebSearchDecision` — piggybacks on existing web search trigger call (zero extra cost). Memory search takes priority over web search when LLM returns both.
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

### Extended File Upload Support (2026-05-18)
- Added 8 new file types: `.md`, `.json`, `.yaml`/`.yml`, `.log`, `.html`, `.xml`, `.xlsx`
- XLSX via `openpyxl` (sheet-per-markdown-table); JSON/YAML/HTML/XML code-fenced; MD/TXT/LOG plain text
- Full requirements.txt and daemon.spec audit: added `Pillow`, `orjson`, `python-docx`, `openpyxl`; removed unused `anthropic`; added missing hiddenimports and collect_submodules for all project packages
- Fixed agentic search file upload bug: uploaded file content now passed via `merged_input` (was using `user_text`)
- Added `tests/conftest.py` to cap torch/numpy threads at half CPU cores during test runs

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
- Latest results (2026-05-13): Recall@1: 0.58, Recall@3: 1.00, MRR: 0.79, intent accuracy: 100% (19/19)

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

---

## Principles

1. **The filter is the product.** Connection generation is cheap. Identifying which connections are novel and meaningful is the entire value proposition. Every infrastructure decision should be evaluated by whether it improves filtering quality.

2. **Infrastructure is ready — generate.** Consolidation is done. The synthesis and proposals pipelines exist and are validated. The bottleneck is now turning them on, feeding them better inputs, and growing the graph.

3. **Data over code.** The limiting factor is graph density and source diversity, not missing features. Effort spent populating the knowledge graph and connecting wiki to proposals has higher ROI than writing new infrastructure.

4. **The personal assistant earns its keep.** It's not scaffolding — it's a daily-use product AND the live testbed. Both purposes matter.

5. **Measure before changing, measure after changing.** The retrieval benchmark suite exists for a reason. No scoring/weight/retrieval changes without before-and-after benchmark runs.

6. **Ship what's built.** A working desktop executable and a smooth onboarding wizard are prerequisite to anyone else using this. Polish and packaging matter now.
