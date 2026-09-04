# Memory System Operations Guide

*Last verified: 2026-05-27*

Operational guide for Daemon's 5-tier hierarchical memory system. Covers the
full lifecycle from query to retrieval to storage, the scoring algorithm with
concrete examples, fact extraction, truth/staleness tracking, shutdown
processing, and tuning guidance.

For formal notation see `FORMAL_MODEL.md`. For config constants see
`QUICK_REFERENCE.md`. For component inventory see `CLAUDE.md`.

---

## What the Memory System Does

Daemon maintains a persistent, multi-layered memory of everything the user
tells it. Each conversation turn triggers retrieval (pulling relevant context
into the prompt) and storage (persisting the interaction for future use).
At session shutdown, consolidation processes extract facts, generate summaries,
update the knowledge graph, track staleness, and run synthesis dreaming.

The system is designed so the user never needs to repeat themselves. Facts
stated months ago surface when relevant, contradictions are detected and
resolved, and stale information is penalized in ranking.

---

## File Map

### Core Pipeline
| File | Purpose |
|------|---------|
| `memory/memory_coordinator.py` | Thin orchestrator (~639 lines), creates all components, delegates to retriever/storage/shutdown |
| `memory/memory_retriever.py` | Retrieval: collection selection, gating, threshold fallbacks, supersession + per-relation TTL filter |
| `memory/relation_classifier.py` | Single source of truth for relation→TTL: health-transient vs standard-ephemeral vs durable (+ permanent-condition overrides). Used by user_profile + memory_retriever. 2026-08-03: also hosts `MULTI_VALUED_RELATIONS` + `GENERIC_PREDICATES` + `is_multi_valued_relation()` — the cross-dedup contradiction arm's cluster exclusions (plural values are not conflicts; generic `is`/`has` relation names carry no claim identity) |
| `memory/memory_scorer.py` | Scoring algorithm (6 weighted factors + 8 additive bonuses/penalties incl. health-framing decay + timeline bonus) with intent overrides, graph boost, size penalty |
| `memory/memory_storage.py` | Storage: ChromaDB + corpus writes, fact extraction hook, graph ingestion, reflection embedding cleanup |
| `memory/skill_activation.py` | SkillActivationPolicy (post-retrieval skill filter) + SkillCooldownStore (JSON-backed TTL) |
| `core/prompt/builder.py` | UnifiedPromptBuilder: thin orchestrator for parallel task dispatch, intent overrides, budget |
| `core/prompt/context_gatherer.py` | Mixin compositor (composes gatherer_web, gatherer_memory, gatherer_knowledge) |
| `core/prompt/gatherer_web.py` | WebSearchMixin: web search retrieval + trigger logic |
| `core/prompt/gatherer_memory.py` | MemoryRetrievalMixin: 17 memory/summary/reflection/facts/profile retrieval methods |
| `core/prompt/gatherer_knowledge.py` | KnowledgeRetrievalMixin: 16 knowledge retrieval methods (notes, docs, git, graph, threads, insights, wiki, codebase) |
| `core/prompt/formatter.py` | PromptFormatter: `_assemble_prompt()` section assembly + `_build_feature_inventory()` + `_staleness_prefix` |
| `core/prompt/hygiene.py` | ContentHygiene: `_hygiene_and_caps()` dedup + caps + `_backfill_recent_conversations()` |
| `core/prompt/token_manager.py` | Budget enforcement, middle-out compression, priority ordering |
| `processing/gate_system.py` | Multi-stage gating: cosine similarity + cross-encoder reranking |

### Fact & Truth Pipeline
| File | Purpose |
|------|---------|
| `memory/fact_extractor.py` | Multi-stage extraction: corrections > spaCy > REBEL > regex, dual budget. Pre-canonicalization filters: ephemeral predicate blocking (config-driven), boolean-only value rejection, junk-object + polarity guards (2026-08-02); object cap `FACT_OBJECT_MAX_CHARS`=300 + salutation salvage (2026-08-26: "Hi Morgan,\n<email>" object → `email_sent=Morgan` instead of a silent drop) |
| `memory/llm_fact_extractor.py` | LLM-assisted triple extraction with entity support; accepts existing profile facts for relation reuse; attaches source_excerpt via keyword matching. `_normalize_triple()` applies `_is_ephemeral_relation()` + `_is_boolean_noise()` guards; LLM prompt explicitly discourages transient state extraction. 2026-08-05: 30-turn window, newest-first budget (see "Extraction coverage" below); `CORE_RELATIONS` constant + learned-relation promotion. 2026-08-26: long-object cap + salutation salvage (this path had NO length cap — a ~700-char pasted email was stored wholesale as a fact object) |
| `memory/fact_verification.py` | Pre-storage conflict checking: ephemeral > candidates > trust > LLM adjudication |
| `memory/truth_scorer.py` | Stateless truth computation: initial score + adjustments + time decay |
| `memory/claim_tracker.py` | Claim extraction, hashing, reverse index, staleness cascade |
| `memory/stance_classifier.py` | Single-source deterministic stance core (2026-08-23): objective / appraisal / reported / inferred / unknown. `classify_triple_stance`, `classify_utterance_stance`, `scope_unresolved_referent`, `classify_for_storage`, `effective_stance`, `capture_tone_from_level`; `EVALUATIVE_LEXICON` of thick evaluative terms (evil, abusive, toxic, worthless...) |

### Persistence Safety (2026-07-14)

The five critical JSON stores (knowledge graph, entity aliases, user profile,
corpus, claim index) share one safety contract via `utils/safe_json.py`:

- **Atomic writes**: temp file + `os.replace` — a crash mid-write can never
  truncate an existing store.
- **Strict loads**: a missing file is a fresh start; a 0-byte file is a fresh
  start with a warning; an existing file with unparseable content is copied to
  `<path>.corrupt-<timestamp>` and raises `CorruptStoreError`. Startup then
  aborts with an actionable message (`main.py` catches it) instead of silently
  running with empty state — which would overwrite the user's data on the next
  save. `memory_coordinator.py` re-raises `CorruptStoreError` through its
  otherwise-broad init exception handlers so the error can't be swallowed.
- **Schema versions**: `knowledge_graph.json` and `claim_index.json` carry a
  `schema_version` field (missing = 1, the pre-versioning format). Loads
  refuse files written by a newer build (`StoreVersionError` — same
  startup-abort path) instead of silently dropping unknown fields. The user
  profile has its own `SCHEMA_VERSION` (2.0) with in-place migration.
- **Backups**: `utils/backup_manager.py` runs as the final shutdown phase —
  JSON stores every shutdown, the ChromaDB tree on a 12h throttle, retention
  keeps the newest 5 plus the newest chroma backup. Restore (dry-run-first,
  never deletes current data) via `scripts/restore_backup.py`; portable
  export via `scripts/export_user_data.py`.

### Knowledge Graph
| File | Purpose |
|------|---------|
| `memory/graph_memory.py` | NetworkX DiGraph as a TOPOLOGY index (one edge per node pair; attrs advisory) + relation-keyed `_edge_index` with `_out_keys`/`_in_keys` adjacency maps as the source of truth for every relation read (2026-09-03 — 146 of 982 live edges had been invisible); BFS traversal with an opt-in hub barrier for the prompt path; read-side suppression (`edge_is_suppressed`: quarantine flag / species conflict); JSON persistence |
| `memory/graph_models.py` | Pydantic models: GraphNode, GraphEdge |
| `memory/entity_resolver.py` | Alias resolution + relation normalization (synonym table + family-collapse patterns, e.g. `asked_about_*` → `asked_about`) |
| `memory/graph_utils.py` | Entity extraction, neighbor lookups, expansion ranking |
| `knowledge/wikidata_enrichment.py` | Anchored shutdown-time Wikidata typed edges: personal entities that exact-match the offline cache get whitelisted relations (`instance_of`, `part_of`, ...) as 1-hop edges; taxonomic relations forward-only; capped per entity/run |

**Graph ingestion paths (both feed `_ingest_fact_to_graph`):** the per-turn
regex extractor (`extract_and_store_facts`) and — since 2026-07-14 — the
shutdown LLM triple path (`_extract_llm_facts`), previously ChromaDB-only.
The LLM extractor prompt explicitly requests entity–entity relations
("Sam sibling_of Biscuit") so lateral (non-user) edges accumulate; an
object that resolves to an existing graph entity is always edge-worthy,
bypassing the junk-object heuristics that would demote it to node metadata.
Maintenance: `scripts/graph_junk_cleanup.py` (curated node removal),
`scripts/graph_relation_normalize.py` (re-canonicalize edge relations through
the deployed `normalize_relation()`; dry-run first, backs up on `--apply`).

### Shutdown & Consolidation
| File | Purpose |
|------|---------|
| `memory/shutdown_processor.py` | Session-end processing: 3 parallel phases (A: extraction, B: generation, C: persistence) |
| `memory/thread_manager.py` | Thread detection for conversation continuity |
| `memory/thread_store.py` | ChromaDB-backed thread persistence + priority ranking + deadline-aware staleness + per-turn regex resolution |
| `memory/thread_extractor.py` | LLM-based thread extraction + resolution detection |

### Storage Layer
| File | Purpose |
|------|---------|
| `memory/storage/multi_collection_chroma_store.py` | ChromaDB wrapper: 14 collections (lazy init via `_get_collection`), `add_to_collection`, `query_collection`, `get_by_id`, `update_metadata`, `get_ids_by_timestamp_range` |

---

## 14 ChromaDB Collections

| Collection | Content | Protected | Deduped |
|------------|---------|-----------|---------|
| `conversations` | Raw Q/A turns | Yes | Never |
| `facts` | Extracted triples (user + entity) | No | Yes |
| `summaries` | Compressed conversation blocks | No | Yes |
| `reflections` | Session-end reflections (embedded with boilerplate stripped) | No | Yes |
| `wiki_knowledge` | Wikipedia content | Yes | Never |
| `obsidian_notes` | User's personal notes | Yes | Never |
| `reference_docs` | Uploaded docs + auto-seeded docs/ | Yes | Never |
| `procedural` | Git commits, how-to knowledge | No | Yes |
| `procedural_skills` | Reusable problem-solving patterns | No | Yes |
| `proposals` | Goal-directed code change proposals | No | Yes |
| `threads` | Open threads (commitments, deadlines) | No | Yes |
| `synthesis_results` | Cross-domain synthesis insights (+ human audit: two-layer grading (3 binary screening + 1-5 slider); see `docs/grading_plan.md`) | No | Yes |
| `visual_memories` | CLIP-embedded image metadata for visual recall | No | No |
| `daemon_self_notes` | Daemon's own session notes (decisions, risks, next steps); `ground_truth: False` in metadata | No | Yes |

**Protected** = never scanned by cross-collection deduplicator.

`daemon_self_notes` is explicitly not ground truth -- all entries carry
`ground_truth: False` metadata to distinguish them from user-stated facts.
Retrieved via `get_daemon_self_notes()` in the context gatherer (max 2 per
prompt), displayed in `[DAEMON SELF-NOTES]` with a caveat label. Scoring
applies `COLLECTION_BOOSTS['daemon_self_notes'] = -0.05` (slight demotion).
`DaemonNote` carries per-step persistence flags (`disk_written` / `embedded` /
`indexed`) and a `fully_persisted` property, so the GUI save path can report a
*partial* save honestly (disk written but embed/index failed) instead of
claiming success — anti-confabulation support for the action guard.
Registered in eval `section_registry` (assembly_order=28, ablatable).

---

## End-to-End Lifecycle: One Conversation Turn

### Phase 1: Retrieval (building the prompt)

User sends: *"How's my squat progress looking?"*

```
UnifiedPromptBuilder.build_prompt()
  │
  ├─ 1. Query Analysis
  │     IntentClassifier → FACTUAL_RECALL (pattern: "how's my")
  │     Retrieval overrides: max_mems=20, max_recent=5, max_wiki=5
  │     Weight overrides: truth=0.30 (boosted for factual recall)
  │
  ├─ 2. Graph Query Expansion
  │     extract_graph_entities("squat progress") → ["squat"]
  │     graph neighbors: ["powerlifting", "deadlift", "bench press"]
  │     expanded query: "How's my squat progress looking? powerlifting deadlift bench press"
  │
  ├─ 3. Parallel Retrieval (22 async tasks, 30s timeout)
  │     ├── recent_conversations (15)     ← corpus_manager
  │     ├── semantic_memories (20)        ← ChromaDB [conversations, summaries, reflections]
  │     ├── user_profile (3000 tokens)    ← UserProfile (categorized facts)
  │     ├── facts (30)                    ← ChromaDB facts collection
  │     ├── summaries (recent+semantic)   ← ChromaDB summaries (recent side =
  │     │     get_recent() timestamp-sorted + is_junk_summary filter since
  │     │     2026-07-25 — was an empty-string semantic query returning junk)
  │     ├── reflections (recent+semantic) ← ChromaDB reflections
  │     ├── graph_context (12 sentences)  ← GraphMemory BFS traversal
  │     ├── procedural_skills (5, over-fetched 3x) ← ChromaDB procedural_skills
  │     ├── unresolved_threads (3)        ← ThreadStore
  │     ├── proactive_insights (2)        ← ContextSurfacer
  │     ├── wiki_content (3)              ← FAISS (41M wiki vectors; ChromaDB fallback)
  │     ├── reference_docs (5)            ← ChromaDB reference_docs
  │     ├── personal_notes (5)            ← Obsidian vault
  │     ├── git_commits (10)              ← ChromaDB procedural
  │     ├── web_search (if triggered)     ← Tavily API
  │     └── codebase_changes (first msg)  ← git diff
  │
  ├─ 3.5. Skill Activation (procedural_skills)
  │     SkillActivationPolicy: intent suppression → score threshold →
  │     STM topic bonus → cooldown filter → cap to 3
  │
  ├─ 4. Multi-Stage Gating (semantic memories)
  │     Top 2 recent → bypass gating
  │     Remaining → cosine similarity → cross-encoder reranking
  │     Forced minimum: 8 memories even if below threshold
  │     (Gate scoring uses the store's bge embedder — device is GPU
  │      auto-detected since 2026-07-15 (`_resolve_embed_device`,
  │      CHROMA_DEVICE env overrides); CPU-pinned it was ~5s per gate call.)
  │
  ├─ 5. Scoring (MemoryScorer.rank_memories)
  │     12-step algorithm → final_score per memory
  │     Sorted descending, top K selected
  │
  ├─ 6. Token Budget Enforcement
  │     Priority-based trimming (profile > conversations > memories > wiki)
  │     Middle-out compression for oversized items
  │     LLM pre-compression for items >=3x over limit
  │
  └─ 7. Prompt Assembly
        [SYSTEM PROMPT] + [USER PROFILE] + [RECENT CONVERSATIONS] +
        [MEMORIES] + [FACTS] + [KNOWLEDGE GRAPH] + [THREADS] + ... +
        [USER QUERY]
```

### Phase 2: Storage (after LLM responds)

```
MemoryStorage.store_interaction(query, response)
  │
  ├─ 0. Thinking-leak storage guard: ResponseParser.sanitize_for_storage()
  │     runs before ANY persistence; all-thinking responses skipped (return None)
  │     — final defense layer so reasoning artifacts can't persist and be replayed
  │     (see docs/THINKING_BLOCKS_IMPLEMENTATION.md). Since 2026-08-03 also strips
  │     the kimi-3 trailing-'e' stream artifact ("…landed?e" was stored verbatim);
  │     both add_summary paths (chroma + corpus) apply the same strip — the
  │     endpoint's non-streaming outputs had it too ("…impress them.e")
  ├─ 1. Skip gate: reject file-error responses + API-error sentinels
  │     ("[API Error]", "[CREDITS EXHAUSTED]", "[RATE LIMITED]", ... — prefixes
  │      from model_manager._classify_api_error; transport failures are never
  │      persisted as Daemon replies) [NEW 2026-07-03]
  │      2026-08-21: an EMPTY assistant response is likewise never stored
  │      (pairs with the reasoning-only-recovery quality floor in
  │      response_generator — "dy won't"-class recovered garbage is discarded)
  ├─ 2. Thread detection: assign thread_id + depth
  ├─ 3. Corpus storage: JSON persistence (immediate; corpus add_summary accepts
  │     dict payloads since 2026-08-21, normalizing content/timestamp/
  │     created_at/type — memory_storage passes a dict and reflections
  │     previously crashed out of the in-memory corpus)
  ├─ 4. Topic detection: primary topic → tag enrichment
  ├─ 5. Score calculation: truth_score + importance_score
  ├─ 6. Metadata assembly: timestamp, tags, thread, provenance
  ├─ 7. ChromaDB storage: conversations collection
  │
  ├─ 7b. Per-turn thread resolution (check_quick_resolutions)
  │     Pure regex: completion signals × open thread keywords, ~1ms
  │     Skips DB query if no completion signal detected in message
  │     Both sides use _norm_keywords (digit tokens kept — "done with hw6"
  │     can't resolve the hw7 thread on generic word overlap, 2026-07-05)
  │
  ├─ 8. Fact extraction (if FACTS_EXTRACT_EACH_TURN)
  │     ├── FactExtractor: corrections > spaCy > REBEL > regex
  │     ├── Dual budget: 6 user facts + 4 entity facts
  │     ├── FactVerifier: ephemeral check → conflict query → verdict
  │     │     STORE → add to facts collection
  │     │     STORE_AND_FLAG → add + mark old as superseded
  │     │     REJECT → skip
  │     │     SKIP → store without verification (ephemeral)
  │     └── Graph ingestion: entity resolution → add nodes/edges
  │
  └─ 9. Consolidation trigger (if threshold met and not shutdown-only)
```

### Phase 3: Shutdown (session end)

```
ShutdownProcessor.process_shutdown_memory()
  │
  ├─ Block summaries (N=10 conversations per block)
  │     LLM consolidation → micro-summary fallback
  │     Claim extraction → ClaimIndex registration
  │     Source doc IDs via get_ids_by_timestamp_range() (not list_all scan)
  │     [2026-08-22] _store_summary writes via raw add_to_collection('summaries',…),
  │     bypassing chroma_store.add_summary (where the stream-artifact strip +
  │     is_junk_summary check live) — 2 summaries landed with leading <|sep|>
  │     after the 08-21 fix; it now sanitizes + junk-rejects at entry, and
  │     scripts/strip_special_token_artifacts.py also scans/repairs the summaries
  │     collection + corpus content fields
  │     (tests/unit/test_stream_artifacts.py::TestShutdownSummaryPathSanitized)
  │
  ├─ Phase A (parallel via asyncio.gather) ────────────────────────
  │  ├─ Session fact extraction (rule-based, last 10 turns)
  │  │     Fact verification gate (REJECT/STORE/STORE_AND_FLAG)
  │  │     On STORE_AND_FLAG: mark conflicting facts as superseded
  │  ├─ LLM fact extraction (neural triples) + batch verification
  │  ├─ Behavioral pattern extraction (cross-turn habit detection)
  │  └─ Procedural skill extraction
  │
  ├─ Phase B (parallel via asyncio.gather) ────────────────────────
  │  ├─ Code proposal generation
  │  ├─ Implementation tracking (lightweight file check)
  │  ├─ Open thread processing
  │  │     Detect resolutions (cascade to duplicate siblings via topics_equivalent)
  │  │     → extract new (prompt lists ALREADY TRACKED threads; extracted dupes of
  │  │       resolved/open threads dropped, open twins get last_referenced refreshed)
  │  │     → enforce cap
  │  ├─ Wiki-to-graph enrichment (session wiki articles → graph nodes)
  │  └─ Wikidata typed-edge enrichment (personal entities → whitelisted
  │        typed edges from the offline cache; anchored, 1-hop, capped)
  │
  ├─ Phase C (sequential — must follow Phase B) ──────────────────
  │  ├─ Knowledge graph + alias + category cache save (JSON flush)
  │  └─ Cross-collection dedup
  │        Mode depends on DAEMON_MODE (config.yaml currently sets mode: dev;
  │        the code-level default when unset is "user"):
  │          Dev mode: dry_run=True — preview/log only, never auto-executes
  │          User mode: auto-executes (CROSS_DEDUP_AUTO_EXECUTE=True)
  │        Execution [2026-08-03]: duplicates DELETED; contradiction losers
  │        SUPERSEDED (is_current=False + superseded_by — reversible, never deleted);
  │        clusters exclude multi-valued/generic/junk-subject/superseded facts
  │        Live execution also available via GUI Preview/Run buttons in Status tab
  │        Double-run guard: class-level _dedup_ran flag prevents running twice per process
  │
  └─ Consolidation trigger (if threshold met and not shutdown-only)

Synthesis dreaming — SEPARATE standalone step [CHANGED 2026-06-19] ──
  Driven by main.py AFTER process_shutdown_memory returns, under its own
  SYNTHESIS_DREAM_TIMEOUT_S (240s) budget — NOT inside Phase B. Its slow
  per-candidate LLM coherence judge was being cancelled by the shared
  SHUTDOWN_TASK_TIMEOUT_S (60s) reflection/fact budget, so no candidate
  ever persisted. Entry: MemoryCoordinator.run_synthesis_dreaming().
    ├─ Auto-halt check: skips if audit FP rate > SYNTHESIS_AUDIT_FP_HALT_THRESHOLD
    ├─ PooledConceptSynthesisGenerator — the SOLE active generator (2026-06-30):
    │     pairs prominent curated concepts (48-concept CONCEPT_POOL) in the
    │     non-obvious cosine band 0.2-0.45 (config: synthesis_pooled — enabled: false
    │     since 2026-07-15, API-cost pause; toggle + count slider in Settings)
    ├─ Retired tiers (config enabled: false): Tier 0 RetrievalSynthesisGenerator,
    │     Tier 1 GraphWalkGenerator, Tier 2 SynthesisGenerator
    ├─ Candidates → SynthesisFilter → SynthesisMemory
    └─ On acceptance: provisional bridge edge created (weight=0.0, status="provisional")
```

---

## Scoring Algorithm

Every retrieved memory gets a `final_score` from `MemoryScorer.rank_memories()`.

### Default Weights

```
# LIVE values from config.yaml gating.score_weights (what SCORE_WEIGHTS resolves to).
# The 0.35/0.25/0.20 vector in app_config.py is only a fallback if the YAML key is
# absent — it is not, so that fallback is dead.
relevance:  0.30    # Pre-gate semantic similarity
recency:    0.22    # Temporal decay (active-day aware)
truth:      0.18    # Evidence-based correctness
importance: 0.05    # Retention priority
continuity: 0.10    # Token overlap with current conversation
topic_match:0.10    # USED — multiplied by the topic_match signal in rank_memories()
structure:  0.05    # In SCORE_WEIGHTS dict but UNUSED — actual structure is additive: 0.15 * density_alignment
```

### Step-by-Step

1. **Base relevance** — Embedding similarity from ChromaDB query + collection boost
2. **Recency decay** — Uses `time_manager.current()` for consistent reference time. Active-day aware: `1/(1 + decay_rate * age_hours)`. Temporal anchor override for TEMPORAL_RECALL queries uses a two-regime decay: small anchors (<=48h, e.g. "today"/"yesterday") get a flat plateau inside the window so relevance/truth differentiate; large anchors (>48h, e.g. "last week") peak near the anchor and penalize too-recent memories. **Timestamp fallback (2026-07-08):** a missing/empty top-level `timestamp` falls back to `metadata['timestamp']` before defaulting to now (tz-aware stragglers normalized to naive local). Before this, the hybrid/semantic retrieval path — which set no top-level timestamp — scored every memory recency=1.0, silently disabling the recency term for the main retrieval path (months-old memories ranked as fresh); the retriever now also surfaces the metadata timestamp at top level. Tests: `tests/unit/test_recency_metadata_fallback.py`
3. **Truth score** — `TruthScorer.compute_effective_truth(metadata)`: stored score + time decay from last confirmation
4. **Importance** — Stored importance score (default 0.5)
5. **Continuity** — Token overlap with last exchange (+0.3 * overlap) + recency bonus (+0.1 if within 10 minutes). Tokens are stemmed via `_stem()` (minimal suffix stripping for common mismatches like anxious/anxiety, deployed/deployment). Tag-keyword bonus: if query stems match memory tags, adds up to +0.15 to continuity (scales with number of tag hits, capped at 3)
6. **Structural alignment** — `0.15 * density_alignment` where density_alignment measures numeric/operator density match between query and memory. Added as direct bonus, not through weighted sum
7. **Penalties** — Analogy penalty (-0.1 for mathy queries matching analogies) + size penalty (see below)
8. **Anchor bonus** — Salient token overlap with conversation context. Deictic queries ("explain that", "what about it") get +0.2·overlap bonus, or a -`DEICTIC_ANCHOR_PENALTY` penalty (live config.yaml 0.25; code default 0.1) when anchor overlap < 0.05; non-deictic queries get +0.1·overlap
9. **Tone adjustment** — Dismissive language in memory → truth reduced by 0.2
10. **Topic match** — 1.0 exact, 0.5 unknown, 0.2 mismatch (live weight 0.10 from config.yaml `gating.score_weights`)
11. **Meta-conversational bonus** — +0.15 for episodic memories when query is about recall ("did we discuss...")
12. **Graph proximity bonus** — +0.05 per knowledge graph neighbor mentioned in memory, capped at 0.15
13. **Staleness penalty** — `staleness_ratio * STALENESS_WEIGHT`, 2x multiplier at >=0.8 ratio, reflections at 60% weight, capped at 0.4
14. **Health-framing decay** — stale free-text illness/recovery narrative in personal-narrative collections past the health-transient TTL gets a ramped penalty (`HEALTH_FRAMING_DECAY_WEIGHT=0.25` base, capped at 0.4)
15. **Timeline bonus** — +0.15 for summaries/reflections when the query is a progression query ("how long", "over time")

### Size Penalty (Large Document Demotion)

Large documents (>10KB) that lack keyword relevance get a scaled penalty:

```
Threshold: LARGE_DOC_SIZE_THRESHOLD = 10,000 bytes
Keyword check: if keyword_score > 0.3 → no penalty (document is keyword-relevant)
Formula: penalty = -0.25 * (size_bytes / 10,000)
Cap: -1.0 (prevents extreme penalties for very large docs)

Examples:
  5KB doc  → 0.0 (under threshold)
  20KB doc, keyword_score=0.1 → -0.25 * 2.0 = -0.50
  95KB doc, keyword_score=0.0 → -0.25 * 9.5 = -2.375 → capped at -1.0
  20KB doc, keyword_score=0.5 → 0.0 (keyword-relevant, no penalty)
```

These constants are defined in `memory_scorer.py` (module-level, not in `app_config.py`):
`LARGE_DOC_SIZE_THRESHOLD`, `LARGE_DOC_KEYWORD_THRESHOLD`, `LARGE_DOC_BASE_PENALTY`.

**Final guardrail:** Deictic queries with low continuity AND low anchor overlap get a 15% penalty to prevent drift.

### Concrete Example

Memory: *"User's squat is 365lb, set last month at the gym"*

```
relevance:    0.30 * 0.82 = 0.246    (high semantic match to "squat progress")
recency:      0.22 * 0.45 = 0.099    (3 weeks old, moderate decay)
truth:        0.18 * 0.85 = 0.153    (confirmed once, slight time decay)
importance:   0.05 * 0.60 = 0.030    (moderate importance)
continuity:   0.10 * 0.15 = 0.015    ("squat" token overlap)
structure:    0.15 * 0.90 = 0.135    (direct additive bonus — high numeric density alignment — "365lb")
graph_bonus:  0.05                    (1 neighbor "powerlifting" mentioned)
staleness:    0.00                    (no stale claims)
penalties:    0.00
─────────────────────────────────────
final_score:  0.728
```

### Intent-Driven Weight Overrides

The IntentClassifier detects query intent and overrides scoring weights:

| Intent | Key Overrides | Effect |
|--------|--------------|--------|
| FACTUAL_RECALL | relevance=0.40, recency=0.05, truth=0.30, continuity=0.10 | Prioritize confirmed facts, suppress recency |
| TEMPORAL_RECALL | recency=0.40, continuity=0.20, `_temporal_anchor_hours` | Reshape decay curve around time window |
| EMOTIONAL_SUPPORT | recency=0.15, continuity=0.40, truth=0.10, max_skills=0 | Prioritize conversation flow over recency, suppress procedural skills |
| TECHNICAL_HELP | relevance=0.40, recency=0.10, continuity=0.20 | Prioritize semantic match, lower recency |
| CREATIVE_EXPLORATION | recency=0.10, continuity=0.20, importance=0.15 | Suppress recency, favor importance/continuity |
| PROJECT_WORK | relevance=0.40, recency=0.10, truth=0.20, continuity=0.15 | Prioritize relevance+truth, lower recency |
| CASUAL_SOCIAL | recency=0.20, continuity=0.25 | Balanced recent + conversational flow |

### Fact Retrieval Ranking

`MemoryRetriever.get_facts()` uses a separate scoring formula from the
main `MemoryScorer.rank_memories()` pipeline. Facts are ranked with
semantic relevance as the primary signal:

```
score = 0.60 * semantic + 0.20 * confidence + 0.20 * recency

Semantic floor: if relevance_score < 0.30, recency weight drops to 0.05
→ prevents irrelevant-but-recent facts from dominating relevant older ones

Candidate pool: limit * 2 pulled from ChromaDB, top `limit` returned after ranking
```

This replaced an earlier formula (`0.7*confidence + 0.3*recency`) that
ignored the ChromaDB semantic similarity score entirely, causing facts to
be ranked primarily by extraction confidence rather than query relevance.

### Transient Fact TTL + Supersession in Retrieval

After scoring and sorting, `get_facts()` filters the facts collection on two
signals, so stale transient state (yesterday's mood, *last month's illness*) and
explicitly retracted facts don't pollute retrieval:

```
For each ranked fact:
  1. Supersession: if metadata is_current is False or superseded_by is set
     → drop immediately, regardless of age (set by fact verification or the
     scripts/cleanup_stale_illness.py remediation).
  2. Per-relation TTL: _fact_ephemeral_ttl(content) parses the predicate from
     "subject | predicate | object" and asks the shared classifier for its TTL.
     Parse timestamp, strip tz, compare age → if older than the TTL, drop.
  3. Durable facts (TTL is None) pass through unaffected.
```

**Single source of truth — `memory/relation_classifier.py`.** Relation→TTL now
lives in one module that the profile section (`UserProfile.get_category`) and the
facts retriever both call, ending an earlier three-way drift where each path had
its own copy. `ephemeral_ttl_hours(relation)` returns:

| Tier | Examples | TTL |
|------|----------|-----|
| Health-transient | `illness`/`recover`/`sick`/`symptom`-named, `health_status`, `*condition` | `PROFILE_HEALTH_TRANSIENT_TTL_HOURS` (default 96h / a few days) |
| Standard ephemeral | `current_*`, `woke_*`, `*_status`, mood/activity | `PROFILE_EPHEMERAL_TTL_HOURS` (default 24h) |
| Durable | `name`, `birthday`, `brother_name`, … | `None` (never expires) |

Health-transient is checked **before** standard-ephemeral (so `recovery_status`,
which also matches the `_status` suffix, gets the longer horizon). Permanent
conditions / disabilities (`disability`, `chronic_condition`, `diagnosis`) are
pinned durable via `_DURABLE_OVERRIDES` — a disability is not an illness episode
and must not age out, even though `chronic_condition` matches the `_condition`
suffix.

> The fact extractors' **storage-side** ephemeral block intentionally still uses
> the exact `PROFILE_EPHEMERAL_RELATIONS` list — health-transient facts *should*
> be stored (useful for a few days), just aged out on read.

---

## Fact Pipeline

### Extraction

> **Personal vocabulary is config-driven and gitignored.** Owner-specific
> vocabulary (preference slots, entity casing, generic subjects, project-area
> keywords, category tokens, relation→category overrides) is merged at **import**
> from `user_profile.personal_vocabulary` (`PROFILE_PERSONAL_*` constants) across
> `fact_extractor`, `user_profile_schema`, `memory_storage`, and `context_surfacer`.
> The committed `config.yaml` ships this block empty; real values live in the
> **gitignored `config/config.local.yaml`**, deep-merged over `config.yaml` at load
> (`app_config.load_local_overrides()`). An empty/absent override = a fully generic install.

User says: *"My brother Sam just got a golden retriever named Biscuit"*

```
FactExtractor.extract_facts()
  │
  ├─ Stage 1: Correction detection → no correction patterns
  ├─ Stage 2: spaCy dependency parsing
  │     "Sam" (PERSON) "got" "golden retriever named Biscuit"
  │     → user | brother | Sam          (user fact, confidence 0.75)
  │     → Sam | has_pet | Biscuit       (entity fact, confidence 0.65)
  │     → Biscuit | is_a | golden retriever (entity fact, confidence 0.70)
  ├─ Stage 3: REBEL neural extraction (if available)
  ├─ Stage 4: Regex fallback (if < 5 triples)
  │
  └─ Pre-storage Filters (applied to every triple before budget/dedup):
        ├─ Ephemeral predicate filter: predicate checked against
        │     PROFILE_EPHEMERAL_RELATIONS config list (loaded once per call).
        │     Blocks transient state like current_mood, woke_up_time, greeting, etc.
        │     Applied before canonicalization so raw relation names are caught.
        ├─ Boolean noise filter: objects that are just "true"/"false"/"yes"/"no"
        │     are dropped — no informational content.
        ├─ Junk-object filter (2026-08-02): _is_junk_object (in _clean_triple
        │     AND llm_fact_extractor._normalize_triple since 2026-08-03 — the LLM
        │     path had no junk check and let dad_show_up=for a bit-class junk in)
        │     drops adverbial/temporal/negation-fragment objects ("for a bit",
        │     "with food", "yesterday", "not good", profanity-intensifier rants);
        │     schedule relations exempt; communication/status/access relations
        │     exempt from the NEGATION check only (2026-08-05 —
        │     "doctor_communication | no patient portal": the negation IS the content).
        └─ Polarity guard (2026-08-02): _polarity_conflict blocks positive-
              preference triples the source text negates ("hate my fucking life"
              had stored user | likes | my fucking life). The _canonicalize_preferences
              "i like " rewrite is clause-scoped (a bare substring hit used to
              rewrite ANY co-occurring triple to likes|obj). Historical junk:
              scripts/purge_junk_facts.py (166 purged 2026-08-02, pre-image backup).
              Tests: tests/unit/test_fact_junk_polarity_guard.py.

Dual budget applied:
  User facts (cap 6): [user | brother | Sam]
  Entity facts (cap 4): [Sam | has_pet | Biscuit, Biscuit | is_a | golden retriever]
```

### LLM Fact Extraction Filtering

`llm_fact_extractor.py` applies the same ephemeral and boolean guards during
`_normalize_triple()`, which runs on every triple the LLM returns:

1. **`_is_ephemeral_relation(rel)`** — loads `PROFILE_EPHEMERAL_RELATIONS` from
   config, rejects matching predicates.
2. **`_is_boolean_noise(obj)`** — rejects objects that are just "true"/"false"/"yes"/"no".

Both checks happen after pronoun normalization and snake_case conversion but
before category classification, so the triple is discarded early.

The LLM prompt itself explicitly discourages transient state extraction. The
RULES section lists examples of ephemeral predicates to avoid:
`current_activity`, `current_mood`, `current_feeling`, `feeling`, `feels`,
`woke_at`, `walked_to`, `showered`, `tidied`, `will_drive_to`, `greeting`,
and generic predicates like `is`, `was`, `has`, `thinks`, `plans`, `wants`,
`needs`. Boolean-value facts (e.g., "showered=true", "has_energy=true") are
also called out. The few-shot examples were updated to remove ephemeral
patterns (e.g., "current_activity: testing" was removed).

### Verification

Each fact passes through `FactVerifier.verify()`:

```
1. Ephemeral check: "brother" not in PROFILE_EPHEMERAL_RELATIONS → continue
2. Candidate query: search facts for subject="user", predicate="brother"
3. No existing "user | brother | Sam" → verdict: STORE
4. If existing "user | brother | Steve" found:
   a. User trust override? (source=user_stated, confidence >= threshold) → STORE_AND_FLAG
   b. LLM adjudication: "Is 'Sam' an update to 'Steve' or a second brother?"
      - UPDATE → STORE_AND_FLAG (mark Steve entry superseded)
      - COMPLEMENTARY → STORE (both coexist)
      - IMPLAUSIBLE → REJECT
```

### Graph Ingestion

After fact storage, `_ingest_fact_to_graph()`:

```
Fact: "user | brother | Sam"
  → _is_graph_worthy_object("Sam") → True (single proper noun)
  → entity_resolver.resolve_or_create("user") → "user"
  → entity_resolver.resolve_or_create("Sam") → "sam"
  → graph_memory.add_relation(user → brother → sam)

Fact: "Biscuit | is_a | golden retriever"
  → _is_graph_worthy_object("golden retriever") → False (2+ words, descriptive)
  → Store as node metadata: sam.metadata["has_pet"] = "Biscuit"
  → Store as node metadata: biscuit.metadata["is_a"] = "golden retriever"
```

### Relation Normalization

`entity_resolver.py` maps raw relation strings to canonical forms via
`RELATION_SYNONYMS` (24 canonical forms with 90+ synonyms). This prevents
duplicate edges from phrasing variations:

```
"lives in" / "resides in" / "based in"   → lives_in
"works on" / "building" / "developing"   → works_on
"likes" / "enjoys" / "loves" / "fond of" → likes
"sibling of" / "brother of" / "sister of" → sibling_of
"spouse of" / "married to" / "partner of" → spouse_of
"speaks" / "fluent in"                    → speaks
"born in" / "from" / "originally from"   → born_in
"skilled at" / "good at" / "proficient in" → skilled_at
```

Unknown relations pass through with spaces replaced by underscores
(e.g., "favorite color" becomes "favorite_color").

**2026-08-02 extension:** a scan found 630/696 stored relations were single-use
inventions. The synonym table grew (drinks/attends/completed/celebrates/...),
family-collapse patterns now cover `likes_*`/`dislikes_*`/`talked_about_*`/
`mentioned_*` (with a negative lookahead protecting `mentioned_alongside`, a
code-level structural relation used by wiki enrichment), and the LLM fact
extractor prompt carries a ~40-relation core vocabulary plus a "specifics
belong in the OBJECT" rule (inflow-side fix). Historical edges re-canonicalized
via `scripts/graph_relation_normalize.py --apply` (24 renamed, 0 collisions).
The remaining singleton tail is mostly content-bearing and can't collapse safely.

Possessive alias patterns (`_POSSESSIVE_RE`) auto-detect phrases like
"my cat", "my boss", "my brother" and register them as entity aliases
during `learn_alias()` calls.

### Graph Query Expansion + Junk Maintenance (2026-06)

`_expand_query_with_graph()` (in `core/prompt/gatherer_memory.py`) calls
`graph_utils.rank_expansion_candidates()` to append neighbour names to the
search query (see `PROMPT_BUILDING_PIPELINE.md`). Two read-side guards keep
expansion focused:

- **Hub-aware BFS.** A hub node — one in `skip_ids` (e.g. "user") or with
  degree >= `GRAPH_EXPANSION_HUB_DEGREE` (default 30) — may be *reached* but is
  never expanded *through*; only seed entities fan out freely. Fixes a bug where
  an incidental token linking to the "user" star hub (~797 of 889 edges) dumped
  the whole personal neighbourhood (pets, project terms) into unrelated queries.
  `extract_graph_entities()` also drops common-noun/participle stopwords
  (video/videos/done/homework/stuff) that had been wrongly resolving as entities.
- **Read-time TTL.** Stale transient edges (mood/activity/illness past their
  per-relation horizon, via `GraphMemory._edge_is_stale_transient` →
  `relation_classifier`) are dropped from both traversal and scoring — the same
  read-side staleness the profile, `facts` collection and graph-context
  (`get_context_sentences`) already apply. Nothing is deleted; a fresh mention
  refreshes `last_seen` and the edge returns.

`GraphMemory.remove_entity()` removes a node plus its incident edges from the
authoritative `_edge_index` and the alias index, so `save()` (which serialises
edges from that index) never re-persists dangling edges. The maintenance tool
`scripts/graph_junk_cleanup.py` is dry-run-first: a default run only writes a
tiered candidate file for review; `--apply` removes only the ids left
uncommented, after a timestamped JSON backup (respects the never-auto-delete
rule).

### Schedule Facts (2026-05)

Facts with `fact_type="schedule"` metadata extracted from schedule statements ("I work Friday 3-10pm"). Five categories: `work_schedule`, `class_schedule`, `exam_date`, `shift_pattern`, `day_off`. Metadata includes `schedule_scope` (recurring/one_off/ambiguous), `schedule_days`, `schedule_start`/`end` (HH:MM), `parser_confidence`, `resolution_basis`, `needs_confirmation`. Supersession via `FactVerifier._check_schedule_supersession()` when same kind + same day detected.

### Truth Score Lifecycle

```
Initial:     0.70  (source: llm_extracted)
             0.80  (source: user_stated)
             0.85  (source: corrected)

Confirmation: +0.08 (user restates the fact)
Correction:   -0.25 (user contradicts the fact, sharp penalty)
Contradiction: -0.15 (dedup finds conflicting claim)

Time decay:   -0.02 per week since last confirmation
              Floor: 0.30 (never decays below this)
              Confirmation resets the decay clock
```

### Staleness Tracking

When a fact is corrected, staleness cascades to summaries that embedded it:

```
1. User says "Actually I moved to Denver" (correction of "lives in Atlanta")
2. Correction event → claim_key = ClaimKey(subject="user", relation="lives_in")
3. ClaimIndex.cascade_staleness(claim_key):
   a. Find all docs containing this claim hash
   b. For each affected summary:
      - stale_claims: add this hash to the stale set
      - staleness_ratio = |stale_claims| / |total_claims|
      - Update ChromaDB metadata
4. During scoring: staleness_ratio * 0.15 penalty (2x at >=0.8)
5. In prompt: items with staleness_ratio >= 0.6 get prefix
   "[HISTORICAL — PARTIALLY OUTDATED]"
```

### Post-Response Truth Pipeline — GUI Wiring (2026-08-23)

The whole post-response pipeline above (correction/confirmation detection →
truth events → staleness cascade) was DEAD on the production GUI path: only
the unused `process_user_query` flow called `_run_post_response_detectors`
(same class as the 08-18 EscalationTracker GUI-wiring bug).
`orchestrator.run_post_response_detectors()` is now PUBLIC and called per
turn from `gui/handlers._write_turn_telemetry`; the flow-based wrapper
delegates to it so both paths run THE same pipeline. Same batch: a
terse-numeric-swap correction pattern joined the detector ("6 weeks off
vryalr not 1."), and `correction_detector.detect_correction_signal()`
(message-level, fact-list-INDEPENDENT) scores correction confidence even
when the correction matches no stored fact. Tests:
`tests/unit/test_correction_gui_wiring.py`.

### Narrative Staleness Flag (2026-08-23)

A correction can invalidate the cached narrative context
(`data/narrative_context.txt`, regenerated only at shutdown) even when it
matches NO stored fact — the 08-23 "6 weeks off vryalr not 1." case: the
wrong "day 8" framing lived in the narrative, and kept re-entering every
prompt's [TEMPORAL GROUNDING] for the rest of the session.
`utils/narrative_staleness.py` (NEW): a `detect_correction_signal()` score
≥ 0.6 in `run_post_response_detectors()` calls `mark_stale()` (atomic flag
write to `data/narrative_stale.json`; keeps the EARLIEST mark);
`corpus_manager.get_narrative_context()` appends a CAUTION line ("the user
corrected a factual detail after this narrative was generated … prefer the
user's most recent statements") while `is_stale(narrative_mtime)` holds; a
fresh narrative save calls `clear()`. The flag doesn't rewrite the
narrative (an LLM job) — it makes the prompt HONEST about it until the next
regeneration. Lenient derived state: corrupt/missing flag = not stale,
never raises (tone_state.json doctrine). Tests:
`tests/unit/test_narrative_staleness.py`.

---

## Stance / Epistemic Tagging (2026-08-23)

`memory/stance_classifier.py` is the single deterministic source for a
fact/edge's epistemic stance: **objective** / **appraisal** (value judgment)
/ **reported** / **inferred** / **unknown**. `LEGACY_STANCE_DEFAULT="unknown"`
covers all pre-existing untagged data; consumers act conservatively —
suppression fires only on EXPLICIT appraisal/inferred, and standing-granting
(settledness, see below) requires EXPLICIT non-elevated evidence. The
`EVALUATIVE_LEXICON` of thick evaluative terms (evil, abusive, toxic,
worthless, ...) drives the deterministic classification.

### Write Path

- `llm_fact_extractor` triples carry a `"stance"` field — the LLM fills
  gaps, but the deterministic classifier OVERRIDES on lexicon hits.
- `fact_extractor._clean_triple` preserves evaluative pronoun-subject
  triples user-scoped instead of dropping them.
- `memory_storage.extract_and_store_facts` forwards `stance` +
  `capture_tone` into fact metadata; `_ingest_fact_to_graph` writes them
  into `GraphEdge.metadata`.
- **Referent scoping**: `scope_unresolved_referent` — an evaluative claim
  with a pronoun/role subject ("she is abusive") re-scopes to
  "user's unnamed referent" / "user's last partner" and NEVER fuzzy-binds
  to a named entity. User-scoped role subjects become verbatim
  `entity_type="role"` graph nodes, bypassing the alias resolver.
- The shutdown LLM path joins each triple to its session corpus entry's
  `is_heavy_topic` flag for `capture_tone` (unmatched → unknown).

### Consumers

1. **Query expansion** — `graph_utils.rank_expansion_candidates` excludes
   explicit appraisal/inferred edges (fixes the "evil" expansion leak).
2. **Rendering** — `GraphEdge.to_natural_language` renders appraisals as
   "you described X as '...' (your words at the time, DATE)" — settled ones
   as "you've consistently described..." — and inferred edges with an
   "(assistant inference)" marker; `memory_retriever._present_fact_content`
   rewrites explicit-appraisal fact content the same way at the retrieval
   boundary (objective/legacy output stays byte-identical).
3. **Dedup** — `cross_deduplicator._find_fact_contradictions` skips
   explicit-appraisal facts (perspectives coexist; they are not
   contradictions to resolve).
4. **Profile** — `user_profile.add_fact` stores the stance, never promotes
   explicit appraisals into the quick profile, and applies a deterministic
   lexicon backstop when the caller passes no stance.
5. **Settledness** — `graph_memory.add_relation` tracks
   `appraisal_days`/`appraisal_tones` distinct-ISO-day lists: an appraisal
   restated on ≥3 DISTINCT non-elevated days gains metadata `settled=True`
   (elevated/unknown-tone days never count).

### Backfill

`scripts/backfill_stance.py` — dry-run default, refuses `--apply` while
Daemon runs (`utils/daemon_guard`), pre-image backups to
`data/backups/stance_backfill_<ts>/`, classifies all facts + graph edges
with THE deployed classifier. Hard sentinel: exits nonzero unless
`fact_e1f5f920_20260818_135210570` (casey|is|evil) classifies appraisal.
Dry-run validated on live data 2026-08-23 (3268 facts, 20 appraisal; 987
edges); `--apply` owner-gated.

Tests: `tests/unit/test_stance_classifier.py`, `test_stance_write_path.py`,
`test_stance_consumers.py`, `test_appraisal_settledness.py`,
`test_backfill_stance.py`.

---

## User Profile

The `UserProfile` class (`memory/user_profile.py`) manages persistent user
facts with append-only storage, categorized by life domain.

### Relation Canonicalization

Before storage, every relation name passes through
`canonicalize_profile_relation(rel, value)` in `user_profile_schema.py`:

1. `normalize_relation()` (from `entity_resolver.py`): spaces to underscores, synonym lookup
2. `SAFE_RELATION_ALIASES` auto-merge: ~40 variant names map to canonical forms
   (e.g., `pet` / `has_pet` / `owns_pet` all become `pet_name`)
3. Value-aware disambiguation: `job` with value "quit" becomes `job_status`;
   `job` with value "bartender" becomes `occupation`

### Categorization Cascade

`categorize_relation()` uses a 5-layer cascade to assign each relation to one
of 12 `ProfileCategory` values (identity, education, career, projects, health,
fitness, preferences, hobbies, study, finance, relationships, goals):

1. **Direct lookup** — `RELATION_CATEGORY_MAP` (~100 entries, exact match)
2. **Prefix lookup** — first underscore-delimited token checked against `_PREFIX_CATEGORY_MAP` (~60 entries)
3. **Cache check** — persistent `data/category_cache.json` checked before heavier layers
4. **Token overlap** — relation tokens scored against per-category keyword sets (`_CATEGORY_TOKENS`, ~30 categories with keyword sets); requires >= 2 matching tokens
5. **Embedding similarity** — `all-MiniLM-L6-v2` (the shared ModelManager embedder) cosine similarity against per-category exemplar phrases; threshold 0.30. Results cached persistently in `data/category_cache.json`
6. **Default** — falls back to `PREFERENCES`. For batch/cleanup, `categorize_relation_deep()` adds an LLM micro-call (gpt-4o-mini, 10 tokens) before defaulting

### Ephemeral vs Snapshot Relations

- **`EPHEMERAL_RELATIONS`** — truly transient state (`current_feeling`, `current_activity`,
  `plans_today`, `appointment_time`, etc.). Detection uses 4 layers: config list,
  exact-match set (`meal`, `drank_alcohol`, `meeting_with`, `appointment`, etc.), prefix patterns
  (`scheduled_`, `signed_up_`, `meeting_with_`, `current_`, `recent_`, etc.), and
  suffix patterns (`_appointment`, `_meeting`, `_intake`, `_consumption`, etc.).
  *Note (2026-06):* bare **`appointment`** is now in the exact-match set (24h TTL,
  sibling to `meeting` / `meeting_with` and the `*_appointment` suffix), so a
  one-time event ages out instead of surfacing forever — this fixed the
  "phantom standing appointment" bug where a stale `user → appointment →
  psychiatry` edge resurfaced indefinitely. (Durable attributes like
  `appointment_length` stay; future-dated deadlines/dates such as `exam_date`
  are *not* swept here — those need date-aware expiry, not a last-seen TTL.)
  Subject to aggressive TTL-based expiry (`PROFILE_EPHEMERAL_TTL_HOURS`, default 24h).
  Historical entries pruned when category exceeds `PROFILE_CATEGORY_SOFT_CAP`
  (default 200), keeping at most `PROFILE_EPHEMERAL_MAX_HISTORY` (default 20)
  old entries per ephemeral relation.

- **`SNAPSHOT_RELATIONS`** — measurements/states valid until explicitly changed
  (`current_weight`, `current_bench`, `current_medication`). NOT expired by TTL;
  kept until superseded by a new value.

- **`SEMANTIC_RELATION_NEIGHBORS`** — related but distinct relations shown as
  hints in LLM extraction prompts (e.g., `bench_max` hints: `current_bench`,
  `previous_bench_max`, `goal_bench`). Never auto-merged.

### Supersedes Chain

Profile storage is append-only. When a user updates a fact:

```
Existing: occupation=bartender (is_current=True, fact_id="abc123")
User says: "I quit my job"

1. canonicalize_profile_relation("job", "quit") → "job_status"
   Different canonical relation → Case 3 (new relation), not a supersede.

User says: "I'm a software engineer now"
1. canonicalize_profile_relation("job", "software engineer") → "occupation"
2. Find existing is_current=True facts with relation="occupation"
3. Mark old fact: is_current=False, truth_score reduced by correction penalty
4. New fact: occupation=software engineer, supersedes="abc123", is_current=True
```

Old facts are never deleted. `is_current=False` facts serve as historical record
and are excluded from active profile injection.

**Extraction coverage + learned relations (2026-08-05):** the shutdown LLM
extractor now sees ~30 turns per shutdown (responses truncated to a 250-char
context snippet; newest-first budget selection — previously full responses +
oldest-first `break` meant ~4-5 pairs and the newest turns dropped, which is
why weeks of "my doctor doesn't respond" never produced a fact). Recurring
invented relations that survive the extraction gates are tracked in
`memory/learned_relations.py` (`data/learned_relations.json`) and promoted
into the extractor prompt's preferred-vocabulary list after appearing on ≥3
distinct days — the vocabulary grows itself; guards (gate-surviving triples
only, canonicalization collapse first, shape check, ephemeral exclusion,
multi-day recurrence, cap 15) keep the 2026-08-02 single-use-relation
explosion from returning through this door.

**Confirmation re-currents (2026-08-05):** re-stating an exact value that was
previously superseded doesn't just boost its confidence — the confirmed fact
becomes `is_current=True` again and any conflicting current facts for the
relation are superseded with a truth correction. Before this, a later junk
value (live: `age=19` over the true `33`, `timezone="your time"` over
`Central`) stayed current forever; no number of user confirmations could
displace it. Owner-curated facts can be asserted through the same deployed
path via `scripts/add_profile_fact.py` (dry-run by default); curated
deletions by fact_id via `scripts/purge_profile_facts.py` (dry-run +
pre-image backup, owner-gated `--apply`).

### Temporal Resolution

When a fact value contains relative temporal references ("tomorrow", "next Monday",
"in 3 days"), `resolve_temporal_references()` from `utils/temporal_resolver.py`
converts them to absolute dates at storage time:

```
Input:  "work tomorrow and the following day" (reference: 2026-03-12)
Output: "work on Thu 2026-03-13 and Fri 2026-03-14"
```

This prevents facts from becoming semantically incorrect as time passes.

### Profile Context Injection

`get_context_injection()` formats facts for the `[USER PROFILE]` prompt section:
- Facts grouped by category, filtered to `is_current=True`
- Hybrid ranking: 2/3 semantic relevance (keyword overlap with query) + 1/3 recency
- Timestamps shown as relative labels ("today", "3 days ago") via `format_relative_timestamp()`
- Source excerpts appended as `(said: "...")` when available (80-char truncated)

---

## Multi-Stage Gating

After retrieval, semantic memories pass through `MultiStageGateSystem`:

```
Input: 30 candidate memories + query

Stage 1: Separation
  Episodic (type=="episodic") → always included, bypass gating
  Others → continue to Stage 2

Stage 2: Blended Scoring (retrieval space)
  Encode query + all memory texts (batch, cached per model) with the SAME
  bge-small embedder ChromaDB retrieved with (injected as retrieval_embedder
  in main.py — fixes the old MiniLM-space mismatch, 2026-07-02)
  For each memory:
    blended = 0.85 * cosine_sim + 0.15 * truth_score + entity_boost
    entity_boost: +0.18 (1 entity match) or +0.25 (multiple)
  Threshold: gate_rel_threshold_retrieval (0.60 bge-blended ≈ old 0.18 MiniLM,
  quantile-matched via scripts/probe_gate_embedding_mismatch.py); per-intent
  MiniLM-space overrides are delta-translated (×0.38 std ratio)
  Deictic queries: floor raised to gate_deictic_min_retrieval (0.61)

Stage 3: Forced Minimum
  If < 8 passed, force-add highest-scoring non-passed items
  (clamped to min_results since 2026-08-21 — the caller's intent budget:
   gatherer_memory passes min_gated=limit → memory_coordinator →
   memory_retriever → gate; the floor can only LOWER, so a bare "Hey" at
   casual_social max_mems=3 no longer gets 8 below-threshold memories forced
   in. Tests: tests/unit/test_gate_min_results_cap.py)
  (QUALITY floor since 2026-08-23: forced items must still score above
   threshold − GATE_FORCED_FLOOR_MARGIN (env, default 0.05) — fail-soft
   means "don't starve an ordinary turn", not "force junk in". Near-misses
   may be rescued; a turn with 0 natural passes now renders
   fewer-but-honest memories instead of the floor's worth of junk.
   Tests: tests/unit/test_gate_forced_quality_floor.py)

Stage 4: Cross-Encoder Reranking (if available, > 5 items)
  Rerank by cross-encoder score

Stage 5: Cap
  Final = episodic + gated[:20 - len(episodic)]
```

**Timing:** ~200ms total (ChromaDB HNSW candidate generation ~50ms, cosine ~50ms, cross-encoder ~100ms). No FAISS in the live memory path — FAISS is used only for the wiki index and visual memory.

### Insight Evidence Sweep — the UNGATED Exception (2026-08-23)

Insight / evidence-assembly turns (`core/insight/sweep.py` — see
`AGENTIC_SEARCH.md`) deliberately BYPASS this gate: a theme-sweep request
("gather everything I've said about X") needs an evidence SET whose members
are individually low-pairwise-similarity but collectively signal-bearing —
per-doc cosine gating structurally cannot pass such sets. The sweep instead
uses generous caps (`per_facet_cap`=10, `total_evidence_cap`=80) over
conversations/summaries/reflections/facts/obsidian_notes/threads, plus a
word-boundary keyword scan via the new read-only
`corpus_manager.search_keyword(terms, start, end, max_results, context_chars)`
(case-insensitive over episodic entries; query and response scanned
SEPARATELY for accurate speaker attribution; newest-first; junk-filtered),
graph 1-hop expansion (hub/stale/appraisal-edge skips), and `MemoryExpander`
windows around top conversation hits. Tests:
`tests/unit/test_insight_sweep.py`, `tests/unit/test_corpus_keyword_search.py`.

---

## Procedural Skill Activation

After parallel retrieval, procedural skills from the `procedural_skills`
collection pass through `SkillActivationPolicy` (`memory/skill_activation.py`),
a post-retrieval filter that prevents irrelevant or repetitive workflows
from consuming prompt budget.

### Filtering Pipeline

```
Input: N candidates from ChromaDB procedural_skills (over-fetched at 3x)

1. Intent suppression
   EMOTIONAL_SUPPORT, CASUAL_SOCIAL → return [] immediately

2. Minimum score threshold
   Drop candidates with relevance_score < 0.25

3. STM topic bonus
   If skill tags/trigger match STM topics → +0.10 relevance boost

4. Cooldown filter
   Skip skills surfaced within past 48 hours
   Tracked in data/skill_cooldown.json (JSON-backed TTL store)

5. Cap
   Return top 3 by adjusted score

Output: 0-3 activated skills for prompt injection
```

### Cooldown Store

`SkillCooldownStore` persists surfacing history at `data/skill_cooldown.json`.
Each entry stores the last-surfaced ISO timestamp and a running count.
Entries older than 30 days are cleaned up via `cleanup_old()`.

### Configuration

| Constant | Default | Purpose |
|----------|---------|---------|
| `SKILL_ACTIVATION_ENABLED` | True | Master toggle |
| `SKILL_ACTIVATION_MAX_SKILLS` | 3 | Max skills surfaced per turn |
| `SKILL_ACTIVATION_MIN_SCORE` | 0.25 | Minimum relevance threshold |
| `SKILL_ACTIVATION_COOLDOWN_HOURS` | 48.0 | Re-surfacing cooldown |
| `SKILL_ACTIVATION_FETCH_MULTIPLIER` | 3 | Over-fetch factor for wider candidate pool |
| `SKILL_ACTIVATION_STM_BONUS` | 0.10 | Bonus for STM topic overlap |
| `SKILL_ACTIVATION_USE_STM` | True | Enable STM bonus |

YAML section: `skill_activation:` in `config/config.yaml`.

---

## Token Budget Management

The prompt has a finite token budget (default 10,000 tokens, floor 8K,
ceiling 16K, context fraction 0.12 — config.yaml `token_budget`). Sections
are prioritized:

```
Priority 10: STM summary (metadata, never trimmed)
Priority  9: User profile (identity, naturally bounded)
Priority  8: Narrative state (temporal grounding, hard cap 500 tokens), web search results
Priority  7: Recent conversations, graph context, unresolved threads
Priority  6: Semantic chunks, personal notes, user uploads
Priority  5: Reference docs, memories
Priority  4: Procedural skills, facts
Priority  3: Summaries, proposed_features, git commits, proactive insights
Priority  2: Reflections, dreams, codebase changes
Priority  1: Wiki
```

**Compression hierarchy:**
1. LLM compression — items >=3x over limit get intelligent reduction
2. Middle-out — keeps 60% head + 40% tail, trims middle
3. Removal — lowest priority sections dropped first

---

## Prompt Assembly Order

The final prompt is assembled with these sections (in attention-optimized order):

```
[RECENT CONVERSATION]                  ← last N turns (session continuity)
[RELEVANT MEMORIES]                    ← scored semantic memories
[RECENT SUMMARIES]                     ← compressed recent history
[SEMANTIC SUMMARIES]                   ← query-relevant compressed history
[RECENT REFLECTIONS]                   ← meta insights, recent
[SEMANTIC REFLECTIONS]                 ← meta insights, query-relevant
[BACKGROUND KNOWLEDGE]                 ← wiki snippets
[WEB SEARCH RESULTS]                   ← real-time web (if triggered)
[RELEVANT INFORMATION]                 ← semantic chunks
[DREAMS]                               ← synthesis insights (if enabled)
[USER'S PERSONAL NOTES]                ← Obsidian vault
[USER UPLOADED ITEMS]                  ← uploaded documents
[VISUAL MEMORIES]                      ← CLIP-matched image metadata
[DAEMON DOCUMENTATION]                 ← reference docs
[PROJECT COMMIT HISTORY]               ← git commits
[ADAPTIVE WORKFLOWS]                   ← procedural skills
[PROPOSED FEATURES]                    ← code proposals
[KNOWLEDGE GRAPH]                      ← entity relationships (natural language)
[UNRESOLVED THREADS]                   ← open commitments/deadlines
[UPCOMING SCHEDULE]                    ← schedule facts for next N days (intent-gated)
[GOOGLE CALENDAR]                      ← upcoming calendar events (if enabled)
[DAEMON SELF-NOTES]                    ← Daemon's own session notes (caveat-labeled)
[PROACTIVE INSIGHTS]                   ← cross-domain connections
[USER PROFILE]                         ← categorized facts (high-attention zone)
[ACTIVE FEATURES]                      ← feature inventory (always)
[CODEBASE CHANGES SINCE LAST SESSION]  ← git diff (first message only)
[TIME CONTEXT]                         ← current datetime (high-attention zone)
[TEMPORAL GROUNDING]                   ← narrative context
[SHORT-TERM CONTEXT SUMMARY]           ← STM analysis (24h window + daily notes injection + reference_type / temporal_facts disambiguation)
[CURRENT USER QUERY]                   ← always last, protected from compression
```

---

## Tuning Guide

### Retrieval Returning Irrelevant Memories

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Old unrelated memories ranking high | Recency weight too low | Increase `recency` in `SCORE_WEIGHTS` |
| `[API unavailable]`/"test" turns surfacing | Junk docs stored pre-2026-07-03 guard | Retrieval already filters them (`memory/utils.is_junk_conversation_doc`); purge stored docs via `scripts/purge_error_memories.py` (dry-run first) |
| Memories from wrong topic | Topic weight too low | Increase `topic_match` weight (live 0.10) |
| Too many low-quality results | Gate threshold too low | Raise `GATE_REL_THRESHOLD_RETRIEVAL` (default 0.60, bge space — small moves shift pass rate a lot) |
| Large docs drowning out small facts | Size penalty too weak | Lower `LARGE_DOC_SIZE_THRESHOLD` in `memory_scorer.py` (default 10KB) or raise `LARGE_DOC_BASE_PENALTY` (default -0.25) |

### Retrieval Missing Relevant Memories

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Facts stated recently not found | Fact extraction disabled | Enable `FACTS_EXTRACT_EACH_TURN` |
| Semantic search missing vocabulary | Graph expansion disabled | Enable `GRAPH_QUERY_EXPANSION_ENABLED` |
| Gating too aggressive | Threshold too high | Lower `GATE_REL_THRESHOLD_RETRIEVAL` (default 0.60) or increase `MIN_GATED_MEMORIES` |
| Deictic follow-ups losing context | Anchor bonus too weak | Check `DEICTIC_ANCHOR_PENALTY` and continuity weights |

### Fact Extraction Problems

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Entity facts polluting results | Entity confidence too low | Raise `ENTITY_FACT_MIN_CONFIDENCE` (default 0.55) |
| User corrections not taking effect | Correction detection missed | Check patterns in `fact_extractor.py` Stage 1 |
| Contradictions not flagged | Fact verification disabled | Enable `FACT_VERIFICATION_ENABLED` |
| Old facts not penalized | Staleness tracking disabled | Enable `STALENESS_ENABLED` |

### Truth Score Issues

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Old facts never decay | Decay rate too low | Increase `TRUTH_SCORER_DECAY_RATE` (default 0.02/week) |
| Confirmed facts decaying too fast | Confirmation boost too small | Increase `TRUTH_SCORER_CONFIRMED_BOOST` (default +0.08) |
| Corrections not penalizing enough | Correction penalty too mild | Increase `TRUTH_SCORER_CORRECTION_PENALTY` (default -0.25) |

### Token Budget Issues

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Context too short | Budget too small | Increase `PROMPT_TOKEN_BUDGET` or `PROMPT_TOKEN_BUDGET_CONTEXT_FRACTION` |
| Important memories getting trimmed | Priority wrong | Check `token_manager.py` priority ordering |
| Prompt dominated by one section | Per-section cap missing | Add/lower `PROMPT_MAX_*` for that section |

---

## Key Configuration

### Scoring
| Constant | Default | Purpose |
|----------|---------|---------|
| `SCORE_WEIGHTS` | see above | 6-factor weights dict |
| `RECENCY_DECAY_RATE` | 0.05 | Exponential decay speed (config.yaml `memory.recency_decay_rate`) |
| `DEICTIC_THRESHOLD` | 0.25 | Acceptance threshold for follow-up queries (live config.yaml; code default 0.60) |
| `NORMAL_THRESHOLD` | 0.15 | Acceptance threshold for normal queries (live config.yaml; code default 0.35) |
| `COSINE_SIMILARITY_THRESHOLD` | 0.15 | Minimum cosine gate (live config.yaml; code default 0.25) |

### Gating
| Constant | Default | Purpose |
|----------|---------|---------|
| `GATE_REL_THRESHOLD` | 0.18 | Gate threshold in legacy MiniLM space (used when no retrieval embedder is injected, e.g. fallback gates) |
| `GATE_REL_THRESHOLD_RETRIEVAL` | 0.60 | Gate threshold in retrieval (bge) space — the LIVE memory-gate threshold; quantile-matched to 0.18 MiniLM |
| `GATE_DEICTIC_MIN_RETRIEVAL` | 0.61 | Deictic follow-up floor in retrieval space (≈ 0.20 MiniLM) |
| `GATE_COSINE_WEIGHT` | 0.85 | Weight of cosine vs truth in blended gate score (env var in gate_system.py, NOT app_config) |
| `MIN_GATED_MEMORIES` | 8 | Forced minimum even if below threshold (env var in gate_system.py, NOT app_config) |
| `GATE_FORCED_FLOOR_MARGIN` | 0.05 | Quality floor for the forced-minimum backfill (2026-08-23): items below threshold − margin are never forced (env var in gate_system.py, NOT app_config) |

### Facts & Truth
| Constant | Default | Purpose |
|----------|---------|---------|
| `FACTS_EXTRACT_EACH_TURN` | varies | Extract facts every turn vs shutdown only |
| `USER_FACTS_PER_TURN_CAP` | 6 | Max user facts per extraction |
| `ENTITY_FACTS_PER_TURN_CAP` | 4 | Max entity facts per extraction |
| `ENTITY_FACT_MIN_CONFIDENCE` | 0.55 | Minimum confidence for entity facts |
| `TRUTH_SCORER_DECAY_RATE` | 0.02 | Per-week decay rate |
| `TRUTH_SCORER_DECAY_FLOOR` | 0.30 | Minimum truth score after decay |
| `STALENESS_WEIGHT` | 0.15 | Scoring penalty per unit staleness |
| `STALENESS_STEEP_THRESHOLD` | 0.80 | Ratio at which penalty doubles |

### Token Budget
| Constant | Default | Purpose |
|----------|---------|---------|
| `PROMPT_TOKEN_BUDGET_DEFAULT` | 10000 | Base token budget (model-aware; LOCAL=12000, FLOOR=8000, CEILING=16000) |
| `PROMPT_TOKEN_BUDGET_CONTEXT_FRACTION` | 0.12 | Fraction of model context window |
| `PROMPT_MAX_RECENT` | 10 | Max recent conversations (live config.yaml `prompt_max_recent`; code default 15) |
| `PROMPT_MAX_MEMS` | 30 | Max semantic memories (live config.yaml `prompt_max_mems`; code default 15) |
| `PROMPT_MAX_RECENT_FACTS` | 30 | Max facts (builder default; no YAML key) |

### Graph
| Constant | Default | Purpose |
|----------|---------|---------|
| `KNOWLEDGE_GRAPH_ENABLED` | True | Master toggle |
| `KNOWLEDGE_GRAPH_RETRIEVAL_DEPTH` | 2 | BFS traversal depth |
| `GRAPH_SCORING_BOOST_ENABLED` | True | Enable graph-proximity bonus in scoring |
| `GRAPH_SCORING_BOOST_CAP` | 0.15 | Max graph bonus per memory |
| `GRAPH_QUERY_EXPANSION_MAX_TERMS` | 8 | Max neighbor names appended to query |
| `GRAPH_EXPANSION_HUB_DEGREE` | 30 | Degree at/above which a node is treated as a hub — reachable but never expanded through |
| `GRAPH_EXPANSION_MIN_MENTIONS` | 2 | Evidence bar for expansion candidates (2026-08-23): nodes with `mention_count` below this never become expansion terms (env-overridable in graph_utils.py; 0 disables; nodes without the field pass) |

### Profile Namespace
| Constant | Default | Purpose |
|----------|---------|---------|
| `PROFILE_EPHEMERAL_RELATIONS` | list | Canonical source for ephemeral predicate definitions. Used by: fact_extractor (extraction blocking), llm_fact_extractor (extraction blocking), memory_retriever (TTL filter), cross_deduplicator (dedup skip) |
| `PROFILE_EPHEMERAL_TTL_HOURS` | 24 | Hours before ephemeral facts are dropped from retrieval results (memory_retriever TTL filter) |
| `PROFILE_EPHEMERAL_MAX_HISTORY` | 20 | Max historical entries kept per ephemeral relation |
| `PROFILE_CATEGORY_SOFT_CAP` | 200 | Max facts per category before pruning kicks in |

### Insight Mode (2026-08-23)
| Constant | Default | Purpose |
|----------|---------|---------|
| `INSIGHT_MODE_ENABLED` | True | Master toggle (YAML `insight_mode.enabled`) |
| `INSIGHT_MAX_FACETS` | 6 | Max facet queries from decomposition (incl. mandatory counter-evidence facet) |
| `INSIGHT_PER_FACET_CAP` | 10 | Max evidence items per facet |
| `INSIGHT_TOTAL_EVIDENCE_CAP` | 80 | Global evidence cap across the sweep |
| `INSIGHT_EVIDENCE_SNIPPET_CHARS` | 280 | Per-item snippet length |
| `INSIGHT_KEYWORD_SCAN_MAX` | 50 | Max hits from the corpus keyword scan |
| `INSIGHT_EXPAND_TOP_K` | 3 | Top conversation hits expanded via MemoryExpander |
| `INSIGHT_EXPAND_WINDOW` | 2 | Expansion window per hit |
| `INSIGHT_SWEEP_TIMEOUT_S` | 45 | Sweep timeout (partial results returned) |
| `INSIGHT_OFFER_ENABLED` | True | Consent-offer arming on insight-shaped statements |
| `INSIGHT_DOC_ON_AGREEMENT` | True | Save doc on agree/partial assessment |
