# Memory System Operations Guide

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
| `memory/memory_coordinator.py` | Thin orchestrator (~632 lines), creates all components, delegates to retriever/storage/shutdown |
| `memory/memory_retriever.py` | Retrieval: collection selection, gating, threshold fallbacks |
| `memory/memory_scorer.py` | Scoring algorithm (6 weighted factors + 7 additive bonuses/penalties) with intent overrides + graph boost |
| `memory/memory_storage.py` | Storage: ChromaDB + corpus writes, fact extraction hook, graph ingestion |
| `core/prompt/builder.py` | UnifiedPromptBuilder: parallel retrieval, token budgets, prompt assembly |
| `core/prompt/context_gatherer.py` | 20+ parallel retrieval tasks, graph expansion, web search |
| `core/prompt/token_manager.py` | Budget enforcement, middle-out compression, priority ordering |
| `processing/gate_system.py` | Multi-stage gating: cosine similarity + cross-encoder reranking |

### Fact & Truth Pipeline
| File | Purpose |
|------|---------|
| `memory/fact_extractor.py` | Multi-stage extraction: corrections > spaCy > REBEL > regex, dual budget |
| `memory/llm_fact_extractor.py` | LLM-assisted triple extraction with entity support; accepts existing profile facts for relation reuse |
| `memory/fact_verification.py` | Pre-storage conflict checking: ephemeral > candidates > trust > LLM adjudication |
| `memory/truth_scorer.py` | Stateless truth computation: initial score + adjustments + time decay |
| `memory/claim_tracker.py` | Claim extraction, hashing, reverse index, staleness cascade |

### Knowledge Graph
| File | Purpose |
|------|---------|
| `memory/graph_memory.py` | NetworkX DiGraph: CRUD, BFS traversal, JSON persistence |
| `memory/graph_models.py` | Pydantic models: GraphNode, GraphEdge |
| `memory/entity_resolver.py` | Alias resolution + relation normalization |
| `memory/graph_utils.py` | Entity extraction, neighbor lookups, expansion ranking |

### Shutdown & Consolidation
| File | Purpose |
|------|---------|
| `memory/shutdown_processor.py` | 10-step session-end processing |
| `memory/thread_manager.py` | Thread detection for conversation continuity |
| `memory/thread_store.py` | ChromaDB-backed thread persistence + priority ranking + per-turn regex resolution |
| `memory/thread_extractor.py` | LLM-based thread extraction + resolution detection |

### Storage Layer
| File | Purpose |
|------|---------|
| `memory/storage/multi_collection_chroma_store.py` | ChromaDB wrapper: 12 collections, `add_to_collection`, `query_collection`, `get_by_id`, `update_metadata` |

---

## 12 ChromaDB Collections

| Collection | Content | Protected | Deduped |
|------------|---------|-----------|---------|
| `conversations` | Raw Q/A turns | Yes | Never |
| `facts` | Extracted triples (user + entity) | No | Yes |
| `summaries` | Compressed conversation blocks | No | Yes |
| `reflections` | Session-end reflections | No | Yes |
| `wiki_knowledge` | Wikipedia content | Yes | Never |
| `obsidian_notes` | User's personal notes | Yes | Never |
| `reference_docs` | Uploaded docs + auto-seeded docs/ | Yes | Never |
| `procedural` | Git commits, how-to knowledge | No | Yes |
| `procedural_skills` | Reusable problem-solving patterns | No | Yes |
| `proposals` | Goal-directed code change proposals | No | Yes |
| `threads` | Open threads (commitments, deadlines) | No | Yes |
| `synthesis_results` | Cross-domain synthesis insights (+ human audit: two-layer grading (3 binary screening + 1-5 slider); see `docs/grading_plan.md`) | No | Yes |

**Protected** = never scanned by cross-collection deduplicator.

---

## End-to-End Lifecycle: One Conversation Turn

### Phase 1: Retrieval (building the prompt)

User sends: *"How's my squat progress looking?"*

```
UnifiedPromptBuilder.build_prompt()
  │
  ├─ 1. Query Analysis
  │     IntentClassifier → FACTUAL_RECALL (pattern: "how's my")
  │     Retrieval overrides: max_facts=30, max_mems=20
  │     Weight overrides: truth=0.30 (boosted for factual recall)
  │
  ├─ 2. Graph Query Expansion
  │     extract_graph_entities("squat progress") → ["squat"]
  │     graph neighbors: ["powerlifting", "deadlift", "bench press"]
  │     expanded query: "How's my squat progress looking? powerlifting deadlift bench press"
  │
  ├─ 3. Parallel Retrieval (20+ async tasks, 30s timeout)
  │     ├── recent_conversations (15)     ← corpus_manager
  │     ├── semantic_memories (20)        ← ChromaDB [conversations, summaries, reflections]
  │     ├── user_profile (3000 tokens)    ← UserProfile (categorized facts)
  │     ├── facts (30)                    ← ChromaDB facts collection
  │     ├── summaries (recent+semantic)   ← ChromaDB summaries
  │     ├── reflections (recent+semantic) ← ChromaDB reflections
  │     ├── graph_context (12 sentences)  ← GraphMemory BFS traversal
  │     ├── procedural_skills (5)         ← ChromaDB procedural_skills
  │     ├── unresolved_threads (3)        ← ThreadStore
  │     ├── proactive_insights (2)        ← ContextSurfacer
  │     ├── wiki_content (3)              ← FAISS (40M wiki vectors; ChromaDB fallback)
  │     ├── reference_docs (5)            ← ChromaDB reference_docs
  │     ├── personal_notes (5)            ← Obsidian vault
  │     ├── git_commits (10)              ← ChromaDB procedural
  │     ├── web_search (if triggered)     ← Tavily API
  │     └── codebase_changes (first msg)  ← git diff
  │
  ├─ 4. Multi-Stage Gating (semantic memories)
  │     Top 2 recent → bypass gating
  │     Remaining → cosine similarity → cross-encoder reranking
  │     Forced minimum: 8 memories even if below threshold
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
  ├─ 1. Skip gate: reject file-error responses
  ├─ 2. Thread detection: assign thread_id + depth
  ├─ 3. Corpus storage: JSON persistence (immediate)
  ├─ 4. Topic detection: primary topic → tag enrichment
  ├─ 5. Score calculation: truth_score + importance_score
  ├─ 6. Metadata assembly: timestamp, tags, thread, provenance
  ├─ 7. ChromaDB storage: conversations collection
  │
  ├─ 7b. Per-turn thread resolution (check_quick_resolutions)
  │     Pure regex: completion signals × open thread keywords, ~1ms
  │     Skips DB query if no completion signal detected in message
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
  ├─ Step 1: Block summaries (N=10 conversations per block)
  │     LLM consolidation → micro-summary fallback
  │     Claim extraction → ClaimIndex registration
  │     Source doc IDs stored for expand_memory drill-down
  │
  ├─ Step 2: Session fact extraction (rule-based, last 10 turns)
  │     Fact verification gate (REJECT/STORE/STORE_AND_FLAG)
  │     On STORE_AND_FLAG: mark conflicting facts as superseded
  │
  ├─ Step 3: LLM fact extraction (neural triples)
  │     Batch verification before storage
  │
  ├─ Step 4: Procedural skill extraction
  ├─ Step 5: Code proposal generation
  ├─ Step 6: Implementation tracking (lightweight file check)
  │
  ├─ Step 7: Open thread processing
  │     Detect resolutions → extract new → enforce cap
  │
  ├─ Step 8: Synthesis dreaming (three-tier parallel generation)
  │     Auto-halt check: skips if audit FP rate > SYNTHESIS_AUDIT_FP_HALT_THRESHOLD
  │     Tier 0: RetrievalSynthesisGenerator (structural query → FAISS → adversarial eval)
  │     Tier 1: GraphWalkGenerator (biased Markov walks, hub-dampened, cross-domain)
  │     Tier 2: SynthesisGenerator (cross-store sampling, FAISS wiki search)
  │     All → SynthesisFilter → SynthesisMemory
  │     On acceptance: provisional bridge edge created (weight=0.0, status="provisional")
  │     Composite-rejected candidates stored for FN audit review
  │
  ├─ Step 9: Knowledge graph save (JSON flush)
  └─ Step 10: Cross-collection dedup (dry-run preview only)
```

---

## Scoring Algorithm

Every retrieved memory gets a `final_score` from `MemoryScorer.rank_memories()`.

### Default Weights

```
relevance:  0.35    # Pre-gate semantic similarity (config.yaml: 0.30)
recency:    0.25    # Temporal decay (active-day aware) (config.yaml: 0.22)
truth:      0.20    # Evidence-based correctness (config.yaml: 0.18)
importance: 0.05    # Retention priority
continuity: 0.10    # Token overlap with current conversation
structure:  0.05    # In SCORE_WEIGHTS dict but UNUSED — actual structure is additive: 0.15 * density_alignment
topic:      0.00    # Disabled by default (config.yaml: 0.10)
```

### Step-by-Step

1. **Base relevance** — Embedding similarity from ChromaDB query + collection boost
2. **Recency decay** — Active-day aware: `1/(1 + decay_rate * age_hours)`. Temporal anchor override for TEMPORAL_RECALL queries reshapes the curve
3. **Truth score** — `TruthScorer.compute_effective_truth(metadata)`: stored score + time decay from last confirmation
4. **Importance** — Stored importance score (default 0.5)
5. **Continuity** — Token overlap with last exchange (+0.3 * overlap) + recency bonus (+0.1 if within 10 minutes)
6. **Structural alignment** — `0.15 * density_alignment` where density_alignment measures numeric/operator density match between query and memory. Added as direct bonus, not through weighted sum
7. **Penalties** — Analogy penalty (-0.1 for mathy queries matching analogies), size penalty (scales from 10KB+, caps at -1.0)
8. **Anchor bonus** — Salient token overlap with conversation context. Deictic queries ("explain that", "what about it") get +0.2 bonus or -0.15 penalty based on overlap
9. **Tone adjustment** — Dismissive language in memory → truth reduced by 0.2
10. **Topic match** — 1.0 exact, 0.5 unknown, 0.2 mismatch (usually weight=0.0)
11. **Meta-conversational bonus** — +0.15 for episodic memories when query is about recall ("did we discuss...")
12. **Graph proximity bonus** — +0.05 per knowledge graph neighbor mentioned in memory, capped at 0.15
13. **Staleness penalty** — `staleness_ratio * STALENESS_WEIGHT`, 2x multiplier at >=0.8 ratio, reflections at 60% weight, capped at 0.4

**Final guardrail:** Deictic queries with low continuity AND low anchor overlap get a 15% penalty to prevent drift.

### Concrete Example

Memory: *"User's squat is 365lb, set last month at the gym"*

```
relevance:    0.35 * 0.82 = 0.287    (high semantic match to "squat progress")
recency:      0.25 * 0.45 = 0.113    (3 weeks old, moderate decay)
truth:        0.20 * 0.85 = 0.170    (confirmed once, slight time decay)
importance:   0.05 * 0.60 = 0.030    (moderate importance)
continuity:   0.10 * 0.15 = 0.015    ("squat" token overlap)
structure:    0.15 * 0.90 = 0.135    (direct additive bonus — high numeric density alignment — "365lb")
graph_bonus:  0.05                    (1 neighbor "powerlifting" mentioned)
staleness:    0.00                    (no stale claims)
penalties:    0.00
─────────────────────────────────────
final_score:  0.805
```

### Intent-Driven Weight Overrides

The IntentClassifier detects query intent and overrides scoring weights:

| Intent | Key Override | Effect |
|--------|-------------|--------|
| FACTUAL_RECALL | truth=0.30 | Prioritize confirmed facts |
| TEMPORAL_RECALL | recency=0.40, `_temporal_anchor_hours` | Reshape decay curve around time window |
| EMOTIONAL_SUPPORT | continuity=0.20 | Prioritize recent conversation flow |
| TECHNICAL_HELP | relevance=0.45 | Prioritize semantic match |
| CASUAL_SOCIAL | recency=0.35 | Prioritize recent over deep |

---

## Fact Pipeline

### Extraction

User says: *"My brother Auggie just got a golden retriever named Biscuit"*

```
FactExtractor.extract_facts()
  │
  ├─ Stage 1: Correction detection → no correction patterns
  ├─ Stage 2: spaCy dependency parsing
  │     "Auggie" (PERSON) "got" "golden retriever named Biscuit"
  │     → user | brother | Auggie          (user fact, confidence 0.75)
  │     → Auggie | has_pet | Biscuit       (entity fact, confidence 0.65)
  │     → Biscuit | is_a | golden retriever (entity fact, confidence 0.70)
  ├─ Stage 3: REBEL neural extraction (if available)
  └─ Stage 4: Regex fallback (if < 5 triples)

Dual budget applied:
  User facts (cap 6): [user | brother | Auggie]
  Entity facts (cap 4): [Auggie | has_pet | Biscuit, Biscuit | is_a | golden retriever]
```

### Verification

Each fact passes through `FactVerifier.verify()`:

```
1. Ephemeral check: "brother" not in PROFILE_EPHEMERAL_RELATIONS → continue
2. Candidate query: search facts for subject="user", predicate="brother"
3. No existing "user | brother | Auggie" → verdict: STORE
4. If existing "user | brother | Steve" found:
   a. User trust override? (source=user_stated, confidence >= threshold) → STORE_AND_FLAG
   b. LLM adjudication: "Is 'Auggie' an update to 'Steve' or a second brother?"
      - UPDATE → STORE_AND_FLAG (mark Steve entry superseded)
      - COMPLEMENTARY → STORE (both coexist)
      - IMPLAUSIBLE → REJECT
```

### Graph Ingestion

After fact storage, `_ingest_fact_to_graph()`:

```
Fact: "user | brother | Auggie"
  → _is_graph_worthy_object("Auggie") → True (single proper noun)
  → entity_resolver.resolve_or_create("user") → "user"
  → entity_resolver.resolve_or_create("Auggie") → "auggie"
  → graph_memory.add_relation(user → brother → auggie)

Fact: "Biscuit | is_a | golden retriever"
  → _is_graph_worthy_object("golden retriever") → False (2+ words, descriptive)
  → Store as node metadata: auggie.metadata["has_pet"] = "Biscuit"
  → Store as node metadata: biscuit.metadata["is_a"] = "golden retriever"
```

### Truth Score Lifecycle

```
Initial:     0.70  (source: llm_extracted)
             0.85  (source: user_stated)
             0.90  (source: corrected)

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

---

## Multi-Stage Gating

After retrieval, semantic memories pass through `MultiStageGateSystem`:

```
Input: 30 candidate memories + query

Stage 1: Separation
  Episodic (type=="episodic") → always included, bypass gating
  Others → continue to Stage 2

Stage 2: Blended Scoring
  Encode query + all memory texts (batch, cached)
  For each memory:
    blended = 0.85 * cosine_sim + 0.15 * truth_score + entity_boost
    entity_boost: +0.18 (1 entity match) or +0.25 (multiple)
  Deictic queries: threshold lowered to min 0.20

Stage 3: Forced Minimum
  If < 8 passed, force-add highest-scoring non-passed items

Stage 4: Cross-Encoder Reranking (if available, > 5 items)
  Rerank by cross-encoder score

Stage 5: Cap
  Final = episodic + gated[:20 - len(episodic)]
```

**Timing:** ~200ms total (FAISS ~50ms, cosine ~50ms, cross-encoder ~100ms)

---

## Token Budget Management

The prompt has a finite token budget (default 25% of model context window,
floor 8K, ceiling 60K). Sections are prioritized:

```
Priority 10: STM summary (metadata, never trimmed)
Priority  9: User profile (identity, naturally bounded)
Priority  8: Narrative state (temporal grounding, hard cap 500 tokens)
Priority  7: Recent conversations, graph context, unresolved threads
Priority  6: Semantic chunks, personal notes, user uploads
Priority  5: Reference docs, memories, web search results
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
[DAEMON DOCUMENTATION]                 ← reference docs
[PROJECT COMMIT HISTORY]               ← git commits
[ADAPTIVE WORKFLOWS]                   ← procedural skills
[PROPOSED FEATURES]                    ← code proposals
[KNOWLEDGE GRAPH]                      ← entity relationships (natural language)
[UNRESOLVED THREADS]                   ← open commitments/deadlines
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
| Memories from wrong topic | Topic filtering disabled | Set `topic_match` weight > 0 |
| Too many low-quality results | Gate threshold too low | Raise `GATE_REL_THRESHOLD` (default 0.18) |
| Large docs drowning out small facts | Size penalty too weak | Lower size penalty threshold (default 10KB) |

### Retrieval Missing Relevant Memories

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Facts stated recently not found | Fact extraction disabled | Enable `FACTS_EXTRACT_EACH_TURN` |
| Semantic search missing vocabulary | Graph expansion disabled | Enable `GRAPH_QUERY_EXPANSION_ENABLED` |
| Gating too aggressive | Threshold too high | Lower `GATE_REL_THRESHOLD` (default 0.18) or increase `MIN_GATED_MEMORIES` |
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
| Old facts never decay | Decay rate too low | Increase `TRUTH_SCORER_DECAY_RATE` (default 0.05/week) |
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
| `RECENCY_DECAY_RATE` | varies | Exponential decay speed |
| `DEICTIC_THRESHOLD` | 0.60 | Acceptance threshold for follow-up queries |
| `NORMAL_THRESHOLD` | 0.35 | Acceptance threshold for normal queries |
| `COSINE_SIMILARITY_THRESHOLD` | 0.25 | Minimum cosine gate |

### Gating
| Constant | Default | Purpose |
|----------|---------|---------|
| `GATE_REL_THRESHOLD` | 0.18 | Multi-stage gate threshold (app_config.py) |
| `GATE_COSINE_WEIGHT` | 0.85 | Weight of cosine vs truth in blended gate score (env var in gate_system.py, NOT app_config) |
| `MIN_GATED_MEMORIES` | 8 | Forced minimum even if below threshold (env var in gate_system.py, NOT app_config) |

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
| `PROMPT_TOKEN_BUDGET_DEFAULT` | 40000 | Base token budget (model-aware; LOCAL=12000, FLOOR=8000, CEILING=60000) |
| `PROMPT_TOKEN_BUDGET_CONTEXT_FRACTION` | 0.25 | Fraction of model context window |
| `PROMPT_MAX_RECENT` | 15 | Max recent conversations |
| `PROMPT_MAX_MEMS` | 15 | Max semantic memories |
| `PROMPT_MAX_FACTS` | 30 | Max facts |

### Graph
| Constant | Default | Purpose |
|----------|---------|---------|
| `KNOWLEDGE_GRAPH_ENABLED` | True | Master toggle |
| `KNOWLEDGE_GRAPH_MAX_DEPTH` | 2 | BFS traversal depth |
| `GRAPH_SCORING_BOOST_CAP` | 0.15 | Max graph bonus per memory |
| `GRAPH_QUERY_EXPANSION_MAX_TERMS` | 8 | Max neighbor names appended to query |
