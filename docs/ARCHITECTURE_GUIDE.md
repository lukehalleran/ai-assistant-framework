# Daemon Architecture Guide

**Purpose**: Narrative architectural walkthrough of the entire Daemon system.
Covers every major subsystem with enough detail to understand behavior, data
flow, and key algorithms — without the per-method exhaustiveness of
`PROJECT_SKELETON.md` or the signature-level compression of `QUICK_REFERENCE.md`.

Read this document to understand **how the system works as a whole**. For
deep dives into specific subsystems, see the cross-references to companion
docs throughout.

**Last Updated**: 2026-05-09

**Related docs**:
- `README.md` — external audience, feature highlights, getting started
- `GOALS.md` — mission, roadmap, active goals, design principles
- `FORMAL_MODEL.md` — mathematical specification of the complete agent
- `MEMORY_SYSTEM.md` — deep dive on memory lifecycle, scoring, fact pipeline
- `SYNTHESIS_FILTER.md` — deep dive on synthesis pipeline stages, calibration
- `grading_plan.md` — two-layer grading protocol, audit queue design, classifier bootstrap
- `QUICK_REFERENCE.md` — API signatures, function-level compressed reference
- `PROJECT_SKELETON.md` — per-file method signatures and implementation details

---

## Table of Contents

1. [System at a Glance](#1-system-at-a-glance)
2. [Request Lifecycle](#2-request-lifecycle)
3. [Context Pipeline](#3-context-pipeline)
4. [Intent Classification](#4-intent-classification)
5. [Memory Architecture](#5-memory-architecture)
6. [Knowledge Graph](#6-knowledge-graph)
7. [Retrieval Pipeline](#7-retrieval-pipeline)
8. [Multi-Stage Gating](#8-multi-stage-gating)
9. [Memory Scoring](#9-memory-scoring)
10. [Prompt Assembly](#10-prompt-assembly)
11. [Response Generation](#11-response-generation)
12. [Agentic Tool System](#12-agentic-tool-system)
13. [Per-Turn State Updates](#13-per-turn-state-updates)
14. [Fact Pipeline](#14-fact-pipeline)
15. [Truth Scoring & Correction Detection](#15-truth-scoring--correction-detection)
16. [Memory Staleness Cascade](#16-memory-staleness-cascade)
17. [Escalation FSM](#17-escalation-fsm)
18. [Thread Surfacing](#18-thread-surfacing)
19. [Proactive Cross-Domain Insights](#19-proactive-cross-domain-insights)
20. [Cross-Collection Deduplication](#20-cross-collection-deduplication)
21. [Synthesis Pipeline](#21-synthesis-pipeline)
22. [Session Lifecycle](#22-session-lifecycle)
23. [Knowledge Sources](#23-knowledge-sources)
24. [Personality & Tone System](#24-personality--tone-system)
25. [Provenance & Audit Trail](#25-provenance--audit-trail)
26. [Configuration Architecture](#26-configuration-architecture)
27. [Production Deployment](#27-production-deployment)
28. [Testing & Benchmarks](#28-testing--benchmarks)
29. [Prompt Eval System](#29-prompt-eval-system)

---

## 1. System at a Glance

Daemon is a personal cognitive assistant with persistent memory, agentic tool
use, and knowledge synthesis infrastructure. It is a single-user, locally-run
system — all data stays on disk, API calls go to LLM providers only.

### Key Numbers

```
Python lines:           ~143,000 (incl. tests)
Python files:           392
Test files:             173
Test functions:         3,559
ChromaDB collections:   13
Prompt sections:        27 (conditional)
Intent types:           9
Parallel retrieval:     19 async tasks
Memory tiers:           5
Agentic tools:          12
Gating latency:         ~200ms
Config options:         180+
```

### Entry Points

| Entry | Command | What Happens |
|-------|---------|-------------|
| GUI | `python main.py` | Gradio web UI at `http://localhost:7860` |
| CLI | `python main.py cli` | Terminal-based chat loop |
| Wizard | `python main.py wizard` | First-run onboarding (mode, keys, vault, wiki index) |
| Docker | `docker-compose up -d` | Containerized deployment |
| Desktop | `dist\Daemon\Daemon.exe` | Frozen executable (built via `pyinstaller daemon.spec`) |
| Installer | `installer\output\DaemonSetup-1.0.0.exe` | Windows installer (built via Inno Setup) |

### User vs Developer Mode (DAEMON_MODE)

The first-run wizard asks users to choose Personal or Developer mode. This gates features at the config level (`config/app_config.py`):

| Feature | User Mode | Dev Mode |
|---------|-----------|----------|
| Chat, memory, web search, computation | Yes | Yes |
| Obsidian daily/weekly/monthly notes | Yes (if vault configured) | Yes |
| Synthesis pipeline | Disabled | Enabled |
| Code proposals + prompt surfacing | Disabled | Enabled |
| Reference docs (architecture) in context | Disabled | Enabled |
| Auto-dedup on shutdown | Executed | Dry-run preview |
| GUI: Debug Trace, Logs tabs | Hidden | Visible |
| GUI: Proposals, Synthesis tabs | Hidden | Visible |
| GUI: Memory Maintenance section | Hidden | Visible |
| System prompt in Provenance | Hidden | Visible |

### Module Layout

```
core/                    # Request orchestration, context pipeline, agentic loop
├── orchestrator.py      # Main controller — wires everything together
├── context_pipeline.py  # 8-stage pre-retrieval query analysis
├── intent_classifier.py # Regex-first intent detection (9 types)
├── best_of_handler.py   # Best-of-N, duel, ensemble generation
├── escalation_tracker.py# Crisis cooldown FSM (4 states)
├── response_planner.py  # Pre-answer planning + post-answer review gate
├── agentic/             # ReAct tool loop
│   ├── controller.py    # Loop orchestration, prompt building, quality heuristics
│   ├── tools.py         # ToolExecutor: dispatch routing + 10 execute methods
│   ├── formatters.py    # AgenticFormatter: 17 pure formatting methods
│   ├── types.py         # Tool definitions, state types
│   └── protocols.py     # Native + XML tool calling
└── prompt/              # Prompt assembly pipeline
    ├── builder.py       # Thin orchestrator: parallel task dispatch, intent overrides, budget
    ├── context_gatherer.py  # Mixin compositor: init + properties (composes 3 gatherer mixins)
    ├── gatherer_web.py      # WebSearchMixin: web search retrieval + trigger logic
    ├── gatherer_memory.py   # MemoryRetrievalMixin: 17 memory/summary/reflection/facts methods
    ├── gatherer_knowledge.py # KnowledgeRetrievalMixin: 16 knowledge retrieval methods
    ├── formatter.py     # 26-section assembly + attention ordering + feature inventory
    ├── hygiene.py       # ContentHygiene: dedup, caps, backfill
    └── token_manager.py # Priority-based budget management

memory/                  # 5-tier memory system
├── memory_coordinator.py    # Thin orchestrator (~551 lines)
├── memory_retriever.py      # Parallel ChromaDB retrieval
├── memory_scorer.py         # 12-step composite scoring
├── memory_storage.py        # Persistence + fact extraction + graph ingestion
├── shutdown_processor.py    # 12-step session-end pipeline
├── graph_memory.py          # NetworkX knowledge graph
├── entity_resolver.py       # Alias resolution + relation normalization
├── fact_extractor.py        # Dual-budget fact extraction
├── fact_verification.py     # Pre-storage conflict detection
├── truth_scorer.py          # Evidence-based truth tracking
├── claim_tracker.py         # Staleness cascade
├── cross_deduplicator.py    # Duplicate + contradiction detection
├── context_surfacer.py      # Proactive insight generation
├── thread_store.py          # Open thread persistence
├── thread_extractor.py      # LLM thread extraction/resolution
├── memory_expander.py       # Temporal expansion for agentic tool
├── synthesis_memory.py      # Synthesis results + convergence
└── storage/
    └── multi_collection_chroma_store.py  # ChromaDB wrapper (13 collections)

knowledge/               # External knowledge integration
├── web_search_manager.py      # Tavily API + caching + numbered web citations
├── wolfram_manager.py         # Wolfram Alpha + rate limiting
├── sandbox_manager.py         # E2B code sandbox
├── synthesis_generator.py     # Cross-store synthesis candidates (Tier 2)
├── synthesis_retriever.py     # Structural query + FAISS synthesis (Tier 0)
├── synthesis_filter.py        # 7-stage synthesis filter
├── graph_walk_generator.py    # Biased Markov walk synthesis (Tier 1)
├── synthesis_models.py        # Synthesis pipeline data models + enums
├── implementation_detector.py # Proposal implementation tracking
├── reference_docs_manager.py  # Auto-seeded docs/
├── wiki_tracker.py            # Session-level Wikipedia article tracking
├── wiki_enrichment.py         # Shutdown: tracked wiki articles → graph nodes
├── wikidata_resolver.py       # Personal ↔ Wikidata entity resolution
├── wikidata_models.py         # Pydantic models for Wikidata import
├── git_memory.py              # Git commit history extractor
├── git_memory_loader.py       # Git → PROCEDURAL ChromaDB loader
└── proposal_generator.py      # Goal-directed code proposal generation

eval/                    # Prompt section ablation & eval system
├── schema.py              # Pure data models (no Daemon imports)
├── section_registry.py    # 28-entry canonical section registry
├── snapshots.py           # Snapshot capture, replay, save/load
├── variants.py            # LOO, AOI, bundle, reorder variant generation
├── corpus.py              # 27-query seed corpus (3 per intent type)
├── utilization.py         # Per-section presence/token utilization analysis
├── harness.py             # Batch generation with rate limiting + resume
├── judge.py               # Pairwise A/B judging with position randomization
├── checks.py              # 5 automated objective checks (no LLM)
├── no_store_generation.py # Side-effect-free LLM generation
└── persistence_guard.py   # State fingerprinting (prevents mutations)

processing/gate_system.py  # Multi-stage retrieval gating
models/model_manager.py    # Multi-provider LLM abstraction
gui/handlers.py            # Request routing + streaming
```

---

## 2. Request Lifecycle

This section traces a single user query through the entire system. Each
subsection below expands on the components introduced here.

### End-to-End Flow

```
User types: "How's my squat progress looking?"
    │
    ├─ 1. GUI handler receives input
    │     handlers.py routes based on mode (raw, duel, agentic, enhanced)
    │
    ├─ 2. Context Pipeline (pre-retrieval analysis)
    │     8 stages: topics → tone → files → heavy check → intent →
    │     query rewrite → STM analysis → identity/thread injection
    │     Result: ContextResult with intent=FACTUAL_RECALL, tone=CONVERSATIONAL
    │
    ├─ 3. Query Expansion (knowledge graph)
    │     "squat" resolves to graph entity → neighbors: powerlifting, deadlift
    │     Expanded query: "How's my squat progress looking? powerlifting deadlift"
    │
    ├─ 4. Parallel Retrieval (19 async tasks, 30s timeout)
    │     Recent conversations, semantic memories, facts, summaries,
    │     reflections, wiki, personal notes, graph context, threads,
    │     proactive insights, procedural skills, web search (if triggered)...
    │
    ├─ 5. Multi-Stage Gating
    │     Batch cosine similarity → cross-encoder reranking (~200ms)
    │     Intent overrides gating threshold
    │
    ├─ 6. Composite Scoring
    │     12-step algorithm: relevance + recency + truth + importance +
    │     continuity + structure + graph bonus − staleness penalty
    │     Intent FACTUAL_RECALL boosts truth weight to 0.30
    │
    ├─ 7. Prompt Assembly (27 conditional sections)
    │     Token-budgeted: priority-based trimming, middle-out compression,
    │     LLM compression for oversized items
    │     High-signal sections placed at end for transformer attention
    │     ResponsePlanner runs in parallel with retrieval (Step 4), plan injected into system prompt
    │
    ├─ 8. Agentic Gate Check
    │     3-tier: keyword heuristic → entity match → LLM fallback
    │     If triggered → ReAct loop (think → tool → observe, max 5 rounds)
    │     If not → standard generation
    │
    ├─ 9. Response Generation
    │     Standard streaming | Best-of-N | Duel (A vs B + judge) | Ensemble
    │     Thinking block extracted and stored separately
    │
    ├─ 9b. Uncertainty Fallback (if enabled)
    │      UncertaintyDetector checks for "I don't know" responses (keyword regex + semantic)
    │      If uncertain → automatic retry via agentic ReAct loop with memory search hint
    │
    ├─ 9c. Post-Answer Review Gate (if enabled)
    │      ResponsePlanner.review_answer() checks response against pre-answer plan
    │      If review fails with confidence >= 0.90 → silent retry via agentic search (similarity-guarded)
    │
    ├─ 10. Post-Response State Update
    │      Store to corpus + ChromaDB with provenance metadata
    │      Run correction/confirmation detection → truth score adjustments
    │      Optional per-turn fact extraction → verification → graph ingestion
    │      Escalation FSM transition based on tone
    │
    └─ Response streamed to user
```

### Mode Routing in handlers.py

The GUI handler decides the response generation mode before calling the
orchestrator:

| Mode | Condition | Behavior |
|------|-----------|----------|
| **Raw** | User toggled raw mode | Skip memory, go straight to LLM |
| **Duel** | `BEST_OF_DUEL_MODE` + 2 generators + 1 judge | Two models generate in parallel, judge picks winner |
| **Agentic** | 3-tier gate triggers | ReAct loop with web search, code sandbox, memory search |
| **Enhanced** | Default | Full retrieval pipeline + streaming generation |

The agentic gate runs before enhanced mode. If any of the three tiers
trigger (keyword match, entity match in knowledge graph, or LLM decision),
the request routes through the agentic controller instead.

---

## 3. Context Pipeline

**File**: `core/context_pipeline.py`

The context pipeline transforms raw user input into a structured
`ContextResult` before any memory retrieval happens. It runs 8 stages in
sequence (some stages depend on earlier outputs):

### Stage Sequence

| Stage | What It Does | Deterministic? |
|-------|-------------|----------------|
| 1. Topic Extraction | spaCy NER + optional LLM fallback → topics list | Mostly (LLM rare) |
| 2. Tone Detection | 250+ weighted keywords, semantic analysis, LLM fallback → crisis level | No (LLM) |
| 3. File Processing | PDF/DOCX/CSV text extraction via pdfplumber | Yes |
| 4. Heavy Topic Check | Sensitive content detection + inline fact extraction | Partially |
| 4.5. Intent Classification | Regex-first, 9 types, no LLM → IntentResult | Yes |
| 5. Query Rewriting | LLM rewrites casual queries for better retrieval | No (LLM) |
| 6. STM Analysis | LLM analyzes recent conversation context → structured JSON | No (LLM) |
| 6b. STM Intent Refinement | Low-confidence intents upgraded via STM free-text | Yes |
| 7. Identity Injection | User name + pronouns from UserProfile | Yes |
| 8. Thread Context | Active conversation thread metadata | Yes |

### ContextResult

The pipeline produces a `ContextResult` dataclass carrying all processed
information downstream:

```
ContextResult:
  processed_query    — rewritten query (or original if rewrite skipped)
  original_query     — always preserved unmodified
  tone_level         — CRISIS / ELEVATED / CONCERN / CONVERSATIONAL
  tone_instructions  — mode-specific response guidelines
  topics             — extracted topic list
  primary_topic      — main topic string
  file_context       — processed file content (if files uploaded)
  thread_context     — active thread metadata (id, depth, topic)
  stm_summary        — STM analysis result (topic, intent, tone, reference_type, temporal_facts, open_threads)
  identity_block     — user identity context string
  is_heavy_topic     — heavy/sensitive topic flag
  extracted_facts    — inline facts (if heavy topic triggered extraction)
  intent             — IntentResult with type, confidence, weight/retrieval/gate overrides
  is_small_talk      — True when CASUAL_SOCIAL intent >= 0.70 (skips planning, reduces retrieval)
```

### Tone Detection

The `ToneDetector` (`utils/tone_detector.py`) uses a 250+ keyword weighted
dictionary to detect crisis levels. Keywords are weighted by severity — "I
can't go on" scores higher than "I'm stressed." The detector distinguishes
personal crisis ("I want to die") from world observation ("People are
suffering") by checking for first-person pronouns and personal context.

Four crisis levels drive system behavior:

| Level | System Prompt Mode | Response Length | Escalation |
|-------|-------------------|----------------|------------|
| CRISIS (HIGH) | Full therapeutic | Multi-paragraph | Escalation FSM engaged |
| ELEVATED (MEDIUM) | Empathetic validation | 2-3 paragraphs | Escalation FSM engaged |
| CONCERN | Brief acknowledgment | 2-4 sentences | Monitored |
| CONVERSATIONAL | Direct, concise | Max 3 sentences | Normal |

### STM Analysis

Short-term memory analysis runs an LLM pass over a 24-hour time-windowed
slice of recent conversation (capped at `STM_MAX_RECENT_MESSAGES=30`,
truncated to 200 chars per message). It internally injects the last 2
daily notes from the Obsidian vault (`STM_INJECT_DAILY_NOTES_DAYS=2`)
read directly from disk via `utils.daily_notes_generator.read_daily_note()`
— this gives STM cross-day recall disambiguation without waiting for the
narrative-context refresh chain. Read-only: never triggers note generation.

The structured output includes: topic, user_question, intent, tone,
**reference_type**, **temporal_facts**, open_threads, constraints.
`reference_type` ∈ {new_event, recall, clarification, correction, unclear}
classifies whether the current message is a fresh report or a restatement
of prior context — defaults to `unclear` when uncertain. `temporal_facts`
is a list of normalized current-state facts produced under a
collapse-toward-fewer-events disambiguation rule (e.g. "did not sleep" +
"fight mode all night" same morning = ONE night, not two).

This summary is injected at the very end of the prompt (maximum
transformer attention window) as a `[SHORT-TERM CONTEXT SUMMARY]` section.
When `reference_type` is anything other than `new_event`, the renderer
appends an explicit WARNING directive instructing the main response model
not to count the current message as a separate occurrence or fabricate
patterns from a single underlying event. This addresses a class of
failures where the main model would stitch a recall + a retrieved memory
into a phantom "twice in close proximity, that's a pattern" framing.

STM has a secondary role: when the intent classifier produces a
low-confidence result (< 0.50), STM's free-text intent field can upgrade
the classification via keyword matching.

**Graceful degradation**: when a session starts before the daily-note
catch-up thread has finished writing yesterday's note (≤20s window),
STM falls back to the recent-conversation-only mode. Mock-based test
fixtures fall through cleanly via class-level `hasattr` check on the
new `CorpusManager.get_recent_within_hours` method.

---

## 4. Intent Classification

**File**: `core/intent_classifier.py`

Intent classification is one of the most impactful subsystems — it
parameterizes retrieval counts, scoring weights, and gating thresholds
without changing any of the underlying algorithms. The same retrieval
pipeline serves all query types; only the parameters change.

### The 9 Intent Types

| Intent | Example Patterns | Key Behavior |
|--------|-----------------|--------------|
| FACTUAL_RECALL | "what's my X", "do you remember" | Boost truth weight (0.30), increase fact retrieval |
| TEMPORAL_RECALL | "last week", "history of", "progression" | Boost recency (0.40), reshape decay curve around temporal anchor |
| EMOTIONAL_SUPPORT | Crisis keywords, "I feel" | Boost continuity (0.20), increase recent conversation retrieval |
| CASUAL_SOCIAL | "hi", "ok", "thanks", short queries | Reduce all retrieval limits, lightweight response |
| TECHNICAL_HELP | "fix", "bug", "how do I", code references | Boost relevance (0.45), increase procedural skill retrieval |
| CREATIVE_EXPLORATION | "brainstorm", "imagine", "what if" | Balanced weights, wider retrieval |
| META_CONVERSATIONAL | "tell me about yourself", "your capabilities" | Boost meta bonus for episodic memories |
| PROJECT_WORK | "let's build", "add feature", file refs | Standard weights, include git commits and proposals |
| GENERAL | Fallback | Default weights, no overrides |

### How Intent Drives the System

Each intent maps to a profile with three override sets:

**Weight overrides** reshape the scoring function. FACTUAL_RECALL boosts
truth to 0.30 (from default 0.20) so confirmed facts rank higher.
TEMPORAL_RECALL boosts recency to 0.40 and injects a `_temporal_anchor_hours`
parameter that reshapes the recency decay curve — memories within the
time window get gentle decay (1.0 → 0.7), memories outside get standard
exponential decay.

**Retrieval overrides** adjust how many items each parallel retrieval task
fetches. CASUAL_SOCIAL reduces max memories to 3, reflections/reference_docs/
narrative/user_uploads/proactive to 0, and summaries to 1. EMOTIONAL_SUPPORT
increases max recent to 20 and keeps reflections at 8. These overrides flow
through `builder.py` into every parallel task launch — setting any count to
0 skips the async retrieval task entirely. Profile values were calibrated
against eval data (50 snapshots, 1799 pairs, two judge models) in Phase 8.
Gated by `PROMPT_SECTION_GATING_ENABLED` config flag (default True).

**Gate threshold overrides** adjust the cosine similarity threshold in
multi-stage gating. Some intents lower the threshold to let more
borderline-relevant memories through.

### Classification Method

Classification is regex-first — no LLM calls. Compiled regex patterns are
checked in priority order, and the highest-confidence match wins. This
keeps classification fast (~0ms) and deterministic.

Tone bias: when the tone detector reports HIGH or MEDIUM crisis level,
ambiguous queries are biased toward EMOTIONAL_SUPPORT.

Low-confidence results (< 0.50) can be refined by STM analysis at
stage 6b of the context pipeline.

---

## 5. Memory Architecture

**Files**: `memory/` directory (12+ files)
**Deep dive**: `MEMORY_SYSTEM.md`

### 5 Memory Tiers

Daemon's memory is modeled on cognitive architecture with five tiers:

| Tier | What It Stores | Retrieval Bias | ChromaDB Collections |
|------|---------------|----------------|---------------------|
| Episodic | Raw conversation turns | Recency | `conversations` |
| Semantic | Extracted fact triples + Wikipedia | Truth + relevance | `facts`, `wiki_knowledge` |
| Procedural | Git commits + learned skill patterns | Pattern matching | `procedural`, `procedural_skills` |
| Summary | LLM-compressed conversation blocks | Relevance | `summaries` |
| Meta | Reflections, threads, proposals, insights | Priority / urgency | `reflections`, `threads`, `proposals`, `synthesis_results` |

Two additional collections serve as reference stores: `obsidian_notes`
(user's personal notes, synced from Obsidian vault) and `reference_docs`
(uploaded documents + auto-seeded docs/ directory).

### 13 ChromaDB Collections

All persistent memory is stored in ChromaDB via
`multi_collection_chroma_store.py`, which wraps the ChromaDB client and
provides a unified interface:

- `add_to_collection(name, text, metadata)` → returns doc_id
- `query_collection(name, query_text, n_results)` → flat list of result dicts
- `get_by_id(collection_name, doc_id)` → single document or None
- `update_metadata(collection_name, doc_id, metadata_updates)` → merge updates

Collections are registered as `None` placeholders in `__init__()` and
auto-initialized on first use by `_initialize_collections()`.

### Protected vs. Deduplicable Collections

| Category | Collections | Protected | Deduped |
|----------|-------------|-----------|---------|
| Core | `conversations` | Yes | Never |
| Semantic | `facts` | No | Yes |
| Knowledge | `wiki_knowledge` | Yes | Never |
| User content | `obsidian_notes`, `reference_docs` | Yes | Never |
| Derived | `summaries`, `reflections`, `procedural`, `procedural_skills`, `proposals`, `threads`, `synthesis_results` | No | Yes |
| Visual | `visual_memories` | No | No |

Protected collections are never scanned by the cross-collection
deduplicator. The `conversations` collection is additionally exempt
from all modification — raw turns are the ground truth.

### MemoryCoordinator — The Thin Orchestrator

`memory_coordinator.py` (~551 lines) is a pure delegation layer. It
creates ~16 components in `__init__()` and exposes ~24 methods that
forward to the appropriate component:

```
MemoryCoordinator
  ├── MemoryRetriever    — retrieval + semantic top memories
  ├── MemoryStorage      — persistence + fact extraction + graph ingestion
  ├── MemoryScorer       — composite scoring with intent overrides
  ├── ShutdownProcessor  — 12-step session-end pipeline
  ├── ThreadManager      — conversation thread tracking
  ├── HybridRetriever    — query rewrite + keyword search
  ├── UserProfile        — categorized user fact profile
  ├── GraphMemory        — knowledge graph (NetworkX DiGraph)
  ├── EntityResolver     — alias resolution + relation normalization
  ├── FactVerifier       — pre-storage conflict detection
  ├── ClaimIndex         — claim staleness tracking
  ├── ContextSurfacer    — proactive cross-domain insights
  ├── ThreadStore        — open thread persistence (ChromaDB-backed)
  ├── FactExtractor      — rule-based fact extraction
  ├── MemoryConsolidator — summary block generation
  └── TopicManager       — topic extraction
```

No business logic lives in the coordinator itself. All scoring, storage,
retrieval, and shutdown logic is in the delegated components.

---

## 6. Knowledge Graph

**Files**: `memory/graph_memory.py`, `memory/graph_models.py`,
`memory/entity_resolver.py`, `memory/graph_utils.py`

The knowledge graph is a NetworkX DiGraph that provides a secondary index
over the fact store. It enables three capabilities that ChromaDB alone
cannot: multi-hop traversal, query expansion via entity neighbors, and
graph-boosted scoring.

### Structure

- **Nodes**: Entities (people, places, concepts). Each has a display name
  and optional metadata (key-value facts about the entity).
- **Edges**: Relations between entities (e.g., user → brother → auggie).
  Edges carry relation labels and weights. Duplicate edges strengthen
  weight rather than creating parallel edges (DiGraph, not MultiDiGraph).
- **Persistence**: JSON at `data/knowledge_graph.json` (nodes + edges) and
  `data/entity_aliases.json` (alias table). Dirty-flag optimization —
  only writes when changes exist.

### Entity Resolution

The `EntityResolver` maps surface forms to canonical entity IDs:

- Exact match lookup against alias table
- Possessive pattern extraction ("my brother Auggie" → alias "brother" → auggie)
- Trigram → bigram → unigram matching against known aliases
- `normalize_relation()` standardizes predicates (e.g., "is the brother of" → "brother")

### Ingestion

Facts are ingested into the graph after ChromaDB storage, in
`memory_storage.py:_ingest_fact_to_graph()`:

1. Extract subject, relation, object from the fact triple
2. Check if object is "graph-worthy" via `_is_graph_worthy_object()`:
   - Reject: temporal ("2 years"), measurements ("20lbs"), verb phrases
     ("stopped being religious"), generic terms, 4+ word phrases
   - These are stored as subject-node metadata instead of creating nodes
3. Resolve subject and object through EntityResolver → canonical IDs
4. Add or strengthen edge: `graph_memory.add_relation(subj → rel → obj)`

### Query Expansion

Before retrieval, queries are expanded using graph neighbors. This bridges
vocabulary gaps — if the user asks about "my brother," expansion appends
related entities (Auggie, Mom, Flapjack) to the search query.

The expansion algorithm in `gatherer_memory.py:_expand_query_with_graph()`:

1. Extract entity IDs from query via alias resolution
2. BFS to depth 2 from matched entities (traverses through hubs like "user")
3. Rank candidates by lateral connectivity:
   `score = min(non_hub_edges * 0.3, 1.0)` + single-word bonus (+0.1) / 3+ word penalty (-0.1)
4. Filter junk: ≤2 chars, 4+ words, digit-starting, temporal, measurements,
   verb phrases
5. Append top K display names to original query (default K=8)

Example: `"what about my brother"` → `"what about my brother Auggie Mom Flapjack"`

### Graph-Boosted Scoring

During memory scoring, each memory gets a `graph_bonus`:

- For each graph-connected entity mentioned in the memory's content: +0.05
- Capped at `GRAPH_SCORING_BOOST_CAP` (default 0.15)
- This means a memory mentioning 3+ related entities gets the maximum boost

### Graph Context in Prompt

The `[KNOWLEDGE GRAPH]` prompt section contains natural-language sentences
generated from BFS traversal of relevant entities. For example:
"Auggie is user's brother. Auggie has a pet named Biscuit. Biscuit is a
golden retriever."

### Bridge Quality Cleanup

`GraphMemory.prune_garbage_bridges()` removes 5 categories of noise edges
between personal and wikidata nodes:

1. **mentioned_alongside** — low-signal edges from wiki enrichment
2. **Low-confidence embedding** — bridge edges with embedding similarity < 0.80
3. **Misrouted personal relations** — personal-domain predicates (e.g. "brother") incorrectly linking to wikidata nodes
4. **Short-name/acronym collisions** — entity aliases that matched wikidata entries by coincidence (e.g. "AI" matching both a personal project and a wikidata concept)
5. **Unstructured predicates** — freeform text edges that don't follow normalized relation patterns

The `WikidataEntityMapper` embedding threshold was also raised from 0.60 to
0.80, with an exact-match blocklist added to prevent known false positives.

---

## 7. Retrieval Pipeline

**Files**: `core/prompt/builder.py` (task dispatch), `core/prompt/context_gatherer.py` (mixin compositor),
`core/prompt/gatherer_memory.py` (memory retrieval), `core/prompt/gatherer_knowledge.py` (knowledge retrieval),
`core/prompt/gatherer_web.py` (web search), `memory/memory_retriever.py`
**Deep dive**: `MEMORY_SYSTEM.md`

### Parallel Retrieval Architecture

When a prompt is being built, `builder.py` launches 19 async retrieval
tasks via `asyncio.gather()` with a 30-second timeout. Each task is
implemented across three gatherer mixins (composed by `context_gatherer.py`)
and fetches from a different source or collection:

| Task | Source | Count | Notes |
|------|--------|-------|-------|
| Recent conversations | Corpus (recency) | 15 | Recency-ordered, no gating |
| Semantic memories | ChromaDB multi-collection | 15 | Cosine + cross-encoder gated |
| Recent summaries | ChromaDB `summaries` | 5 | Time-ordered |
| Semantic summaries | ChromaDB `summaries` | 5 | Relevance-ordered |
| Recent reflections | ChromaDB `reflections` | 3 | Time-ordered |
| Semantic reflections | ChromaDB `reflections` | 3 | Relevance-ordered |
| Graph context | Knowledge graph BFS | 12 sentences | Natural language from traversal |
| Procedural skills | ChromaDB `procedural_skills` | 5 | Adaptive workflows |
| Unresolved threads | ChromaDB `threads` | 3 | Priority-ranked |
| Proactive insights | ContextSurfacer | 2 | LLM once/session, cached |
| Wiki content | FAISS (40M vectors, IVFPQ index) | 3 | Gated at 0.30 threshold; falls back to ChromaDB if FAISS unavailable |
| Reference docs | ChromaDB `reference_docs` | 5 | Auto-seeded from docs/ |
| Personal notes | ChromaDB `obsidian_notes` | 5 | Gated at 0.30 threshold |
| Git commits | ChromaDB `procedural` | 10 | Project history |
| Web search | Tavily API | if triggered | Cached 72 hours |
| Codebase changes | git diff | first msg only | Session-start awareness |
| User profile | UserProfile | 3000 tokens | Categorized facts |

All retrieval counts are overridable per-intent. CASUAL_SOCIAL reduces
most limits (including `max_visual_memories=0` to skip image retrieval
for greetings); EMOTIONAL_SUPPORT and META_CONVERSATIONAL also disable
visual memories. Image passing is further gated by `supports_vision()` —
non-vision models (DeepSeek, GLM) never receive image data.

### Small-Talk Short Circuit

Before launching expensive retrieval, the builder checks for small talk
(greetings, single words, very short queries). If detected, retrieval is
short-circuited to a lightweight context with only recent conversations
and basic profile — saving ~200ms of gating and embedding work.

### How Builder Coordinates Retrieval

The builder sets up transient state on the scorer before retrieval and
cleans it up after:

```
1. Set scorer._intent_weight_overrides from intent profile
2. Set scorer._graph_memory and scorer._entity_resolver for graph boost
3. Launch parallel retrieval tasks
4. Await all results
5. Clear scorer state
```

This pattern avoids passing intent overrides through every layer of the
retrieval stack — the scorer reads them from its own instance attributes
during the gather window.

---

## 8. Multi-Stage Gating

**File**: `processing/gate_system.py`

After retrieval, semantic memories pass through a multi-stage filter that
eliminates irrelevant results before scoring. The goal is to prevent
low-relevance memories from consuming prompt budget.

### Gating Stages

```
Input: ~30 candidate memories + query

Stage 1: Separation
  Episodic memories (type=="episodic") → bypass gating, always included
  All others → continue to Stage 2

Stage 2: Blended Scoring
  Encode query + all memory texts (batch, cached embeddings)
  For each memory:
    blended = 0.85 * cosine_sim + 0.15 * truth_score + entity_boost
    entity_boost: +0.18 (1 entity match) or +0.25 (multiple)
  Deictic queries ("explain that", "what about it"):
    threshold lowered to min 0.20

Stage 3: Forced Minimum
  If fewer than 8 memories passed, force-add highest-scoring rejects
  This prevents empty context on edge cases

Stage 4: Cross-Encoder Reranking (optional, if > 5 items)
  Rerank surviving candidates by cross-encoder score
  Higher quality but slower (~100ms)

Stage 5: Cap
  Final = episodic + gated[:20 - len(episodic)]
```

Total gating latency: ~200ms (embedding ~50ms, cosine ~50ms,
cross-encoder ~100ms).

The gating threshold is overridable per-intent via `gate_threshold_override`
in the intent profile. Some intents (e.g., emotional support) lower the
threshold to let more borderline-relevant memories through.

---

## 9. Memory Scoring

**File**: `memory/memory_scorer.py`
**Deep dive**: `MEMORY_SYSTEM.md` (Scoring Algorithm section)

Every retrieved memory gets a `final_score` from
`MemoryScorer.rank_memories()`. The scoring function is a weighted sum
of six factors plus additive bonuses and subtractive penalties — all
parameterized by intent.

### Default Weight Vector

```
relevance:  0.35    # Embedding similarity from ChromaDB
recency:    0.25    # Temporal decay (active-day aware)
truth:      0.20    # Evidence-based correctness via TruthScorer
importance: 0.05    # Content-based retention priority
continuity: 0.10    # Token overlap with current conversation
structure:  0.05    # Numeric/operator density alignment
```

### 12-Step Scoring Algorithm

1. **Base relevance** — Embedding similarity + per-collection boost
   (facts +0.15, summaries +0.10, semantic +0.05, wiki +0.05)
2. **Recency decay** — `1/(1 + decay_rate * age_hours)`, with temporal
   anchor override for TEMPORAL_RECALL
3. **Truth score** — `TruthScorer.compute_effective_truth(metadata)`:
   stored score + time decay from last confirmation
4. **Importance** — Stored importance score (default 0.5)
5. **Continuity** — Token overlap with last exchange (+0.3) + recency
   bonus (+0.1 if within 10 minutes)
6. **Structural alignment** — Numeric/operator density match between
   query and memory
7. **Penalties** — Analogy penalty (-0.1), size penalty (scales from 10KB+)
8. **Anchor bonus** — Salient token overlap with conversation context;
   deictic queries get +0.2 or -0.15 based on overlap
9. **Tone adjustment** — Dismissive language in memory → truth reduced 0.2
10. **Topic match** — 1.0 exact / 0.5 unknown / 0.2 mismatch (usually
    weight=0.0, enabled per-intent)
11. **Meta-conversational bonus** — +0.15 for episodic memories when query
    is about recall ("did we discuss...")
12. **Graph proximity bonus** — +0.05 per knowledge graph neighbor in
    memory content, capped at 0.15
13. **Staleness penalty** — `staleness_ratio * 0.15`, 2x multiplier at
    ≥0.8 ratio, reflections at 60% weight, capped at 0.4

### Temporal-Aware Recency

For TEMPORAL_RECALL queries with a detected time window (e.g., "last
week"), the classifier extracts a temporal anchor in hours. The recency
decay curve is reshaped:

- Within window: gentle decay (1.0 → 0.7)
- Outside window: standard exponential decay from 0.7

This makes memories from the target time period rank much higher than
they would with standard decay.

### Concrete Scoring Example

Memory: *"User's squat is 365lb, set last month at the gym"*

```
relevance:    0.35 × 0.82 = 0.287    (high semantic match)
recency:      0.25 × 0.45 = 0.113    (3 weeks old)
truth:        0.20 × 0.85 = 0.170    (confirmed once)
importance:   0.05 × 0.60 = 0.030    (moderate)
continuity:   0.10 × 0.15 = 0.015    ("squat" token overlap)
structure:    0.15 × 0.90 = 0.135    (numeric density "365lb")
graph_bonus:  0.05                    (1 neighbor "powerlifting")
staleness:    0.00                    (no stale claims)
─────────────────────────────────────
final_score:  0.805
```

---

## 10. Prompt Assembly

**Files**: `core/prompt/formatter.py` (`_assemble_prompt()`), `core/prompt/hygiene.py` (`ContentHygiene`),
`core/prompt/token_manager.py`

### 27 Conditional Sections

The prompt is assembled from up to 27 sections, ordered by transformer
attention patterns — high-signal, low-token sections are placed at the
end for maximum attention weight:

```
[RECENT CONVERSATION]              — always (recent turns with timestamps)
[RELEVANT MEMORIES]                — always (scored semantic memories)
[RECENT SUMMARIES]                 — if available
[SEMANTIC SUMMARIES]               — if available
[BACKGROUND KNOWLEDGE]             — if available (Wikipedia)
[WEB SEARCH RESULTS]               — if triggered
[RELEVANT INFORMATION]             — if available
[RECENT REFLECTIONS]               — if available
[SEMANTIC REFLECTIONS]             — if available
[DREAMS]                           — if enabled
[USER'S PERSONAL NOTES]            — if available (Obsidian, gated 0.30)
[USER UPLOADED ITEMS]              — if files uploaded
[VISUAL MEMORIES]                  — if available (CLIP-matched images)
[DAEMON DOCUMENTATION]             — if available (reference docs)
[PROJECT COMMIT HISTORY]           — if available (git)
[ADAPTIVE WORKFLOWS]               — if available (procedural skills)
[PROPOSED FEATURES]                — if available (code proposals)
[KNOWLEDGE GRAPH]                  — if available (entity relationships)
[UNRESOLVED THREADS]               — if available (open commitments)
[PROACTIVE INSIGHTS]               — if available (cross-domain)
[USER PROFILE]                     — if available (categorized facts)
[ACTIVE FEATURES]                  — always (compact feature inventory)
[CODEBASE CHANGES SINCE SESSION]   — first message only (git diff)
[TIME CONTEXT]                     — always (current datetime)
[TEMPORAL GROUNDING]               — if available (narrative context)
[SHORT-TERM CONTEXT SUMMARY]       — if available (STM analysis)
[CURRENT USER QUERY]               — always (last exchange + query)
```

### Token Budget Management

The prompt has a finite token budget: `min(context_window * 0.25, ceiling)`
clamped to `[floor, ceiling]`, with separate caps for local vs API models.
Default budget: 40,000 tokens (API models), 12,000 tokens (local models).

Sections are assigned priorities for trimming:

```
Priority 10: STM summary (metadata, never trimmed)
Priority  9: User profile (identity, naturally bounded)
Priority  8: Narrative state (hard cap 500 tokens)
Priority  7: Recent conversations, graph context, threads
Priority  6: Semantic chunks, personal notes, user uploads
Priority  5: Reference docs, memories, web search results
Priority  4: Procedural skills, facts
Priority  3: Summaries, proposals, git commits, proactive insights
Priority  2: Reflections, dreams, codebase changes
Priority  1: Wiki
```

### Three-Tier Compression

When the assembled prompt exceeds the budget:

1. **LLM compression** — Items ≥3x over their allocation (`LLM_COMPRESSION_RATIO_THRESHOLD`)
   get intelligent reduction via parallel async LLM calls (gpt-4o-mini,
   3s timeout per item, max 8 batch). Falls back to middle-out on timeout.
   Config: `LLM_COMPRESSION_ENABLED`, `LLM_COMPRESSION_MODEL`,
   `LLM_COMPRESSION_TIMEOUT`; YAML section `llm_compression`
2. **Middle-out compression** — Keeps 60% head + 40% tail, trims middle.
   Inserts snip marker: `… [middle-out snipped N chars] …`
3. **Section removal** — Lowest priority sections dropped first, 25% of
   items removed per pass (max 3 passes)

### Post-Budget Floors

Critical sections have guaranteed minimums that survive trimming:
- Recent conversations: min 5 (even if budget is tight)
- Summaries: min 10 items
- Reflections: min 10 items

### Staleness Prefixes

Items with `staleness_ratio >= 0.6` are prefixed with
`[HISTORICAL — PARTIALLY OUTDATED]` so the LLM knows to treat them
cautiously.

---

## 11. Response Generation

**Files**: `core/response_generator.py`, `core/best_of_handler.py`

### Standard Streaming

The default mode: the assembled prompt is sent to the active LLM model
via `response_generator.generate_streaming_response()`. Chunks stream
back to the GUI as they arrive. Thinking blocks are suppressed from
output via three streaming-time gates:

1. **Tag-based**: `has_incomplete_thinking_block()` detects `<thinking>`/`<think>` open tags before the close tag arrives.
2. **API-level**: Models with native reasoning (Claude, DeepSeek-R1) send thinking via `delta.reasoning_content`; synthetic `<thinking>` tags are emitted by the response generator.
3. **Heuristic**: `likely_untagged_thinking()` catches untagged chain-of-thought during streaming (≥2 distinct meta-reasoning patterns like "I should...", "The user wants..."). Suppresses display until `parse_thinking_block()` can cleanly split thinking from answer.

The thinking content is captured separately for provenance storage.

### Best-of-N / Duel / Ensemble

`BestOfHandler` provides three multi-model generation modes:

**Single mode**: Generate N candidates from one model with temperature
variation (e.g., temp 0.7 vs temp 0.9), then a selector model picks the
best. Useful for quality improvement without multiple model providers.

**Duel mode**: Two different models (e.g., claude-opus-4.6 and gpt-5)
generate in parallel. A judge model reads both responses and picks the
winner. The system prompt tells the judge to evaluate for accuracy,
completeness, and naturalness.

**Ensemble mode**: N generators produce candidates, M judges score all
candidates. Weighted blend or top-1 selection. Most expensive but highest
quality.

All three modes have latency budget enforcement — if generation times out,
the system falls back to standard streaming.

### Response Parsing

After generation, responses pass through `ResponseParser`:

- `parse_thinking_block()` — Extract thinking content and final answer via three layers:
  1. **Tag-based**: `<thinking>`/`<think>`/`<output>` tag extraction
  2. **Heuristic fallback**: `_detect_untagged_thinking()` catches chain-of-thought dumped without tags (meta-reasoning phrases, instruction echoes). Requires ≥2 distinct pattern hits and a clean split point.
  3. **Tag cleanup**: `strip_thinking_tag_leaks()` removes partial/malformed tags
- `likely_untagged_thinking()` — Fast streaming-time check (no split required). Returns True when ≥2 heuristic patterns are present. Used by `handlers.py` to suppress thinking before `parse_thinking_block()` can find a clean split.
- `strip_reflection_blocks()` — Remove `<reflect>` and quality reflection blocks
- `strip_xml_wrappers()` — Remove `<result>`, `<answer>`, `<output>` wrappers
- `strip_prompt_artifacts()` — Remove echoed prompt headers

For models with native reasoning support (Claude, DeepSeek-R1), thinking is
separated at the API level via `extra_body={"reasoning": {"effort": "medium"}}`
in `model_manager.py`. Thinking arrives via `delta.reasoning_content` in
streaming chunks, not in the text body. The prompt-based `<thinking>` tag
instruction is skipped for these models to prevent instruction echoing.

---

## 12. Agentic Tool System

**Files**: `core/agentic/controller.py`, `core/agentic/tools.py`,
`core/agentic/formatters.py`, `core/agentic/types.py`,
`core/agentic/protocols.py`

When a query needs more than stored memory — real-time information,
computation, or deeper memory exploration — Daemon enters a multi-round
ReAct (Reason + Act) loop.

### Triggering: 3-Tier Gate

The agentic gate in `gui/handlers.py` decides whether to enter the loop:

1. **Keyword heuristic** (instant) — 20+ computation keywords ("calculate",
   "compute") and memory keywords ("do you remember", "my notes",
   "search your memory")
2. **Entity match** (instant) — Query terms checked against knowledge
   graph alias index. Mentions of known entities (Flapjack, Auggie, etc.)
   auto-route to agentic memory search
3. **LLM fallback** — Piggybacks on the web search trigger LLM call (zero
   extra cost). The `WebSearchDecision` model includes a
   `needs_memory_search` field

If any tier triggers, the request routes through the agentic controller.
Memory-only queries set `skip_initial_search=True` to skip the initial
web search and go straight to `search_memory`. Git-related queries
("commits", "contributors", "recent changes") route through `git_stats`
when a `GitStatsManager` is available.

### Uncertainty Fallback (Post-Generation)

If a standard (non-agentic) response is detected as uncertain — "I don't
have information about that", "I can't recall" — the `UncertaintyDetector`
(`core/uncertainty_detector.py`) triggers a **silent** retry via the agentic
loop. Detection uses two layers: keyword regex (~18 patterns, confidence
>=0.70) and semantic embedding similarity against 8 anchor sentences
(threshold configurable, default 0.70). Long responses (>400 chars after
hedge-stripping) skip detection. Config: `UNCERTAINTY_FALLBACK_ENABLED`,
`UNCERTAINTY_SEMANTIC_THRESHOLD`, `UNCERTAINTY_MAX_LENGTH`.

### Post-Answer Review Gate

After uncertainty fallback, the `ResponsePlanner` (`core/response_planner.py`)
can review the generated answer against the pre-answer plan. If the review
fails with high confidence (>= 0.90), the system retries via agentic search
with the review feedback as a hint. Responses under 120 chars skip review.
The review is a lightweight LLM call (~300 tokens, 5s timeout). Config:
`RESPONSE_REVIEW_ENABLED`, `RESPONSE_REVIEW_CONFIDENCE_THRESHOLD`,
`RESPONSE_REVIEW_TIMEOUT`.

### Silent Retry Protocol

Both the uncertainty fallback and review gate follow a silent retry protocol:

- **No UI feedback** during retry — no progress messages, no streaming chunks
- **Similarity guard**: retry result only accepted if word overlap with the
  original response is < 70% (Jaccard on word sets). If too similar, the
  retry is silently discarded and the original stays visible.
- This prevents the jarring UX of showing the same response twice.

### 12 Tools (9 action + done_searching + file_read/file_grep/file_list as 3)

| Tool | Implementation | Key Feature |
|------|---------------|-------------|
| Web Search | Tavily API | Query decomposition, 72hr cache, daily credit tracking |
| Wolfram Alpha | LLM API | Token bucket rate limiting, MD5 result cache |
| Code Sandbox | E2B Firecracker microVMs | Persistent sessions (variables survive across rounds) |
| Memory Search | ChromaDB (13 collections) + FAISS (wiki vectors) | wiki_knowledge routes through FAISS; all other collections via ChromaDB |
| Memory Expansion | MemoryExpander | Summary drill-down via source_doc_ids, temporal neighbors. Collection-aware char limits: long-form (obsidian, reference_docs) use 3000/2000 vs default 600/300. |
| Full Document | ReferenceDocsManager | Reassemble all chunks of an uploaded doc by title; fuzzy title matching |
| File Read / File Grep / File List | Local filesystem | 3 separate tools, sandboxed to project directory |
| Git Stats | GitStatsManager | Read-only local git commands: commit counts, contributors, files changed, diff stats. Keyword-based intent parsing with temporal phrase extraction. |
| Recall Image | VisualRetriever | Search visual memory for CLIP-matched images by text query. |
| Done Searching | Control signal | Model declares search complete, triggers final synthesis |

### ReAct Loop Structure

```
Round 1 (automatic):
  Execute initial web search (unless skip_initial_search)
  Compress results, check quality
  If low quality → generate relaxation hint for next round

Rounds 2–N (model-driven, max 5):
  1. Build iteration prompt with [TIME CONTEXT] + accumulated context + context inventory
  2. LLM generates thought + tool selection
  3. Parse decision: wants_search / wants_code / wants_memory / wants_end
  4. Execute selected tool, capture observation
  5. Append observation to accumulated context
  6. Budget enforcement: trim oldest rounds if over context_budget_tokens

Final synthesis:
  Generate response with full accumulated context
  Build final prompt with [TIME CONTEXT], trim low-value sections if over ceiling
```

### Context Inventory

Before each iteration, the controller computes a `context_inventory` —
a summary of what the RAG pipeline already gathered. This is injected
into the iteration prompt to prevent redundant searches. The LLM sees
what collections have been searched, how many results were found, and
what topics are already covered.

### Collection Diversity Tracking

`memory_search_counts` tracks how many times each ChromaDB collection
has been searched within a session. After 2+ searches of the same
collection, hints steer the LLM toward under-explored collections.

### Protocol Support

The controller supports multiple LLM calling conventions:

- **Native function calling** (OpenAI, Anthropic) — Tools defined as
  JSON schemas, parsed from `response.tool_calls`
- **XML markers** (local models) — Tools wrapped in XML tags, parsed
  by regex
- **Legacy** — Fallback text parsing

Protocol is auto-detected from model name via `detect_protocol()`.

Supported tool names (native / XML marker):
`web_search` / `<web_search>`, `wolfram` / `<wolfram>`,
`code_sandbox` / `<code_sandbox>`, `search_memory` / `<search_memory>`,
`expand_memory` / `<expand_memory>`, `file_operation` / `<file_operation>`,
`git_stats` / `<git_stats>`.

### Budget Enforcement

Two levels of budget control:

1. **Accumulated context budget** (`context_budget_tokens`, default 8000) —
   When adding a new round's observations would exceed this, the oldest
   round is trimmed first
2. **Final prompt ceiling** — Before the synthesis generation,
   `_build_final_prompt()` checks total token count and drops
   non-critical sections in order: dreams, reflections, docs, summaries

---

## 13. Per-Turn State Updates

**Files**: `memory/memory_storage.py`, `core/orchestrator.py`

After every user query and LLM response, the system updates its state.
This happens synchronously (except for background storage).

### store_interaction Flow

```
1. Skip gate: reject file-error responses (prevent false memories)
2. Thread detection: assign thread_id + depth for conversation continuity
3. Corpus storage: JSON persistence (immediate write)
4. Topic detection: primary topic → tag enrichment
5. Score calculation: truth_score + importance_score
6. Metadata assembly: timestamp, tags, thread, provenance
7. ChromaDB storage: conversations collection
8. Fact extraction (if FACTS_EXTRACT_EACH_TURN enabled):
   a. FactExtractor: corrections > spaCy > REBEL > regex
   b. Dual budget: 6 user facts + 4 entity facts
   c. FactVerifier: check for conflicts → STORE / STORE_AND_FLAG / REJECT / SKIP
   d. Graph ingestion: entity resolution → add nodes/edges to knowledge graph
9. Consolidation trigger (if threshold met)
```

### Post-Storage Processing (in orchestrator)

After `store_interaction()`, the orchestrator runs additional updates:

- **Correction detection**: Pattern-based detection of user corrections
  ("actually it's X", "no, I meant Y") → truth score adjustments on
  affected memories
- **Confirmation detection**: Pattern-based detection of confirmations
  ("yes, that's right") → truth score boosts
- **Escalation FSM transition**: Update the escalation state machine
  based on the detected tone level
- **Staleness cascade**: If correction events are detected, cascade
  staleness to summaries/reflections that cited the corrected fact

---

## 14. Fact Pipeline

**Files**: `memory/fact_extractor.py`, `memory/llm_fact_extractor.py`,
`memory/fact_verification.py`
**Deep dive**: `MEMORY_SYSTEM.md` (Fact Pipeline section)

### Multi-Stage Extraction

Facts are extracted from conversation through a prioritized pipeline:

1. **Correction detection** — Check for correction patterns first
   ("actually", "no, I meant"). Corrections override existing facts.
2. **spaCy dependency parsing** — NER-based triple extraction. Identifies
   entities (PERSON, ORG, GPE) and their relationships.
3. **REBEL neural extraction** — If available, neural relation extraction
   for more complex triples.
4. **Regex fallback** — Pattern-based extraction for common fact
   structures ("I am X", "my X is Y").

### Dual Budget System

Each extraction pass has separate budgets for user facts and entity facts:

- **User facts** (cap 6 per turn): Subject = "user". These flow to both
  ChromaDB and UserProfile.
- **Entity facts** (cap 4 per turn): Subject = entity name. These stay
  in ChromaDB only — never written to UserProfile.

Entity facts carry additional metadata: `fact_scope` (user/entity),
`entity_type` (from spaCy NER: PERSON, ORG, etc.),
`user_connection` (possessive pattern: "my brother", "our dog").

### Fact Verification Gate

Before any fact is stored, it passes through `FactVerifier.verify()`:

```
1. Ephemeral check: Is this an ephemeral relation (current_feeling, mood)?
   → SKIP (store without verification, these change frequently)

2. Find candidates: Query ChromaDB for existing facts with same
   (subject, predicate)

3. No candidates → STORE (safe, no conflict)

4. Object matches existing → STORE (re-confirmation, may boost truth)

5. Conflict detected (different object):
   a. User-trust override: If source is user_stated + high confidence
      → STORE_AND_FLAG (mark old as superseded)
   b. Entity-scope + low confidence → REJECT (likely extraction error)
   c. LLM adjudication (if enabled): "Is this an UPDATE, COMPLEMENTARY
      fact, or IMPLAUSIBLE?"
      - UPDATE → STORE_AND_FLAG
      - COMPLEMENTARY → STORE (both coexist)
      - IMPLAUSIBLE → REJECT
   d. Fallback → STORE_AND_FLAG (trust newer)
```

The STORE_AND_FLAG verdict marks the old fact with `superseded_by` metadata
via `update_metadata()` — the old fact is never deleted, just flagged.

---

## 15. Truth Scoring & Correction Detection

**Files**: `memory/truth_scorer.py`, `core/correction_detector.py`

### Evidence-Based Truth

The `TruthScorer` is a stateless utility that computes truth scores based
on evidence history rather than access counts (the old echo-chamber system
was removed).

**Initial scores** by source:
- `user_stated`: 0.85 (user directly said it)
- `corrected`: 0.90 (user corrected a previous fact)
- `llm_extracted`: 0.70 (LLM inferred it)
- `inferred`: 0.60 (system deduced it)

**Score adjustments**:
- Confirmation: +0.08 (user restates the fact)
- Correction: −0.25 (user contradicts the fact)
- Contradiction: −0.15 (dedup finds conflicting claim)

**Time decay**: −0.02 per week since last confirmation, floor at 0.30.
Confirmation resets the decay clock. Decay is computed at read time
(`compute_effective_truth()`), not written back — the stored score
represents the last known adjustment.

### Correction Detection

The `CorrectionDetector` uses pattern-based detection to identify when
the user corrects or confirms stored facts:

**Correction patterns**: "actually it's X", "no I meant Y", "that's wrong",
"I changed my X", "it's not X it's Y"

**Confirmation patterns**: "yes that's right", "correct", "exactly",
user restating a stored fact

Detection produces `CorrectionEvent` objects with the affected fact,
the correction type, and the new value. These events trigger truth
score adjustments and staleness cascade.

---

## 16. Memory Staleness Cascade

**File**: `memory/claim_tracker.py`

When a fact is corrected, the correction doesn't just affect that fact —
it can make summaries and reflections that cited the old fact partially
outdated. The staleness cascade tracks this.

### Claim Tracking

The `ClaimIndex` maintains a reverse index from claims to documents:

- **Claim hashing**: Hash on `(subject, relation)` — not the full triple.
  This catches all value changes for a given predicate.
- **Canonicalization**: Uses EntityResolver for subjects,
  `normalize_relation()` for predicates.
- **Persistence**: JSON at `data/claim_index.json`.

### Cascade Mechanism

When a correction event occurs:

```
1. Build ClaimKey from the corrected fact: (subject, relation)
2. Look up all documents containing this claim hash in ClaimIndex
3. For each affected document (summary or reflection):
   a. Add claim hash to stale_claims set
   b. Compute: staleness_ratio = |stale_claims| / |total_claims|
   c. Update ChromaDB metadata with new staleness_ratio
4. During scoring: staleness_ratio × 0.15 penalty
   - 2x multiplier when ratio ≥ 0.80 (nearly all claims outdated)
   - Reflections at 60% weight (they're more abstract)
   - Capped at 0.40
5. In prompt: items with ratio ≥ 0.60 get prefix
   "[HISTORICAL — PARTIALLY OUTDATED]"
```

### Registration at Summary Creation

When summaries are created at shutdown, claims are extracted from the
summary text and registered in the ClaimIndex. This establishes the
reverse mapping that enables future cascade.

---

## 17. Escalation FSM

**File**: `core/escalation_tracker.py`

The escalation tracker is a finite state machine that monitors emotional
momentum across conversation turns and modifies response behavior during
crisis situations.

### 4 States

```
VALIDATE_AND_SUGGEST (VAS) → Normal operation
GROUNDING_PRESENCE (GP)    → Pure acknowledgment, no advice (2-3 sentences)
QUIET_COMPANIONSHIP (QC)   → Just be present, minimal (1-2 sentences)
GENTLE_REENGAGEMENT (GR)   → Tentative re-engagement (2-4 sentences, 1 suggestion)
```

### Tracked Variables

- `consecutive_elevated_count` — Increments on CRISIS/ELEVATED tone,
  resets on calm
- `consecutive_calm_count` — Increments on CONCERN/CONVERSATIONAL, resets
  on elevated
- `ignored_suggestion_count` — Tracks unengaged suggestions (caps at +2/-1)
- `last_need_type` — PRESENCE or PERSPECTIVE (from emotional context)

### State Transitions

```
De-escalation path (tone just dropped):
  consecutive_calm > window  → VAS (fully de-escalated)
  need_type == PERSPECTIVE   → VAS (analytical shift, skip gentle)
  else                       → GR  (genuine calming, gentle re-engage)

No escalation (tone not elevated): → VAS

Escalation path (tone elevated):
  count < threshold          → VAS (not yet escalated)
  count > threshold AND
    ignored ≥ 2              → QC  (suggestions ignored, quiet presence)
  else                       → GP  (grounding presence)
```

### PERSPECTIVE Nuance

When a user shifts from emotional distress to analytical/meta thinking
(need_type = PERSPECTIVE), this is an intensity shift, not genuine calming.
The FSM skips GENTLE_REENGAGEMENT and returns directly to VAS. This
prevents the system from treating "okay let me think about this rationally"
as still being in crisis.

### Prompt Modifications

Each escalation state injects specific instructions into the system prompt
and overrides the token budget:

| State | Instruction | Token Budget |
|-------|------------|--------------|
| VAS | (none — standard tone) | (none) |
| GP | "Max 2-3 sentences, pure acknowledgment, no advice" | 500 |
| QC | "Max 1-2 sentences, just be present, no suggestions" | 300 |
| GR | "2-4 sentences, ONE small concrete suggestion allowed" | 800 |

---

## 18. Thread Surfacing

**Files**: `memory/thread_store.py`, `memory/thread_extractor.py`,
`memory/thread_models.py`

Daemon tracks commitments, deadlines, unanswered questions, and unfinished
topics across sessions. These are surfaced proactively at session start.

### Thread Types

| Type | Priority Weight | Examples |
|------|----------------|---------|
| DEADLINE | 1.0 | "I need to submit by Friday" |
| COMMITMENT | 0.8 | "I'll look into that for you" |
| QUESTION | 0.6 | "What happened with the interview?" |
| UNFINISHED | 0.4 | "We were talking about your project" |

### Lifecycle

**Extraction** (at shutdown): `ThreadExtractor` uses LLM to identify new
threads from session conversation. Each thread gets urgency scoring and
optional deadline extraction.

**Per-turn resolution** (real-time): `check_quick_resolutions()` in
`thread_store.py` runs after every `store_interaction()` call. Pure regex
matching of completion signals against open thread topic keywords (~1ms).
Fast pre-check skips DB query if no completion signal in the message.

**Resolution detection** (at shutdown): The extractor also checks if
existing threads were addressed during the session. Resolved threads
are marked as complete.

**Staleness**: `is_stale(stale_days, deadline_grace_hours)` checks both
time-based staleness (no reference in `stale_days`) and deadline-aware
expiry (deadline date passed + grace period).

**Priority scoring**: `TYPE_PRIORITY[type] × urgency × recency_decay`.
Higher-priority, more-urgent, more-recent threads rank first.

**Cap enforcement**: When total open threads exceed `THREAD_MAX_OPEN`
(default 50), lowest-priority threads are pruned. This is the **only
unconditional deletion** in the system — everything else defaults to
dry-run.

### Prompt Integration

The top 3 threads (by priority score) are surfaced in the
`[UNRESOLVED THREADS]` prompt section. On the first message of a
session (detected via `time_since_previous_message()` returning "N/A"),
the orchestrator injects a `## PROACTIVE THREAD SURFACING` instruction
asking the LLM to weave thread awareness into the conversation naturally.

---

## 19. Proactive Cross-Domain Insights

**File**: `memory/context_surfacer.py`, `memory/surfacing_models.py`,
`memory/surfacing_history.py`

The `ContextSurfacer` analyzes the knowledge graph to find non-obvious
connections across different domains of the user's life.

### How It Works

1. **Classify user edges**: Walk all edges from the "user" node, classify
   by domain (health, career, fitness, relationships, hobbies, etc.)
   using `categorize_relation()` (5-layer cascade: direct→prefix→token→embedding→LLM) and override dict

2. **Identify active domains**: Match query entities to domains via
   entity resolution + keyword fallback

3. **Select bridge candidates**: Score non-active domains by:
   `entity_count(0.3) + avg_weight(0.3) + lateral_edges(0.4)`
   Filter out recently shown via `SurfacingHistory`

4. **LLM synthesis**: Single LLM call batching all bridge candidates →
   returns JSON array of insight texts (or null for weak connections)

5. **Record history**: Update cooldown tracking to avoid repetition

### Session Caching

Insights are generated once per session. The first call runs the full
pipeline; subsequent calls return cached results. This keeps LLM cost
fixed regardless of conversation length.

### Novelty Tracking

`SurfacingHistory` persists to `data/surfacing_history.json` and enforces
a configurable cooldown (default 72 hours). Bridge candidates that were
recently surfaced are filtered out before the LLM call.

### Graph Sparsity Guard

The surfacer skips entirely if the knowledge graph has fewer than 20
nodes or 15 edges — there isn't enough structure for meaningful
cross-domain analysis.

---

## 20. Cross-Collection Deduplication

**File**: `memory/cross_deduplicator.py`, `memory/dedup_models.py`

The `CrossCollectionDeduplicator` detects two types of redundancy across
ChromaDB collections:

### Duplicate Detection

Pairwise cosine similarity across target collections (facts, summaries,
skills, proposals, reflections). Pairs with similarity ≥ 0.92 are flagged
as duplicates.

**Keep/delete priority**: When two items are duplicates, the system keeps
the one from the higher-priority collection:
summaries(5) > reflections(4) > skills(3) > proposals(2) > facts(1).
Ties broken by recency.

### Contradiction Detection

For facts specifically: group by `(subject, predicate)` and find entries
with different objects. For example: "user | lives_in | Atlanta" vs
"user | lives_in | Denver".

**Ephemeral skip**: Predicates in `PROFILE_EPHEMERAL_RELATIONS`
(current_feeling, is, has, thinks, etc.) are excluded — their history
is meaningful, not contradictory.

### Safety Model

**Shutdown runs dry_run=True only** — the deduplicator logs findings but
never auto-deletes. Live deletions require explicit GUI action via
Preview/Run buttons in the Status tab.

A double-run guard (`_dedup_ran` class-level flag) prevents the
deduplicator from running twice per process.

### Staleness Integration

After finding contradiction clusters, the deduplicator cascades staleness
penalties to summaries that cited the older (potentially outdated) facts
via the `ClaimIndex`.

---

## 21. Synthesis Pipeline

**Files**: `knowledge/synthesis_retriever.py`, `knowledge/graph_walk_generator.py`,
`knowledge/synthesis_generator.py`, `knowledge/synthesis_filter.py`,
`memory/synthesis_memory.py`, `knowledge/synthesis_models.py`
**Deep dive**: `SYNTHESIS_FILTER.md`

The synthesis pipeline is Daemon's long-term value proposition — automated
discovery of non-obvious connections between concepts from different
domains. It runs as a "dreaming" step during session shutdown.

> **Current status (2026-04):** All three generators (retrieval, graph walk,
> cross-store) are **disabled** in `config.yaml` pending grading validation.
> No new synthesis candidates are being generated. Existing results in the
> audit queue are available for grading. See `docs/grading_plan.md` for the
> validation protocol.

### Candidate Generation (Three-Tier)

Three generators run in parallel at shutdown, each with independent quotas.
All produce `SynthesisCandidate` objects for the same filter pipeline.

**Tier 0 — RetrievalSynthesisGenerator** (`knowledge/synthesis_retriever.py`):
Structural query extraction (few-shot LLM) → FAISS semantic search (40M
vectors) → adversarial evaluation. Highest-quality candidates because the
retrieval query targets structural patterns rather than surface similarity.
Drop-in replacement interface for `SynthesisGenerator`.

**Tier 1 — GraphWalkGenerator** (`knowledge/graph_walk_generator.py`):
Biased Markov random walks on the unified personal+wikidata graph.
Node2Vec-style return bias (2.0x toward personal nodes). Hub dampening
(log-scale penalty for degree > `GRAPH_WALK_HUB_DEGREE_THRESHOLD`=15)
prevents walks from being dominated by highly-connected nodes. Cross-domain
walk constraint requires walks to touch >= `GRAPH_WALK_MIN_DOMAINS`=2
distinct domain categories. Activated only when bridge edges >= 40.

**Tier 2 — SynthesisGenerator** (`knowledge/synthesis_generator.py`):
Cross-store sampling from ChromaDB facts + FAISS wiki. Forms cross-domain
pairs, uses LLM to articulate bridges. Namespace resolution via
`_resolve_to_graph()` with 4-strategy fallback (EntityResolver → direct ID
→ slug form → display_name index).

### 8-Stage Filter Pipeline

Candidates pass through 8 stages, ordered cheap-to-expensive. Any stage
failure immediately rejects the candidate (short-circuit):

| Stage | Gate | Cost | What It Does |
|-------|------|------|-------------|
| 0 | Text Sanity | ~0ms | Min tokens, verb detection, repetition filter |
| 1 | Domain Crossing | ~1ms | Require ≥2 distinct domains |
| 2 | Semantic Distance | ~5ms | Endpoint distance in [0.20, 0.90] |
| 3 | External Novelty | ~15ms | 3 sub-checks: claim sim vs wiki, concept co-occurrence vs wiki, template specificity vs generic patterns |
| 4 | Internal Novelty | ~10ms | Check synthesis memory — new paths to same insight pass (convergence) |
| 5 | Coherence Judge | ~1-4s | Two-pass LLM: structural coherence (Pass 1), factual skeptic (Pass 2, MODERATE only) |
| 6 | Composite Score | ~0ms | Weighted: coherence(0.30) + novelty(0.40) + distance(0.15) + structural(0.15) ≥ 0.65 |
| 7 | Storage | ~10ms | ChromaDB write to `synthesis_results`; composite-rejected stored for FN audit |

### Stage 3: External Novelty (Three Sub-Checks)

This stage catches three distinct failure modes:

- **Claim similarity**: Full articulated claim searched against wiki
  via FAISS semantic search. Catches direct rehashes. Hard gate: sim > 0.88 → reject (IVFPQ-calibrated, was 0.60).
- **Co-occurrence**: Bare concept conjunction ("concept_a concept_b")
  searched against wiki via FAISS. Catches "known connection, novel phrasing."
  Hard gate: sim > 0.85 → reject (IVFPQ-calibrated, was 0.60).
- **Template specificity**: Regex detection of vacuous bridge language
  ("both involve", "share structural similarities", "operates on
  similar principles"). No hard gate — feeds into composite score.

### Stage 5: Two-Pass Coherence Judge

**Pass 1 (Structural)**: LLM evaluates whether a shared structural
pattern genuinely exists. Rates INVALID / WEAK / MODERATE / STRONG.
Minimum MODERATE to pass. Recalibrated prompt distinguishes WEAK (no
mechanism named, vague bridging language) from MODERATE (names real
mechanism applied concretely to both domains). Generality is not a flaw
if application is concrete. `max_tokens` raised 250 → 400.

**Pass 2 (Factual Skeptic)**: Fires only on MODERATE results. Targets
pseudoscience wrapped in structural language. Binary PASS/FAIL. On FAIL,
coherence is downgraded to WEAK → rejected.

### Convergence Detection

The key signal is independent rediscovery. When different random walks
find the same insight via different paths, that's evidence of a genuine
connection:

```
convergence_strength = |unique_paths| × |unique_sources|

Promoted to CONVERGING when:
  unique_paths ≥ 3 AND unique_sources ≥ 2
```

When Stage 4 finds an existing match with a new path_hash, it passes
rather than rejecting. At storage time, convergence metadata is updated.

### Audit Queue (Human-in-the-Loop)

Accepted results and composite-rejected candidates are queued for blind
human grading in the GUI "Synthesis" tab. See `docs/grading_plan.md` for
the full grading protocol.

**Two-layer grading:**

- **Layer 1 — Binary screening** (three yes/no questions):
  - `changes_thinking`: "Does this make me think about something differently?"
  - `mechanism_real`: "Is the mechanism it describes real?" (yes/no/unsure)
  - `heard_before`: "Have I heard this connection before?"
- **Layer 2 — Gut-feel slider** (1-5): 1=Nonsense, 2=Surface metaphor,
  3=True but obvious, 4=Real insight, 5=Breakthrough.
- **Classification**: Grades 1-3 = invalid (FP if accepted), grades 4-5 = valid.

`SynthesisMemory.get_audit_stats()` tracks FP rate. At shutdown, if
`fp_rate > SYNTHESIS_AUDIT_FP_HALT_THRESHOLD` (default 0.50) with at
least `SYNTHESIS_AUDIT_MIN_GRADED` (default 10) graded results, synthesis
dreaming auto-halts to prevent accumulating miscalibrated output. Config:
`SYNTHESIS_AUDIT_ENABLED`, `SYNTHESIS_AUDIT_FP_HALT_THRESHOLD`,
`SYNTHESIS_AUDIT_MIN_GRADED`; YAML section `synthesis_audit`. 34 tests in
`tests/unit/test_synthesis_audit.py`.

### Benchmark Performance

Tested against 72 labeled candidates (7 tiers). Current coherence judge
model: `claude-opus-4.6` (upgraded from `gpt-4o-mini`, then `sonnet-4.5`).

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Claude Opus 4.6 (two-pass) | Current production model | — | — |
| Sonnet 4.5 (two-pass) | 90.9% | 100% | 95.2% |
| GPT-4o-mini (two-pass) | 83.3% | 100% | 90.9% |

---

## 22. Session Lifecycle

**Files**: `memory/shutdown_processor.py`, `gui/launch.py`,
`core/orchestrator.py`

### Session Start

When Daemon starts (GUI launch or CLI start):

1. **Reference docs auto-seed**: Background thread seeds `docs/` into
   `reference_docs` ChromaDB collection (mtime-based idempotency)
2. **Git change detection**: Compares git state since last session
   (committed + uncommitted changes)
3. **Feature inventory**: Builds compact summary of active features
4. **Narrative freshness check**: Detects if temporal context is stale
   (>24 hours since last update)
5. **First-message injection**: `[CODEBASE CHANGES]` and
   `[ACTIVE FEATURES]` sections added to prompt; thread surfacing
   instruction injected

### Per-Turn Updates

See [Section 13: Per-Turn State Updates](#13-per-turn-state-updates).

### Session Shutdown (12-Step Pipeline)

`ShutdownProcessor.process_shutdown_memory()` runs a strict sequence:

```
Step 1:  Block summaries — LLM compression of N-turn conversation blocks
         Claim extraction → ClaimIndex registration
         Source doc IDs stored for expand_memory drill-down

Step 2:  Session fact extraction — Rule-based, last 10 turns
         Each fact passes through FactVerifier gate
         source_excerpt forwarded from MemoryNode metadata to ChromaDB

Step 3:  LLM fact extraction — Neural triple extraction, last 12 turns
         Batch verification before storage
         source_excerpt attached via keyword matching (_attach_source_excerpts)
         Graph ingestion for entity-worthy facts

Step 4:  Behavioral pattern extraction — Cross-turn habit detection
         Single LLM call identifies recurring cross-domain behaviors the user
         exhibits but never states explicitly (e.g., "codes at the gym").
         Stores as profile facts. Min 3 turns required, max 3 patterns.
         Config: behavioral_patterns.enabled (default true).

Step 5:  Procedural skill extraction — WHEN/THEN adaptive workflows

Step 6:  Code proposal generation — Self-improvement suggestions
         Filtered against GOALS.md for relevance

Step 7:  Implementation tracking — Lightweight file-existence check
         for previously proposed features

Step 8:  Open thread processing — Three phases:
         a. Resolution detection (check if threads addressed)
         b. New thread extraction (commitments, deadlines, questions)
         c. Cap enforcement (prune lowest-priority if over max)

Step 9:  Synthesis dreaming — Three-tier parallel candidate generation:
         Tier 0 RetrievalSynthesisGenerator, Tier 1 GraphWalkGenerator,
         Tier 2 SynthesisGenerator → 7-stage filter → convergence tracking
         → provisional bridge creation on acceptance

Step 10: Wiki-to-graph enrichment — Tracked wiki articles from session
         added as graph nodes, linked to existing entities via
         extract_graph_entities(). Edges use mentioned_alongside relation.

Step 11: Knowledge graph save — JSON flush (dirty-flag optimization)

Step 12: Cross-collection dedup — Dry-run preview only (never auto-deletes)
```

**Critical invariant**: No user data is auto-deleted at shutdown. Dedup
runs dry-run only. Thread cap enforcement (Step 7c) is the only deletion,
and it removes lowest-priority threads when over the cap.

### Session-End Reflection

After the 12-step pipeline, `run_shutdown_reflection()` generates a
meta-reflection about the session — what was discussed, what stood out,
what patterns emerged. This is stored in the `reflections` collection
for future context.

---

## 23. Knowledge Sources

Daemon integrates multiple external knowledge sources beyond its own
memory system:

### Web Search (Tavily API)

**File**: `knowledge/web_search_manager.py`

Real-time web search with query decomposition, caching, and rate limiting:

- **Depths**: QUICK (snippets), STANDARD (search + extract top pages),
  DEEP (search + extract + LLM link following)
- **Query decomposition**: Complex multi-entity queries decomposed into
  parallel sub-queries (max 4), results merged
- **Caching**: 72-hour cache in ChromaDB to avoid repeated API calls
- **Rate limiting**: Daily credit budget (default 100 credits/day)
- **News detection**: Keyword + semantic similarity detection for broad
  news briefings. `_is_broad_news_query()` uses keyword phrases, `news + ?`
  fallback, and a semantic embedding check against 5 news anchor phrases.
  Mid-sentence named entities trigger specific-query rejection.
- **Multi-term search**: `multi_search(sub_queries=[...])` accepts pre-computed
  sub-queries for parallel execution (used when LLM generates multiple terms)
- **Crisis suppression**: Web search is suppressed during HIGH/MEDIUM
  crisis tone levels

### Wolfram Alpha

**File**: `knowledge/wolfram_manager.py`

Symbolic computation, unit conversions, scientific data:

- Token bucket rate limiting
- MD5-keyed result caching
- Assumption parsing from Wolfram response

### Code Sandbox (E2B)

**File**: `knowledge/sandbox_manager.py`

Secure Python execution in ephemeral Firecracker microVMs:

- Persistent sessions — variables survive across agentic loop rounds
- Pre-installed: NumPy, Pandas, SciPy
- Output capture (stdout, stderr, plots)

### Wikipedia

**Files**: `knowledge/semantic_search.py`, `knowledge/wiki_tracker.py`,
`knowledge/wiki_enrichment.py`, `knowledge/wikidata_resolver.py`,
`knowledge/wikidata_models.py`

6.5M+ articles (40M+ vectors) semantically indexed with FAISS:

- Pipeline: download dump → parse XML → chunk (512 tokens) →
  embed (all-MiniLM-L6-v2) → build FAISS IVFPQ index
- IVFPQ compression: 48 subquantizers × 8 bits = 48 bytes/vector
  (~32x reduction from 1536-byte float32), index fits in ~2 GB RAM
- Zero-copy metadata: parquet file read on-demand per query via
  row-group offset index — no DataFrame loaded into RAM. Footprint:
  FAISS index (~2.2 GB) + embedder (~0.4 GB)
- Requires ~102GB storage for raw wiki data (optional — system works without it)
- All wiki vector searches (agentic search, prompt retrieval, synthesis
  pipeline) route through FAISS; ChromaDB `wiki_knowledge` used only as fallback
- **Session tracking**: `WikiArticleTracker` records articles accessed during
  queries. At shutdown, `WikiGraphEnricher` creates graph nodes (source="wiki_retrieved")
  for tracked articles and links them to existing entities. Edges use
  `mentioned_alongside` relation at weight 0.5.
- **Wikidata integration**: `WikidataEntityMapper` resolves personal entities
  against Wikidata via 3 strategies (exact alias, embedding similarity, domain
  filtering). `wikidata_models.py` defines Pydantic models for import.

### Obsidian Vault

**File**: `knowledge/obsidian_manager.py`

User's personal notes synced from Obsidian:

- Mtime-based re-embedding: changed files detected by comparing
  file mtime vs stored mtime metadata
- Changed files: old chunks deleted, re-embedded (not skip-if-exists)
- Multimodal support for embedded images
- Stored in `obsidian_notes` collection (protected, never deduped)
- Higher relevance gate (0.30 vs general 0.18) to prevent topically
  similar but contextually irrelevant notes from leaking

### Git Memory

**Files**: `knowledge/git_memory.py`, `knowledge/git_memory_loader.py`

Git commit history extracted as procedural knowledge:

- Commit messages, authors, timestamps, optional diffs
- Stored in `procedural` collection
- Surfaced during project-related queries

### Reference Docs

**File**: `knowledge/reference_docs_manager.py`

Daemon self-knowledge documentation:

- Auto-seeded from `docs/` directory on GUI startup (background thread)
- Mtime-based idempotency: unchanged files skipped, modified re-uploaded
- Stored in `reference_docs` collection (protected, never deduped)

---

## 24. Personality & Tone System

**Files**: `config/prompts/default_personality.txt`,
`config/prompts/operating_principles.txt`

### File-Based Personality

The system prompt is composed from two separate text files:

- **Personality file** (`default_personality.txt` or user's
  `custom_personality.txt`) — Editable personality with warm, grounded
  tone and examples. Users can create custom personalities via the GUI
  "Personality" tab.
- **Operating principles** (`operating_principles.txt`) — Immutable
  behavioral rules: AI limitations, facts handling, claim grounding,
  response modes, knowledge source instructions.

Placeholder substitution replaces `{USER_NAME}`, `{USER_PRONOUNS}`,
`{PRONOUN_SUBJ}`, `{PRONOUN_OBJ}`, `{PRONOUN_POSS}` with actual user
profile values.

Personality text is truncated to `PERSONALITY_MAX_CHARS` (15000) to
prevent prompt budget blowout.

### Tone-Driven Modifications

The tone detection system (Section 3) produces crisis levels that modify
the system prompt with mode-specific instructions:

- **CRISIS_SUPPORT**: Full therapeutic mode with resources
- **ELEVATED_SUPPORT**: Empathetic validation, 2-3 paragraphs
- **LIGHT_SUPPORT**: Brief acknowledgment, 2-4 sentences
- **CONVERSATIONAL**: Direct and concise, max 3 sentences

The escalation FSM (Section 17) can further override these instructions
with tighter response length constraints during sustained crisis.

---

## 25. Provenance & Audit Trail

**Files**: `memory/memory_storage.py`, `gui/handlers.py`,
`core/agentic/controller.py`

Every stored interaction carries provenance metadata enabling
prompt-to-response traceability:

### Provenance Record

| Field | Source | Purpose |
|-------|--------|---------|
| `session_id` | MemoryCoordinator | Groups turns within a session |
| `response_mode` | handlers.py | One of: agentic, enhanced, best_of, best_of_duel, fallback |
| `model_name` | ModelManager | LLM alias that produced the response |
| `thinking_block` | ResponseParser | Captured `<thinking>` content (truncated to 4000 chars) |
| `cited_memory_ids` | Citation extraction | Memory doc IDs referenced via `[MEM_RECENT_N]` markers |
| `prompt_hash` | SHA-256[:16] | Hash of final assembled prompt |
| `agentic_summary` | AgenticSearchSession | Tools used, round count, searches, expansions |

### Citation Markers

During prompt assembly, memories are tagged with citation markers
(`[MEM_RECENT_0]`, `[MEM_SEMANTIC_1-2]`). If the LLM includes these
markers in its response, they're extracted during post-processing and
mapped back to memory document IDs via the `memory_id_map`.

### GUI Display

The "Provenance" tab in the GUI displays per-turn provenance: response
mode, model, citations, token counts, and agentic rounds (if applicable).

---

## 26. Configuration Architecture

**Files**: `config/app_config.py`, `config.yaml`

### Pattern: YAML → Module Constants

All configuration follows a consistent pattern:

1. `config.yaml` defines hierarchical settings by section
2. `app_config.py` reads YAML, applies environment variable overrides,
   and exports module-level constants
3. Components import constants directly: `from config.app_config import X`

```python
# config/app_config.py pattern
CFG = config.get("section_name", {})
SOME_CONSTANT = int(os.getenv("SOME_CONSTANT", CFG.get("key_name", default_value)))
```

### Key Configuration Sections

| YAML Section | Controls | Key Constants |
|-------------|----------|--------------|
| `daemon` | Version, data directory | `DATA_DIR`, `CORPUS_FILE` |
| `memory` | Retrieval limits, corpus size | `CORPUS_MAX_ENTRIES`, `PROMPT_MAX_*` |
| `models` | Default model, max tokens | `DEFAULT_MODEL`, `DEFAULT_MAX_TOKENS` |
| `gating` | Similarity thresholds, weights | `COSINE_SIMILARITY_THRESHOLD`, `SCORE_WEIGHTS` |
| `features` | Feature toggles | `USE_STM_PASS`, `ENABLE_QUERY_REWRITE`, `ENABLE_BEST_OF` |
| `web_search` | Tavily config, caching | `WEB_SEARCH_ENABLED`, `WEB_SEARCH_DAILY_CREDIT_LIMIT` |
| `knowledge_graph` | Graph behavior | `KNOWLEDGE_GRAPH_ENABLED`, `GRAPH_SCORING_BOOST_CAP` |
| `intent_classifier` | Intent thresholds | `INTENT_ENABLED`, `INTENT_STM_REFINEMENT_THRESHOLD` |
| `entity_facts` | Extraction budgets | `ENTITY_FACTS_ENABLED`, `ENTITY_FACTS_PER_TURN_CAP` |
| `fact_verification` | Verification gate | `FACT_VERIFICATION_ENABLED` |
| `staleness` | Staleness tracking | `STALENESS_ENABLED`, `STALENESS_WEIGHT` |
| `escalation` | Crisis FSM | `ESCALATION_ENABLED`, `ESCALATION_THRESHOLD` |
| `thread_surfacing` | Thread tracking | `THREAD_SURFACING_ENABLED`, `THREAD_MAX_OPEN` |
| `proactive_surfacing` | Insight generation | `PROACTIVE_SURFACING_ENABLED` |
| `cross_dedup` | Deduplication | `CROSS_DEDUP_ENABLED`, `CROSS_DEDUP_DUPLICATE_THRESHOLD` |
| `synthesis` | Filter pipeline | `SYNTHESIS_ENABLED`, `SYNTHESIS_COHERENCE_MODEL` |
| `synthesis_generator` | Candidate generation | `SYNTHESIS_GENERATOR_ENABLED` |
| `synthesis_audit` | Human grading + auto-halt | `SYNTHESIS_AUDIT_ENABLED`, `SYNTHESIS_AUDIT_FP_HALT_THRESHOLD` |
| `provenance` | Audit trail | `PROVENANCE_ENABLED` |
| `git_stats` | Git activity queries | `GIT_STATS_ENABLED`, `GIT_STATS_TIMEOUT`, `GIT_STATS_MAX_OUTPUT_LINES` |
| `response_planning` | Pre-answer plan + post-answer review | `RESPONSE_PLANNING_ENABLED`, `RESPONSE_REVIEW_ENABLED`, `RESPONSE_REVIEW_CONFIDENCE_THRESHOLD` |
| `llm_compression` | Smart token compression | `LLM_COMPRESSION_ENABLED`, `LLM_COMPRESSION_MODEL`, `LLM_COMPRESSION_TIMEOUT` |
| `behavioral_patterns` | Cross-turn habit detection | `behavioral_patterns.enabled` |
| `wiki_enrichment` | Session wiki articles to graph | `WIKI_ENRICHMENT_ENABLED`, `WIKI_ENRICHMENT_MAX_PER_SESSION` |
| `wikidata_import` | Wikidata entity resolution | `WIKIDATA_*` constants |
| `graph_walk` | Biased Markov walk synthesis | `GRAPH_WALK_ENABLED`, `GRAPH_WALK_MIN_BRIDGE_EDGES` |
| `uncertainty_fallback` | "I don't know" detection + retry | `UNCERTAINTY_FALLBACK_ENABLED`, `UNCERTAINTY_SEMANTIC_THRESHOLD` |

### Environment Variable Overrides

Every constant can be overridden via environment variable of the same name.
This enables Docker deployment and per-environment tuning without modifying
config files.

---

## 27. Production Deployment

### GUI Mode (Gradio)

Default entry point. Dark theme with JetBrains Mono font, configured in
`gui/launch.py:get_dark_theme()`. Tabs for chat, status, personality,
provenance, and thread management.

Startup hooks run background tasks: reference docs auto-seed, narrative
freshness check, daily/weekly/monthly notes catch-up.

### CLI Mode

`python main.py cli` — Terminal chat loop. Same orchestrator and memory
system, different I/O layer.

### Docker

`docker-compose.yml` provides containerized deployment:
- Volume mounts for persistent data (ChromaDB, corpus, config)
- Port mapping for Gradio UI (7860)
- Environment variable injection for API keys

### Desktop Executable (PyInstaller)

`pyinstaller daemon.spec` builds a frozen executable with:
- Embedded data files and model weights
- Bootstrap module (`utils/bootstrap.py`) handles frozen-path resolution
- First-run wizard for API key and model configuration
- Build output: `dist/Daemon/Daemon`

### Resource Usage

| Resource | Idle | Active Query |
|----------|------|--------------|
| CPU | ~5% | 30-50% |
| RAM | ~500MB | ~1.5GB |
| GPU VRAM | — | 2-8GB (optional) |

Storage: Corpus JSON ~10MB, ChromaDB ~50MB, Wikipedia raw data ~102GB
(optional), Wikipedia FAISS IVFPQ index ~2 GB, Logs ~1MB/day.

---

## 28. Testing & Benchmarks

### Test Structure

3,559 tests across 173 files:

```
tests/
├── unit/           # Component tests (most test files here)
├── test_eval/      # Eval system tests (246 tests)
├── benchmarks/     # Retrieval quality (real embeddings)
└── fixtures/       # Seed data (30 memories, 72 synthesis candidates)
```

**Last full run**: 2026-05-01 — 3248 passed, 5 wizard-only failures (pre-existing).

Key test counts by subsystem:

| Subsystem | Tests | File |
|-----------|-------|------|
| Intent classifier | 74 | `tests/unit/test_intent_classifier.py` |
| Knowledge graph | 70 | `tests/unit/test_knowledge_graph.py` |
| Implementation tracking | 66 | `tests/unit/test_implementation_detector.py` |
| Proactive surfacing | 55 | `tests/unit/test_context_surfacer.py` |
| Escalation tracker | 51 | `tests/unit/test_escalation_tracker.py` |
| Graph integration | 50 | `tests/unit/test_graph_integration.py` |
| Cross-dedup | 47 | `tests/unit/test_cross_deduplicator.py` |
| Claim tracker | 47 | `tests/unit/test_claim_tracker.py` |
| File access | 44 | `tests/unit/test_file_access_manager.py` |
| Fact verification | 39 | `tests/unit/test_fact_verification.py` |
| Thread system | 137 | 4 files (models + store + extractor + integration) |
| Eval system | 246 | `tests/test_eval/` (7 files) |

### Retrieval Quality Benchmarks

`tests/benchmarks/` provides an end-to-end retrieval quality suite:

- **30 seed memories** with realistic content (fitness PRs, relationships,
  project notes, emotional conversations)
- **19 test cases** across all 9 intent types
- **Real embeddings** via all-MiniLM-L6-v2 (not mocks)
- **Metrics**: recall@K and Mean Reciprocal Rank (MRR)
- **Runtime**: ~5 seconds

Run: `pytest tests/benchmarks/ -m benchmark -v`
Exclude: `pytest -m "not benchmark"`

### Testing Conventions

- **Framework**: pytest + pytest-asyncio
- **Markers**: `slow`, `semantic`, `benchmark`
- **Config patching**: Config values imported inside functions — patch
  `config.app_config.CONSTANT`, not `module.CONSTANT`
- **Debug dict**: `memory_scorer.py` debug dict only populated at DEBUG
  log level — tests must set logger level explicitly
- **Excluded dirs**: `venv/`, `data/`, `integration.bak/`

---

## 29. Prompt Eval System

**Files**: `eval/` directory (16 files)
**Plan**: `eval/PLAN.md` (8-phase architecture)

The eval system provides prompt section ablation and quality measurement
infrastructure. It answers the question: "Which prompt sections actually
help, and which hurt?" without modifying production code or persisting
any data.

### Architecture (8 Phases)

| Phase | Status | What It Does |
|-------|--------|-------------|
| 1 | Complete | Snapshot capture and deterministic replay |
| 2 | Complete | Variant generation, query corpus, utilization analysis |
| 3 | Skipped | Manual baseline collection |
| 4 | Complete | Batch generation harness with rate limiting and resume |
| 5 | Complete | Pairwise A/B judging with position randomization |
| 6 | Complete | Automated objective checks (length, filler, grounding, citations, thinking leak) |
| 7 | Planned | Longitudinal tracking |
| 8 | Complete | Intent-conditioned section gating (eval-driven prompt optimization) |

### Pipeline

```
SnapshotCapture (builder.py hook)
  → save to eval/snapshots/*.json
  → SnapshotReplay (deterministic from stored formatted_text)
  → VariantGenerator (LOO, AOI, bundle, reorder strategies)
  → GenerationHarness (batch model calls with PersistenceGuard)
  → PairwiseJudge (blind A/B, 5-criterion rubric)
  → ObjectiveChecks (5 automated checks, no LLM)
  → Section impact report (per-section win/loss/tie rates)
```

### Integration Point

`builder.py:_maybe_capture_eval_snapshot()` is a gated hook at the end of
prompt assembly. Disabled by default (`DAEMON_EVAL_CAPTURE=0`). When
enabled, it lazy-imports eval modules and captures the post-hygiene layer
with formatted sections, prompt text, and git provenance. Zero overhead
when disabled.

### Safety

`PersistenceGuard` fingerprints ChromaDB collections and JSON data files
before and after eval runs. `before.assert_same_as(after)` raises on any
mutation, ensuring eval never modifies production state.

### Key Result (Phase 8)

Phase 8 used Phase 5 judge verdicts and Phase 6 objective checks to
calibrate per-intent retrieval overrides. CASUAL_SOCIAL now skips most
retrieval; EMOTIONAL_SUPPORT increases recent conversation retrieval.
Gated by `PROMPT_SECTION_GATING_ENABLED` config flag (default True).

### Test Coverage

246 tests in `tests/test_eval/` covering registry, snapshots, replay,
generation, persistence, variants, corpus, utilization, harness, judge,
and checks.

---

## Appendix: Component Interaction Map

This shows the primary data flow relationships between major components.
Arrows indicate "calls" or "data flows to."

```
┌─────────────────────────────────────────────────────────────────┐
│ GUI (handlers.py)                                               │
│  ├── Mode routing (raw / duel / agentic / enhanced)             │
│  ├── Streaming response chunks to user                          │
│  └── Background storage with provenance                         │
└──────┬──────────────────────────────────────┬───────────────────┘
       │                                      │
       ▼                                      ▼
┌──────────────┐                    ┌──────────────────────┐
│ Orchestrator │                    │ AgenticController    │
│  ├─ Context  │                    │  ├─ ReAct loop       │
│  │  Pipeline │                    │  ├─ Tool execution   │
│  ├─ Prompt   │                    │  ├─ Budget enforce   │
│  │  Builder  │                    │  └─ Final synthesis  │
│  └─ Response │                    └──────────┬───────────┘
│     Generator│                               │
└──────┬───────┘                               │
       │                                       │
       ├───────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ PromptBuilder                                                │
│  ├─ ContextGatherer (19 parallel retrieval tasks)            │
│  │    ├─ MemoryRetriever (ChromaDB queries)                  │
│  │    ├─ GraphMemory (BFS traversal + query expansion)       │
│  │    ├─ WebSearchManager (Tavily API)                       │
│  │    ├─ ThreadStore (open threads)                          │
│  │    ├─ ContextSurfacer (proactive insights)                │
│  │    └─ UserProfile (identity facts)                        │
│  ├─ GateSystem (cosine + cross-encoder filtering)            │
│  ├─ MemoryScorer (12-step composite scoring)                 │
│  ├─ TokenManager (priority-based budget + compression)       │
│  ├─ Formatter (26-section assembly)                          │
│  └─ EvalSnapshot (optional capture, gated by env var)        │
└──────┬───────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ Memory System                                                │
│  ├─ MemoryCoordinator (thin orchestrator, ~16 components)    │
│  │    ├─ MemoryStorage (persist + fact extraction + graph)    │
│  │    ├─ FactExtractor (dual budget: user + entity)          │
│  │    ├─ FactVerifier (STORE / FLAG / REJECT / SKIP)         │
│  │    ├─ TruthScorer (evidence-based, time decay)            │
│  │    ├─ ClaimIndex (staleness cascade)                      │
│  │    ├─ GraphMemory (NetworkX DiGraph, JSON persistence)    │
│  │    ├─ EntityResolver (alias table, relation normalization)│
│  │    └─ CrossDeduplicator (duplicates + contradictions)     │
│  │                                                           │
│  └─ ShutdownProcessor (12-step session-end pipeline)         │
│       ├─ Block summaries + claim registration                │
│       ├─ Fact extraction (regex + LLM) + verification        │
│       ├─ Behavioral pattern extraction (cross-turn habits)   │
│       ├─ Skills, proposals, implementation tracking          │
│       ├─ Thread extraction + resolution + cap enforcement    │
│       ├─ Synthesis dreaming (3-tier) + filter + audit queue  │
│       ├─ Wiki-to-graph enrichment                            │
│       ├─ Knowledge graph persistence                         │
│       └─ Cross-dedup preview (dry-run only)                  │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ Storage Layer                                                │
│  ├─ MultiCollectionChromaStore (13 collections)              │
│  ├─ CorpusManager (JSON persistence)                         │
│  ├─ Knowledge Graph (data/knowledge_graph.json)              │
│  ├─ Entity Aliases (data/entity_aliases.json)                │
│  ├─ Claim Index (data/claim_index.json)                      │
│  └─ Surfacing History (data/surfacing_history.json)           │
└──────────────────────────────────────────────────────────────┘
```

---

## Appendix: Multi-Provider LLM Abstraction

**File**: `models/model_manager.py`

`ModelManager` provides a unified interface across multiple LLM providers:

### Supported Providers

| Provider | Access Via | Models |
|----------|-----------|--------|
| Anthropic | Direct API | Claude Opus 4.6, Sonnet 4.6, Haiku 4.5 |
| OpenAI | Direct API / OpenRouter | GPT-5, GPT-4o, GPT-4o-mini |
| DeepSeek | OpenRouter | DeepSeek-R1, DeepSeek-V3 |
| Google | Direct API | Gemini models |
| Local | Transformers | Any HuggingFace model |

### Key Methods

- `generate_once(prompt, model_name, system_prompt, max_tokens, temperature)` —
  Primary async generation method. Handles extended thinking extraction
  for Claude models (text blocks from content arrays). Passes
  `extra_body={"reasoning": {"effort": "medium"}}` for reasoning models.
- `generate_async(prompt, raw, images, **kwargs)` — Async streaming. Passes
  `extra_body={"reasoning": {"effort": "medium"}}` for reasoning models so
  thinking arrives via `delta.reasoning_content` (API-level separation).
- `generate_once_with_tools(prompt, tools, tool_choice)` — For agentic
  workflows. Returns raw completion with `response.tool_calls`.
- `supports_tools(model_name)` — Whether the model supports function calling.
- `supports_vision(model_name)` — Whether the model supports image/vision
  input (GPT-4o+, Claude, Gemini). Returns False for text-only models
  (DeepSeek, GLM). `generate_async()` silently drops images for non-vision
  models to prevent API errors.
- `supports_reasoning(model_name)` — Whether the model may return extended
  thinking blocks (Claude, DeepSeek-R1). Used by generate_async/generate_once
  to enable native reasoning, and by orchestrator to skip prompt-based
  thinking instruction.
- `get_embedder()` — Cached SentenceTransformer singleton (all-MiniLM-L6-v2).

### Error Classification

API errors are mapped to user-friendly categories:

- `[CREDITS EXHAUSTED]` — quota/billing errors (429 with quota message)
- `[RATE LIMITED]` — standard rate limits (429)
- `[AUTH ERROR]` — invalid API keys (401)
- `[MODEL NOT FOUND]` — model not available (404)
- `[SERVER ERROR]` — provider outage (500+)

### Graceful Degradation

When primary model fails, the system can fall back through configured
model chains. If all API models fail, generation returns an error message
rather than crashing.

---

## Appendix: Implementation Tracking

**File**: `knowledge/implementation_detector.py`

The implementation tracker detects whether previously generated code
proposals have been built. It uses a 4-stage pipeline:

| Stage | Method | Latency | What It Checks |
|-------|--------|---------|---------------|
| 1 | File existence | ~50ms | Do files mentioned in the proposal exist? |
| 2 | Grep | ~200ms | Does the codebase contain key patterns from the proposal? |
| 3 | Git history | ~300ms | Do recent commits mention related terms? |
| 4 | LLM judgment | ~1-2s | Borderline cases get LLM adjudication |

**Lightweight mode** (file existence only) is used at shutdown to minimize
latency. The full 4-stage pipeline runs on-demand via CLI or GUI.

**Confidence thresholds**:
- Confirmed (≥ 0.85): High certainty the proposal was implemented
- Likely (≥ 0.60): Strong evidence
- Uncertain (≥ 0.30): Some signals but not conclusive
- Not implemented (< 0.30): No evidence found

**Cooldown**: Default 86400 seconds (24 hours) between checks for the
same proposal.

**CLI**: `python main.py check-proposals [--id UUID] [--verbose]`
**GUI**: Batch check + single-check buttons in the Proposals tab.

---

## Appendix: User Profile System

**File**: `memory/user_profile.py`, `memory/user_profile_schema.py`

The `UserProfile` maintains categorized user facts with temporal history:

### 12 Fact Categories

Health, career, fitness, education, relationships, hobbies, finance,
goals, projects, identity, location, preferences.

Each category stores facts as key-value pairs with timestamps. The
profile is append-only — updates add new timestamped entries, old entries
are preserved for temporal history.

### Hybrid Retrieval

Profile facts are retrieved using a 2/3 semantic + 1/3 recent blend:
- Semantic: embed the query, find closest facts by similarity
- Recent: return most recently added facts regardless of query

This ensures that both topically relevant and freshly stated facts
appear in context.

### Profile Boundary

Only user facts (subject = "user") flow to the UserProfile. Entity
facts (subject = entity name) stay in ChromaDB only. This prevents
entity information from polluting the user's self-model.

### Prompt Integration

The user profile is rendered as the `[USER PROFILE]` prompt section,
organized by category with source excerpts when available (`(said: "...")`).
An inline anti-confabulation instruction prevents the LLM from embellishing
vague facts with fabricated specifics. It sits in the high-attention zone
of the prompt (near the end) and is naturally bounded at ~1-3K tokens.

---

## Appendix: Key Design Decisions

### Why Regex-First Intent Classification

LLM-based intent classification would add 200-500ms latency and cost per
query. The regex approach is instant, deterministic, and covers 90%+ of
queries accurately. The STM refinement path handles the remaining edge
cases at no additional cost (STM analysis runs anyway for other purposes).

### Why DiGraph, Not MultiDiGraph

Duplicate edges (same subject-relation-object) strengthen the edge weight
rather than creating parallel edges. This reflects the semantic reality:
multiple mentions of the same fact increase confidence, not multiplicity.

### Why Dry-Run-Only Deduplication

Auto-deleting user data at shutdown creates an unacceptable trust risk.
The deduplicator previews what it would do; the user decides whether to
act via the GUI. This is the single most important safety invariant.

### Why Per-Intent Weight Overrides (Not Separate Pipelines)

A single scoring function with swappable weight vectors is more
maintainable than 9 separate retrieval pipelines. The same infrastructure
serves all query types — only the parameters change. This also means
every infrastructure improvement (faster gating, better embeddings)
automatically benefits all intent types.

### Why Token Budget Floors

Without floors, aggressive budget enforcement could eliminate all recent
conversations during a token-heavy session (many retrieved memories,
large user profile). The floor guarantees ensure the LLM always has
minimal conversational context, even when the budget is tight.

### Why Source-Doc Backlinks in Summaries

When the agentic tool `expand_memory` drills into a summary, it needs
to recover the original conversations. Source doc IDs stored at summary
creation time enable direct lookup. Without them, the system falls back
to temporal anchor range queries, which are less precise.

### Why Staleness Cascades (Not Just Fact Correction)

Correcting a fact fixes one ChromaDB entry. But summaries and reflections
that incorporated the old fact still carry outdated information. The
staleness cascade propagates the correction to downstream documents,
ensuring the scoring function penalizes stale summaries rather than
presenting them as current.
