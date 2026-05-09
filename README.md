# Daemon

**A formally-specified cognitive agent with persistent memory, knowledge graph reasoning, agentic tool use, and a long-term vision for automated cross-domain knowledge synthesis.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-3%2C559-brightgreen.svg)](#testing)
[![Docker Ready](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

> 143K lines of code · 444 files · 3,559 tests · Solo-built over 12 months, part-time

---

## What Is This?

Daemon is a **personal AI agent with deep, persistent memory** — not a stateless chatbot wrapper. It remembers your projects, relationships, preferences, emotional state, and conversation history across sessions, then uses a multi-stage retrieval pipeline to surface relevant context for every response.

It is also **a fully agentic system**. When a query requires real-time information, computation, or deeper memory exploration, Daemon enters a ReAct reasoning loop — thinking, selecting tools, observing results, and iterating until it has what it needs. The agent can search the web, run sandboxed Python code, query Wolfram Alpha, search its own memory collections, and drill into compressed conversation summaries to recover original context.

But the agent is also infrastructure for a harder problem: **automated knowledge synthesis**. The long-term goal is to use random walks across large embedded knowledge bases (Wikipedia, arXiv, PubMed) to generate candidate cross-domain connections, then filter them through a multi-stage novelty pipeline. Most output will be noise. The entire value is in the filter — and that filter is what the assistant pipeline tests daily against real retrieval problems.

### Demo

https://github.com/user-attachments/assets/38e4c2ff-d992-4152-a98e-df58faa54533

---

## What Makes This Different

Most "AI memory" projects store conversation history and do cosine similarity lookups. Daemon implements a full cognitive pipeline with formal mathematical specification, agentic tool use, and self-improving infrastructure.

### Formally Specified Agent Architecture

The entire agent is [formally modeled](docs/FORMAL_MODEL.md) as a composition of mathematical functions — perceive, interpret, expand, remember, plan, act, learn. Every scoring weight, gating threshold, state transition, and prompt section is documented with its mathematical definition and code location. The agent state, retrieval function, scoring function, and session lifecycle are all specified with enough precision to verify behavior against the codebase.

### Agentic Tool System (ReAct Loop)

When a query needs more than stored memory, Daemon enters a multi-round ReAct loop with seven tools:

| Tool | Purpose |
|------|---------|
| **Web Search** | Tavily API with query decomposition, caching, and rate limiting |
| **Wolfram Alpha** | Symbolic computation, unit conversions, scientific data |
| **Code Sandbox** | Secure Python execution in ephemeral Firecracker microVMs (E2B) |
| **Memory Search** | Targeted search across 12 ChromaDB collections from within the loop |
| **Memory Expansion** | Drill into a search hit — retrieve chronological neighbors or decompress summaries back to original conversations |
| **Git Stats** | Read-only local git commands: commit counts, contributors, files changed, diff stats with temporal windows |
| **Done** | Signal synthesis complete |

The agent receives a **context inventory** summarizing what RAG already gathered, preventing redundant searches. A 3-tier agentic gate (keyword heuristic → knowledge graph entity match → LLM fallback) decides when to enter the loop at all. The loop runs up to 5 rounds with token-budgeted context compression between iterations. A git stats tool handles temporal repository questions (commit counts, contributors, diff stats) via safe, read-only subprocess calls — no LLM needed for intent parsing.

### Intent-Parameterized Retrieval

A single scoring function serves all 9 query intent types (factual recall, emotional support, temporal recall, project work, etc.) by swapping weight vectors, retrieval limits, and gating thresholds per-intent. No separate pipelines — just different lenses on the same infrastructure. Intent classification is regex-first with no LLM calls, refined by short-term memory analysis when confidence is low.

### Knowledge Graph Query Expansion

Before retrieval, queries are expanded using a NetworkX knowledge graph built from extracted facts. Entity resolution with alias tables (trigram → bigram → unigram matching) turns "what about my brother" into "what about my brother Auggie Mom Flapjack" — surfacing related entities the user didn't explicitly mention. Expansion candidates are ranked by lateral connectivity and filtered for junk nodes.

### Evidence-Based Truth Scoring & Fact Verification

Memories aren't trusted equally. A TruthScorer tracks confirmation and correction events across sessions, and a CorrectionDetector adjusts scores when the user corrects stored facts. Before any new fact is stored, a FactVerifier runs conflict detection against existing knowledge — producing STORE, STORE_AND_FLAG (supersede old), REJECT, or SKIP verdicts with optional LLM adjudication.

### Memory Staleness Cascade

When a fact is corrected, a ClaimIndex traces which summaries and reflections cited it, then cascades staleness ratios to downstream documents. Stale memories receive scoring penalties with a steep curve (2x multiplier above 80% staleness) and are prefixed with `[HISTORICAL — PARTIALLY OUTDATED]` in the prompt so the LLM knows to treat them cautiously.

### Proactive Cross-Domain Insight Surfacing

A ContextSurfacer analyzes the knowledge graph to find non-obvious connections across different domains of the user's life — e.g., linking programming skills to academic coursework, or work schedule patterns to fitness goals. Insights are generated once per session with novelty tracking and cooldown periods to avoid repetition.

### Crisis-Aware Escalation FSM

A 4-state finite state machine (VALIDATE → GROUNDING → QUIET → GENTLE) tracks emotional momentum across turns and modifies response length, tone, and token budget. It distinguishes genuine calming from analytical shifts (PERSPECTIVE), avoiding premature de-escalation. The 250+ keyword tone detector separates personal crisis ("I can't go on") from world observation ("People are suffering due to deportations").

### Self-Improving Proposals Pipeline

At session end, Daemon generates its own improvement proposals filtered against a GOALS.md file. Proposals are stored in ChromaDB with deduplication, then surfaced during project-related conversations through a 10-stage pipeline: project-relevance check → goal-aligned retrieval → keyword dedup → semantic dedup → gate scoring → composite scoring → novelty penalty (git overlap) → diversity selection → optional LLM pairwise ranking. A 4-stage implementation detector (file existence → code content → git history → LLM judgment) tracks which proposals have been built.

### Retrieval Quality Benchmarks

30 seed memories and 19 test cases with real embeddings (all-MiniLM-L6-v2) measure recall@K and MRR across all intent types. No scoring or weight changes ship without before-and-after benchmark runs.

---

## Example Conversation

See [`conversation_logs/gold_star_example.txt`](./conversation_logs/gold_star_example.txt) for a full session transcript. Highlights:

**Context tracking across turns:**
> *User:* "Just got to the gym, going out after with Dad and sister... low battery"
> *(70 minutes later)*
> *User:* "What time is it?"
> *Daemon:* "5:21 PM, December 23rd. Battery holding up okay for the last few sets?"

**Real-time web search with cited sources:**
> *User:* "Search for news about eastern European economies and cite a source"
> *Daemon:* "**Poland** is having a strong moment. According to Bloomberg (Dec 23, 2025), Poland's economy topped $1 trillion this year... Sources: Bloomberg, IMF, Euronews - all from today."

**Intellectual honesty about limitations:**
> *User:* "Can you cite a source about the current state of the economies of eastern Europe?"
> *Daemon:* "Honestly, I can't pull up the specific source right now - the web search results that informed that response aren't in my current context window."

**Deep personal memory retrieval:**
> *User:* "Tell me everything you know about me"
> *Daemon:* Returns comprehensive profile including identity, health history, pets (Flapjack, Mr. Pickle's AirTag), fitness PRs, personal relationships, and ongoing projects.

**Emotional context understanding:**
> *User:* "What does Paczki mean to me?"
> *Daemon:* "Paczki was your lifeline... You've told me she saved your life during a period when you were in a really dark place. She gave you a reason to keep going when you couldn't find one yourself."

---

## Architecture at a Glance

```
User Query
    │
    ├─ Context Pipeline ──── 8 stages: topic extraction (spaCy NER + LLM fallback) →
    │                        tone detection (250+ keywords, 4 crisis levels) →
    │                        intent classification (9 types, regex-first, no LLM) →
    │                        query rewriting → STM analysis → intent refinement
    │
    ├─ Knowledge Graph ───── Entity resolution (alias tables, trigram matching) +
    │                        BFS expansion (depth 2) + lateral connectivity ranking
    │
    ├─ Parallel Retrieval ── 18 async tasks across 12 ChromaDB collections
    │                        (conversations, facts, summaries, reflections, wiki,
    │                         obsidian notes, git commits, procedural skills, threads,
    │                         proposals, reference docs, proactive insights, ...)
    │
    ├─ Multi-Stage Gating ── Batch cosine similarity → Cross-encoder reranking (~200ms)
    │                        Per-intent threshold overrides
    │
    ├─ Composite Scoring ─── 6 weighted factors + graph/anchor/meta bonuses +
    │                        staleness/size/deictic penalties, all parameterized by intent
    │                        Temporal-aware recency decay for time-windowed queries
    │
    ├─ Prompt Assembly ───── 26 conditional sections, token-budgeted (40K default)
    │                        Two-tier compression: LLM summary (≥3x oversized) +
    │                        middle-out slicing (mildly oversized)
    │                        Post-budget floors: min 5 recent conversations guaranteed
    │                        Attention-aware ordering (high-signal sections at end)
    │                        Codebase diff injection on session start
    │
    ├─ Agentic Tool Loop ─── ReAct pattern: Think → Tool → Observe → Repeat (max 5 rounds)
    │                        Tools: Tavily + Wolfram Alpha + E2B sandbox +
    │                        memory search + memory expansion + git stats + done
    │                        Context inventory prevents redundant re-searches
    │                        Budget-enforced context accumulation (oldest rounds trimmed)
    │
    ├─ Generation ─────────── Standard streaming | Best-of-N | Duel (A vs B + judge) |
    │                        Multi-model ensemble with voter selection
    │
    └─ State Transition ──── Per-turn: memory storage + truth events + fact verification +
                                       graph ingestion + escalation FSM
                             Per-session: block summaries + fact extraction (regex + LLM) +
                                          knowledge graph update + thread extraction/resolution +
                                          staleness cascade + procedural skills + proposals +
                                          implementation tracking + cross-collection dedup (dry-run) +
                                          session reflection
```

> For the full formal model with mathematical definitions, see [FORMAL_MODEL.md](docs/FORMAL_MODEL.md).
> For a compressed code-level walkthrough, see [PROJECT_SKELETON.md](docs/PROJECT_SKELETON.md).

---

## Memory System

Five memory tiers modeled on cognitive architecture, stored across 12 ChromaDB collections:

| Tier | What It Stores | Retrieval Bias |
|------|---------------|----------------|
| **Episodic** | Raw conversation turns | Recency |
| **Semantic** | Extracted fact triples + Wikipedia (6.5M articles) | Truth score + relevance |
| **Procedural** | Git commits + learned skill patterns | Pattern matching |
| **Summary** | LLM-compressed conversation blocks (with source backlinks) | Relevance |
| **Meta** | Session reflections + open threads + code proposals + proactive insights | Priority / urgency |
| **Synthesis** | Accepted cross-domain connections with convergence tracking | Composite score + convergence |

The scoring function for ranking retrieved memories:

```
σ_ι(d, x) = Σ wᵢ(ι) · fᵢ(d, x) + bonuses - penalties

where weights wᵢ are overridden per-intent ι:
  relevance (0.35) + recency (0.25) + truth (0.20) +
  importance (0.05) + continuity (0.10) + structure (0.05)

bonuses:  anchor_bonus + meta_bonus + graph_bonus (0.05 per entity, capped 0.15)
penalties: staleness (cascade from corrected facts, steep at 80%, capped 0.4) +
           size penalty + deictic drift + analogy penalty

temporal-aware recency (for TEMPORAL_RECALL with anchor α hours):
  within window:  1.0 - (age/α) × 0.3      (gentle decay)
  outside window: 0.7 / (1 + λ(age - α))   (standard decay)
```

Cross-collection deduplication detects both cosine duplicates (threshold 0.92) and fact contradictions (same subject+predicate, different object). Runs dry-run only by default — live deletions require explicit GUI action.

---

## Session Lifecycle

Daemon doesn't just chat — it performs structured cognitive maintenance at session boundaries.

**Session start:** Detects git changes since last session (committed + uncommitted), builds a compact feature inventory, and injects codebase awareness into the first prompt.

**Per-turn:** Stores episodic memory, updates truth scores from correction/confirmation detection, runs optional fact extraction with verification gate, ingests facts into the knowledge graph, and transitions the escalation FSM.

**Session end (11-step pipeline):**
1. Block summaries — LLM-compressed conversation history with source document backlinks
2. Regex-based fact extraction (last 10 turns) with conflict verification
3. LLM-assisted fact extraction (last 12 turns) with batch verification and graph ingestion
4. Behavioral pattern extraction — cross-turn habit detection (e.g., "codes at the gym")
5. Procedural skill extraction (learned trigger → action patterns)
6. Self-improvement proposal generation (filtered against GOALS.md)
7. Cross-collection deduplication scan (dry-run only, never auto-deletes)
8. Open thread processing — resolution detection + new thread extraction + cap enforcement
9. Implementation tracking — checks if previously proposed features have been built
10. Synthesis dreaming — 3-tier generator pipeline (retrieval, graph walk, cross-store), currently disabled pending grading validation
11. Session reflection — LLM meta-reflection stored for future context

**Critical invariant:** No user data is auto-deleted at shutdown.

---

## Agentic Tool System

The agentic system uses a ReAct (Reason + Act) pattern with dual protocol support — native function calling for API models (OpenAI, Anthropic) and XML markers for local models.

**Triggering:** A 3-tier gate decides when to enter the agentic loop:
1. Keyword heuristic — computation or memory-recall triggers
2. Entity match — query terms checked against knowledge graph alias index
3. LLM fallback — catches structurally obvious recall/search queries the heuristics missed

**Tool details:**

| Tool | Implementation | Key Feature |
|------|---------------|-------------|
| Web Search | Tavily API | Query decomposition for multi-entity queries, 72hr ChromaDB cache, daily credit tracking |
| Wolfram Alpha | LLM API | Token bucket rate limiting, MD5-keyed result cache, assumption parsing |
| Code Sandbox | E2B Firecracker microVMs | Persistent sessions (variables survive across loop iterations), NumPy/Pandas/SciPy pre-installed |
| Memory Search | ChromaDB (12 collections) | Per-collection search descriptions, diversity tracking prevents redundant queries |
| Memory Expansion | Temporal + backlink strategies | For summaries: recovers original conversations via source_doc_ids or temporal anchors. For other collections: ±N chronological neighbors. Session-gated, cached. |
| Git Stats | GitStatsManager | Keyword-based intent parsing → safe git subprocess calls. Temporal windows ("this week", "today"). Read-only subcommand allowlist. |

---

## Open Thread Tracking

Daemon tracks commitments, deadlines, and unresolved questions across sessions using a ChromaDB-backed thread store:

- **Extraction:** LLM identifies new threads from conversation (commitments, deadlines, open questions)
- **Resolution detection:** LLM checks if existing threads were addressed during the session
- **Priority scoring:** Threads ranked by urgency and surfaced in the prompt as `[UNRESOLVED THREADS]`
- **Cap enforcement:** Lowest-priority threads pruned when over the maximum (only auto-deletion in the system)
- **Prompt injection:** Active threads appear in the high-attention zone of the prompt

---

## Prompt Architecture

The prompt is assembled from 26 conditional sections, ordered by transformer attention patterns (high-signal, low-token sections placed at the end for maximum attention weight):

```
[RECENT CONVERSATION]              [KNOWLEDGE GRAPH]
[RELEVANT MEMORIES]                 [UNRESOLVED THREADS]
[RECENT SUMMARIES]                  [PROACTIVE INSIGHTS]
[SEMANTIC SUMMARIES]                [USER PROFILE]
[BACKGROUND KNOWLEDGE]              [ACTIVE FEATURES]
[WEB SEARCH RESULTS]                [CODEBASE CHANGES SINCE LAST SESSION]
[RELEVANT INFORMATION]              [TIME CONTEXT]
[RECENT REFLECTIONS]                [TEMPORAL GROUNDING]
[SEMANTIC REFLECTIONS]              [SHORT-TERM CONTEXT SUMMARY]
[USER'S PERSONAL NOTES]             [CURRENT USER QUERY]
[USER UPLOADED ITEMS]
[DAEMON DOCUMENTATION]
[PROJECT COMMIT HISTORY]
[ADAPTIVE WORKFLOWS]
[PROPOSED FEATURES]
```

Token budget allocation is governed by intent — e.g., CASUAL_SOCIAL reduces max memories, EMOTIONAL_SUPPORT increases continuity weight. Default budget: 40,000 tokens with two-tier compression: heavily oversized items (≥3x over limit) get LLM summary, while mildly oversized items use middle-out character slicing (preserves start and end, compresses middle). Escalation FSM states override the token budget (300–800 tokens during crisis states).

---

## Knowledge Synthesis Pipeline

> *This is the primary long-term goal. Everything else is infrastructure.*

The idea: use **Markov chain random walks** across large embedded knowledge bases to generate candidate cross-domain connections, then filter through a multi-stage pipeline to surface genuinely novel, non-obvious links between ideas.

**Three generator tiers** (implemented, currently disabled pending grading validation):
- **Tier 0 — Retrieval** (`synthesis_retriever.py`): Structural query extraction → FAISS semantic search (40M vectors) → adversarial evaluation
- **Tier 1 — Graph Walk** (`graph_walk_generator.py`): Biased Markov random walks across unified personal + Wikidata graph (30K nodes, 7K edges, 41 bridges). Hub dampening, cross-domain walk constraint
- **Tier 2 — Cross-Store** (`synthesis_generator.py`): Entity sampling from facts + wiki_knowledge, LLM bridge articulation

**Data sources**: Wikipedia (41M vectors via FAISS IVFPQ, embedded and indexed), personal knowledge graph (facts, entities, relations). arXiv/PubMed planned for future.

**Filtering pipeline** (7 stages — `knowledge/synthesis_filter.py`):

| Stage | Gate | Cost | What It Does |
|-------|------|------|-------------|
| 0 | Text Sanity | ~0ms | Min tokens, verb detection, repetition filter |
| 1 | Domain Crossing | ~1ms | Require ≥2 distinct domains |
| 2 | Semantic Distance | ~5ms | Endpoint distance in [0.20, 0.90] — not too close, not nonsensical |
| 3 | External Novelty | ~10ms | Wiki corpus search — reject if similarity > 0.80 (already known) |
| 4 | Internal Novelty | ~10ms | Synthesis memory check — new paths to same insight pass (convergence signal) |
| 5 | Coherence Judge | ~500ms | LLM rates INVALID/WEAK/MODERATE/STRONG — minimum MODERATE to pass |
| 6 | Composite Score | ~0ms | Weighted composite ≥ 0.65: coherence(0.30) + novelty(0.40) + distance(0.15) + structural(0.15) |

**Convergence detection**: The real signal isn't raw frequency — it's independent rediscovery. When different random walks find the same insight via different paths, that's evidence of a genuine connection. The pipeline tracks unique walk paths and concept pairs per insight, promoting to CONVERGING status at 3+ independent paths from 2+ distinct source pairs.

**Synthesis memory**: Accepted insights are stored in a dedicated ChromaDB collection (`synthesis_results`) with full metadata — coherence rating, novelty scores, convergence tracking. Duplicate detection prevents redundant storage; instead, new paths to existing insights update convergence metadata. Provisional graph edges (weight=0.0) are created on acceptance and mature on independent rediscovery.

**Human audit queue**: A blind review interface (GUI "Synthesis" tab) presents candidates without generator labels. Two-layer grading: 3 binary screening questions (changes thinking? mechanism real? heard before?) + 1-5 gut-feel slider. Auto-halt threshold: synthesis dreaming disabled if FP rate exceeds 50% with 10+ graded results.

**Current status**: All three generators are disabled in config.yaml pending sufficient grading data. The eval system (246 tests) provides infrastructure for validating re-enablement.

The personal assistant earns its keep as a daily testbed for every component in this pipeline — gating, scoring, reranking, truth scoring, deduplication — at conversational scale before batch synthesis workloads stress them at knowledge-base scale.

---

## Quick Start

### Prerequisites

- Python 3.11+
- 4GB RAM minimum (8GB recommended)
- At least one LLM API key (OpenAI, Anthropic, DeepSeek, or Google)

### Installation

```bash
git clone https://github.com/lukehalleran/ai-assistant-framework.git
cd ai-assistant-framework

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Set your API key (at least one provider)
export OPENAI_API_KEY=sk-your-key-here
# Or: export ANTHROPIC_API_KEY=sk-ant-...
# Or create .env file with your keys
```

### Launch

```bash
# GUI mode (recommended) — opens at http://localhost:7860
python main.py

# CLI mode
python main.py cli

# First-run wizard (for testing wizard flow)
python main.py wizard
```

On first launch, the onboarding wizard collects:
- **Mode** (Personal or Developer — gates advanced features)
- **API keys** (OpenRouter required; Tavily, Wolfram, E2B optional)
- **Communication style** (warm/balanced/direct)
- **Identity** (name, pronouns)
- **Obsidian vault path** (strongly recommended for daily summaries)
- **Wikipedia index path** (optional ~2GB download for cross-domain knowledge)

### Docker

```bash
# Build and run
docker-compose up -d

# Access at http://localhost:7860
docker-compose logs -f daemon-gui
```

### Desktop Executable & Installer

```bash
# Build executable
rm -rf build/ dist/
pyinstaller daemon.spec --clean --noconfirm

# Build Windows installer (requires Inno Setup 6)
cd installer && build_installer.bat
# Output: installer/output/DaemonSetup-1.0.0.exe (~293MB)
```

See [docs/BUILD_GUIDE.md](docs/BUILD_GUIDE.md) for full build guide, test checklist, and troubleshooting.

---

## Testing

```bash
python -m pytest -q                     # Run full suite
python -m pytest tests/benchmarks/      # Retrieval quality benchmarks
python -m pytest -m "not slow"          # Exclude slow tests
python -m pytest --cov=. --cov-report=html  # With coverage
```

**3,559 tests** across 173 test files — unit, integration, eval, and retrieval quality benchmarks. Key areas:

- 246 eval system tests (registry, snapshots, replay, variants, corpus, harness, judge, checks)
- 74 intent classification tests
- 70 knowledge graph tests
- 66 implementation tracking tests
- 55 proactive context surfacing tests
- 51 escalation tracker + 51 git stats tests
- 50 graph integration tests
- 47 cross-deduplication + 47 claim tracker tests
- 44 file access manager tests
- 39 fact verification + 38 graph walk generator tests
- 34 synthesis audit + 33 uncertainty detector tests
- 26 profile namespace + 23 web search citation tests
- 21 source excerpt pipeline + 20 response planner tests
- 19 retrieval quality benchmarks (30 seed memories, real embeddings, recall@K + MRR)

---

## Project Stats

```
Lines of code:          ~143,000 (across all source files)
Python lines:           ~139,000
Total files:            444
Python files:           392
Test files:             173
Test functions:         3,559
ChromaDB collections:   12
Prompt sections:        26 (conditional)
Intent types:           9
Retrieval tasks:        18 (parallel)
Memory tiers:           5 + synthesis
Agentic tools:          7 + done signal
Gating latency:         ~200ms
Config options:         180+
```

---

## Project Structure

```
core/                        # Request orchestration
├── orchestrator.py          # Main controller
├── best_of_handler.py       # Best-of-N/duel/ensemble generation
├── context_pipeline.py      # 8-stage query analysis (tone, topic, intent, STM)
├── intent_classifier.py     # Regex-first intent classification (9 types)
├── escalation_tracker.py    # Crisis cooldown FSM (4 states)
├── correction_detector.py   # User correction/confirmation detection
├── git_stats_manager.py     # Read-only git repo stats for agentic loop
├── response_parser.py       # Thinking block + tag stripping utilities
├── uncertainty_detector.py   # "I don't know" detection for agentic retry fallback
├── response_planner.py      # Pre-answer planning + post-answer review gate
├── agentic/                 # ReAct agentic tool loop
│   ├── controller.py        # ReAct loop orchestration (5-round max)
│   ├── tools.py             # ToolExecutor: dispatch routing + execution for all tools
│   ├── formatters.py        # AgenticFormatter: pure stateless result formatting
│   ├── types.py             # Tool definitions, state machine, progress events
│   └── protocols.py         # Native function calling + XML marker handlers
└── prompt/                  # Modular prompt system
    ├── builder.py           # Thin orchestrator: parallel task dispatch, intent overrides
    ├── context_gatherer.py  # Mixin compositor (init + properties + utilities)
    ├── gatherer_web.py      # WebSearchMixin: web search trigger + retrieval
    ├── gatherer_memory.py   # MemoryRetrievalMixin: conversations, summaries, facts, profile
    ├── gatherer_knowledge.py # KnowledgeRetrievalMixin: notes, docs, wiki, graph, threads
    ├── formatter.py         # PromptFormatter: 26-section assembly + attention ordering
    ├── hygiene.py           # ContentHygiene: dedup, caps, backfill
    ├── token_manager.py     # Priority-based budget management
    └── proposal_filter.py   # 10-stage proposal injection pipeline

memory/                      # 5-tier memory system
├── memory_coordinator.py    # Thin orchestrator (~500 lines, 24 delegation methods)
├── memory_retriever.py      # Parallel ChromaDB retrieval pipeline
├── memory_scorer.py         # Composite scoring with intent overrides + graph boost
├── memory_storage.py        # Persistence with fact verification gate
├── shutdown_processor.py    # Session-end processing pipeline (11 major steps)
├── truth_scorer.py          # Evidence-based truth tracking
├── graph_memory.py          # NetworkX knowledge graph (JSON persistence)
├── entity_resolver.py       # Entity alias resolution + relation normalization
├── fact_extractor.py        # Dual-budget fact extraction (user + entity)
├── fact_verification.py     # Pre-storage conflict detection (STORE/FLAG/REJECT/SKIP)
├── claim_tracker.py         # ClaimIndex reverse index + staleness cascade
├── cross_deduplicator.py    # Cross-collection dedup + contradiction detection
├── context_surfacer.py      # Proactive cross-domain insight generation
├── memory_expander.py       # Temporal expansion + summary drill-down
├── synthesis_memory.py      # Synthesis results persistence + convergence + audit queue
├── user_profile.py          # Categorized fact storage with namespace canonicalization
├── user_profile_schema.py   # 5-layer relation categorization, safe aliases, ephemeral TTL
├── procedural_skill.py      # ProceduralSkill dataclass + SkillCategory enum
├── thread_store.py          # ChromaDB-backed thread CRUD + priority ranking
├── thread_extractor.py      # LLM-based thread extraction + resolution detection
└── storage/
    └── multi_collection_chroma_store.py  # ChromaDB wrapper (12 collections)

processing/
└── gate_system.py           # FAISS → cosine → cross-encoder pipeline

models/
├── model_manager.py         # Multi-provider LLM (OpenAI, Anthropic, DeepSeek, Google, local)
└── tokenizer_manager.py

knowledge/
├── web_search_manager.py    # Tavily API + query decomposition + caching
├── wolfram_manager.py       # Wolfram Alpha LLM API + rate limiting
├── sandbox_manager.py       # E2B Firecracker sandbox + persistent sessions
├── git_memory.py            # Git commit extraction + procedural indexing
├── obsidian_manager.py      # Obsidian vault sync + multimodal image support
├── reference_docs_manager.py # Auto-seeded docs/ knowledge
├── wiki_manager.py          # FAISS-indexed Wikipedia (6.5M articles)
├── implementation_detector.py # 4-stage proposal implementation tracking
├── synthesis_models.py      # Synthesis pipeline data models + enums
├── synthesis_filter.py      # 7-stage synthesis filter pipeline
├── synthesis_generator.py   # Cross-store sampling + LLM bridge articulation (Tier 2)
├── synthesis_retriever.py   # Structural query + FAISS search + adversarial eval (Tier 0)
├── graph_walk_generator.py  # Biased Markov walk synthesis generator (Tier 1)
├── wiki_tracker.py          # Session-level Wikipedia article tracking
├── wiki_enrichment.py       # Shutdown: tracked wiki articles → graph nodes
├── wikidata_resolver.py     # Personal ↔ Wikidata entity resolution
└── wikidata_models.py       # Pydantic models for Wikidata import

utils/
├── tone_detector.py         # 250+ weighted keywords, 4 crisis levels
├── topic_manager.py         # 3-stage: heuristic → spaCy NER → LLM fallback
├── web_search_trigger.py    # 3-tier agentic gate (keyword → entity → LLM)
├── daily_notes_generator.py # Auto daily summaries with LLM tag generation
├── weekly_notes_generator.py # Weekly aggregation + folder organization
└── monthly_notes_generator.py # Monthly summaries with folder migration

gui/                         # Gradio web interface
├── launch.py                # Dark theme, tab layout, startup hooks
├── handlers.py              # Chat streaming + agentic routing + fast mode
├── wizard.py                # 14-step onboarding wizard (mode, keys, vault, wiki index)
└── tabs/                    # Modular tab UI
    ├── proposals.py         # Code proposal management tab
    ├── settings.py          # Settings tab
    └── synthesis.py         # Synthesis audit queue + blind review

eval/                        # Prompt section ablation & eval system (246 tests)
├── schema.py                # Pure data models (no Daemon imports)
├── section_registry.py      # 27-entry canonical section registry
├── snapshots.py             # Snapshot capture, replay, save/load
├── variants.py              # LOO, AOI, bundle, reorder variant generation
├── corpus.py                # 27-query seed corpus (3 per IntentType)
├── utilization.py           # Per-section presence/token analysis
├── harness.py               # Batch generation orchestration with resume
├── judge.py                 # Pairwise blind A/B judging with position randomization
├── checks.py                # 5 automated objective checks (no LLM)
├── no_store_generation.py   # Side-effect-free LLM generation
└── persistence_guard.py     # State fingerprinting (ChromaDB + JSON files)

tests/                       # 173 test files
├── unit/                    # Component tests (~60 files)
├── test_eval/               # Eval system tests (246 tests)
├── benchmarks/              # Retrieval quality (real embeddings, recall@K, MRR)
└── fixtures/                # Benchmark seed data (30 memories, 19 test cases)
```

---

## Wikipedia Knowledge Base (Optional)

Daemon supports Wikipedia-scale knowledge retrieval with 6.5M+ articles (~41M vectors) semantically indexed via FAISS IVFPQ.

### Pre-built Index (Recommended)

Download the pre-built index from HuggingFace (~14.5 GB):

```bash
pip install huggingface_hub

huggingface-cli download PaczkiLives/daemon-wiki-faiss \
    --repo-type dataset \
    --local-dir ~/daemon-wiki-data/wiki_data

export WIKI_DATA_ROOT=~/daemon-wiki-data
```

**Runtime requirements:** ~2.6 GB RAM (index + embedding model), ~14.5 GB disk. No GPU needed.

### Build From Scratch

If you prefer to build the index yourself:

```bash
python scripts/build_faiss_index.py

# Pipeline: download dump → parse XML → chunk (512 tokens) →
#   embed (all-MiniLM-L6-v2) → build FAISS IVFPQ index → store metadata
```

**Build requirements:** ~102 GB disk for raw dump + ~60 GB for intermediate embeddings.

The assistant works without Wikipedia — features gracefully degrade when the index is not present.

---

## Configuration

Core settings via `config.yaml` with environment variable overrides:

```yaml
# config.yaml
daemon:
  version: v4
  data_dir: ./data

models:
  default: sonnet-4.5
  default_max_tokens: 9984

memory:
  corpus_max_entries: 2000
  prompt_max_recent: 15
  prompt_max_mems: 15
  prompt_max_facts: 30

gating:
  cosine_similarity_threshold: 0.15
  score_weights:
    relevance: 0.30
    recency: 0.22
    truth: 0.18
    importance: 0.05
    continuity: 0.10
    structure: 0.05
    topic_match: 0.10
```

```bash
# Environment variables (required)
OPENAI_API_KEY=sk-...      # Or ANTHROPIC_API_KEY, etc.

# Environment overrides (optional)
CORPUS_MAX_ENTRIES=2000
PROMPT_TOKEN_BUDGET=40000
CHROMA_DEVICE=cpu          # Or "cuda" for GPU
```

See [config/app_config.py](config/app_config.py) for the full list of 180+ configuration options with defaults.

---

## Troubleshooting

**"No such file or directory: corpus_v4.json"**
```bash
mkdir -p data && echo '[]' > data/corpus_v4.json
```

**"ChromaDB dimension mismatch error"**
```bash
rm -rf data/chroma_db_v4    # Restart will rebuild
```

**"CUDA out of memory"**
```bash
export CHROMA_DEVICE=cpu
```

**"Token limit exceeded"**
```bash
export PROMPT_TOKEN_BUDGET=8000
```

**Debug mode:**
```bash
export LOG_LEVEL=DEBUG
python main.py
```

---

## Key Documentation

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE_GUIDE.md](docs/ARCHITECTURE_GUIDE.md) | Narrative architectural walkthrough — how every subsystem works and connects |
| [FORMAL_MODEL.md](docs/FORMAL_MODEL.md) | Mathematical specification of the complete agent |
| [PROJECT_SKELETON.md](docs/PROJECT_SKELETON.md) | Compressed code-level walkthrough for LLM context |
| [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) | Ultra-compressed API reference — function signatures and core logic |
| [MEMORY_SYSTEM.md](docs/MEMORY_SYSTEM.md) | Memory lifecycle, scoring algorithm, fact pipeline, tuning guide |
| [SYNTHESIS_FILTER.md](docs/SYNTHESIS_FILTER.md) | Synthesis pipeline stages, calibration, benchmark results |
| [PROMPT_BUILDING_PIPELINE.md](docs/PROMPT_BUILDING_PIPELINE.md) | Prompt assembly, token budgets, section ordering, intent overrides |
| [AGENTIC_SEARCH.md](docs/AGENTIC_SEARCH.md) | ReAct loop, tool system, protocol handling, provenance |
| [THINKING_BLOCKS_IMPLEMENTATION.md](docs/THINKING_BLOCKS_IMPLEMENTATION.md) | Three-layer thinking separation defense |
| [GOALS.md](docs/GOALS.md) | Active goals, principles, and recent completions |
| [eval/PLAN.md](eval/PLAN.md) | Prompt eval system architecture (8-phase plan) |
| [BUILD_GUIDE.md](docs/BUILD_GUIDE.md) | Desktop executable build guide |

---

## Design Principles

1. **The filter is the product.** Connection generation is cheap. Identifying which connections are novel and meaningful is the entire value proposition.
2. **Subtractive work > additive work.** The codebase must get leaner before it gets bigger. Removing systems that don't pull their weight is more valuable than adding new ones.
3. **Measure before changing, measure after changing.** No scoring or retrieval changes without before-and-after benchmark runs.
4. **The personal assistant earns its keep.** It's a daily-use product AND a live testbed for the synthesis pipeline. Both purposes matter.
5. **Batch scale reveals infrastructure debt.** Every quirk tolerable at conversational volume becomes a critical bug at synthesis volume.

---

## Resource Usage

| Resource | Idle | Active Query |
|----------|------|--------------|
| CPU | ~5% | 30-50% |
| RAM | ~500MB | ~1.5GB |
| GPU VRAM | - | 2-8GB (optional) |

**Storage:**
- Corpus JSON: ~10MB (2000 conversations)
- ChromaDB: ~50MB
- Wikipedia FAISS index + metadata: ~14.5GB (optional, [download from HuggingFace](https://huggingface.co/datasets/PaczkiLives/daemon-wiki-faiss))
- Logs: ~1MB/day
