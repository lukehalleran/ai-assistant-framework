# Daemon

**A local-first AI assistant framework with persistent memory, multi-stage RAG, knowledge graph retrieval, agentic tools, retrieval benchmarks, and production packaging.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/lukehalleran/ai-assistant-framework/actions/workflows/tests.yml/badge.svg)](https://github.com/lukehalleran/ai-assistant-framework/actions/workflows/tests.yml)
[![3,663 Tests](https://img.shields.io/badge/tests-3%2C663-brightgreen.svg)](#testing)
[![Docker Ready](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

> ~143K lines of code | 444 files | 3,663 tests | Solo-built over 12 months, part-time

---

## TL;DR

Daemon is a **conversational AI agent framework** I built solo over ~12 months. It is not a chatbot wrapper — it is a full agent architecture with:

- **Persistent hierarchical memory** across 13 ChromaDB collections (episodic, semantic, procedural, summary, meta)
- **Multi-stage RAG pipeline** with intent-parameterized scoring, multi-stage gating (~200ms), and 18 parallel retrieval tasks
- **Knowledge graph** (NetworkX) with entity alias resolution, BFS query expansion, and graph-boosted scoring
- **ReAct-style agentic tool loop** (web search, code sandbox, memory search, image recall, git stats)
- **Retrieval quality benchmarks** with real embeddings, recall@K, and MRR (run before any scoring changes)
- **Prompt section ablation eval system** (246 tests) — snapshot, replay, variant generation, pairwise judging, objective checks
- **Docker deployment** and **desktop installer** (PyInstaller + Inno Setup)

### Why it matters

This project required designing an async retrieval pipeline, a composite scoring function that adapts per-intent, a fact verification gate with staleness cascade, a multi-stage synthesis filter, a formal mathematical specification of the agent, and production-grade packaging — not just wiring an LLM to a vector database. Every subsystem has unit tests, configuration, and documentation.

---

## Best Code to Review

| Area | Path | What to look for |
|------|------|-----------------|
| **Agentic tool loop** | `core/agentic/` | ReAct orchestration, native + XML protocol handling, tool dispatch, context budgeting |
| **Prompt assembly** | `core/prompt/` | Modular builder with 26 conditional sections, intent overrides, token budget management, parallel retrieval |
| **Memory system** | `memory/` | Composite scoring, truth tracking, knowledge graph, entity resolution, fact verification, staleness cascade |
| **Eval system** | `eval/` | Snapshot capture/replay, variant generation, pairwise judging, objective checks — all side-effect-free |
| **Retrieval benchmarks** | `tests/benchmarks/` | Real embeddings (all-MiniLM-L6-v2), 30 seed memories, recall@K + MRR across intent types |

---

## Demo

https://github.com/user-attachments/assets/38e4c2ff-d992-4152-a98e-df58faa54533

---

## What Makes This Different

Most "AI memory" projects store conversation history and do cosine similarity lookups. Daemon implements a full cognitive pipeline with formal mathematical specification, agentic tool use, and self-improving infrastructure.

### Formally Specified Agent Architecture

The entire agent is [formally modeled](docs/FORMAL_MODEL.md) as a composition of mathematical functions — perceive, interpret, expand, remember, plan, act, learn. Every scoring weight, gating threshold, state transition, and prompt section is documented with its mathematical definition and code location.

### Agentic Tool System (ReAct Loop)

When a query needs more than stored memory, Daemon enters a multi-round ReAct loop with 9 tools:

| Tool | Purpose |
|------|---------|
| **Web Search** | Tavily API with query decomposition, caching, and rate limiting |
| **Wolfram Alpha** | Symbolic computation, unit conversions, scientific data |
| **Code Sandbox** | Secure Python execution in ephemeral Firecracker microVMs (E2B) |
| **Memory Search** | Targeted search across 13 ChromaDB collections from within the loop |
| **Memory Expansion** | Drill into a search hit — retrieve chronological neighbors or decompress summaries back to original conversations |
| **Full Document** | Retrieve and reassemble all chunks of an uploaded document by title (fuzzy matching) |
| **Image Recall** | Entity-resolved visual memory search with CLIP cross-modal matching |
| **Git Stats** | Read-only git commands: commit counts, contributors, diff stats with temporal windows |
| **Done** | Signal synthesis complete |

The agent receives a **context inventory** of what RAG already gathered, preventing redundant searches. A 3-tier agentic gate (keyword heuristic -> knowledge graph entity match -> LLM fallback) decides when to enter the loop at all. Memory search takes priority over web search when the LLM detects recall intent.

### Intent-Parameterized Retrieval

A single scoring function serves all 9 query intent types (factual recall, emotional support, temporal recall, project work, etc.) by swapping weight vectors, retrieval limits, and gating thresholds per-intent. No separate pipelines — just different lenses on the same infrastructure. Intent classification is regex-first with no LLM calls, refined by short-term memory analysis when confidence is low.

### Knowledge Graph Query Expansion

Before retrieval, queries are expanded using a NetworkX knowledge graph built from extracted facts. Entity resolution with alias tables (trigram -> bigram -> unigram matching) turns "what about my brother" into a query that also surfaces related entities the user didn't explicitly mention. Expansion candidates are ranked by lateral connectivity and filtered for junk nodes.

### Evidence-Based Truth Scoring & Fact Verification

Memories aren't trusted equally. A TruthScorer tracks confirmation and correction events across sessions, and a CorrectionDetector adjusts scores when the user corrects stored facts. Before any new fact is stored, a FactVerifier runs conflict detection against existing knowledge — producing STORE, STORE_AND_FLAG (supersede old), REJECT, or SKIP verdicts with optional LLM adjudication.

### Memory Staleness Cascade

When a fact is corrected, a ClaimIndex traces which summaries and reflections cited it, then cascades staleness ratios to downstream documents. Stale memories receive scoring penalties with a steep curve (2x multiplier above 80% staleness) and are prefixed with `[HISTORICAL -- PARTIALLY OUTDATED]` in the prompt so the LLM knows to treat them cautiously.

### Visual Memory (CLIP Cross-Modal Search)

Images are ingested through a CLIP + vision-LLM pipeline: encode with OpenCLIP ViT-B/32, caption with a vision model, extract entity tags, store in dual FAISS + ChromaDB indexes. Retrieval uses entity resolution from the knowledge graph to match queries to the correct images — even when CLIP's generic visual similarity can't distinguish between entities.

### Crisis-Aware Escalation FSM

A 4-state finite state machine (VALIDATE -> GROUNDING -> QUIET -> GENTLE) tracks emotional momentum across turns and modifies response length, tone, and token budget. The 250+ keyword tone detector separates personal crisis from world observation.

---

## Engineering Highlights

- Designed an **async retrieval pipeline** with 18 parallel tasks across 13 ChromaDB collections, with per-intent retrieval limits and gating thresholds
- Built **intent-parameterized ranking** — a single composite scoring function that adapts weights, retrieval counts, and gate thresholds per-intent instead of hardcoding separate retrieval paths
- Added **fact verification and staleness penalties** to reduce outdated memory surfacing, with cascading invalidation across downstream summaries and reflections
- Created **retrieval quality benchmarks** with real embeddings (all-MiniLM-L6-v2), 30 seed memories, and 19 test cases measuring recall@K and MRR — run before any scoring or weight changes
- Built a **prompt section ablation eval system** (246 tests) — snapshot capture, deterministic replay, leave-one-out/add-one-in variants, pairwise blind judging, and 5 automated objective checks
- Implemented **entity-resolved visual memory** using CLIP cross-modal search with knowledge graph alias resolution, direct metadata fallback when CLIP can't rank minority entities, and intent-proximity disambiguation for multi-entity queries
- Packaged the app with **Docker**, a **PyInstaller desktop executable**, and an **Inno Setup Windows installer**

---

## Architecture at a Glance

```
User Query
    |
    +- Context Pipeline ---- 8 stages: topic extraction (spaCy NER + LLM fallback) ->
    |                        tone detection (250+ keywords, 4 crisis levels) ->
    |                        intent classification (9 types, regex-first, no LLM) ->
    |                        query rewriting -> STM analysis -> intent refinement
    |
    +- Knowledge Graph ----- Entity resolution (alias tables, trigram matching) +
    |                        BFS expansion (depth 2) + lateral connectivity ranking
    |
    +- Parallel Retrieval -- 18 async tasks across 13 ChromaDB collections
    |                        (conversations, facts, summaries, reflections, wiki,
    |                         obsidian notes, git commits, procedural skills, threads,
    |                         proposals, reference docs, visual memories, ...)
    |
    +- Multi-Stage Gating -- Batch cosine similarity -> Cross-encoder reranking (~200ms)
    |                        Per-intent threshold overrides
    |
    +- Composite Scoring --- 6 weighted factors + graph/anchor/meta bonuses +
    |                        staleness/size/deictic penalties, all parameterized by intent
    |
    +- Prompt Assembly ----- 26 conditional sections, token-budgeted (40K default)
    |                        Two-tier compression: LLM summary + middle-out slicing
    |                        Post-budget floors: min 5 recent conversations guaranteed
    |
    +- Agentic Tool Loop --- ReAct pattern: Think -> Tool -> Observe -> Repeat (max 5 rounds)
    |                        9 tools + context inventory to prevent redundant re-searches
    |
    +- Generation ---------- Standard streaming | Best-of-N | Duel (A vs B + judge) |
    |                        Multi-model ensemble with voter selection
    |
    +- State Transition ---- Per-turn: memory storage + truth events + fact verification +
                                       graph ingestion + escalation FSM
                             Per-session: block summaries + fact extraction (regex + LLM) +
                                          knowledge graph update + thread resolution +
                                          staleness cascade + procedural skills + proposals +
                                          implementation tracking + cross-collection dedup +
                                          session reflection
```

> For the full formal model with mathematical definitions, see [FORMAL_MODEL.md](docs/FORMAL_MODEL.md).
> For a compressed code-level walkthrough, see [PROJECT_SKELETON.md](docs/PROJECT_SKELETON.md).

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
# GUI mode (recommended) -- opens at http://localhost:7860
python main.py

# CLI mode
python main.py cli
```

On first launch, the onboarding wizard collects API keys, communication style preferences, and optional integrations (Obsidian vault, Wikipedia index).

### Docker

```bash
docker-compose up -d
# Access at http://localhost:7860
```

### Desktop Executable

```bash
rm -rf build/ dist/
pyinstaller daemon.spec --clean --noconfirm
# Output: dist/Daemon/Daemon
```

See [docs/BUILD_GUIDE.md](docs/BUILD_GUIDE.md) for the full build guide and Windows installer instructions.

---

## Testing

```bash
python -m pytest -q                          # Fast default suite (~3,633 tests)
python -m pytest tests/integration/ -v       # Integration tests (excluded by default)
python -m pytest tests/benchmarks/ -v        # Retrieval quality benchmarks (real embeddings)
python -m pytest -m "not slow" -q            # Exclude slow tests explicitly
python -m pytest --cov=. --cov-report=html   # With coverage
```

> **Note:** The default `pytest` command excludes integration tests and benchmarks via `pytest.ini`. Benchmark and integration commands are listed separately above. Markers: `slow`, `semantic`, `benchmark`.

**3,663 tests** across 173 test files. Key areas:

- 246 eval system tests (registry, snapshots, replay, variants, corpus, harness, judge, checks)
- 83 web search trigger tests
- 74 intent classification tests
- 70 knowledge graph + 50 graph integration tests
- 66 implementation tracking tests
- 55 proactive context surfacing tests
- 51 escalation tracker + 51 git stats tests
- 47 cross-deduplication + 47 claim tracker tests
- 39 fact verification + 38 graph walk generator tests
- 34 synthesis audit + 33 uncertainty detector tests
- 26 profile namespace + 23 web search citation tests
- 22 visual memory store + 22 visual memory pipeline tests
- 19 retrieval quality benchmarks (30 seed memories, recall@K + MRR)

---

## Privacy & Safety

Daemon is designed for **local-first personal use**. User memory is stored locally by default (ChromaDB + JSON files in `data/`). API keys are loaded from environment variables or `.env` files — `.env.example` contains placeholders only. No user data is transmitted except to the configured LLM API providers for generation.

Personal examples in this README are anonymized. The memory system's `dry_run=True` default for all deletion operations means no user data is auto-deleted without explicit action.

---

## Memory System

Five memory tiers modeled on cognitive architecture, stored across 13 ChromaDB collections:

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
score(d, x) = sum( w_i(intent) * f_i(d, x) ) + bonuses - penalties

where weights w_i are overridden per-intent:
  relevance (0.35) + recency (0.25) + truth (0.20) +
  importance (0.05) + continuity (0.10) + structure (0.05)

bonuses:  anchor_bonus + meta_bonus + graph_bonus (0.05 per entity, capped 0.15)
penalties: staleness (cascade from corrected facts, steep at 80%, capped 0.4) +
           size penalty + deictic drift + analogy penalty
```

Cross-collection deduplication detects both cosine duplicates (threshold 0.92) and fact contradictions (same subject+predicate, different object). Runs dry-run only by default — live deletions require explicit GUI action.

---

## Session Lifecycle

**Session start:** Detects git changes since last session, builds a compact feature inventory, and injects codebase awareness into the first prompt.

**Per-turn:** Stores episodic memory, updates truth scores from correction/confirmation detection, runs fact extraction with verification gate, ingests facts into the knowledge graph, and transitions the escalation FSM.

**Session end (11-step pipeline):**
1. Block summaries with source document backlinks
2. Regex-based fact extraction with conflict verification
3. LLM-assisted fact extraction with batch verification and graph ingestion
4. Behavioral pattern extraction (cross-turn habit detection)
5. Procedural skill extraction
6. Self-improvement proposal generation (filtered against GOALS.md)
7. Cross-collection deduplication scan (dry-run only, never auto-deletes)
8. Open thread processing (resolution detection + extraction + cap enforcement)
9. Implementation tracking (checks if previously proposed features have been built)
10. Synthesis dreaming (3-tier generator pipeline, currently disabled pending validation)
11. Session reflection

**Critical invariant:** No user data is auto-deleted at shutdown.

---

## Knowledge Synthesis Pipeline

> *This is the primary long-term goal. Everything else is infrastructure.*

The idea: use **Markov chain random walks** across large embedded knowledge bases to generate candidate cross-domain connections, then filter through a multi-stage pipeline to surface genuinely novel, non-obvious links between ideas.

**Three generator tiers** (implemented, currently disabled pending grading validation):
- **Tier 0 -- Retrieval** (`synthesis_retriever.py`): Structural query extraction -> FAISS semantic search (40M vectors) -> adversarial evaluation
- **Tier 1 -- Graph Walk** (`graph_walk_generator.py`): Biased Markov random walks across unified personal + Wikidata graph (30K nodes, 7K edges). Hub dampening, cross-domain walk constraint
- **Tier 2 -- Cross-Store** (`synthesis_generator.py`): Entity sampling from facts + wiki_knowledge, LLM bridge articulation

**Filtering pipeline** (7 stages):

| Stage | Gate | What It Does |
|-------|------|-------------|
| 0 | Text Sanity | Min tokens, verb detection, repetition filter |
| 1 | Domain Crossing | Require 2+ distinct domains |
| 2 | Semantic Distance | Endpoint distance in [0.20, 0.90] |
| 3 | External Novelty | Wiki corpus search -- reject if too similar to known content |
| 4 | Internal Novelty | Synthesis memory check -- convergence signal from independent rediscovery |
| 5 | Coherence Judge | LLM rates INVALID/WEAK/MODERATE/STRONG -- minimum MODERATE |
| 6 | Composite Score | Weighted composite >= 0.65 |

**Human audit queue**: A blind review interface presents candidates without generator labels. Two-layer grading: 3 binary screening questions + 1-5 gut-feel slider. Auto-halt if FP rate exceeds 50%.

---

## Project Structure

```
core/                        # Request orchestration
+-- orchestrator.py          # Main controller
+-- best_of_handler.py       # Best-of-N/duel/ensemble generation
+-- context_pipeline.py      # 8-stage query analysis (tone, topic, intent, STM)
+-- intent_classifier.py     # Regex-first intent classification (9 types)
+-- escalation_tracker.py    # Crisis cooldown FSM (4 states)
+-- uncertainty_detector.py  # "I don't know" detection for agentic retry fallback
+-- response_planner.py      # Pre-answer planning + post-answer review gate
+-- agentic/                 # ReAct agentic tool loop
|   +-- controller.py        # ReAct loop orchestration (5-round max)
|   +-- tools.py             # ToolExecutor: dispatch routing for 9 tools
|   +-- formatters.py        # Stateless result formatting
|   +-- types.py             # Tool definitions, state machine, progress events
|   +-- protocols.py         # Native function calling + XML marker handlers
+-- prompt/                  # Modular prompt system
    +-- builder.py           # Orchestrator: parallel task dispatch, intent overrides
    +-- context_gatherer.py  # Mixin compositor (init + properties + utilities)
    +-- gatherer_web.py      # WebSearchMixin
    +-- gatherer_memory.py   # MemoryRetrievalMixin
    +-- gatherer_knowledge.py # KnowledgeRetrievalMixin (notes, docs, wiki, graph, visual)
    +-- formatter.py         # 26-section assembly + attention ordering
    +-- hygiene.py           # Dedup, caps, backfill
    +-- token_manager.py     # Priority-based budget management

memory/                      # 5-tier memory system
+-- memory_coordinator.py    # Thin orchestrator (~500 lines)
+-- memory_retriever.py      # Parallel ChromaDB retrieval
+-- memory_scorer.py         # Composite scoring with intent overrides + graph boost
+-- memory_storage.py        # Persistence with fact verification gate
+-- shutdown_processor.py    # Session-end processing (11 steps)
+-- truth_scorer.py          # Evidence-based truth tracking
+-- graph_memory.py          # NetworkX knowledge graph (JSON persistence)
+-- entity_resolver.py       # Entity alias resolution + relation normalization
+-- fact_extractor.py        # Dual-budget fact extraction (user + entity)
+-- fact_verification.py     # Pre-storage conflict detection
+-- claim_tracker.py         # ClaimIndex reverse index + staleness cascade
+-- cross_deduplicator.py    # Cross-collection dedup + contradiction detection
+-- context_surfacer.py      # Proactive cross-domain insight generation
+-- memory_expander.py       # Temporal expansion + summary drill-down
+-- synthesis_memory.py      # Synthesis persistence + convergence + audit queue
+-- user_profile.py          # Categorized fact storage with namespace canonicalization
+-- user_profile_schema.py   # 6-layer relation categorization + ephemeral TTL
+-- storage/
    +-- multi_collection_chroma_store.py  # ChromaDB wrapper (13 collections)

knowledge/
+-- web_search_manager.py    # Tavily API + query decomposition + caching
+-- clip_manager.py          # OpenCLIP singleton for cross-modal encoding
+-- visual_memory_store.py   # Dual FAISS + ChromaDB visual memory storage
+-- visual_memory_pipeline.py # Image ingestion: CLIP encode -> caption -> entity tag -> store
+-- visual_retrieval.py      # Entity-resolved CLIP text->image retrieval
+-- synthesis_filter.py      # 7-stage synthesis filter pipeline
+-- synthesis_generator.py   # Cross-store sampling + LLM bridge (Tier 2)
+-- synthesis_retriever.py   # Structural query + FAISS search (Tier 0)
+-- graph_walk_generator.py  # Biased Markov walk synthesis (Tier 1)
+-- implementation_detector.py # 4-stage proposal implementation tracking

eval/                        # Prompt ablation & eval system (246 tests)
+-- schema.py                # Pure data models (no Daemon imports)
+-- section_registry.py      # 27-entry canonical section registry
+-- snapshots.py             # Snapshot capture, replay, save/load
+-- variants.py              # LOO, AOI, bundle, reorder variant generation
+-- corpus.py                # 27-query seed corpus (3 per IntentType)
+-- utilization.py           # Per-section presence/token analysis
+-- harness.py               # Batch generation orchestration with resume
+-- judge.py                 # Pairwise blind A/B judging
+-- checks.py                # 5 automated objective checks (no LLM)

tests/                       # 173 test files, 3,663 tests
+-- unit/                    # Component tests (~60 files)
+-- test_eval/               # Eval system tests (246 tests)
+-- benchmarks/              # Retrieval quality (real embeddings, recall@K, MRR)
+-- integration/             # Integration tests (excluded from default run)
```

---

## Prompt Architecture

The prompt is assembled from 26 conditional sections, ordered by transformer attention patterns (high-signal sections at the end for maximum attention weight):

```
[RECENT CONVERSATION]              [KNOWLEDGE GRAPH]
[RELEVANT MEMORIES]                 [UNRESOLVED THREADS]
[RECENT SUMMARIES]                  [PROACTIVE INSIGHTS]
[SEMANTIC SUMMARIES]                [USER PROFILE]
[BACKGROUND KNOWLEDGE]              [VISUAL MEMORIES]
[WEB SEARCH RESULTS]                [ACTIVE FEATURES]
[RELEVANT INFORMATION]              [CODEBASE CHANGES SINCE LAST SESSION]
[RECENT REFLECTIONS]                [TIME CONTEXT]
[SEMANTIC REFLECTIONS]              [TEMPORAL GROUNDING]
[USER'S PERSONAL NOTES]             [SHORT-TERM CONTEXT SUMMARY]
[USER UPLOADED ITEMS]               [CURRENT USER QUERY]
[DAEMON DOCUMENTATION]
[PROJECT COMMIT HISTORY]
[ADAPTIVE WORKFLOWS]
[PROPOSED FEATURES]
```

Token budget allocation is governed by intent. Default budget: 40,000 tokens with two-tier compression. Escalation FSM states override the token budget (300-800 tokens during crisis states).

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

See [config/app_config.py](config/app_config.py) for the full list of 180+ configuration options with defaults.

---

## Current Refactor Target

`core/prompt/builder.py` (1,619 lines) is the main orchestration hub. The next refactor step is to push compression logic, retrieval scheduling, and intent override application into the existing mixin modules, reducing builder.py to pure orchestration.

---

## Design Principles

1. **The filter is the product.** Connection generation is cheap. Identifying which connections are novel and meaningful is the entire value proposition.
2. **Subtractive work > additive work.** Removing systems that don't pull their weight is more valuable than adding new ones.
3. **Measure before changing, measure after changing.** No scoring or retrieval changes without before-and-after benchmark runs.
4. **The personal assistant earns its keep.** It's a daily-use product AND a live testbed for the synthesis pipeline.
5. **Batch scale reveals infrastructure debt.** Every quirk tolerable at conversational volume becomes a critical bug at synthesis volume.

---

## Key Documentation

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE_GUIDE.md](docs/ARCHITECTURE_GUIDE.md) | Narrative architectural walkthrough |
| [FORMAL_MODEL.md](docs/FORMAL_MODEL.md) | Mathematical specification of the complete agent |
| [PROJECT_SKELETON.md](docs/PROJECT_SKELETON.md) | Compressed code-level walkthrough for LLM context |
| [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) | Ultra-compressed API reference |
| [MEMORY_SYSTEM.md](docs/MEMORY_SYSTEM.md) | Memory lifecycle, scoring algorithm, tuning guide |
| [SYNTHESIS_FILTER.md](docs/SYNTHESIS_FILTER.md) | Synthesis pipeline stages, calibration, benchmarks |
| [PROMPT_BUILDING_PIPELINE.md](docs/PROMPT_BUILDING_PIPELINE.md) | Prompt assembly, token budgets, intent overrides |
| [AGENTIC_SEARCH.md](docs/AGENTIC_SEARCH.md) | ReAct loop, tool system, protocol handling |

---

## Resource Usage

| Resource | Idle | Active Query |
|----------|------|--------------|
| CPU | ~5% | 30-50% |
| RAM | ~500MB | ~1.5GB |
| GPU VRAM | - | 2-8GB (optional) |

**Storage:**
- ChromaDB: ~50MB
- Wikipedia FAISS index + metadata: ~14.5GB (optional, [download from HuggingFace](https://huggingface.co/datasets/PaczkiLives/daemon-wiki-faiss))
- Logs: ~1MB/day

