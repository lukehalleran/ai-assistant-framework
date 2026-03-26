# Daemon

**A cognitive RAG agent with persistent memory, knowledge graph reasoning, and a long-term vision for automated cross-domain knowledge synthesis.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-2%2C800%2B-brightgreen.svg)](#testing)
[![Docker Ready](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

> 102K lines of Python · 284 files · 2,800+ tests · Solo-built over 11 months, part-time

---

## What Is This?

Daemon is a **personal AI assistant with deep, persistent memory** — not a stateless chatbot wrapper. It remembers your projects, relationships, preferences, emotional state, and conversation history across sessions, then uses a multi-stage retrieval pipeline to surface relevant context for every response.

But the personal assistant is also infrastructure for a harder problem: **automated knowledge synthesis**. The long-term goal is to use random walks across large embedded knowledge bases (Wikipedia, arXiv, PubMed) to generate candidate cross-domain connections, then filter them through a multi-stage novelty pipeline. Most output will be noise. The entire value is in the filter — and that filter is what the assistant pipeline tests daily against real retrieval problems.

### Demo

https://github.com/user-attachments/assets/38e4c2ff-d992-4152-a98e-df58faa54533

---

## What Makes This Different

Most "AI memory" projects store conversation history and do cosine similarity lookups. Daemon implements a full cognitive pipeline:

**Formalized retrieval architecture.** The entire agent is [formally modeled](docs/FORMAL_MODEL.md) as a composition of mathematical functions — perceive, interpret, expand, remember, plan, act, learn. Every scoring weight, gating threshold, and state transition is documented with its mathematical definition and code location.

**Intent-parameterized scoring.** A single scoring function serves all 9 query intent types (factual recall, emotional support, temporal recall, etc.) by swapping weight vectors and retrieval limits per-intent. No separate pipelines — just different lenses on the same infrastructure.

**Knowledge graph query expansion.** Before retrieval, queries are expanded using a NetworkX knowledge graph built from extracted facts. Entity resolution with alias tables turns "what about my brother" into "what about my brother Auggie Mom Flapjack" — surfacing related entities the user didn't explicitly mention.

**Evidence-based truth scoring.** Memories aren't trusted equally. A TruthScorer tracks confirmation/correction events across sessions, and a CorrectionDetector adjusts scores when the user corrects stored facts. This replaced an earlier access-count system that created echo chambers.

**Crisis-aware escalation FSM.** A 4-state finite state machine (VALIDATE → GROUNDING → QUIET → GENTLE) tracks emotional momentum across turns and modifies response length, tone, and token budget. It distinguishes genuine calming from analytical shifts (PERSPECTIVE), avoiding premature de-escalation.

**Retrieval quality benchmarks.** 30 seed memories and 19 test cases with real embeddings (all-MiniLM-L6-v2) measure recall@K and MRR across all intent types. No scoring or weight changes ship without before-and-after benchmark runs.

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
    ├─ Context Pipeline ──── 8 stages: topic extraction → tone detection →
    │                        intent classification (9 types) → query rewriting → STM analysis
    │
    ├─ Knowledge Graph ───── Entity resolution + BFS expansion (depth 2)
    │                        Lateral connectivity ranking, junk node prevention
    │
    ├─ Parallel Retrieval ── 18 async tasks across 11 ChromaDB collections
    │                        (conversations, facts, summaries, reflections, wiki,
    │                         obsidian notes, git commits, procedural skills, threads, ...)
    │
    ├─ Multi-Stage Gating ── FAISS → Cosine similarity → Cross-encoder reranking (~200ms)
    │
    ├─ Composite Scoring ─── 6 weighted factors + bonuses/penalties, parameterized by intent
    │                        Temporal-aware recency decay for time-windowed queries
    │
    ├─ Prompt Assembly ───── ~24 conditional sections, token-budgeted (15K default)
    │                        Attention-aware ordering (high-signal sections at end)
    │
    ├─ Agentic Search ────── ReAct loop: Tavily + Wolfram Alpha + E2B sandbox
    │                        Context inventory prevents redundant re-searches
    │
    └─ State Transition ──── Per-turn: memory storage + truth events + staleness cascade + escalation FSM
                             Per-session: block summaries + fact extraction + knowledge graph
                             ingestion + thread extraction + staleness indexing + cross-collection dedup (dry-run)
```

> For the full formal model with mathematical definitions, see [FORMAL_MODEL.md](docs/FORMAL_MODEL.md).
> For a compressed code-level walkthrough, see [PROJECT_SKELETON.md](docs/PROJECT_SKELETON.md).

---

## Memory System

Five memory tiers modeled on cognitive architecture, stored across 11 ChromaDB collections:

| Tier | What It Stores | Retrieval Bias |
|------|---------------|----------------|
| **Episodic** | Raw conversation turns | Recency |
| **Semantic** | Extracted fact triples + Wikipedia | Truth score + relevance |
| **Procedural** | Git commits + learned skill patterns | Pattern matching |
| **Summary** | LLM-compressed conversation blocks | Relevance |
| **Meta** | Session reflections + open threads + code proposals | Priority / urgency |

The scoring function for ranking retrieved memories:

```
σ(d, x) = Σ wᵢ · fᵢ(d, x) + structure + bonuses - penalties

where weights wᵢ are overridden per-intent:
  relevance (0.35) + recency (0.25) + truth (0.20) +
  importance (0.05) + continuity (0.10) + topic_match (0.00)

structure: density alignment bonus (up to ~0.15)
bonuses: anchor_bonus + meta_bonus + graph_bonus (0.05 per related entity, capped 0.15)
penalties: size penalty, deictic drift, staleness penalty (ratio*0.15, steep at 80%, capped 0.4)
```

Cross-collection deduplication detects both cosine duplicates (threshold 0.92) and fact contradictions (same subject+predicate, different object). Runs dry-run only by default — live deletions require explicit GUI action.

---

## Tone Detection

Crisis-aware tone detection adapts response depth using 250+ weighted keywords:

| Level | Trigger Examples | Response Style |
|-------|------------------|----------------|
| **HIGH** | "I want to die", "can't go on" | Full therapeutic presence, multi-paragraph |
| **MEDIUM** | "panic attack", "falling apart" | 2-3 paragraphs, supportive |
| **CONCERN** | "really anxious", "scared" | 2-4 sentences, brief validation |
| **CONVERSATIONAL** | Status updates, technical questions | 1-3 sentences, direct |

The system distinguishes personal crisis from world event observation:
- "I'm suffering and can't go on" → HIGH (personal crisis)
- "People are suffering due to deportations" → CONVERSATIONAL (world observation)

---

## Knowledge Synthesis Vision

> *This is the primary long-term goal. Everything else is infrastructure.*

The idea: use **Markov chain random walks** across large embedded knowledge bases to generate candidate cross-domain connections, then filter through a multi-stage pipeline to surface genuinely novel, non-obvious links between ideas.

**Data sources** (planned): Wikipedia (embedded, query-retrieval working), arXiv abstracts, PubMed abstracts — unified embedding space.

**Filtering pipeline** (the hard problem):
1. **Bulk filter** — cosine similarity against known-good connections, kill ~80-90% of noise
2. **Coherence judge** — smaller LLM: "is this logically sound and interesting?"
3. **Pairwise reranking** — stronger LLM: relative novelty comparison across candidates
4. **Novelty detection** — inverted retrieval: embed winners, search source literature. High similarity = known. Low similarity + high coherence = signal.

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

# First-run wizard (for frozen executable)
python main.py wizard
```

### Docker

```bash
# Build and run
docker-compose up -d

# Access at http://localhost:7860
docker-compose logs -f daemon-gui
```

### Desktop Executable

```bash
rm -rf build/ dist/
pyinstaller daemon.spec --clean --noconfirm
# Output: dist/Daemon/Daemon (Linux) or dist/Daemon/Daemon.exe (Windows)
```

See [docs/BUILD_GUIDE.md](docs/BUILD_GUIDE.md) for Windows installer instructions and performance targets.

---

## Testing

```bash
python -m pytest -q                     # Run full suite
python -m pytest tests/benchmarks/      # Retrieval quality benchmarks
python -m pytest -m "not slow"          # Exclude slow tests
python -m pytest --cov=. --cov-report=html  # With coverage
```

**2,800+ tests** across 137 test files — unit, integration, and retrieval quality benchmarks. Key areas:

- 74 intent classification tests
- 70 knowledge graph tests
- 66 implementation tracking tests
- 55 proactive context surfacing tests
- 51 escalation tracker tests
- 50 graph integration tests
- 47 cross-deduplication tests
- 39 fact verification tests
- 47 claim tracker / staleness tests
- 44 file access manager tests
- 19 retrieval quality benchmarks (30 seed memories, real embeddings, recall@K + MRR)

---

## Project Stats

```
Lines of Python:      ~102,000
Python files:         284
Test files:           137
Test functions:       2,800+
ChromaDB collections: 11
Prompt sections:      ~24 (conditional)
Intent types:         9
Retrieval tasks:      18 (parallel)
Memory tiers:         5
Gating latency:       ~200ms
```

---

## Configuration

Core settings via `config.yaml` with environment variable overrides:

```yaml
# config.yaml
daemon:
  version: v4
  data_dir: ./data

models:
  default_model: sonnet-4.5
  default_max_tokens: 1850
  enable_thinking_blocks: true

memory:
  corpus_max_entries: 2000
  prompt_max_recent: 15
  prompt_max_mems: 15
  prompt_max_facts: 30

gating:
  cosine_threshold: 0.15
  xenc_threshold: 0.55
```

```bash
# Environment variables (required)
OPENAI_API_KEY=sk-...      # Or ANTHROPIC_API_KEY, etc.

# Environment overrides (optional)
CORPUS_MAX_ENTRIES=2000
PROMPT_TOKEN_BUDGET=15000
CHROMA_DEVICE=cpu          # Or "cuda" for GPU
```

See [config/app_config.py](config/app_config.py) for the full list of ~80+ configuration options with defaults.

---

## Project Structure

```
core/                        # Request orchestration
├── orchestrator.py          # Main controller
├── best_of_handler.py       # Best-of-N/duel/ensemble generation
├── context_pipeline.py      # Query analysis (tone, topic, intent, STM)
├── intent_classifier.py     # Regex-first intent classification (9 types)
├── escalation_tracker.py    # Crisis cooldown FSM
├── correction_detector.py   # User correction/confirmation detection
├── agentic/                 # ReAct agentic search loop
└── prompt/                  # Modular prompt system
    ├── builder.py           # UnifiedPromptBuilder
    ├── context_gatherer.py  # Parallel async retrieval
    ├── formatter.py         # Section assembly
    └── token_manager.py     # Budget management

memory/                      # 5-tier memory system
├── memory_coordinator.py    # Thin orchestrator (~500 lines)
├── shutdown_processor.py    # Session-end processing pipeline
├── memory_scorer.py         # Composite scoring with intent overrides
├── truth_scorer.py          # Evidence-based truth tracking
├── graph_memory.py          # NetworkX knowledge graph
├── entity_resolver.py       # Entity alias resolution
├── fact_extractor.py        # Dual-budget fact extraction (user + entity)
├── claim_tracker.py         # Staleness tracking + cascade invalidation
├── cross_deduplicator.py    # Cross-collection dedup (dry-run default)
├── fact_verification.py     # Conflict detection before storage
├── context_surfacer.py      # Proactive cross-domain insights
├── thread_store.py          # Open thread tracking
└── storage/
    └── multi_collection_chroma_store.py  # ChromaDB wrapper (11 collections)

processing/
└── gate_system.py           # FAISS → cosine → cross-encoder pipeline

models/
├── model_manager.py         # Multi-provider LLM (OpenAI, Anthropic, DeepSeek, Google, local)
└── tokenizer_manager.py

knowledge/
├── web_search_manager.py    # Tavily API + caching
├── git_memory.py            # Git commit extraction
├── reference_docs_manager.py # Auto-seeded docs/ knowledge
└── implementation_detector.py # Proposal implementation tracking

scripts/
├── migrate_claims.py        # Backfill claim index from existing summaries
├── migrate_facts_to_graph.py # Populate knowledge graph from ChromaDB facts
└── cleanup_graph_junk.py    # Remove junk nodes from knowledge graph

utils/
├── tone_detector.py         # 250+ weighted keywords, 4 crisis levels
├── topic_manager.py
└── web_search_trigger.py

gui/                         # Gradio web interface
├── launch.py                # Dark theme, tab layout, startup hooks
├── handlers.py              # Chat streaming
└── wizard.py                # First-run onboarding

tests/                       # 137 test files
├── unit/                    # Component tests
├── benchmarks/              # Retrieval quality (real embeddings)
└── fixtures/                # Benchmark seed data
```

---

## Wikipedia Pipeline

Daemon supports Wikipedia-scale knowledge retrieval with 6.5M+ articles semantically indexed:

```bash
python data/pipeline/unified_pipeline.py --download --semantic

# Pipeline: download dump → parse XML → chunk (512 tokens) →
#   embed (all-MiniLM-L6-v2) → build FAISS IVF index → store metadata
```

**Requirements:** ~102GB storage for the full index. Optional — the assistant works without it.

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
| [FORMAL_MODEL.md](docs/FORMAL_MODEL.md) | Mathematical specification of the complete agent |
| [PROJECT_SKELETON.md](docs/PROJECT_SKELETON.md) | Compressed architectural walkthrough for LLM context |
| [GOALS.md](docs/GOALS.md) | Active goals, principles, and recent completions |
| [TAG_GENERATION.md](docs/TAG_GENERATION.md) | Tag generation system documentation |
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
- Wikipedia FAISS: 102GB (optional)
- Logs: ~1MB/day
