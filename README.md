# Daemon

**A local-first personal AI agent framework with persistent multi-collection memory, evaluated retrieval, knowledge-graph reasoning, literature-backed synthesis validation, agentic tool use, and human-gated self-improvement.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/lukehalleran/ai-assistant-framework/actions/workflows/tests.yml/badge.svg)](https://github.com/lukehalleran/ai-assistant-framework/actions/workflows/tests.yml)
[![Tests](https://img.shields.io/badge/tests-8%2C717-brightgreen.svg)](#testing)
[![Docker Ready](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

> ~245K lines of Python (≈188K code) across 760 files | 14 ChromaDB collections | ~7,800 tests | 21 agentic tools | solo-built part-time over ~15 months

Daemon is built around persistent memory, evaluated retrieval, knowledge-graph context, agentic tools, and experimental literature-backed synthesis. It stores your memory locally, retrieves context through a multi-stage RAG pipeline, tracks fact truth and staleness over time, and supports human-gated self-improvement through structured code proposals and isolated agent-branch experiments.

It is a stateful agent architecture, not a chatbot wrapper: every query passes through context analysis, intent classification, parallel retrieval, gating, scoring, prompt assembly, generation, and post-response state updates.

*Solo architected and maintained by Luke U_handle. AI coding assistants were used as development tools, but architecture, review, testing, integration decisions, and commits are human-directed — which is why GitHub lists `@claude` as a contributor.*

---

## TL;DR

- **Persistent hierarchical memory** across **14 ChromaDB collections** (episodic, semantic, procedural, summary, meta, synthesis)
- **Evaluated multi-stage RAG** — intent-parameterized scoring, **20+ parallel retrieval tasks**, multi-stage gating (~200ms), cross-encoder rerank; **retrieval benchmarks gate every scoring change** (versioned in [BENCHMARK_METRICS.md](docs/BENCHMARK_METRICS.md); current numbers in [METRICS_SNAPSHOT.md](docs/METRICS_SNAPSHOT.md))
- **Knowledge-graph reasoning** (NetworkX) with entity alias resolution, BFS query expansion, and graph-boosted scoring
- **ReAct agentic tool loop** — **21 tools** (web, sandbox, memory, files, git/github, academic search, image recall, document generation, action proposals, contact lookup) with a context inventory that prevents redundant searches
- **Literature-backed synthesis** — narrows a large conceptual space into evidence-backed *candidate* connections and validates them against independent corpora (candidates, not discoveries)
- **Human-gated self-improvement** — structured code proposals + isolated agent-branch experiments; the machine may propose and evaluate, **only a human may merge**
- **Prompt-section ablation eval system** — snapshot, replay, variant generation, blind pairwise judging, objective checks
- **Docker deployment** + **desktop installer** (PyInstaller + Inno Setup)

---

## Reviewer Quick Path

If you only have 10 minutes:

1. [`docs/ARCHITECTURE_GUIDE.md`](docs/ARCHITECTURE_GUIDE.md) — system-level walkthrough.
2. [`core/prompt/`](core/prompt/) — parallel retrieval + prompt assembly (the system is a pipeline, not one prompt).
3. [`memory/memory_scorer.py`](memory/memory_scorer.py), [`memory/fact_verification.py`](memory/fact_verification.py), [`memory/claim_tracker.py`](memory/claim_tracker.py) — ranking, truth handling, staleness.
4. [`docs/SYNTHESIS_VALIDATION.md`](docs/SYNTHESIS_VALIDATION.md) + [`knowledge/doc_cooccurrence.py`](knowledge/doc_cooccurrence.py) — the literature-backed synthesis validation.
5. `python -m pytest tests/benchmarks/ -m benchmark -v` — see retrieval evaluation run on real embeddings.

If you only want the single most distinctive subsystem, start with synthesis validation:
[`docs/SYNTHESIS_VALIDATION.md`](docs/SYNTHESIS_VALIDATION.md) → [`knowledge/doc_cooccurrence.py`](knowledge/doc_cooccurrence.py) → `scripts/synthesis_*`.

---

## Demo

https://github.com/user-attachments/assets/38e4c2ff-d992-4152-a98e-df58faa54533

---

## What Makes This Different

Most "AI memory" projects store conversation history and do cosine-similarity lookups. Daemon is a full cognitive pipeline with a formal mathematical specification, evaluated retrieval, agentic tool use, and supervised self-improvement infrastructure.

### Formally specified agent architecture

The entire agent is [formally modeled](docs/FORMAL_MODEL.md) as a composition of mathematical functions — perceive, interpret, expand, remember, plan, act, learn. Every scoring weight, gating threshold, state transition, and prompt section is documented with its mathematical definition and code location.

### Agentic tool system (ReAct loop)

When a query needs more than stored memory, Daemon enters a multi-round ReAct loop with **21 tools**:

| Group | Tools |
|-------|-------|
| **Knowledge** | Web search (Tavily), Wolfram Alpha, academic search (arXiv / PubMed / Hacker News / Stack Exchange), direct URL fetch |
| **Memory** | Cross-collection memory search, memory expansion (chronological neighbors / summary drill-down), full-document reassembly, image recall (CLIP) |
| **Code & repo** | Python code sandbox (E2B microVMs), read-only git stats, read-only GitHub API, file read / grep / list (sandboxed) |
| **Actions** | Research-and-save document generation, internet write-action *proposals* (human-approved), contact lookup (Google Contacts / Gmail) |

The agent receives a **context inventory** of what RAG already gathered, preventing redundant searches. A 4-tier agentic gate (keyword heuristic → knowledge-graph entity match → document/note intent → LLM fallback) decides when to enter the loop at all. Tool calls emitted as plain text or nested XML by proxied models are recovered rather than leaked into the answer.

### Intent-parameterized retrieval

A single scoring function serves all 9 query intent types (factual recall, emotional support, temporal recall, project work, …) by swapping weight vectors, retrieval limits, and gating thresholds per-intent. No separate pipelines — just different lenses on the same infrastructure. Intent classification is regex-first (no LLM calls), refined by short-term-memory analysis when confidence is low.

### Evidence-based truth scoring & staleness cascade

Memories aren't trusted equally. A `TruthScorer` tracks confirmation and correction events across sessions; a `FactVerifier` runs conflict detection before any new fact is stored (STORE / STORE_AND_FLAG / REJECT / SKIP, with optional LLM adjudication and **no auto-deletion**). When a fact is corrected, a `ClaimIndex` cascades staleness ratios to downstream summaries and reflections, which then receive scoring penalties and a `[HISTORICAL — PARTIALLY OUTDATED]` prefix in the prompt.

### Knowledge-graph query expansion

Queries are expanded using a NetworkX knowledge graph built from extracted facts. Entity resolution with alias tables (trigram → bigram → unigram) turns "what about my brother" into a query that also surfaces related entities the user didn't explicitly mention, ranked by lateral connectivity.

### Visual memory (CLIP cross-modal search)

Images are ingested through OpenCLIP ViT-B/32 → vision-LLM caption → entity tags → dual FAISS + ChromaDB indexes. Retrieval uses knowledge-graph entity resolution to match queries to the correct images even when generic CLIP similarity can't distinguish entities.

---

## Evaluation & Benchmarks

Retrieval quality is **measured, not asserted** — no scoring or weight change ships without a before/after benchmark run.

- **Retrieval benchmarks** (`tests/benchmarks/`): real embeddings (BGE-small-en-v1.5, 384d) + cross-encoder rerank (ms-marco-MiniLM-L-6-v2). Two suites: (1) synthetic adversarial cases (openly distributed), and (2) owner-local cases sampled from personal ChromaDB (not distributed — contains personal memory and skips cleanly when absent). Latest rerun (2026-07-23) on the synthetic suite: **283/296 cases pass (95.6%)**; combined **MRR 0.84** (R@1 0.78) over 280 retrieval cases. Historically measured (2026-05-17 baseline snapshot, MRR 0.89, 305/305 on the combined dataset) and retained in [BENCHMARK_METRICS.md](docs/BENCHMARK_METRICS.md) for regression tracking.
- **Prompt-section ablation eval** (`eval/`): snapshot capture → deterministic replay → leave-one-out / add-one-in variants → blind pairwise A/B judging → 5 automated objective checks. Entirely side-effect-free (a persistence guard asserts no ChromaDB/JSON mutation during eval).
- **Synthesis validation** (`scripts/synthesis_*`, `docs/SYNTHESIS_VALIDATION.md`): judge-discrimination tests, the document-co-occurrence oracle hardening (n=99), controlled-distance and discovery-mining experiments — all using literature as ground truth.

> Benchmark numbers are versioned in [docs/BENCHMARK_METRICS.md](docs/BENCHMARK_METRICS.md). All metrics are reproducible locally — no benchmark claim ships without a runnable test.

---

## Current Research Track: Literature-Backed Synthesis

Daemon's synthesis system does not claim to discover truth. It **narrows a large conceptual search space into evidence-backed *candidate* connections**, then validates them against independent corpora. The target signal is:

> **low semantic similarity + corpus co-occurrence = a non-obvious but literature-backed candidate connection.**

Two concepts that sit far apart in embedding space but are *discussed together* in the literature are strong candidates for non-obvious but grounded connections. Cosine similarity alone can't surface these — it equates "known" with "topically close." A separate **document-co-occurrence oracle** can. (Low-cosine co-occurrence is a *signal*, not a guarantee: it can still be a common-word artifact — a failure mode this project's own validation found and now screens for.)

Recent validation work (see [docs/SYNTHESIS_VALIDATION.md](docs/SYNTHESIS_VALIDATION.md)) treats *literature as the ground truth you don't have to be an expert in*:

- The coherence judge is **tested for discrimination** — literature-confirmed connections vs. surface metaphors — not trusted blindly. (A regression here recently found the judge had silently degraded after a model swap, and a stage-3 "known" gate whose signal was *inverted* — both caught by the harness, zero human grading.)
- A **document-co-occurrence oracle**, independent of embedding distance, was built and validated at **~97% recall / ~4% false-positive** on a labeled set. It catches cross-domain connections cosine misses entirely — e.g. `simulated annealing ↔ metallurgy` at cosine **0.09**.
- A **discovery miner** surfaces low-cosine, corpus-supported candidate pairs from Wikipedia.

The long-term direction is **cross-corpus knownness** — labeling each candidate by where it is already documented:

- known in Wikipedia
- known in arXiv
- known in PubMed
- known in personal memory
- unknown but coherent
- low-cosine but multi-corpus-supported

This gives a measurable way to surface candidate hypotheses without pretending the machine knows truth. What it explicitly **does not** do:

- It does not prove a novel candidate is *true* — only that it is non-obvious and corpus-supported.
- It does not replace human or domain-expert review.
- It raises the prior that a surviving candidate is worth a human's time.

**Final validation still requires a human or domain expert. Synthesis outputs are candidates, not discoveries.**

---

## Human-Gated Self-Improvement

Daemon includes a self-improvement pipeline that is **proposal-only and human-gated**. The live proposer generates structured code-change proposals, classifies their risk, stores them in ChromaDB, and surfaces them in the GUI for human review. The experimental `agent_branch/` track runs isolated coding workers in rootless **Podman sandboxes**, computes diffs trust-side, kills unsafe or config-tampering proposals, ranks the survivors, and can feed survivors back into the proposal store as pending proposals.

> **Invariant: the machine may propose and evaluate; only a human may merge.**

Nothing auto-commits, pushes, or merges code. The agent-branch harness is hardened against `.git`-based RCE, nested-`.git` masking, diff-parser evasion, config tampering, and metering bypass. Python-level and shell-level filesystem guards block destructive operations on protected paths during agentic tool dispatch.

---

## Architecture at a Glance

```
User Query
    |
    +- Context Pipeline ---- topic extraction (spaCy NER + LLM fallback) ->
    |                        tone detection (250+ keywords, 4 crisis levels) ->
    |                        intent classification (9 types, regex-first, no LLM) ->
    |                        query rewriting -> STM analysis -> intent refinement
    |
    +- Knowledge Graph ----- Entity resolution (alias tables, trigram matching) +
    |                        BFS expansion (depth 2) + lateral connectivity ranking
    |
    +- Parallel Retrieval -- 20+ async tasks across 14 ChromaDB collections
    |                        (conversations, facts, summaries, reflections, wiki,
    |                         obsidian notes, git commits, procedural skills, threads,
    |                         proposals, reference docs, visual memories, self-notes, ...)
    |
    +- Multi-Stage Gating -- Batch cosine similarity -> cross-encoder reranking (~200ms)
    |                        Per-intent threshold overrides
    |
    +- Composite Scoring --- 6 weighted factors + graph/anchor/meta bonuses +
    |                        staleness/size/deictic penalties, all parameterized by intent
    |
    +- Prompt Assembly ----- 31 conditional sections, token-budgeted (10K default,
    |                        floor 8K / ceiling 16K); two-tier compression
    |                        (LLM summary + middle-out slicing); guaranteed recency floors
    |
    +- Agentic Tool Loop --- ReAct: Think -> Tool -> Observe -> Repeat (max 5 rounds)
    |                        20 tools + context inventory to prevent redundant re-searches
    |
    +- Generation ---------- Standard streaming | Best-of-N | Duel (A vs B + judge) |
    |                        Multi-model ensemble with voter selection
    |
    +- State Transition ---- Per-turn: memory storage + truth events + fact verification +
                                       graph ingestion + escalation FSM
                             Per-session: summaries + fact extraction + graph update +
                                          thread resolution + staleness cascade +
                                          skills + proposals + dedup + reflection
```

> Full formal model: [FORMAL_MODEL.md](docs/FORMAL_MODEL.md) · Code-level walkthrough: [PROJECT_SKELETON.md](docs/PROJECT_SKELETON.md)

---

## Best Code to Review

| Area | Path | What to look for |
|------|------|-----------------|
| **Agentic tool loop** | `core/agentic/` | ReAct orchestration, native + XML protocol handling, dispatch parity (one routing table), context budgeting |
| **Prompt assembly** | `core/prompt/` | Modular builder, 31 conditional sections, intent overrides, token budget, parallel retrieval |
| **Memory system** | `memory/` | Composite scoring, truth tracking, knowledge graph, fact verification, staleness cascade |
| **Synthesis validation** | `knowledge/doc_cooccurrence.py`, `knowledge/synthesis_filter.py` | The literature-as-ground-truth oracle + 7-stage filter |
| **Eval system** | `eval/` | Snapshot/replay, variant generation, pairwise judging, objective checks — side-effect-free |
| **Self-improvement** | `agent_branch/`, `knowledge/proposal_generator.py` | Sandboxed evaluation, trust-side diffing, human-gated merge |

---

## Quick Start

### Prerequisites
- Python 3.11 (pyproject pins `>=3.11,<3.12`; the systemd timer templates assume a pyenv 3.11.8 env)
- 4 GB RAM minimum (8 GB recommended; 16 GB if running the synthesis pipeline)
- At least one LLM API key (OpenAI, Anthropic, DeepSeek, Google, or an OpenRouter-routed model)

### Installation
```bash
git clone https://github.com/lukehalleran/ai-assistant-framework.git
cd ai-assistant-framework

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Set at least one provider key
export OPENAI_API_KEY=sk-your-key-here
# Or: export ANTHROPIC_API_KEY=sk-ant-...   (or create a .env file)
```

### Launch
```bash
python main.py        # GUI (recommended) -> http://localhost:7860
python main.py cli    # CLI mode
python main.py wizard # First-run onboarding wizard
```

### Docker
```bash
docker-compose up -d   # -> http://localhost:7860
```

### Desktop executable
```bash
pyinstaller daemon.spec --clean --noconfirm   # -> dist/Daemon/Daemon
```

See [docs/BUILD_GUIDE.md](docs/BUILD_GUIDE.md) for the full build guide and Windows installer instructions.

---

## Development

### Pre-commit hooks (optional)

To guard against accidentally committing sensitive terms, install the privacy check:

```bash
ln -s ../../hooks/pre-commit-privacy .git/hooks/pre-commit
```

This runs `gitleaks protect --staged` if installed and greps staged files against the privacy term list.

---

## Testing

```bash
python -m pytest -q                          # Fast default suite
python -m pytest tests/benchmarks/ -m benchmark -v   # Retrieval quality benchmarks (real embeddings)
python -m pytest -m "not slow" -q            # Exclude slow tests
python -m pytest --cov=. --cov-report=html   # With coverage
```

> The default `pytest` run excludes integration tests and benchmarks via `pytest.ini`. Markers: `slow`, `semantic`, `benchmark`.
>
> **CI runs a curated fast subset**, not the full suite: `.github/workflows/tests.yml` additionally `--ignore`s 17 heavyweight test files (model-loading / real-store integration tests) on top of the `pytest.ini` exclusions. The full suite runs locally in memory-capped batches (see `CLAUDE.md`); the badge above counts collected tests, not CI-executed tests.

**7,444 tests across 344 test files** (exact live count in the badge above and `docs/METRICS_SNAPSHOT.md`; run `pytest --collect-only`). Coverage spans every subsystem — prompt-section eval (246), synthesis audit (40), knowledge graph, intent classification, web-search trigger, fact verification, escalation FSM, cross-deduplication, claim tracking, visual memory, and retrieval benchmarks (real embeddings, recall@K + MRR), among others.

<!-- METRICS:BEGIN -->
<!-- Generated by scripts/generate_doc_metrics.py — do not edit by hand. -->
_Snapshot: 2026-09-05 · git `6e0aaea`_

- **Python:** 831 files · 266,701 total lines · 204,755 non-blank/non-comment  <sub>(git ls-files '*.py', excl venv/ data/)</sub>
- **Tests:** 8717 collected across 407 test files  <sub>(python -m pytest --collect-only -q; pytest.ini exclusions applied)</sub>
- **Agentic tools:** 23 dispatch-table types (22 exposed in the loop; recall_image excluded)
- **ChromaDB collections:** 14
- **Retrieval benchmark (ledger):** MRR=0.8402 · R@1=0.7821 · n=280  <sub>(docs/BENCHMARK_METRICS.md (Combined row))</sub>
- **Retrieval benchmark (CSV cross-check):** MRR=0.8402 · R@1=0.7821 · n=280  <sub>(data/benchmark_per_case.csv (computed))</sub>
<!-- METRICS:END -->

_Volatile counts above are regenerated from primary sources by `scripts/generate_doc_metrics.py` (see `docs/METRICS_SNAPSHOT.md`)._

---

## Privacy & Safety

Daemon is **local-first because persistent personal memory is sensitive.** Conversation memory, ChromaDB indexes, user profile, notes, and generated artifacts stay on disk by default (ChromaDB + JSON in `data/`). API keys load from environment variables or `.env` — `.env.example` holds placeholders only. No user data leaves the machine except in the API calls made to the configured LLM providers for generation (and to any tool — e.g. web search — you explicitly enable).

Owner-specific personal vocabulary lives in a **gitignored** `config/config.local.yaml`, deep-merged over a fully generic committed config — a fresh clone ships with no personal data. All deletion operations default to `dry_run=True`; **no user data is auto-deleted** without explicit GUI action, never on shutdown.

---

## What This Is Not

- **Not AGI**, and not an autonomous self-modifying system — the machine proposes and evaluates; a human merges.
- **Not a truth-discovery machine** — synthesis surfaces *candidate* connections, not facts. Final validation is human.
- **Not a cloud-hosted memory product** — memory lives on your disk, not a service.
- **Not a benchmark claim without local tests** — every metric in this README is reproducible from `tests/`.

Daemon proposes, retrieves, validates, and queues; humans remain responsible for truth judgments and code merges.

---

## Memory System

Five memory tiers modeled on cognitive architecture — plus a synthesis store — stored across **14 ChromaDB collections**:

| Tier | What It Stores | Retrieval Bias |
|------|---------------|----------------|
| **Episodic** | Raw conversation turns | Recency |
| **Semantic** | Extracted fact triples + full English Wikipedia (41M+ vectors) | Truth score + relevance |
| **Procedural** | Git commits + learned skill patterns | Pattern matching |
| **Summary** | LLM-compressed conversation blocks (with source backlinks) | Relevance |
| **Meta** | Session reflections + open threads + code proposals + proactive insights | Priority / urgency |
| **Synthesis** | Cross-domain candidate connections + rediscovery captures + convergence tracking | Composite score |

Memory ranking:

```
score(d, x) = sum( w_i(intent) * f_i(d, x) ) + bonuses - penalties

weights w_i (config.yaml gating.score_weights, LIVE; per-intent overridable):
  relevance (0.30) + recency (0.22) + truth (0.18) +
  importance (0.05) + continuity (0.10) + topic_match (0.10)
  (structure 0.05 is in the dict but UNUSED — structure is an additive 0.15·density term)

bonuses:  anchor + meta + graph (0.05 per related entity, capped 0.15)
penalties: staleness (cascade from corrected facts, steep at 80%, capped 0.4) +
           size + deictic drift + analogy
```

Cross-collection deduplication detects cosine duplicates (≥0.92) and fact contradictions (same subject+predicate, different object). Dry-run only by default; live deletions require explicit GUI action.

---

## Session Lifecycle

**Session start:** detects git changes since last session, builds a compact feature inventory, injects codebase awareness into the first prompt.

**Per-turn:** stores episodic memory, updates truth scores from correction/confirmation detection, runs fact extraction through the verification gate, ingests facts into the knowledge graph, transitions the escalation FSM.

**Session end (multi-step pipeline):** block summaries with backlinks → regex + LLM fact extraction with conflict verification → behavioral pattern + procedural skill extraction → self-improvement proposal generation → cross-collection dedup (dry-run) → open-thread resolution → implementation tracking → synthesis dreaming (bounded, checkpointed) → session reflection.

**Critical invariant:** no user data is auto-deleted at shutdown.

---

## Synthesis Pipeline (detail)

A generator feeds a 7-stage filter; the filter is the product (generation is cheap, *deciding what's evidence-backed and non-obvious* is the value):

- **Primary — Pooled Concept** (`synthesis_pooled_generator.py`, 2026-06-30): pairs *prominent* curated cross-domain concepts in the non-obvious cosine band (0.2–0.45). The lever is concept prominence, not anchoring; it is the sole generator dreaming uses when `SYNTHESIS_POOLED_ENABLED`.
- **Retired tiers** (`enabled: false`): Tier 0 Retrieval (`synthesis_retriever.py`), Tier 1 Graph Walk (`graph_walk_generator.py`), Tier 2 Cross-Store (`synthesis_generator.py`) — kept for reference; each paired thin/low-prominence concepts and yielded ≈0 accepts.

| Stage | Gate | What it does |
|-------|------|-------------|
| 0 | Text sanity | Min tokens, verb detection, repetition filter |
| 1 | Domain crossing | Require 2+ distinct domains |
| 2 | Semantic distance | Endpoint distance in [0.20, 0.90] |
| 3 | External novelty | Claim-vs-wiki similarity + direct `cos(A,B)` concept-co-occurrence gate (replaced an inverted bigram signal; see validation doc) |
| 4 | Internal novelty | Synthesis-memory convergence signal from independent rediscovery |
| 5 | Coherence judge | LLM rates INVALID/WEAK/MODERATE/STRONG (verdict-first, reasoning-robust); min MODERATE |
| 6 | Composite score | Novelty-ranked composite ≥ 0.70 (0.35 coherence / 0.60 novelty / 0.05 distance / 0.0 structural) |

A blind **human audit queue** presents candidates without generator labels: 3 binary screening questions + a 1–5 slider, with an auto-halt if the human-graded false-positive rate exceeds threshold. Candidates the system rediscovers as already-in-literature are captured separately as proof-of-concept evidence, not failures.

---

## Project Structure

```
core/                        # Request orchestration
+-- orchestrator.py          # Main controller
+-- context_pipeline.py      # Query analysis (tone, topic, intent, STM)
+-- intent_classifier.py     # Regex-first intent classification (9 types)
+-- escalation_tracker.py    # Crisis cooldown FSM (4 states)
+-- agentic/                 # ReAct agentic tool loop (20 tools in-loop; 21-row dispatch table)
+-- prompt/                  # Modular prompt system (builder + gatherer mixins + formatter)
+-- actions/                 # Human-in-the-loop internet write actions

memory/                      # 5-tier memory system
+-- memory_coordinator.py    # Thin orchestrator
+-- memory_scorer.py         # Composite scoring with intent overrides + graph boost
+-- truth_scorer.py          # Evidence-based truth tracking
+-- graph_memory.py          # NetworkX knowledge graph (JSON persistence)
+-- fact_verification.py     # Pre-storage conflict detection
+-- claim_tracker.py         # ClaimIndex reverse index + staleness cascade
+-- synthesis_memory.py      # Synthesis persistence + convergence + audit queue
+-- storage/multi_collection_chroma_store.py   # ChromaDB wrapper (14 collections)

knowledge/                   # External knowledge + synthesis
+-- web_search_manager.py    # Tavily API + decomposition + caching
+-- synthesis_filter.py      # 7-stage synthesis filter
+-- doc_cooccurrence.py      # Document co-occurrence oracle (literature-as-ground-truth)
+-- clip_manager.py / visual_memory_*.py        # CLIP visual memory
+-- proposal_generator.py    # Goal-directed code proposals

eval/                        # Prompt ablation & eval system
+-- section_registry.py      # 31-entry canonical section registry
+-- snapshots.py / variants.py / harness.py / judge.py / checks.py

agent_branch/                # Sandboxed, human-gated self-modification harness
scripts/                     # Validation harnesses (synthesis_*, oracle, miner)
tests/                       # 344 test files, ~7,400 tests
```

---

## Prompt Architecture

The prompt is assembled from **31 conditional sections**, ordered by transformer attention patterns (high-signal sections at the end). Default token budget **10,000** (floor 8K, ceiling 16K) with two-tier compression (LLM summary + middle-out slicing); the escalation FSM overrides the budget during crisis states. The stable personality/identity prefix is split at a cache breakpoint so per-turn churn doesn't invalidate prompt caching.

---

## Wikipedia Knowledge Base (Optional)

Daemon supports Wikipedia-scale retrieval over the full English Wikipedia (~41M embedded vectors) indexed via FAISS IVFPQ, with zero-copy on-demand metadata reads (the 33 GB metadata parquet is never loaded into RAM).

```bash
pip install huggingface_hub
huggingface-cli download MochiLives/daemon-wiki-faiss \
    --repo-type dataset --local-dir ~/daemon-wiki-data/wiki_data
export WIKI_DATA_ROOT=~/daemon-wiki-data
```

**Runtime:** ~2.6 GB RAM (index + embedder), ~14.5 GB disk, no GPU required. The assistant degrades gracefully when the index is absent.

---

## Configuration

Central config: `config/config.yaml` (63 sections) → Pydantic v2 validation (`config/schema.py`) → ~500 module-level constants (`config/app_config.py`) with environment-variable overrides. The active model is multi-provider and config-selectable.

```yaml
memory:
  prompt_max_recent: 10
  semantic_retrieval_limit: 100
gating:
  cosine_similarity_threshold: 0.15
  score_weights: { relevance: 0.30, recency: 0.22, truth: 0.18,
                   importance: 0.05, continuity: 0.10, structure: 0.05, topic_match: 0.10 }
token_budget:
  default: 10000   # floor 8000, ceiling 16000
```

---

## Roadmap: Sleep Mode (Background Cognition)

The next architectural direction moves heavy non-interactive work out of live chat and into idle-time background processing. **Awake mode** stays latency-sensitive — answer, retrieve, use tools, store the turn. **Sleep mode** handles the slow cognitive work:

- memory consolidation
- fact extraction and verification
- knowledge-graph maintenance
- synthesis candidate generation and filtering
- cross-corpus knownness checks (Wikipedia / arXiv / PubMed / personal memory)
- arXiv / PubMed indexing
- self-improvement proposal generation
- retrieval / eval benchmarks
- stale-claim refresh
- audit-queue preparation

This turns today's shutdown-only processing into a first-class **background cognition scheduler**. All sleep outputs are **checkpointed, resumable, and auditable** — background work produces artifacts for review, it does not silently mutate ground truth.

---

## Design Principles

1. **The filter is the product.** Generating connections is cheap; identifying which are evidence-backed and non-obvious is the value.
2. **Measure before and after.** No scoring or retrieval change without before/after benchmark runs.
3. **Literature is the ground truth you don't have to be an expert in.** Validate candidates against corpora; reserve human/expert time for the last mile.
4. **Human-gated by default.** The machine proposes and evaluates; a human merges. Nothing destructive happens silently.
5. **Subtractive work > additive work.** Removing systems that don't pull their weight beats adding new ones.

---

## Status

Daemon is an **active solo research/engineering project**, not a polished SaaS product. The core paths — memory, retrieval, GUI, agentic tools, benchmarks, Docker, and desktop packaging — are implemented and tested. Synthesis validation, cross-corpus knownness, and sleep-mode background cognition are **active research tracks**.

A few subsystems are research/developer features rather than always-on defaults, and degrade gracefully when their data is absent:

- **Synthesis pipeline** — generators feed a human audit queue; dreaming runs at session-end only, behind an auto-halt.
- **Code proposals + `agent_branch/`** — sandboxed, human-gated; nothing auto-commits, pushes, or merges.
- **External corpora** — the Wikipedia FAISS index (~41M vectors) and arXiv/PubMed knownness are **optional**; the assistant runs fully without them.
- **Debug / provenance tooling** — surfaced in the GUI for inspection, not required for normal chat.

---

## Key Documentation

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE_GUIDE.md](docs/ARCHITECTURE_GUIDE.md) | Narrative architectural walkthrough |
| [FORMAL_MODEL.md](docs/FORMAL_MODEL.md) | Mathematical specification of the complete agent |
| [PROJECT_SKELETON.md](docs/PROJECT_SKELETON.md) | Compressed code-level walkthrough |
| [MEMORY_SYSTEM.md](docs/MEMORY_SYSTEM.md) | Memory lifecycle, scoring algorithm, tuning |
| [SYNTHESIS_FILTER.md](docs/SYNTHESIS_FILTER.md) | Synthesis pipeline stages, calibration |
| [SYNTHESIS_VALIDATION.md](docs/SYNTHESIS_VALIDATION.md) | Literature-as-ground-truth validation methodology + results |
| [AGENTIC_SEARCH.md](docs/AGENTIC_SEARCH.md) | ReAct loop, tool system, protocol handling |

---

## Resource Usage

| Resource | Idle | Active Query |
|----------|------|--------------|
| CPU | ~5% | 30–50% |
| RAM | ~500 MB | ~1.5 GB (≈2.6 GB with Wikipedia index loaded) |
| GPU VRAM | — | 2–8 GB (optional) |

**Storage:** ChromaDB ~50 MB · Wikipedia FAISS index + metadata ~14.5 GB (optional) · logs ~1 MB/day.
