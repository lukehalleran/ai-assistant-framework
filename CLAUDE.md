# CLAUDE.md

## Project Overview

Daemon is a production-grade conversational RAG system with a 5-tier hierarchical memory architecture. Features multi-stage semantic gating, crisis-aware tone detection, multi-provider LLM support, knowledge graph, synthesis dreaming, visual memory, and agentic search.

**Stats:** ~143K lines across 450+ files (395+ Python), 177 test files, 3,600+ tests. Last full run: 2026-05-17 (2417 unit + 305 benchmark passed, 0 failures).

## Critical Rules

- **NEVER auto-delete user data.** Deletions must default to `dry_run=True`. Only explicit GUI action triggers real deletes — never on shutdown, never silently.
- **NEVER load large datasets fully into memory.** 16GB RAM. Never `pd.read_parquet()` on wiki parquet (41M+ rows) without row limits. Use row-group reads or pyarrow zero-copy.

## Commands

```bash
python main.py              # GUI at http://localhost:7860
python main.py cli          # CLI mode
python main.py wizard       # First-run wizard
python -m pytest -q         # All tests
pytest -m "not slow"        # Exclude slow
pytest --cov=. --cov-report=html  # Coverage
pyinstaller daemon.spec --clean --noconfirm  # Build executable
```

## Request Flow

```
Query → ContextPipeline (tone‖topic parallel, intent gates heavy-topic skip)
      → UnifiedPromptBuilder (context assembly + web search, cached query embedding)
        + ResponsePlanner (parallel: lightweight plan)
      → Agentic search (if triggered, replaces standard generation)
        17 tools: web, wolfram, sandbox, memory, files, git, github, fetch_url,
                  stackexchange, arxiv, pubmed, hackernews, recall_image, ...
      → BestOfHandler | ResponseGenerator (streaming)
      → ResponseParser (thinking block, artifact stripping)
      → MemoryCoordinator (persist)
      → CorrectionDetector (truth score updates + staleness cascade)
      → Response
```

## Memory System

**13 ChromaDB Collections:** `conversations` (protected), `facts`, `summaries`, `reflections`, `wiki_knowledge` (protected), `obsidian_notes` (protected), `reference_docs` (protected), `procedural`, `procedural_skills`, `proposals`, `threads`, `synthesis_results`, `visual_memories`

**Knowledge Graph:** NetworkX DiGraph at `data/knowledge_graph.json`. Entity alias resolution at `data/entity_aliases.json`. Ingestion via `memory_storage.py:_ingest_fact_to_graph()`. Retrieval via BFS in `context_gatherer.py`. Graph-boosted scoring (0.05/entity, cap 0.15) + query expansion with graph neighbors.

**Scoring:** `final = w.relevance*R + w.recency*T + w.truth*Tr + w.importance*I + w.continuity*C + w.structure*S + graph_bonus`. Defaults: 0.35/0.25/0.20/0.05/0.10/0.05. Per-intent weight overrides via IntentClassifier. Two-regime temporal decay. Stemmed continuity + tag-keyword bonus.

**Multi-Stage Gating (~200ms):** FAISS→50 candidates → Cosine filter (0.15) → CrossEncoder→top K

**Embedding model:** BAAI/bge-small-en-v1.5 (384d). Cross-encoder reranker: ms-marco-MiniLM-L-6-v2 (top-15 post-scoring rerank in retriever).

## Prompt Building

```
UnifiedPromptBuilder (thin orchestrator)
├── ContextGatherer (parallel async: web, memory, knowledge mixins)
├── ContentHygiene (dedup, caps, backfill)
├── MultiStageGateSystem (filtering)
├── PromptFormatter (section assembly + feature inventory)
├── TokenManager (budget, middle-out compression)
└── LLMSummarizer (dynamic compression)
```

## Key Config

Central: `config/app_config.py` (module-level constants) + `config/schema.py` (Pydantic v2 validation) + `config/config.yaml` (45 sections). Config pattern: YAML → schema validation → app_config constants with env var overrides.

Key values: `PROMPT_TOKEN_BUDGET_DEFAULT=15000` (floor 8K, ceiling 16K), `COSINE_SIMILARITY_THRESHOLD=0.25`, 9 intent types, dual fact budget (user=6, entity=4). Web search requires `TAVILY_API_KEY`. Currently disabled: synthesis generators, graph walk.

## Module Structure

```
core/                         # Request orchestration
├── orchestrator.py           # Main controller
├── best_of_handler.py        # Best-of-N/duel/ensemble
├── context_pipeline.py       # Query analysis (tone, topic, intent, STM)
├── intent_classifier.py      # Regex-first, 9 types, drives retrieval/scoring
├── escalation_tracker.py     # Crisis cooldown strategies
├── correction_detector.py    # Pattern-based correction/confirmation + content attribution
├── uncertainty_detector.py   # "I don't know" detection → agentic retry
├── content_type_detector.py  # Regex detection of shared content (lyrics, poems, code, quotes)
├── ambiguity_detector.py     # Cross-session phrase ambiguity detection → disambiguation notes
├── response_planner.py       # Pre-answer plan + post-answer review gate
├── git_stats_manager.py      # Read-only git repo stats (agentic tool)
├── github_manager.py         # Read-only GitHub API via gh CLI (agentic tool)
├── response_generator.py
├── agentic/                  # ReAct search loop
│   ├── controller.py         # Loop orchestration, [TOOL STATUS] injection
│   ├── tools.py              # 17 tool types + get_tool_health() (incl. GitHub, StackExchange, arXiv, PubMed, HN)
│   ├── formatters.py         # Stateless result formatting
│   ├── types.py              # SearchDecision, tool definitions
│   └── protocols.py          # Native tools + XML parsing
└── prompt/                   # Modular prompt system
    ├── builder.py            # Thin orchestrator + ambiguity detection wiring
    ├── base.py               # Base utilities + fallback classes
    ├── context_gatherer.py   # Mixin compositor
    ├── gatherer_web.py       # Web search
    ├── gatherer_memory.py    # Conversations, summaries, reflections, facts, profile
    ├── gatherer_knowledge.py # Notes, docs, wiki, git, graph, threads, insights
    ├── formatter.py          # Section assembly + session boundaries + content object rendering
    ├── hygiene.py            # Dedup, caps, backfill
    ├── proposal_filter.py    # Code proposal retrieval + filtering
    ├── summarizer.py         # LLM dynamic compression
    └── token_manager.py

memory/                       # Memory system
├── memory_coordinator.py     # Thin orchestrator
├── shutdown_processor.py     # Session-end processing + topic-specific reflection generation
├── memory_storage.py         # Persistence + graph ingestion + reflection retrieval text + content type metadata
├── memory_scorer.py          # Scoring with intent overrides + graph boost
├── memory_retriever.py       # Retrieval + reflection v2 subsystem + semantic-primary fact ranking
├── fact_extractor.py         # Dual-budget fact extraction (user + entity)
├── llm_fact_extractor.py     # LLM-assisted extraction with relation reuse
├── truth_scorer.py           # Evidence-based truth (confirmation/correction/contradiction)
├── graph_memory.py           # NetworkX DiGraph: CRUD, BFS, JSON persistence
├── graph_models.py           # Pydantic: GraphNode, GraphEdge
├── graph_utils.py            # Entity extraction, neighbor lookups
├── entity_resolver.py        # Alias resolution + relation normalization
├── memory_expander.py        # Temporal-window expansion (agentic tool)
├── cross_deduplicator.py     # Cross-collection dedup (dry-run only on shutdown)
├── dedup_models.py           # DedupPlan, DuplicatePair, ContradictionCluster
├── claim_tracker.py          # Staleness tracking via claim hashing
├── synthesis_memory.py       # Synthesis persistence + audit queue
├── procedural_skill.py       # ProceduralSkill + SkillCategory
├── skill_activation.py       # Post-retrieval skill filtering + cooldown
├── user_profile.py           # User profile with source excerpts + relative timestamps
├── user_profile_schema.py    # Relation canonicalization, 5-layer categorization
├── thread_manager.py
├── thread_store.py           # ChromaDB thread persistence + quick resolution
├── thread_extractor.py       # LLM thread extraction + resolution detection
├── fact_verification.py      # Store/flag/reject verdicts, never auto-deletes
├── surfacing_models.py       # Proactive insight data models
├── context_surfacer.py       # Cross-domain insight generation from graph
├── surfacing_history.py      # Insight surfacing history (JSON-backed)
├── memory_interface.py       # Protocol contracts for memory components
├── memory_consolidator.py    # Session consolidation
├── hybrid_retriever.py       # Hybrid retrieval for summaries/reflections
├── corpus_manager.py         # In-memory corpus management
├── code_proposal.py          # Code proposal data models + supervision fields (RiskLevel, ProposalOutcome)
├── proposal_store.py         # ChromaDB proposal storage
├── thread_models.py          # Thread data models + enums
├── utils.py                  # Memory utility functions
└── storage/
    └── multi_collection_chroma_store.py  # ChromaDB wrapper (13 collections)

knowledge/                    # External knowledge
├── web_search_manager.py     # Tavily API + caching + WEB_N citations
├── git_memory.py             # Git commit extraction
├── git_memory_loader.py      # Git → PROCEDURAL loader
├── reference_docs_manager.py # Self-knowledge docs + auto-seed from docs/
├── synthesis_filter.py       # 7-stage filter (Opus coherence judge)
├── synthesis_generator.py    # Cross-store sampling (Tier 2, disabled)
├── synthesis_retriever.py    # FAISS structural query (Tier 0, disabled)
├── graph_walk_generator.py   # Biased Markov walks (Tier 1, disabled)
├── clip_manager.py           # OpenCLIP ViT-B/32 singleton (disabled)
├── visual_memory_store.py    # ChromaDB + FAISS for images (disabled)
├── visual_memory_pipeline.py # CLIP → caption → entity → store (disabled)
├── visual_retrieval.py       # CLIP text→image search (disabled)
├── wiki_tracker.py           # Session-level wiki article tracking
├── wiki_enrichment.py        # Shutdown: wiki articles → graph nodes
├── wikidata_resolver.py      # Personal ↔ Wikidata entity resolution
├── proposal_generator.py     # Goal-directed code proposals
├── implementation_detector.py # 4-stage proposal status detection
└── synthesis_models.py       # Synthesis data models + enums

config/                       # Configuration
├── app_config.py             # YAML loader + ~280 module-level constants
├── schema.py                 # Pydantic v2 validation (44 section models)
├── config.yaml               # 44 sections, ~334 keys
├── feature_registry.yaml     # Retrospective shipped-feature catalog (branch supervision)
└── feature_registry.py       # Typed loader: dependency resolution, conflict detection

gui/                          # Gradio web interface
├── launch.py                 # Main GUI, dark theme, startup tasks
├── handlers.py               # Chat submission, streaming, agentic gate
└── wizard.py                 # First-run onboarding

eval/                         # Prompt ablation & eval (Phases 1-2, 4-6)
├── schema.py                 # Pure data models
├── section_registry.py       # 27-entry canonical section registry
├── snapshots.py              # Capture + replay
├── variants.py               # LOO/AOI/bundle/reorder variant generation
├── corpus.py                 # 27-query seed corpus (3 per intent)
├── utilization.py            # Per-section utilization analysis
├── harness.py                # Generation harness with resume
├── judge.py                  # Pairwise A/B judge (5-criterion rubric)
├── checks.py                 # 5 automated objective checks
├── no_store_generation.py    # Side-effect-free LLM generation
└── persistence_guard.py      # ChromaDB + JSON mutation detection

models/
├── model_manager.py          # Multi-provider LLM (OpenAI, Anthropic, DeepSeek, local)
└── tokenizer_manager.py

processing/
└── gate_system.py            # FAISS → Cosine → CrossEncoder

utils/
├── tone_detector.py          # Crisis detection (250+ keywords)
├── web_search_trigger.py     # Keyword + semantic + LLM trigger detection
├── file_processor.py         # File upload (PDF/DOCX/XLSX/CSV/JSON/YAML/HTML/XML/MD/TXT/LOG/PY) + table extraction
├── text_chunking.py          # chunk_by_headers + chunk_by_size fallback
├── bootstrap.py              # Frozen executable setup
├── fs_snapshot.py            # Filesystem manifest for agent session safety
├── destructive_op_guard.py   # Git command classifier (blocks destructive ops)
├── python_fs_guard.py        # Python-level fs guard (os.remove, shutil.rmtree, shutil.copy/copy2/copyfile, etc.)
└── shell_cmd_guard.py        # Shell command classifier (rm, mv, chmod, etc.)
```

## Important Patterns

- **Protocols:** Memory components use `memory/memory_interface.py` Protocol contracts
- **Async:** `asyncio.gather()` for parallel retrieval throughout
- **Tone-driven prompts:** System prompts vary by crisis level (HIGH/MEDIUM/CONCERN/CONVERSATIONAL)
- **Token budget:** Middle-out compression (preserve start + end, compress middle)
- **Graceful degradation:** LLM→micro-summaries, semantic→keyword, API→stubs
- **Thinking leak defense:** 3-layer (API reasoning params → `<thinking>` tags → heuristic detection with 13 line-level + 9 sentence-level patterns)
- **Anti-confabulation:** Source excerpts on profile facts, inline warnings at all profile injection points

## Agent Session Safety

Before agent edits: `bash scripts/agent_session_start.sh`
After agent edits: `bash scripts/agent_session_audit.sh`
Use `scripts/safe_git.sh` instead of raw `git` for destructive-capable operations.
Use `scripts/safe_cmd.sh` instead of raw `rm`, `mv`, `chmod`, etc. for potentially destructive shell commands.
Python-level destructive operations (`os.remove`, `shutil.rmtree`, `Path.unlink`, `os.rename`, `os.replace`, `shutil.move`, `shutil.copyfile`, `shutil.copy`, `shutil.copy2`) are guarded by `utils/python_fs_guard.py`, which monkey-patches these 10 functions at startup and blocks protected-path operations during agentic tool dispatch. Child Python interpreters inherit the guard via `scripts/bin/usercustomize.py` when PYTHONPATH includes `scripts/bin/`.

Agents may **not** run `git restore`, `git reset --hard`, `git clean`, `git push`, `rm -rf`, `mv` on protected paths, or `chmod 000` without explicit human unlock. See `docs/AGENT_SAFETY.md` for full details.

## Testing

Config in `pytest.ini`. Markers: `slow`, `semantic`, `benchmark`. Excluded dirs: `venv/`, `data/`, `integration.bak/`. Tests in `tests/` and `tests/unit/` (named `test_<component>.py`). Eval tests in `tests/test_eval/` (246 tests).
