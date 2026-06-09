# CLAUDE.md

## Project Overview

Daemon is a production-grade conversational RAG system with a 5-tier hierarchical memory architecture. Features multi-stage semantic gating, crisis-aware tone detection, multi-provider LLM support, knowledge graph, synthesis dreaming, visual memory, and agentic search.

**Stats:** 541 files, ~171K lines of code (217K total incl. comments/blanks; test corpus excluded), 216 test files, ~3,875 tests. Last full run: 2026-05-17 (2417 unit + 305 benchmark passed, 0 failures); newer suites (action guard, relation classifier, health-transient TTL) verified via targeted runs.

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
        20 tools: web, wolfram, sandbox, memory, files, git, github, fetch_url,
                  stackexchange, arxiv, pubmed, hackernews, recall_image,
                  generate_document, propose_action, lookup_contact, ...
      → BestOfHandler | ResponseGenerator (streaming)
      → ResponseParser (thinking block, artifact stripping)
      → MemoryCoordinator (persist)
      → CorrectionDetector (truth score updates + staleness cascade)
      → Response
```

## Memory System

**14 ChromaDB Collections:** `conversations` (protected), `facts`, `summaries`, `reflections`, `wiki_knowledge` (protected), `obsidian_notes` (protected), `reference_docs` (protected), `procedural`, `procedural_skills`, `proposals`, `threads`, `synthesis_results`, `visual_memories`, `daemon_self_notes`

*Note:* document-level parent/child hierarchy (`parent_id`/`child_ids`) is currently **disabled** in the Chroma store (it caused retrieval issues — see `multi_collection_chroma_store.py:411`); retrieval is flat-within-collection. The "hierarchy" is the abstraction tiers (conversations → summaries → reflections), not document links.

**Knowledge Graph:** NetworkX DiGraph at `data/knowledge_graph.json`. Entity alias resolution at `data/entity_aliases.json`. Ingestion via `memory_storage.py:_ingest_fact_to_graph()`. Retrieval via BFS in `context_gatherer.py`. Graph-boosted scoring (0.05/entity, cap 0.15) + query expansion with graph neighbors.

**Scoring:** `final = w.relevance*R + w.recency*T + w.truth*Tr + w.importance*I + w.continuity*C + w.structure*S + graph_bonus`. Defaults: 0.35/0.25/0.20/0.05/0.10/0.05. Per-intent weight overrides via IntentClassifier. Two-regime temporal decay. Stemmed continuity + tag-keyword bonus.

**Multi-Stage Gating (~200ms):** ChromaDB HNSW→~50 candidates → cosine filter → CrossEncoder rerank→top K. (No FAISS in the live memory path — candidate generation is ChromaDB's HNSW; FAISS is used only for the wiki index and visual memory.)

**Embedding model:** ChromaDB retrieval uses BAAI/bge-small-en-v1.5 (384d). Note: the gate, tone detector, and web-search trigger re-embed text with `all-MiniLM-L6-v2` (the shared `ModelManager` SentenceTransformer, also 384d) — a *different* model from the one ChromaDB retrieved with. Cross-encoder reranker: ms-marco-MiniLM-L-6-v2 (top-15 post-scoring rerank in retriever).

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

Central: `config/app_config.py` (module-level constants) + `config/schema.py` (Pydantic v2 validation) + `config/config.yaml` (52 sections). Config pattern: YAML → schema validation → app_config constants with env var overrides.

Key values: `PROMPT_TOKEN_BUDGET_DEFAULT=15000` (floor 8K, ceiling 16K), `COSINE_SIMILARITY_THRESHOLD=0.25`, 9 intent types, dual fact budget (user=6, entity=4). Web search requires `TAVILY_API_KEY`. Google Calendar/Gmail/Contacts require `GOOGLE_CLIENT_ID` + `GOOGLE_CLIENT_SECRET` env vars. Contact resolution: `GOOGLE_CONTACTS_ENABLED`, `GOOGLE_OTHER_CONTACTS_ENABLED`, `GOOGLE_GMAIL_SEARCH_ENABLED` (all default true). Action guard (anti-confabulation): `PENDING_PROPOSAL_ENABLED`, `PENDING_PROPOSAL_TTL_TURNS`, `ACTION_CLAIM_GUARD_ENABLED`, `ACTION_CLAIM_SELF_REPAIR_ENABLED` — YAML section `action_guard`. Memory TTL: `PROFILE_EPHEMERAL_TTL_HOURS=24`, `PROFILE_HEALTH_TRANSIENT_TTL_HOURS=96` (illness/recovery facts age out in ~days, not weeks; durable conditions never expire — see `relation_classifier.py`). Shutdown: `SHUTDOWN_TASK_TIMEOUT_S=60` (bounds the session-end reflection+summary gather so a hung LLM call can't block exit). Personal vocabulary is externalized — owner-specific terms (medications, personal relations, project terms) live in the **gitignored** `config/config.local.yaml`, deep-merged over `config.yaml` at load via `app_config.load_local_overrides()` → `PROFILE_PERSONAL_*` constants. Committed `config.yaml` ships generic (`personal_vocabulary: {}`, blank `smtp_user`/`smtp_from`, generic vault/notes/system-prompt paths); a fresh clone with no override is fully generic. Any config key can be overridden in `config.local.yaml` — see the committed `config/config.local.example.yaml` template. Currently disabled: synthesis generators, graph walk.

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
├── action_claim_guard.py     # Anti-confabulation: detect action proposals/completion claims,
│                             #   verify claims vs what actually ran (note/doc self-repairable;
│                             #   email/calendar/etc. external → corrected, never auto-executed)
├── pending_proposal.py       # Capture a "Want me to save this?" offer + content; execute it on
│                             #   a later affirmation ("sure") — closes the proposal→affirm→do loop
├── response_planner.py       # Pre-answer plan + post-answer review gate
├── git_stats_manager.py      # Read-only git repo stats (agentic tool)
├── github_manager.py         # Read-only GitHub API via gh CLI (agentic tool)
├── response_generator.py
├── actions/                  # Internet actions (human-in-the-loop write actions)
│   ├── types.py              # ActionType enum, ActionProposal, ActionResult, PendingActionsStore
│   ├── registry.py           # ACTION_SPECS — SINGLE source of truth per write action (executor,
│   │                         #   required/optional params, intent patterns, backfill, tool-health,
│   │                         #   enable-flag). Adding an action = one spec here. Consumed by
│   │                         #   executors.py, agentic/protocols.py (parse), agentic/tools.py
│   │                         #   (tool-health), agentic/controller.py (detect/backfill/instruction).
│   ├── executors.py          # ActionExecutorRegistry: routes via ACTION_SPECS to each executor
│   ├── telegram.py           # Telegram Bot API sender
│   ├── discord.py            # Discord webhook sender
│   ├── email.py              # Gmail API primary + SMTP fallback email sender + contact name resolution
│   ├── google_auth.py        # Google OAuth2 manager (token persistence, refresh, scope checking, contacts + gmail scopes)
│   ├── google_calendar.py    # Fetch upcoming Google Calendar events (read-only, 5-min cache)
│   ├── google_calendar_create.py  # Create Google Calendar events via Calendar API
│   ├── google_contacts.py    # People API saved + other contacts search, resolve_contact() with Gmail fallback
│   ├── gmail_search.py       # Gmail header search (From/To) for contact resolution fallback
│   └── audit.py              # Append-only JSONL audit log (logs/actions_audit.jsonl)
├── agentic/                  # ReAct search loop
│   ├── gate.py               # 4-tier agentic gate: AgenticDecision + evaluate_agentic_gate()
│   │                         #   Tier 1 also routes file/saved-document RETRIEVAL intent
│   │                         #   (keywords + regex + pronoun/affirmation continuation) to tools
│   ├── controller.py         # Loop orchestration, [TOOL STATUS] injection
│   ├── tools.py              # 20 tool types + get_tool_health() + DISPATCH_TABLE (the single
│   │                         #   decision→handler routing table; ToolExecutor.dispatch_single AND
│   │                         #   controller._dispatch_single_inner both iterate it — no drift)
│   ├── formatters.py         # Stateless result formatting
│   ├── types.py              # SearchDecision, tool definitions, LOOKUP_CONTACT_TOOL_DEFINITION
│   └── protocols.py          # Native tools + XML parsing + contact lookup aliases
│                             #   (native handler recovers XML tool markers from plain
│                             #   text when no tool_calls; nested <file_read><path>X</path>
│                             #   ...</file_read> child-tag forms for file/doc/url tools)
└── prompt/                   # Modular prompt system
    ├── builder.py            # Thin orchestrator + ambiguity detection wiring
    ├── base.py               # Base utilities + fallback classes
    ├── context_gatherer.py   # Mixin compositor
    ├── gatherer_web.py       # Web search
    ├── gatherer_memory.py    # Conversations, summaries, reflections, facts, profile
    ├── gatherer_knowledge.py # Notes, docs, wiki, git, graph, threads, insights, calendar
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
│                             #   + facts-collection supersession (is_current=False) + per-relation TTL filter
├── relation_classifier.py    # Single source of truth: relation→TTL (health-transient ~days /
│                             #   ephemeral ~24h / durable) + permanent-condition (disability) overrides
├── fact_extractor.py         # Dual-budget fact extraction (user + entity) + schedule extraction (5 patterns, _enrich_schedule_metadata)
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
├── user_profile.py           # User profile with source excerpts + relative timestamps + per-relation transient TTL
├── user_profile_schema.py    # Relation canonicalization, 5-layer categorization (+ personal_vocabulary merge from config)
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
├── clip_manager.py           # OpenCLIP ViT-B/32 singleton (enabled)
├── visual_memory_store.py    # ChromaDB + FAISS for images (enabled)
├── visual_memory_pipeline.py # CLIP → caption → entity → store (enabled)
├── visual_retrieval.py       # CLIP text→image search (enabled; recall_image excluded from agentic loop to save API credits)
├── wiki_tracker.py           # Session-level wiki article tracking
├── wiki_enrichment.py        # Shutdown: wiki articles → graph nodes
├── wikidata_resolver.py      # Personal ↔ Wikidata entity resolution
├── proposal_generator.py     # Goal-directed code proposals
├── document_generator.py     # Research & save markdown docs (report/summary). Bounded
│                             #   trigger (doc-noun must be object of save-verb; buried
│                             #   mid-body of a long msg = incidental, answers in chat) +
│                             #   content-aware (pasted material = PRIMARY [INPUT_1] source,
│                             #   web/wiki suppressed) + LLM API-error sentinel guard
├── daemon_notes_manager.py   # Daemon self-notes for future sessions (non-ground-truth)
├── implementation_detector.py # 4-stage proposal status detection
└── synthesis_models.py       # Synthesis data models + enums

config/                       # Configuration
├── app_config.py             # YAML loader + ~300 module-level constants
├── schema.py                 # Pydantic v2 validation (59 section models)
├── config.yaml               # 52 sections, ~380 keys
├── feature_registry.yaml     # Retrospective shipped-feature catalog (branch supervision)
└── feature_registry.py       # Typed loader: dependency resolution, conflict detection

gui/                          # Gradio web interface
├── launch.py                 # Main GUI, dark theme, startup tasks
├── handlers.py               # Chat submission, streaming, action approve/reject
└── wizard.py                 # First-run onboarding

eval/                         # Prompt ablation & eval (Phases 1-2, 4-6)
├── schema.py                 # Pure data models
├── section_registry.py       # 31-entry canonical section registry
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
└── gate_system.py            # Cosine (over ChromaDB HNSW candidates) → CrossEncoder

utils/
├── tone_detector.py          # Crisis detection (250+ keywords)
├── web_search_trigger.py     # Keyword + semantic + LLM trigger detection
├── file_processor.py         # File upload (PDF/DOCX/XLSX/CSV/JSON/YAML/HTML/XML/MD/TXT/LOG/PY) + table extraction
├── text_chunking.py          # chunk_by_headers + chunk_by_size fallback
├── bootstrap.py              # Frozen executable setup
├── fs_snapshot.py            # Filesystem manifest for agent session safety
├── destructive_op_guard.py   # Git command classifier (blocks destructive ops)
├── python_fs_guard.py        # Python-level fs guard (os.remove, shutil.rmtree, shutil.copy/copy2/copyfile, etc.)
├── shell_cmd_guard.py        # Shell command classifier (rm, mv, chmod, etc.)
└── temporal_resolver.py      # Relative→absolute date conversion + schedule time normalization (normalize_time_range, expand_day_abbreviations, resolve_date_expression)
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

**Tool/action wiring parity** (`tests/unit/test_tool_wiring_parity.py`): asserts every SearchDecision tool flag has a `DISPATCH_TABLE` row, both routers iterate that table, and every `ActionType`/`propose_action` enum entry has a registered executor. If you add a tool or action and forget a wiring point, this fails loudly instead of the call being silently dropped (the failure mode that historically made additions painful).
