# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Daemon is a memory-augmented conversational AI system with hierarchical memory, semantic search, and Wikipedia-scale knowledge retrieval. Built from first principles, it implements a full RAG pipeline with multi-stage gating, temporal memory decay, and cognitive-inspired memory architecture.

**Important**: This is the developer's first Python project. Be mindful of knowledge gaps and explain non-obvious patterns when making changes.

**Architecture Reference**: For detailed system architecture, data flows, and component relationships, see `PROJECT_SKELETON.md` - a compressed overview designed for LLM context windows that captures essential structure and patterns without full implementation details.

## Commands

### Development
```bash
# Setup
pip install -r requirements.txt

# Run GUI (default)
python main.py

# Run with performance profiles
make -f Makefile.fast run      # Fast profile (CPU, reduced context)
make -f Makefile.balanced run  # Balanced profile
make -f Makefile.max run       # Maximum quality (GPU)

# CLI test mode
python main.py cli

# Test specific features
python main.py test-summaries
python main.py inspect-summaries
python main.py test-prompt-summaries
```

### Testing
```bash
# Run all tests (if pytest installed)
python -m pytest -q

# Individual test scripts
python tests/test_basic_pipeline.py
python tests/test_file_processor.py
python tests/test_memory_coordinator.py
python tests/test_gated_prompt.py
python tests/test_summaries.py
python tests/gui_smoke.py
```

### Wikipedia Pipeline
```bash
# Process Wikipedia dump (sample mode)
make -f Makefile.fast pipeline

# Full processing (manual)
python data/pipeline/unified_pipeline.py --download --semantic
```

## Architecture

### Core Request Flow
```
User Query
    ↓
Orchestrator (core/orchestrator.py)
  - Topic extraction via spaCy NER
  - Command handling
  - Query routing
    ↓
Memory Coordinator (memory/memory_coordinator.py)
  - Hierarchical memory retrieval
  - 5 ChromaDB collections (episodic, semantic, procedural, summary, meta)
  - Temporal decay scoring
    ↓
Multi-Stage Gate System (processing/gate_system.py)
  1. FAISS semantic search (top 50)
  2. Cosine similarity filter (threshold ~0.45-0.65)
  3. Cross-encoder reranking (MS-MARCO)
    ↓
Prompt Builder (core/prompt_builder.py)
  - Token budget management (default 2048-4096)
  - Context prioritization
  - Personality injection
    ↓
Response Generator (core/response_generator.py)
  - Async streaming via ModelManager
  - Multiple backends (OpenAI, Claude, local models)
    ↓
Memory Storage
  - Corpus persistence to JSON
  - Vector embeddings to ChromaDB
  - Automatic consolidation at shutdown
```

### Memory System

**5 Memory Types** (memory/corpus_manager.py):
- `EPISODIC`: Raw conversation turns
- `SEMANTIC`: Extracted facts (via fact_extractor.py)
- `PROCEDURAL`: Learned patterns and behaviors
- `SUMMARY`: Compressed conversation blocks
- `META`: Reflections and meta-patterns

**Hierarchical Organization**:
- Parent-child relationships between memories
- Temporal decay: `1.0 / (1.0 + decay_rate * age_days)`
- Importance boosting based on access recency
- Automatic summarization every N conversations (configurable)

### Key Components

**core/**
- `orchestrator.py`: Main request handler; prepares prompts, invokes models, persists interactions
- `response_generator.py`: Streaming generation with best-of-N and duel modes
- `prompt_builder.py`: Token-budgeted context assembly
- `competitive_scorer.py`: Judge-based response selection

**memory/**
- `memory_coordinator.py`: Unified memory interface; coordinates retrieval/storage across all memory types
- `corpus_manager.py`: JSON-based memory persistence
- `fact_extractor.py`: Entity/fact extraction from conversations
- `storage/multi_collection_chroma_store.py`: Vector storage backend

**processing/**
- `gate_system.py`: Multi-stage relevance filtering (FAISS → cosine → cross-encoder)

**models/**
- `model_manager.py`: Multi-provider LLM client (OpenAI, Anthropic, OpenRouter)
- `tokenizer_manager.py`: Token counting and budget tracking

**gui/**
- `launch.py`: Gradio web interface
- `handlers.py`: UI event handlers

**knowledge/**
- `WikiManager.py`: Wikipedia embedding search with FAISS
- `semantic_search.py`: General semantic retrieval utilities

**utils/**
- `logging_utils.py`: Centralized logging setup
- `file_processor.py`: PDF/DOCX/CSV ingestion
- `topic_manager.py`: Hybrid topic extraction (spaCy + optional LLM fallback)
- `time_manager.py`: Temporal context and decay calculations

**config/**
- `app_config.py`: Configuration loader with YAML/env merging; defines all paths, thresholds, and system prompt

## Configuration

**Environment Variables** (override defaults):
- `OPENAI_API_KEY`: Required for OpenAI models
- `CORPUS_FILE`: Path to memory JSON (default: `./data/corpus.json`)
- `CHROMA_PATH`: ChromaDB directory (default: `./chroma_db`)
- `CHROMA_DEVICE`: `cpu` or `cuda` (affects sentence-transformers)
- `PROMPT_TOKEN_BUDGET`: Max tokens for context (default: 2048)
- `SUMMARY_EVERY_N`: Conversations between auto-summaries (default: 20)
- `GATE_COSINE_THRESHOLD`: Relevance cutoff for gating (0.0-1.0)

**Performance Profiles** (Makefile.fast/balanced/max):
- Control memory limits, timeout thresholds, embedding batch sizes
- Example: `CHROMA_DEVICE=cpu PROMPT_MAX_MEMS=20 python main.py`

## Development Patterns

### Adding New Memory Types
1. Add enum to `memory/corpus_manager.py::MemoryType`
2. Update `memory_coordinator.py::add_memory()` to handle new type
3. Add ChromaDB collection in `storage/multi_collection_chroma_store.py`
4. Update prompt builder to include new memories in context

### Modifying Prompt Structure
- Edit `core/prompt_builder.py::build_prompt()`
- Respect token budgets (use `tokenizer_manager.count_tokens()`)
- Follow section ordering: recent → memories → facts → summaries → wiki → personality

### Changing Gating Thresholds
- Adjust `GATE_COSINE_THRESHOLD` in config or env
- Modify stage logic in `processing/gate_system.py`
- Lower threshold = more results (slower, noisier)
- Higher threshold = fewer results (faster, more precise)

### Testing Memory Flows
- Use `python main.py cli` for quick REPL-style testing
- Check `conversation_logs/` for persisted conversations
- Inspect ChromaDB: `python tests/inspect_chroma.py`
- Verify summaries: `python main.py inspect-summaries`

## Common Issues

**GPU Out of Memory**:
- Set `CHROMA_DEVICE=cpu` in environment
- Reduce `PROMPT_TOKEN_BUDGET` or `MODEL_MAX_TOKENS`
- Use `Makefile.fast` instead of `Makefile.max`

**Slow Retrieval**:
- Check Wikipedia index size (may need reprocessing)
- Increase `GATE_COSINE_THRESHOLD` to filter more aggressively
- Reduce `PROMPT_MAX_MEMS` or `PROMPT_MAX_SEMANTIC`

**Missing Dependencies**:
- Sentence-transformers models download on first run (~100MB)
- spaCy requires language model: `python -m spacy download en_core_web_sm`

## Critical: Claude Code Performance Guidelines

**IMPORTANT**: This project contains massive data directories (149GB total) that will cause Claude Code to freeze if searched broadly:

- `data/wiki/` - 102GB Wikipedia embeddings
- `data/pipeline/` - 43GB including Google Cloud SDK (17K+ Python files)
- Various vector indices and databases

**When performing searches or code analysis:**

1. **ALWAYS limit searches to source code directories only:**
   - Use patterns like `core/**/*.py`, `memory/**/*.py`, `models/**/*.py`
   - NEVER search the entire project with broad patterns like `**/*.py`

2. **Source code directories (safe to search):**
   - `core/` - Main orchestration and prompt logic
   - `memory/` - Memory management and storage
   - `models/` - LLM interfaces and management
   - `gui/` - Web interface components
   - `utils/` - Utility functions
   - `config/` - Configuration management
   - `tests/` - Test suite
   - `processing/` - Data processing and gating

3. **Data directories (AVOID in searches):**
   - `data/` - Large datasets, embeddings, indices
   - `scripts/` - May contain large external dependencies
   - Any directory with `chroma`, `faiss`, `wiki`, or `pipeline` in the name

**Example of correct search approach:**
- ✅ `glob: "core/**/*.py"` or `path: "core"`
- ✅ `grep pattern "function" path "memory/"`
- ❌ `glob: "**/*.py"` (will include 35K+ files and cause freezing)

This prevents timeouts and ensures responsive performance on simple requests like "add model contracts" or code modifications.

## Notes

- The system prompt is loaded from `core/system_prompt.txt` (header comments stripped)
- Conversation logs are written to `conversation_logs/` with timestamps
- Memory consolidation (summaries, fact extraction) runs at shutdown
- Wikipedia processing creates large files: `embeddings_mmap.dat` (~800MB), FAISS index (~350MB)
- All async operations use `asyncio.run()` or are coordinated by the orchestrator's event loop
