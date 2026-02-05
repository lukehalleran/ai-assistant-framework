# Daemon Project Goals

**Last Updated**: 2026-02-05

---

## Mission

Daemon is a personal cognitive assistant with deep, persistent memory. It should feel like talking to someone who genuinely knows you — your projects, preferences, history, and context — not a stateless chatbot that forgets everything between sessions.

---

## Active Goals (Current Sprint)

### 1. Complete Modular Refactor (`refactor/prompt-modular`)
- **Status**: In progress
- Orchestrator should be a thin coordinator, not a god object
- All query analysis flows through `ContextPipeline`
- All prompt building flows through `UnifiedPromptBuilder`
- Memory system delegates to `MemoryRetriever`, `MemoryStorage`, `MemoryScorer`, `ShutdownProcessor`
- Eliminate remaining calls to deprecated `prepare_prompt`
- Target: orchestrator.py under 400 lines

### 2. Code Proposals System (Goal-Directed)
- **Status**: In progress (Phases 1-3 complete, GUI + shutdown integration done)
- LLM-generated proposals for codebase improvements, stored in ChromaDB
- Proposals generated at shutdown from session context and on-demand from GUI
- Deduplication via semantic similarity
- Next: improve proposal quality with richer project context, CLAUDE.md injection

### 3. Agentic Search Reliability
- **Status**: Stabilizing
- ReAct loop with Tavily + Wolfram + E2B sandbox
- LLM-first trigger (not keyword heuristic) decides when to search
- Query relaxation for failed searches
- Next: improve loop termination, reduce unnecessary search triggers

---

## Medium-Term Goals (Next 1-3 Months)

### 4. Memory Quality Over Quantity
- Summaries, facts, and skills should be high-signal, not noisy
- Improve fact extraction precision (reduce false triples)
- Better deduplication across collections (facts, summaries, skills)
- Prune low-value entries over time (decay + consolidation)

### 5. Temporal Awareness
- Narrative context system provides rolling "life state" grounding
- Daily + weekly notes auto-generated from conversations
- Goal: Daemon should know what you were working on last week without being told

### 6. Multi-Provider LLM Resilience
- Duel mode (2 models + judge) for high-stakes responses
- Best-of-N with diverse temperatures
- Graceful fallback chains: primary API -> secondary API -> local model
- Token budget management across providers with different context limits

### 7. Personal Knowledge Integration
- Obsidian vault with multimodal image support
- Reference document uploads (technical docs, project outlines)
- Git commit history as procedural knowledge
- Goal: all personal knowledge surfaces when relevant, not just chat history

---

## Long-Term Vision

### 8. Proactive Intelligence
- Daemon should surface relevant information before being asked
- Code proposals are the first step — extend to life reminders, project nudges
- Pattern recognition across sessions (recurring themes, unresolved questions)

### 9. Production Readiness
- Desktop executable via PyInstaller (daemon.spec exists)
- Docker deployment (Dockerfile + docker-compose exist)
- Health check endpoint for monitoring
- Graceful shutdown with full memory persistence

### 10. Privacy and Trust
- All data stays local (ChromaDB on disk, JSON corpus)
- API calls only for LLM generation (no telemetry, no cloud storage)
- User profile and identity managed locally
- Crisis-aware tone detection with appropriate response escalation

---

## Non-Goals (Explicit Exclusions)
- **Auto-execution of code proposals** — proposals are advisory only, human reviews and implements
- **Real-time collaboration** — single-user system, not multi-tenant
- **Mobile/web deployment** — desktop-first (Gradio GUI or CLI)
- **Fine-tuning or training** — uses commercial APIs and pre-trained local models only
