# Formal Model: Daemon RAG Agent

**Last verified against codebase**: 2026-05-09

## 1. Primitive Sets

Let the following sets be given:

- **Q** — the set of all natural-language query strings (user inputs)
- **R** — the set of all natural-language response strings (agent outputs)
- **D** — the set of all documents (memory entries), where each d in D is a tuple d = (content, metadata, embedding)
- **T** — the set of all tool calls (web search, memory search, memory expansion, code execution, Wolfram Alpha, file access, git stats)
- **A = R U T** — the action space (the agent either responds or invokes a tool)

---

## 2. State

The agent state at time t is:

```
s_t = (C_t, G_t, H_t, Theta_t, Lambda_t, U_t, E_t, Sigma_t)
```

where:

| Symbol | Name | Type | Code |
|--------|------|------|------|
| C_t | Corpus | P(D) — typed multiset of documents | ChromaDB (13 collections) + JSON corpus |
| G_t | Knowledge Graph | (V, E) — directed labeled graph | `graph_memory.py` (NetworkX DiGraph) + `entity_resolver.py` (alias table) |
| H_t | Conversation History | (Q x R)* — ordered sequence of past turns | Recent corpus entries + STM window |
| Theta_t | Conversation Thread | {thread_id, depth, topic, is_heavy} | `thread_manager.py` |
| Lambda_t | Open Threads | set of (topic, type, urgency, deadline) tuples | `thread_store.py` (ChromaDB `threads` collection) |
| U_t | User Model | key-value map of user facts/profile | `user_profile.py` |
| E_t | Escalation State | FSM state (see Section 11) | `escalation_tracker.py` |
| Sigma_t | Synthesis Memory | set of accepted SynthesisResult records with convergence metadata | `synthesis_memory.py` (ChromaDB `synthesis_results` collection) |

**Initial state** s_0: C loaded from persistent storage, G loaded from `data/knowledge_graph.json`, Lambda loaded from ChromaDB `threads` collection, Sigma loaded from ChromaDB `synthesis_results` collection, H = empty (new session), E = VALIDATE_AND_SUGGEST.

---

## 3. Context Function (Pre-Retrieval)

The context function transforms raw input into a structured representation for downstream use.

```
phi : Q x S -> X
```

where X is the **context space**, a structured record:

```
x = (q', q, tone, topics, intent, stm, identity, thread, facts, files)
```

The pipeline is a composition phi = phi_8 . phi_7 . ... . phi_1 where each stage enriches a partial context:

| Stage | Map | Description | Deterministic? |
|-------|-----|-------------|----------------|
| phi_1 | Q -> topics | Topic extraction (spaCy NER + LLM fallback) | Yes (LLM fallback rare) |
| phi_2 | Q x H -> tone in {CRISIS, ELEVATED, CONCERN, CONVERSATIONAL} | Emotional context analysis (harm scoring + semantic + LLM) | No (LLM) |
| phi_3 | Q -> files | File processing (PDF/DOCX/CSV text extraction via pdfplumber) | Yes |
| phi_4 | Q x tone -> is_heavy, facts | Heavy topic check + inline fact extraction | Partially (LLM for facts) |
| phi_4.5 | Q x tone -> intent | Intent classification (regex-first, 9 types, no LLM) | Yes |
| phi_5 | Q x H -> q' | Query rewriting for retrieval (LLM rewrites casual queries) | No (LLM) |
| phi_6 | H_24h x N_daily x Q -> stm | Short-term memory analysis (topic, intent, tone, reference_type, temporal_facts, open threads); H_24h = 24h time-windowed conversation slice, N_daily = last 2 daily notes from vault for cross-day recall disambiguation | No (LLM) |
| phi_6.5 | intent x stm -> intent' | STM intent refinement (if confidence < 0.50) | Yes |
| phi_7 | U -> identity | User identity injection (name, personality from profile) | Yes |
| phi_8 | Theta -> thread | Thread context (depth, topic, continuity) | Yes |

**Code**: `context_pipeline.py` -> `ContextResult`

---

## 4. Retrieval Function

Retrieval selects a relevant subset of the corpus given the processed context.

```
rho_iota : X x C x G -> D*
```

where D* is an **ordered** sequence of documents (ranked by relevance), and iota (intent) parameterizes both retrieval counts and scoring weights.

This decomposes into four stages:

### 4.0 Query Expansion (Pre-Retrieval)

```
expand : X x G -> Q'
```

Before candidate generation, the search query is expanded using the knowledge graph:

1. Extract entity IDs from query via alias resolution (trigram -> bigram -> unigram matching)
2. BFS to depth 2 from matched entities (traverses through hubs like "user")
3. Rank candidates by lateral connectivity: `score = min(non_hub_edges * 0.3, 1.0)` + single-word bonus (+0.1) / 3+ word penalty (-0.1)
4. Filter junk candidates (<=2 chars, 4+ words, digit-starting, temporal, measurements, verb phrases)
5. Append top K display names to original query (default K=8)

Example: "what about my brother" -> "what about my brother Auggie Mom Flapjack"

**Code**: `gatherer_memory.py:_expand_query_with_graph()` -> `graph_utils.py:rank_expansion_candidates()`

### 4.1 Candidate Generation

```
candidates : X x C -> P(D)
```

19 parallel retrieval tasks via `asyncio.gather()` (30s timeout). Codebase changes are fetched separately before the main gather.
- Conversations (recent by time + semantic by expanded query)
- Facts (user profile: hybrid 2/3 semantic + 1/3 recent)
- Summaries (recent + semantic, separate)
- Reflections (recent + semantic, separate)
- Wiki, personal notes, reference docs, user uploads
- Git commits, procedural skills, code proposals
- Visual memories (CLIP-matched image metadata)
- Knowledge graph BFS traversal (separate from expansion)
- Open threads (priority-ranked)
- Proactive cross-domain insights
- Web search (Tavily API, if triggered and available)

**Code**: `builder.py:build_prompt()` launches all tasks; retrieval methods split across `gatherer_memory.py`, `gatherer_knowledge.py`, and `gatherer_web.py` (composed via `context_gatherer.py`).

### 4.2 Gating

```
gate : X x P(D) -> P(D)
```

Multi-stage filter: Batch cosine similarity -> Cross-encoder reranking.
Each stage eliminates candidates below a threshold (overridable per-intent via `gate_threshold_override`).

**Code**: `processing/gate_system.py`

### 4.3 Scoring and Ranking

```
sigma_iota : D x X x G -> R
```

The composite score for a document d given context x is (parameterized by intent iota for weights, and knowledge graph G for graph bonus):

```
sigma_iota(d, x) = SUM_i w_i(iota) * f_i(d, x)  +  SUM_j b_j(d, x, G)  +  SUM_k p_k(d, x)
```

**Weighted factors** (w_i overridable per-intent via `weight_overrides`):

| Factor f_i | Default weight | Definition |
|-----------|---------------|------------|
| relevance(d, x) + collection_boost | 0.35 | Embedding similarity + per-collection bonus (config.yaml active values: facts +0.10, summaries +0.10, conversations +0.30, semantic +0.05, wiki +0.05) |
| recency(d) | 0.25 | Time decay (see temporal curves below) |
| truth(d) | 0.20 | Evidence-based reliability via TruthScorer.compute_effective_truth() (see truth decay below) |
| importance(d) | 0.05 | Content-based importance in [0,1] |
| continuity(d, Theta) | 0.10 | Thread continuity score (recent 10-min window) |
| topic_match(d, x) | 0.00 | Disabled by default. 1.0 (exact) / 0.5 (neutral) / 0.2 (different) |

**Structure score** (direct additive bonus, not in weight dict):

```
structure = 0.15 * density_alignment
```

**Bonuses** b_j (additive, not weighted):

| Bonus | Value | Condition |
|-------|-------|-----------|
| anchor_bonus (deictic) | 0.2 * anchor_overlap | Deictic follow-up AND anchor_overlap >= 0.05 |
| anchor_bonus (non-deictic) | 0.1 * anchor_overlap | Non-deictic with anchor overlap |
| meta_bonus | 0.15 / 0.12 / 0.10 | is_meta_conversational AND memory type in {EPISODIC / META / SUMMARY} |
| graph_bonus | min(0.05 * entity_matches, 0.15) | 0.05 per graph-connected entity found in memory content, capped at GRAPH_SCORING_BOOST_CAP |

**Penalties** p_k:

| Penalty | Value | Condition |
|---------|-------|-----------|
| deictic_anchor_penalty | -0.10 | Deictic follow-up AND anchor_overlap < 0.05 |
| analogy_penalty | -0.10 | numeric_op_density > 0.08 AND analogy markers AND "analogy" not in query |
| size_penalty | -0.25 * (size_bytes / 10000), capped at -1.0 | Document > 10KB AND keyword_score < 0.30 |
| staleness_penalty | -min(staleness_ratio * STALENESS_WEIGHT, 0.4) | staleness_ratio > 0 (summaries/reflections with outdated claims). 2x multiplier at ratio >= 0.8; reflections at 60% weight |
| deictic_drift | final_score *= 0.85 | Post-calculation: deictic AND continuity < 0.12 AND anchor_bonus < 0.04 |

**Temporal-aware recency** (when intent = TEMPORAL_RECALL with anchor window alpha hours):

```
recency(d) =
  { 1.0 - (age/alpha) * 0.3           if age <= alpha     (gentle decay within window)
  { 0.7 / (1 + lambda*(age - alpha))   if age > alpha      (standard decay from 0.7)
```

Default recency (no temporal anchor): `recency(d) = 1.0 / (1.0 + 0.05 * age_hours)`

**Truth decay** (TruthScorer, computed at read time, never persisted):

Initial scores depend on fact source: `user_stated=0.80`, `corrected=0.85`, `llm_extracted=0.70`, `inferred=0.50`. After creation or last confirmation, truth decays linearly per week:

```
decayed = current_score - (weeks_since_confirmed * DECAY_RATE)
clamped to [DECAY_FLOOR, current_score]
```

where `DECAY_RATE = 0.02` per week and `DECAY_FLOOR = 0.30`. Confirmations reset the decay clock and boost by `+0.08`. Corrections penalize by `-0.25`; cross-collection contradictions by `-0.15`.

**Code**: `memory/truth_scorer.py`

The final retrieval returns: rho(x, C) = top-K documents sorted by sigma descending.

**Code**: `memory_scorer.py` -> `rank_memories()`

---

## 5. Prompt Construction

```
beta : X x D* x iota x E -> P
```

where P is the **prompt space** (system prompt + context + query, token-budgeted). The **system prompt** is composed from two text files: an editable personality file (`config/prompts/default_personality.txt` or user's `custom_personality.txt`) concatenated with immutable operating principles (`config/prompts/operating_principles.txt`), with placeholder substitution for `{USER_NAME}`, `{USER_PRONOUNS}`, etc. This replaces the former JSON-based `PersonalityManager`.

The token budget is **model-aware**: computed as `min(context_window * 0.25, ceiling)` clamped to `[floor, ceiling]`, with separate caps for local vs API models. All ~20 context sections are now governed by the budget via an expanded PRIORITY_ORDER. Intent iota drives token budget allocation and retrieval count overrides. Escalation state E drives system prompt instructions and token budget caps. Post-budget **floor guarantees** ensure critical sections survive trimming: recent conversations (min 5), summaries (min 10), reflections (min 10). The agentic controller enforces its own budget on accumulated search context (`context_budget_tokens` default 8000) and trims low-value sections from the final prompt if total exceeds ceiling.

Assembly produces an ordered sequence of 27 conditional sections:

```
prompt = [
    [RECENT CONVERSATION]                    // always (recent turns with timestamps)
    [RELEVANT MEMORIES]                      // always (semantic hits with timestamps)
    [RECENT SUMMARIES]                       // if available (compressed recent history)
    [SEMANTIC SUMMARIES]                     // if available (relevant compressed history)
    [BACKGROUND KNOWLEDGE]                   // if available (Wikipedia)
    [WEB SEARCH RESULTS]                     // if triggered (Tavily results with sources)
    [RELEVANT INFORMATION]                   // if available (semantic chunks)
    [RECENT REFLECTIONS]                     // if available (recent meta insights)
    [SEMANTIC REFLECTIONS]                   // if available (relevant meta insights)
    [DREAMS]                                 // if enabled
    [USER'S PERSONAL NOTES]                  // if available (Obsidian vault, gated at 0.30)
    [USER UPLOADED ITEMS]                    // if available (files/images)
    [VISUAL MEMORIES]                        // if available (CLIP-matched image metadata)
    [DAEMON DOCUMENTATION]                   // if available (reference docs from docs/)
    [PROJECT COMMIT HISTORY]                 // if available (git commits)
    [ADAPTIVE WORKFLOWS]                     // if available (procedural skills)
    [PROPOSED FEATURES]                      // if available (code proposals)
    [KNOWLEDGE GRAPH]                        // if available (entity relationship sentences)
    [UNRESOLVED THREADS]                     // if available (open commitments/deadlines)
    [PROACTIVE INSIGHTS]                     // if available (cross-domain connections)
    [USER PROFILE]                           // if available (categorized facts + source excerpts + anti-confabulation instruction)
    [ACTIVE FEATURES]                        // always (compact feature inventory)
    [CODEBASE CHANGES SINCE LAST SESSION]    // first message only (git diff since last session)
    [TIME CONTEXT]                           // always (current datetime)
    [TEMPORAL GROUNDING]                     // if available (narrative life context)
    [SHORT-TERM CONTEXT SUMMARY]             // if available (STM: topic, intent, tone, threads)
    [CURRENT USER QUERY]                     // always (last exchange + current query)
]
```

Sections near the end receive higher attention weight in transformer models. The ordering places high-signal, low-token sections (user profile, time, STM, query) in the high-attention zone.

Token budget allocation is governed by intent — e.g., CASUAL_SOCIAL reduces max memories, EMOTIONAL_SUPPORT increases continuity weight. Token budget default: 40,000 tokens (API) / 12,000 (local) with two-tier compression: heavily oversized items (≥3x over token limit) get LLM summary via `_llm_compress_oversized()` (async parallel batch, ~0.3-0.5s), while mildly oversized items use middle-out character slicing (preserves start and end, compresses middle).

**Code**: `prompt/formatter.py` -> `_assemble_prompt()` (assembly), `prompt/builder.py` (orchestration)

---

## 6. The Agent Loop

The agent is a function:

```
Agent : Q x S -> R x S'
```

Internally:

```
AGENT(q, s):
    x    <- phi(q, s)                            // context pipeline (perceive)
    iota <- classify_intent(q, x.tone)           // intent classification (interpret)
    d*   <- rho_iota(x, s.C, s.G)                 // memory retrieval (remember)
                                                  // NOTE: web search runs as a PARALLEL
                                                  // retrieval task here, not post-generation
    pi_plan <- plan(q, x) if should_plan(x)      // response planning (parallel with d*)
                                                  // should_plan() bypasses: CASUAL_SOCIAL intent,
                                                  // crisis/elevated tone, <8 word queries, disabled
    p    <- beta(x, d*, iota, E_t, pi_plan)      // prompt construction (plan injection)
                                                  // Plan injected via format_plan_injection():
                                                  // appends [RESPONSE PLAN] section to system prompt
                                                  // with key_points, tone, avoid, strategy fields
    a_0  <- LLM(p)                                // first generation (act)

    // UNCERTAINTY FALLBACK (post-generation, optional)
    // Dual-layer detection: keyword regex then semantic embedding
    //   Layer 0: Length guard — strip hedge prefixes, skip if substantive > 400 chars
    //   Layer 1: ~18 compiled regex patterns (confidence 0.75–0.90), min confidence 0.70
    //   Layer 2: Cosine similarity of first 300 chars against 8 pre-embedded anchor
    //            sentences (e.g. "I don't have any information about that in my memory."),
    //            threshold 0.70
    // Returns UncertaintyResult(is_uncertain, confidence, trigger_type, matched_pattern)
    if UNCERTAINTY_FALLBACK_ENABLED and is_uncertain(a_0):
        agentic_search_trigger := true               // force agentic retry

    // POST-ANSWER REVIEW GATE (optional)
    // If response planning produced a plan, review the answer against it
    if RESPONSE_REVIEW_ENABLED and pi_plan exists:
        review <- review_answer(a_0, pi_plan)        // lightweight LLM check (~300 tokens)
        if not review.passes and review.confidence >= 0.90:
            agentic_search_trigger := true            // force agentic retry with review feedback

    // AGENTIC SEARCH LOOP (ReAct pattern, optional)
    // Triggered by LLM-first decision OR uncertainty fallback
    if agentic_search_trigger(a_0, x):
        inventory <- compute_context_inventory(d*)    // what RAG already gathered
        observations <- {}
        for i = 1, 2, ..., N_max:                     // default N_max = 5
            thought_i <- LLM(p, observations, inventory)
            tool_i    <- extract_tool_call(thought_i)
            if tool_i = empty: break
            obs_i     <- execute(tool_i)              // Tavily, Wolfram, E2B, memory_search, expand_memory, file_read/grep/list, git_stats
            observations <- observations + {obs_i}
        r <- LLM(p, observations)                     // final synthesis
    else:
        r <- a_0

    // STATE UPDATE
    s' <- delta(s, q, r)
    return (r, s')
```

**Key distinction**: Web search runs as one of the 19 parallel retrieval tasks during rho (Section 4.1). The agentic search loop is a separate, heavier mechanism that fires post-generation when the LLM determines it needs additional real-time information. The agentic loop receives a **context inventory** summarizing what RAG already gathered to prevent redundant searches.

### 6.1 Memory Expansion (expand_memory)

Within the agent loop, the LLM may invoke `expand_memory` to zoom into a search hit:

```
mu : D_id x W x C_name -> D_context

mu(d_id, w, c):
    d_anchor <- get_by_id(c, d_id)                   // fetch anchor document
    if c = "summaries":
        // Source-docs strategy: retrieve original conversations
        if d_anchor.metadata.source_doc_ids exists:
            D_source <- fetch(source_doc_ids)          // direct backlinks
        else:
            D_source <- query_range(conversations,     // temporal anchor fallback
                          d_anchor.temporal_anchor_start,
                          d_anchor.temporal_anchor_end)
        return format(d_anchor, D_source)
    else:
        // Timestamp-window strategy: chronological neighbors
        D_all   <- list_all(c), sorted by (timestamp, doc_id)
        idx     <- index_of(d_anchor, D_all)
        D_window <- D_all[max(0, idx-w) : idx+w+1]
        return format(d_anchor, D_window)
```

Session-gated: `expand_count <= EXPAND_MAX_PER_SESSION` (default 3). Cached per `(d_id, w, c)` tuple.

**Code**: `memory/memory_expander.py` -> `MemoryExpander.expand()`

The agentic search controller is decomposed into three classes:
- `AgenticSearchController` (`core/agentic/controller.py`) — main loop orchestration, prompt building, model interaction
- `ToolExecutor` (`core/agentic/tools.py`) — dispatch routing + execution for all 12 tool types (web, Wolfram, sandbox, memory search/expand, file read/grep/list, git stats, full document, recall image, signal done)
- `AgenticFormatter` (`core/agentic/formatters.py`) — pure stateless formatting for all result types (conversations, memories, web results, etc.)

Protocol dispatch uses `detect_protocol()` (`core/agentic/protocols.py`) to choose between `NativeToolsHandler` (OpenAI/Anthropic function calling) and `XMLMarkerHandler` (local models using XML tags) based on model name. Both handlers parse responses into `SearchDecision` objects with a shared interface.

**Code**: `orchestrator.py` -> `process_user_query()` and `core/agentic/controller.py`

### 6.2 Git Stats (git_stats)

Within the agent loop, the LLM may invoke `git_stats` to query repository history:

```
git_stats : Q_nl -> {success, summary, output, commands_run}

git_stats(q_nl):
    intent  <- keyword_match(q_nl)                     // commit_count | recent_commits |
                                                       // files_changed | contributors |
                                                       // branches | status | diff_stat
    window  <- extract_temporal(q_nl)                  // "this week" -> --since=<ISO date>
    cmd     <- build_command(intent, window)
    assert cmd.subcommand in {log, shortlog, diff,     // safety: allowlist only
              status, branch, rev-list, rev-parse,
              show, describe, tag, stash}
    output  <- subprocess.run(cmd)
    summary <- truncate_and_summarize(output)
    return {success, summary, output, commands_run}
```

**Code**: `core/agentic/tools.py` -> `_execute_git_stats()`

---

## 7. State Transition (Per-Turn)

```
delta : S x Q x R -> S'
```

After each turn, the state is updated:

```
delta(s, q, r):
    // ALWAYS (immediate, synchronous)
    s'.C  <- s.C + {new_episodic_memory(q, r, truth_score, importance)}     // corpus + ChromaDB
    s'.H  <- s.H ++ (q, r)                                                  // conversation context deque
    s'.Theta <- detect_or_create_thread(q, s.Theta)                          // conversation threading
    s'.E  <- escalation_update(s.E, x.tone, r)                              // FSM state transition

    // ALWAYS (lightweight, pattern-based)
    corrections <- detect_corrections(q, s.U)                                // regex patterns
    confirmations <- detect_confirmations(q, s.U)                            // regex patterns
    for event in corrections + confirmations:
        s'.U <- apply_truth_event(s.U, event)                               // truth score adjustment

    // OPTIONAL (if FACTS_EXTRACT_EACH_TURN enabled, default: disabled)
    if FACTS_EXTRACT_EACH_TURN:
        facts <- extract_facts(q)                                            // regex + LLM
        for fact in facts:
            verdict <- fact_verify(fact, s.C)                                // conflict detection
            if verdict != REJECT:
                s'.C <- s'.C + {fact}
                if verdict == STORE_AND_FLAG:
                    mark_old_as_superseded(s'.C, fact)                       // metadata update, no deletion
                if KNOWLEDGE_GRAPH_ENABLED and is_graph_worthy(fact):
                    s'.G <- s'.G + fact_to_edge(fact)                        // graph ingestion

    // PER-TURN THREAD RESOLUTION (regex-based, ~1ms)
    s'.Lambda <- check_quick_resolutions(q, s.Lambda)                        // resolve threads matching completion signals
    s'.U <- s'.U                                                             // profile unchanged (except truth events above)

    return s'
```

**Code**: `memory_storage.py:store_interaction()`, `orchestrator.py` (post-response section), `correction_detector.py`

---

## 8. Session Lifecycle (Shutdown Transition)

At session end, a heavier transition occurs in strict order:

```
delta_shutdown : S -> S'
```

```
delta_shutdown(s):
    // Step 1: Block summaries
    summaries <- consolidate(s.H, block_size=N)              // LLM compression of N-turn blocks
    s'.C <- s.C + summaries

    // Step 2: Session facts (regex-based, last 10 turns)
    facts_regex <- extract_session_facts(s.H[-10:])
    for fact in facts_regex:
        verdict <- fact_verify(fact, s'.C)                    // conflict detection
        if verdict == STORE:
            s'.C <- s'.C + {fact}
        if verdict == STORE_AND_FLAG:
            s'.C <- s'.C + {fact}
            mark_old_as_superseded(s'.C, fact)                // metadata flag, no deletion

    // Step 3: LLM-assisted facts (last 12 turns, batch verified)
    facts_llm <- llm_extract_triples(s.H[-12:])
    verdicts <- fact_verify_batch(facts_llm, s'.C)
    for (fact, verdict) in zip(facts_llm, verdicts):
        if verdict != REJECT:
            s'.C <- s'.C + {fact}
            if fact.subject == "user":
                s'.U <- add_to_profile(s'.U, fact)            // user facts -> UserProfile
            // entity facts (subject != "user") stay in ChromaDB only
            if KNOWLEDGE_GRAPH_ENABLED and is_graph_worthy(fact):
                s'.G <- s'.G + fact_to_edge(fact)

    // Step 4: Procedural skills (0-3 patterns)
    skills <- extract_skills(s.H[-12:])                       // LLM: trigger -> action patterns
    s'.C <- s'.C + skills

    // Step 5: Code proposals (0-5 proposals)
    proposals <- generate_proposals(s.H, s'.C)                // LLM with pipeline-enriched context
    s'.C <- s'.C + proposals

    // Step 6: Open thread processing (3 phases)
    // 6a: Resolution detection
    resolved <- detect_resolutions(s.H, s.Lambda)             // LLM checks if threads addressed
    s'.Lambda <- mark_resolved(s.Lambda, resolved)
    // 6b: New thread extraction
    new_threads <- extract_threads(s.H[-N:])                  // LLM: commitments, deadlines, questions
    s'.Lambda <- s'.Lambda + new_threads
    // 6c: Cap enforcement
    if |s'.Lambda| > THREAD_MAX_OPEN:
        s'.Lambda <- prune_lowest_priority(s'.Lambda, THREAD_MAX_OPEN)   // ONLY unconditional deletion

    // Step 7: Knowledge graph persistence
    save_to_disk(s'.G, "data/knowledge_graph.json")           // dirty-flag optimization
    save_to_disk(aliases, "data/entity_aliases.json")

    // Step 8: Cross-collection deduplication (DRY-RUN ONLY)
    dedup_plan <- scan_duplicates(s'.C, threshold=0.92)       // cosine similarity
    contradictions <- scan_contradictions(s'.C)                // same subject+predicate, diff object
    log_preview(dedup_plan, contradictions)                    // NEVER auto-deletes
    // Live deletions require explicit GUI action (Preview/Run buttons)

    // Step 9: Session-end reflection
    reflection <- reflect(s.H, summaries)                     // LLM meta-reflection
    s'.C <- s'.C + {reflection}

    return s'
```

**Critical invariant**: No user data is auto-deleted at shutdown. Dedup runs dry_run=True only. Thread cap enforcement (Step 6c) is the only deletion, and it removes lowest-priority threads when over the cap.

**Code**: `shutdown_processor.py`

---

## 9. Memory Types as a Typed Corpus

The corpus C is a **typed multiset** partitioned across 13 ChromaDB collections in 8 categories:

```
C = C_episodic  U  C_semantic  U  C_procedural  U  C_summary  U  C_reference  U  C_meta  U  C_synthesis  U  C_visual
```

| Category | Collections | Characteristics |
|----------|-------------|-----------------|
| Episodic | `conversations` | Raw turns. Recency-biased retrieval. Protected from dedup. |
| Semantic | `facts`, `wiki_knowledge` | Triples + external knowledge. Truth-scored. Wiki queries route through FAISS (40M vectors); ChromaDB `wiki_knowledge` is fallback only. Wiki protected from dedup. |
| Procedural | `procedural`, `procedural_skills` | Git commits + reusable patterns. Skill dedup at 0.85 threshold. |
| Summary | `summaries` | Block-compressed conversation history. Relevance-biased. |
| Reference | `obsidian_notes`, `reference_docs` | User notes + system docs. Protected from dedup. Gated at 0.30 threshold. |
| Meta | `reflections`, `threads`, `proposals` | Session insights + open loops + code plans. Priority-scored (threads). |
| Synthesis | `synthesis_results` | Cross-domain insights with convergence tracking. Produced by shutdown dreaming. |
| Visual | `visual_memories` | CLIP-embedded image metadata for visual recall. Intent-gated (disabled for casual/emotional/meta). Images dropped for non-vision models. |

**Knowledge graph** G provides a secondary index over C_semantic. Nodes are entities, edges are relations extracted from facts. Objects with 4+ words are stored as node metadata (not nodes) to prevent junk.

**Protected collections** (never scanned by deduplicator): `conversations`, `obsidian_notes`, `reference_docs`, `wiki_knowledge`.

**Dedup priority** (when choosing which duplicate to keep): summaries(5) > reflections(4) > skills(3) > proposals(2) > facts(1). Ties broken by recency.

---

## 10. Intent as a Scoring Lens

Intent classification selects a **lens** — a set of parameter overrides that reshape scoring without changing the scoring function itself.

```
iota : Q x tone -> IntentResult = (type, confidence, w_override, r_override, g_override)
```

**9 intent types**: FACTUAL_RECALL, TEMPORAL_RECALL, EMOTIONAL_SUPPORT, CASUAL_SOCIAL, TECHNICAL_HELP, CREATIVE_EXPLORATION, META_CONVERSATIONAL, PROJECT_WORK, GENERAL.

Each intent type has a corresponding entry in the `_PROFILES` dict (`intent_classifier.py`) specifying concrete overrides. For example, FACTUAL_RECALL boosts truth weight (0.30 vs default 0.20) and relevance (0.40 vs 0.35) while cutting recency (0.10 vs 0.25); CASUAL_SOCIAL zeroes out wiki, skills, proposals, git, and reference docs retrieval counts; TEMPORAL_RECALL boosts recency weight (0.40) and threads a `_temporal_anchor_hours` key to reshape the scorer's decay curve. GENERAL uses all defaults unchanged.

where:
- **w_override** is a subset of R^6 overriding the weight vector [w_relevance, w_recency, w_truth, w_importance, w_continuity, w_topic_match]
- **r_override** overrides retrieval counts (max memories, max facts, max summaries, etc.)
- **g_override** overrides the gating threshold

This means the scoring function sigma is **parameterized by intent**:

```
sigma_iota(d, x) = SUM_i w_i(iota) * f_i(d, x)  +  bonuses  +  penalties
```

The same retrieval infrastructure serves all 9 intent types — only the parameters change.

**Classification is regex-first** (no LLM calls): patterns checked in priority order, highest confidence wins. Tone bias: HIGH/MEDIUM tone biases toward EMOTIONAL_SUPPORT for ambiguous queries. Low-confidence results (< 0.50) refined by STM free-text intent field.

**Code**: `intent_classifier.py`

---

## 11. Escalation as a Finite State Machine

The escalation tracker is a separate FSM running alongside the agent:

```
States: {VALIDATE_AND_SUGGEST (VAS), GROUNDING_PRESENCE (GP),
         QUIET_COMPANIONSHIP (QC), GENTLE_REENGAGEMENT (GR)}
```

**Tracked variables**:
- `consecutive_elevated_count` — increments on CRISIS/ELEVATED, resets on calm
- `consecutive_calm_count` — increments on CONCERN/CONVERSATIONAL, resets on elevated
- `ignored_suggestion_count` — tracks unengaged suggestions (caps at +2/-1)
- `last_need_type` — PRESENCE or PERSPECTIVE (from emotional context)

**Transitions** (evaluated in this order):

```
// De-escalation path (tone just dropped)
if NOT elevated AND was_recently_elevated:
    if consecutive_calm > deescalation_window:   -> VAS     // window expired
    if need_type == PERSPECTIVE:                 -> VAS     // analytical shift, skip gentle
    else:                                        -> GR      // genuine calming

// No escalation
if NOT elevated:                                 -> VAS

// Escalation path
if consecutive_elevated < threshold:             -> VAS     // not yet escalated
if consecutive_elevated > threshold
   AND ignored_suggestions >= 2:                 -> QC      // suggestions ignored
else:                                            -> GP      // at/past threshold
```

**FSM output modifies prompt construction**:

| State | System prompt instruction | Token budget override |
|-------|--------------------------|----------------------|
| VAS | (none — uses standard tone instructions) | (none) |
| GP | "Maximum 2-3 sentences, pure acknowledgment, no advice" | 500 |
| QC | "Maximum 1-2 sentences, just be present, no suggestions" | 300 |
| GR | "2-4 sentences, ONE small concrete suggestion allowed" | 800 |

**PERSPECTIVE nuance**: When a user shifts from emotional distress to analytical/meta thinking (need_type = PERSPECTIVE), this is an intensity shift, not genuine calming. The FSM skips GENTLE_REENGAGEMENT and returns directly to full engagement (VAS).

**Code**: `escalation_tracker.py`

---

## 12. Provenance Audit Trail

Each stored interaction carries a **provenance record** Pi attached as ChromaDB metadata:

```
Pi : {session_id, response_mode, model_name, thinking_block,
      cited_memory_ids, prompt_hash, agentic_summary}
```

**Fields**:
- `session_id` — UUID generated once per `MemoryCoordinator` instance (i.e., per GUI session)
- `response_mode` — one of `{agentic, enhanced, best_of, best_of_duel, fallback}`
- `model_name` — the LLM alias that produced the response (or the duel winner)
- `thinking_block` — captured `<thinking>...</thinking>` content from extended-thinking models (truncated to `PROVENANCE_THINKING_MAX_CHARS`)
- `cited_memory_ids` — list of memory document IDs referenced by the response (via `[MEM_RECENT_N]` / `[MEM_SEMANTIC_N]` citation markers injected by the agentic controller)
- `prompt_hash` — SHA-256 of the final assembled prompt, enabling prompt-to-response traceability
- `agentic_summary` — one-line summary from `AgenticSearchSession.get_provenance_summary()`: tools used, round count, searches, memory expansions

**Attachment point**: `memory_storage.py:store_interaction()` merges the provenance dict into the metadata of the stored conversation document. The GUI `handlers.py` builds the provenance dict for all 5 response modes before calling `store_interaction()`.

**GUI**: The "Provenance" tab (renamed from "Citations") displays per-turn provenance in the chat interface.

**Config**: `PROVENANCE_ENABLED` (default True), `PROVENANCE_THINKING_MAX_CHARS` (default 4000).

**Code**: `memory/memory_storage.py`, `gui/handlers.py`, `core/agentic/controller.py`, `core/agentic/types.py`, `core/response_parser.py`

---

## 13. Synthesis Filter Pipeline

The synthesis pipeline transforms candidate cross-domain connections into validated insights. It operates on the synthesis memory Sigma_t, independent of the conversational agent loop.

### 13.1 Candidate Generation (Three-Tier)

Three generators run in parallel at shutdown, producing candidates for the shared filter:

```
Tier 0 — RETRIEVAL (RetrievalSynthesisGenerator):
  For each personal fact f in sample(C_facts):
    q_struct <- LLM_few_shot(f)                    // structural query extraction
    results  <- FAISS_search(q_struct, k=5)        // 40M Wikipedia vectors
    claim    <- LLM_adversarial(f, results)        // adversarial evaluation
    c        = (entity(f), wiki_article, claim, [], domains, dist)

Tier 1 — WALK (GraphWalkGenerator):
  walk      <- biased_markov_walk(G, start, steps)  // Node2Vec-style, 2.0x personal return bias
  constraint: |domains(walk)| >= 2, hub_dampening(degree > 15)
  claim     <- LLM_narrate(walk)                    // walk narration prompt
  c         = (walk[0], walk[-1], claim, walk, domains, dist)

Tier 2 — XSTORE (SynthesisGenerator):
  a         <- sample(C_facts)                      // personal entity
  b         <- FAISS_sample(C_wiki)                 // random Wikipedia article
  claim     <- LLM_bridge(a, b)                     // bridge articulation
  c         = (a, b, claim, [], domains, dist)
```

Each generator produces `SynthesisCandidate` objects with a shared schema:

```
c = (a, b, claim, path, domains, dist, generator_tier)
```

where a, b in V (graph nodes or entity references), claim in Q (natural language), path = [v_1, ..., v_k] is the walk sequence (empty for retrieval/xstore), domains subset of {family, health, work, hobby, ...}, dist = cosine_distance(embed(a), embed(b)).

**Key difference**: Tier 0 (retrieval) extracts structural queries from personal facts via few-shot LLM prompting, then searches FAISS for Wikipedia articles that share the structural pattern. The LLM evaluates adversarially rather than inventing connections. This produces candidates that name specific mechanisms ("conditional dependency", "historical layering") rather than surface metaphors.

**Tier 1 gate**: GraphWalkGenerator requires `count_bridge_edges() >= GRAPH_WALK_MIN_BRIDGE_EDGES` (40). If insufficient bridges exist, Tier 1 produces zero candidates and its quota is filled by Tier 2.

### 13.2 Filter Function

The filter is a composition of 7 stages, each a gate g_i : SynthesisResult -> StageResult:

```
F(c) = g_6 . g_5 . g_4 . g_3 . g_2 . g_1 . g_0 (c)
```

The pipeline short-circuits: if any g_i.passed = false, subsequent stages do not execute.

| Stage | Gate | Condition | Cost |
|-------|------|-----------|------|
| g_0 | Text sanity | \|tokens(claim)\| >= 10 AND repetition_ratio <= 0.5 AND has_verb(claim) | O(n) string ops |
| g_1 | Domain crossing | \|domains\| >= 2 | O(1) |
| g_2 | Semantic distance | 0.20 <= dist <= 0.90 | O(1) |
| g_3 | External novelty | 3 sub-checks: (a) max_sim(claim, C_wiki) < 0.88 (near-verbatim only), (b) max_sim("a b", C_wiki) < 0.85 (co-occurrence, 40M-scale recalibrated), (c) template_sim(claim) via regex generic pattern detection | O(log n) vector search |
| g_4 | Internal novelty | path_hash not in existing.unique_paths (convergence pass) OR no match | O(log m) vector search |
| g_5 | Coherence judge (`claude-opus-4.6`) | Two-pass: Pass 1 LLM_structural_coherence(a, b, claim) >= MODERATE; Pass 2 (MODERATE only) LLM_factual_skeptic(claim) = PASS | O(1-2) LLM calls |
| g_6 | Composite score | composite(c) >= 0.65 | O(1) arithmetic |

### 13.3 Composite Scoring

```
score(c) = w_coh * coh(c) + w_nov * nov(c) + w_dist * dist_score(c) + w_str * str(c)

where:
  w_coh = 0.30, w_nov = 0.40, w_dist = 0.15, w_str = 0.15
  coh(c) = CoherenceLevel.value in {0.0, 0.33, 0.66, 1.0}
  nov(c) = w_claim * (1 - claim_sim)           // claim novelty
         + w_cooc * (1 - cooccurrence_sim)      // co-occurrence novelty
         + w_spec * (1 - template_sim)          // specificity
         + w_int  * (1 - max_sim_internal)      // internal novelty
    where w_claim = 0.25, w_cooc = 0.30, w_spec = 0.25, w_int = 0.20
  dist_score(c) = 1 - |dist - mid| / half    (peaks at midpoint of [0.20, 0.90])
  str(c) = min(|domains| / 4, 1.0)
```

### 13.4 Convergence Detection

Independent rediscovery from different paths is the key signal. For a result r in Sigma:

```
convergence_strength(r) = |unique_paths(r)| * |unique_sources(r)|
```

A result is promoted to CONVERGING status when:
```
|unique_paths(r)| >= 3  AND  |unique_sources(r)| >= 2
```

When Stage 4 finds an existing match with a new path_hash, it passes (score 0.5) rather than rejecting. At storage time (Stage 7), the existing result's convergence metadata is updated.

### 13.5 State Transition

After filtering:

```
Sigma_{t+1} = Sigma_t  U  {r}      if F(c).status = ACCEPTED
            = Sigma_t               if F(c).status = REJECTED

For convergence updates (existing result r' matched by new path):
  r'.unique_paths   = r'.unique_paths   U  {c.path_hash}
  r'.unique_sources = r'.unique_sources U  {a|b}
  r'.convergence_strength = |r'.unique_paths| * |r'.unique_sources|
```

### 13.6 Bridge Feedback (Provisional Edges)

On acceptance, a provisional bridge edge is created in the knowledge graph:

```
If F(c).status = ACCEPTED:
  G_{t+1}.E = G_t.E  U  {(c.a, c.b, relation=claim_summary, weight=0.5, provisional=true)}
```

Provisional bridges mature to full weight (1.0) when independently rediscovered via convergence (Section 13.4). This creates a feedback loop: accepted synthesis insights densify the graph, enabling the walk generator (Tier 1) to produce candidates that cross the personal-wikidata boundary.

**Code**: `knowledge/synthesis_models.py`, `knowledge/synthesis_filter.py`, `knowledge/synthesis_retriever.py`, `knowledge/graph_walk_generator.py`, `knowledge/synthesis_generator.py`, `memory/synthesis_memory.py`

---

## 14. Summary: The Complete Agent

Putting it all together:

```
Agent(q, s_t) =
    let x       = phi(q, s_t)                       in    // perceive
    let iota    = classify_intent(q, x.tone)         in    // interpret
    let q'      = expand(q, s_t.G)                   in    // expand (graph-augmented query)
    let d*      = rho_iota(x, q', s_t.C, s_t.G)       in    // remember (19 parallel retrievals)
    let pi_plan = plan(q, x) if should_plan(x)      in    // plan response (parallel with remember)
    let p       = beta(x, d*, iota, s_t.E, pi_plan) in    // assemble (27-section prompt + plan injection)
    let r       = generate_or_search(p)              in    // act (LLM + optional agentic loop)
    let r'      = review_or_retry(r, pi_plan)        in    // review (post-answer gate + optional retry)
    let Pi      = provenance(r', p, session)          in    // audit (provenance metadata)
    let s_{t+1} = delta(s_t, q, r', Pi)               in    // learn (store + truth + escalation + provenance)
    (r', s_{t+1})
```

Ten operations. Perceive, interpret, expand, remember, plan-response, assemble, act, review, audit, learn.

---

## Appendix: Symbol Reference

| Symbol | Meaning | Code Location |
|--------|---------|---------------|
| Q | Query space | User input strings |
| R | Response space | Agent output strings |
| S | State space | Distributed across all persistence |
| C | Corpus (typed multiset, 13 collections) | ChromaDB + corpus JSON |
| G | Knowledge graph | `graph_memory.py` + `entity_resolver.py` |
| H | Conversation history | Recent corpus + STM |
| Theta | Conversation thread state | `thread_manager.py` |
| Lambda | Open threads (proactive surfacing) | `thread_store.py` + `thread_extractor.py` |
| U | User model | `user_profile.py` |
| E | Escalation FSM state | `escalation_tracker.py` |
| X | Context space | `ContextResult` dataclass in `context_pipeline.py` |
| P | Prompt space (27 sections) | `prompt/formatter.py` -> `_assemble_prompt()` (assembly), `prompt/builder.py` (orchestration) |
| A = R U T | Action space | Response or tool call |
| phi | Context function (8 integer stages + 2 half-stages) | `context_pipeline.py` |
| rho_iota | Retrieval function (19 parallel tasks, parameterized by intent) | `context_gatherer.py` (compositor) + `gatherer_memory.py` + `gatherer_knowledge.py` + `gatherer_web.py` + `memory_retriever.py` + `memory_scorer.py` |
| sigma_iota | Scoring function (parameterized by intent + graph) | `memory_scorer.py` -> `rank_memories()` |
| beta | Prompt construction (X x D* x iota x E -> P) | `prompt/builder.py` |
| iota | Intent classification | `intent_classifier.py` |
| delta | State transition (per-turn) | `memory_storage.py:store_interaction()` + `orchestrator.py` |
| delta_shutdown | State transition (session-end) | `shutdown_processor.py` |
| expand | Query expansion (graph-augmented) | `gatherer_memory.py:_expand_query_with_graph()` + `graph_utils.py` |
| mu | Memory expansion (temporal window / summary drill-down) | `memory/memory_expander.py` |
| Pi | Provenance record (session_id, response_mode, prompt_hash, ...) | `memory_storage.py` + `gui/handlers.py` |
| Sigma | Synthesis memory (accepted SynthesisResult set) | `synthesis_memory.py` (ChromaDB `synthesis_results`) |
| F | Synthesis filter (7-stage pipeline g_0 . ... . g_6; g_3 has 3 sub-checks; g_5 has 2-pass structure) | `synthesis_filter.py` |
| Gen_0 | Retrieval synthesis generator (structural query -> FAISS -> adversarial eval) | `knowledge/synthesis_retriever.py` |
| Gen_1 | Graph walk generator (biased Markov walk -> narration) | `knowledge/graph_walk_generator.py` |
| Gen_2 | Cross-store synthesis generator (random pairing -> bridge articulation) | `knowledge/synthesis_generator.py` |
| is_uncertain | Uncertainty detection (length guard + ~18 keyword regex patterns + 8 semantic anchors, threshold 0.70) | `core/uncertainty_detector.py` |
| pi_plan | Response plan (key_points, tone, avoid, strategy) | `core/response_planner.py` |
| plan | Response planning function (lightweight LLM call) | `core/response_planner.py:create_plan()` |
| review_answer | Post-answer review against plan | `core/response_planner.py:review_answer()` |
| gate | Multi-stage gating | `processing/gate_system.py` |
