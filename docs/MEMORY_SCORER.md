# Memory Scorer

Standalone memory/document scoring and ranking engine for RAG systems.

## What it does

Scores and ranks memory documents (conversations, facts, summaries, reflections, etc.) using a **12-step pipeline**:

1. **Base relevance** + collection/source boost
2. **Recency decay** — two-regime temporal anchor (small/large window) or hourly fallback
3. **Evidence-based truth** — TruthScorer with time decay, confirmation boost, correction penalty
4. **Importance** — keyword and length heuristics
5. **Continuity** — stemmed token overlap with query + recency window bonus + tag matching
6. **Structural alignment** — numeric/operator density matching (math-aware)
7. **Penalties** — analogy penalty for math queries, size penalty for large irrelevant docs
8. **Anchor bonus** — deictic follow-up handling ("explain that again")
9. **Topic match** — alignment with current conversation topic
10. **Meta-conversational bonus** — for "what have we discussed" queries
11. **Graph proximity bonus** — optional knowledge-graph entity matching
12. **Staleness penalty** — outdated claim detection with steep threshold

## Quick start

```python
from memory_scorer import MemoryScorer, TruthScorer, ScorerConfig

# Basic usage
scorer = MemoryScorer()
ranked = scorer.rank_memories(memories, query="what is my cat's name")
# -> memories sorted by final_score descending

# Custom config
config = ScorerConfig(
    recency_decay_rate=0.08,
    collection_boosts={"facts": 0.20, "summaries": 0.10},
    staleness_enabled=True,
    debug=True,  # populates 'debug' dict on each memory
)
scorer = MemoryScorer(config=config)

# With conversation context (for continuity scoring)
scorer = MemoryScorer(
    conversation_context=[
        {"query": "how is my cat?", "response": "Flapjack is doing well!"}
    ]
)

# Intent-driven weight overrides
ranked = scorer.rank_memories(
    memories, query="what happened last tuesday",
    weight_overrides={
        "recency": 0.40,
        "relevance": 0.25,
        "_temporal_anchor_hours": 168,  # 1 week
    },
)
```

## Truth Scorer

Standalone evidence-based truth scoring engine. Facts start at a source-dependent initial score, gain truth through confirmations, lose truth through corrections/contradictions, and decay toward a floor when unconfirmed.

```python
ts = TruthScorer()

# Initial scores by source
ts.calculate_initial_score("user_stated")   # 0.80
ts.calculate_initial_score("llm_extracted") # 0.70
ts.calculate_initial_score("inferred")      # 0.50

# Adjustments
ts.apply_confirmation(0.7)    # 0.85
ts.apply_correction(0.7)      # 0.45
ts.apply_contradiction(0.7)   # 0.55

# Time decay (read-only, not persisted)
ts.apply_time_decay(0.8, last_confirmed_at=some_datetime)

# Main entry point — reads metadata dict, applies decay
ts.compute_effective_truth({"truth_score": 0.8, "last_confirmed_at": "..."})
```

## Graph scoring (optional)

Implement the `GraphScorerProtocol` to enable knowledge-graph proximity boosting:

```python
from memory_scorer import GraphScorerProtocol

class MyGraphScorer:
    def get_related_names(self, query: str) -> set[str]:
        # Return entity names related to entities in the query
        return {"flapjack", "cooking", "portland"}

scorer = MemoryScorer()
scorer.graph_scorer = MyGraphScorer()
```

Each related entity name found in a memory's text adds +0.05 to the score (capped at 0.15).

## Memory dict format

The scorer expects memory dicts with these fields (all optional with defaults):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `relevance_score` | float | 0.5 | Semantic similarity from retrieval |
| `timestamp` | str/datetime | now | Creation or last-update time |
| `collection` | str | "" | Source collection name |
| `query` | str | "" | Original query |
| `response` | str | "" | Original response |
| `content` | str | "" | Content (for non-Q/A memories) |
| `metadata` | dict | {} | Contains `truth_score`, `staleness_ratio`, etc. |
| `tags` | str/list | "" | Comma-separated or list of tags |
| `keyword_score` | float | 0.0 | Keyword match score (for size penalty) |
| `importance_score` | float | 0.5 | Importance score |
| `memory_type` | str | "" | e.g. "EPISODIC", "SUMMARY" |

## CLI

```bash
# Score a single memory
python memory_scorer.py score '{"relevance_score": 0.8, "timestamp": "2025-06-01T12:00:00", "metadata": {"truth_score": 0.7}}' --query "test"

# Compute effective truth from metadata
python memory_scorer.py truth '{"truth_score": 0.75, "last_confirmed_at": "2025-05-01T00:00:00"}'

# Run interactive demo
python memory_scorer.py demo
```

## Dependencies

Python 3.9+ standard library only. No external packages required.

## Tests

```bash
pytest tests/test_memory_scorer.py -v
```

94 tests covering all 12 scoring steps, TruthScorer, config, graph protocol, temporal anchoring, staleness, CLI, and edge cases.
