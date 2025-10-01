"""
Minimal competitive field scorer - Prototype for fail-fast validation

This scores candidates from multiple fields and packs them under a token budget
by value-per-token instead of using fixed per-field allocations.

Usage:
    context = await builder._gather_context(...)
    context = apply_competitive_selection(context, query, tokenizer, budget=3000)
    # Now context has competitively selected items
"""

from typing import Dict, Any, List, Optional
import re


def apply_competitive_selection(
    context: Dict[str, Any],
    query: str,
    tokenizer_manager,
    model_name: str = "gpt-4",
    budget: int = 3000
) -> Dict[str, Any]:
    """
    Apply competitive selection across all fields.

    Args:
        context: Dict with fields like memories, facts, semantic_chunks, etc.
        query: User query (for future cosine scoring)
        tokenizer_manager: For token counting
        model_name: Model name for tokenizer
        budget: Total token budget

    Returns:
        New context dict with competitively selected items
    """

    # Field type priorities (higher = more trusted)
    TRUST_SCORES = {
        "facts": 0.30,
        "semantic_chunks": 0.25,
        "recent_conversations": 0.20,
        "memories": 0.15,
        "summaries": 0.10,
        "reflections": 0.08,
        "wiki": 0.05,
        "dreams": 0.02,
    }

    # Flatten all fields into candidates
    candidates = []

    for field_name in ["memories", "facts", "semantic_chunks", "recent_conversations",
                       "summaries", "reflections", "dreams"]:
        items = context.get(field_name, [])
        if not items:
            continue

        for item in items:
            # Extract text content
            text = _extract_text(item)
            if not text or len(text.strip()) < 5:
                continue

            # Get token count
            try:
                tokens = tokenizer_manager.count_tokens(text, model_name)
            except Exception:
                # Fallback: rough estimate
                tokens = len(text.split()) * 1.3

            # Get cosine score if available (from metadata)
            cosine = 0.5  # default
            if isinstance(item, dict):
                metadata = item.get("metadata", {})
                if isinstance(metadata, dict):
                    cosine = metadata.get("cosine_similarity",
                             metadata.get("relevance_score",
                             metadata.get("final_score", 0.5)))

            # Compute composite score
            trust = TRUST_SCORES.get(field_name, 0.0)
            raw_score = (1.0 * cosine) + (0.3 * trust)

            # Value = score per token
            value = raw_score / max(1, tokens)

            candidates.append({
                "field": field_name,
                "item": item,
                "text": text,
                "tokens": tokens,
                "cosine": cosine,
                "trust": trust,
                "score": raw_score,
                "value": value,
            })

    # Sort by value (highest first)
    candidates.sort(key=lambda c: c["value"], reverse=True)

    # Greedy pack under budget
    selected = []
    used_tokens = 0

    for cand in candidates:
        if used_tokens + cand["tokens"] <= budget:
            selected.append(cand)
            used_tokens += cand["tokens"]

    # Rebuild context dict
    new_context = {
        "time_context": context.get("time_context", ""),
        "wiki": context.get("wiki", ""),  # Keep wiki as-is (special case)
        "memories": [],
        "facts": [],
        "semantic_chunks": [],
        "recent_conversations": [],
        "summaries": [],
        "reflections": [],
        "dreams": [],
    }

    # Group selected items back into fields
    for cand in selected:
        field = cand["field"]
        if field in new_context:
            new_context[field].append(cand["item"])

    return new_context


def _extract_text(item: Any) -> str:
    """Extract text from memory/fact/chunk object."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        # Try common text fields
        for key in ["content", "text", "response", "query", "snippet"]:
            if key in item and item[key]:
                return str(item[key])
        # For conversations, combine query+response
        if "query" in item or "response" in item:
            q = item.get("query", "")
            r = item.get("response", "")
            return f"{q} {r}".strip()
    # Fallback
    return str(item)
