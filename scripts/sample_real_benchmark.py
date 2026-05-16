#!/usr/bin/env python3
"""
Sample real memories from chroma_db_v4 to build a ground-truth benchmark.

Semi-automated approach:
  1. Randomly sample memories from real ChromaDB collections
  2. Extract a natural query from each memory's content
  3. Run the query through the real retrieval pipeline
  4. Export a draft YAML for human review/correction

Usage:
    python scripts/sample_real_benchmark.py [--n-per-collection 5] [--seed 42] [--out tests/fixtures/retrieval_benchmarks_real.yaml]

The output YAML has the same schema as retrieval_benchmarks.yaml and can be
loaded by the existing benchmark harness.
"""

import argparse
import asyncio
import json
import random
import re
import sys
import os
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from memory.memory_retriever import MemoryRetriever
from memory.memory_scorer import MemoryScorer
from core.intent_classifier import IntentClassifier


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHROMA_DIR = "data/chroma_db_v4"

# Collections to sample from, with relative weights
SAMPLE_COLLECTIONS = {
    "conversations": 15,
    "facts": 10,
    "summaries": 5,
    "reflections": 5,
}

# Collections we won't sample FROM but will seed (distractors)
DISTRACTOR_COLLECTIONS = ["facts", "summaries", "reflections", "conversations"]

# Queries to generate seed-pool distractors
DISTRACTOR_QUERIES = [
    "tell me about myself",
    "how are you feeling",
    "what happened recently",
    "work projects",
    "family and friends",
    "hobbies and interests",
    "goals and plans",
    "daily routine",
    "technical discussions",
    "emotional conversations",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_query_from_memory(mem: Dict, collection: str) -> Optional[str]:
    """
    Generate a natural retrieval query from a memory.

    Key principle: the query should be something a real user would type
    that should cause this memory to be retrieved.
    """
    md = mem.get("metadata", {})
    content = mem.get("content", "")

    if collection == "conversations":
        return _query_from_conversation(md, content)
    elif collection == "facts":
        return _query_from_fact(md, content)
    elif collection == "summaries":
        return _query_from_summary(content)
    elif collection == "reflections":
        return _query_from_reflection(content)
    return None


def _query_from_conversation(md: Dict, content: str) -> Optional[str]:
    """Use the original user query. Skip trivial greetings."""
    q = md.get("query", "")
    if q and len(q) > 12:
        return q
    m = re.match(r"User:\s*(.+?)(?:\n|$)", content)
    if m and len(m.group(1).strip()) > 12:
        return m.group(1).strip()
    return None


# Predicate → natural question templates for fact queries
_FACT_TEMPLATES = {
    # Personal attributes
    "hobby": "What are my hobbies?",
    "interest": "What am I interested in?",
    "favorite_game": "What's my favorite game?",
    "plays": "What do I play?",
    "location": "Where do I live?",
    "lives_in": "Where do I live?",
    "moved_to": "Where did I move to?",
    "age": "How old am I?",
    "name": "What's my name?",
    "job": "What do I do for work?",
    "occupation": "What's my job?",
    "school": "Where do I go to school?",
    "major": "What's my major?",
    # Relationships
    "family_member": "Tell me about my family",
    "family_relationship": "How's my relationship with my family?",
    "relationship": "Who are the people in my life?",
    "relationship_with": "Tell me about my relationships",
    "family_visit": "Any family visits coming up?",
    "family_connection": "What's going on with my family?",
    "location_of_family": "Where is my family?",
    "dinner_with": "Who did I have dinner with?",
    "pet": "Do I have any pets?",
    "pet_name": "What's my pet's name?",
    # Activities / state
    "current_activity": "What am I up to?",
    "current_mood": "How am I feeling?",
    "feels_about_work": "How do I feel about work?",
    "working": "What am I working on?",
    "goal": "What are my goals?",
    "intention": "What am I planning to do?",
    "plans": "What are my plans?",
    "project_type": "What kind of project am I working on?",
    "project_complexity": "How complex is my project?",
    # Health / routine
    "woke_up_time": "What time did I wake up?",
    "sleep": "How's my sleep been?",
    "exercise": "What exercise have I been doing?",
    "medication": "What medication am I on?",
    # Temporal
    "celebration": "Any celebrations coming up?",
    "vacation": "When was my last vacation?",
    "last_vacation": "When did I last go on vacation?",
    "college_ex_vacation": "What vacation did I take?",
    "week": "What week is it?",
}


def _query_from_fact(md: Dict, content: str) -> Optional[str]:
    """Generate natural query from a fact triple."""
    parts = content.split("|")
    if len(parts) < 3:
        return None

    subj = parts[0].strip()
    pred = parts[1].strip()
    obj = parts[2].strip()

    # Skip degenerate objects
    if not obj or len(obj) < 2:
        return None
    if obj.lower() in {"true", "false", "yes", "no", "none", "working"}:
        return None

    # Skip non-user subjects that produce bad queries
    if subj.lower() not in {"user", "i", "my", "me"}:
        # Entity facts: "What do I know about {subj}?"
        if len(subj) > 2 and len(obj) > 3:
            return f"What do I know about {subj}?"
        return None

    # Try template match first (most natural queries)
    pred_clean = pred.strip().lower()
    if pred_clean in _FACT_TEMPLATES:
        return _FACT_TEMPLATES[pred_clean]

    # Fallback: construct from predicate + object
    pred_human = pred.replace("_", " ")
    if len(obj) > 5:
        # Object is descriptive enough to be the query focus
        return f"Tell me about {obj}"
    else:
        return f"What's my {pred_human}?"


_SUMMARY_BOILERPLATE = {
    "here are the factual bullets",
    "here are the key factual",
    "here are the extractive",
    "based on the excerpts",
    "here are factual bullets",
    "key factual bullets",
    "[api error]",
    "error code:",
}


def _query_from_summary(content: str) -> Optional[str]:
    """
    Extract a topic-based query from a summary.

    Summaries are bullet-point lists of facts from a conversation.
    The query should be about the topic, not echo the summary text.
    """
    lines = content.split("\n")
    substantive = []
    for line in lines:
        cleaned = line.strip().lstrip("•*-# ")
        if len(cleaned) < 25:
            continue
        if any(bp in cleaned.lower() for bp in _SUMMARY_BOILERPLATE):
            continue
        substantive.append(cleaned)

    if not substantive:
        return None

    # Use the first substantive bullet to extract a topic
    first = substantive[0]

    # Try to extract the subject/topic from the bullet
    # Common pattern: "The user [did X]" or "User [did X]"
    m = re.match(
        r"(?:The )?[Uu]ser(?:'s)?\s+(.+?)(?:\.|,|—|;|$)",
        first,
    )
    if m:
        action = m.group(1).strip()
        if len(action) > 10:
            # Convert to a natural query
            # "spent $50 on Uber Eats" → "Tell me about spending on Uber Eats"
            # "underwent 2 echos" → "Tell me about my medical tests"
            return f"What happened with {action[:80]}?"

    # Fallback: use the first substantive line as-is if it's topic-like
    if len(first) > 30 and not first.lower().startswith(("the assistant", "there was")):
        # Truncate to a reasonable query length
        return first[:100]

    return None


_REFLECTION_SKIP = {
    "what went well", "what could improve", "what could be improved",
    "the assistant effectively", "the assistant provided",
    "the assistant demonstrated", "the assistant acknowledged",
    "the assistant recognized", "there was a lack",
    "clear acknowledgment", "the assistant successfully",
    "the assistant showed", "the assistant maintained",
    "overall", "in summary",
}


def _query_from_reflection(content: str) -> Optional[str]:
    """
    Generate a user-facing query from a reflection.

    Reflections are about conversation quality. The query should be
    about the *topic* of the conversation being reflected on, not
    about the assistant's performance.
    """
    lines = content.split("\n")

    # Look for concrete topics mentioned in the reflection
    topics = []
    for line in lines:
        cleaned = line.strip().lstrip("•*-# ")
        if len(cleaned) < 25:
            continue
        if any(cleaned.lower().startswith(skip) for skip in _REFLECTION_SKIP):
            continue

        # Extract topic references from the line
        # Patterns: "user's X", "regarding X", "about X", "around X"
        for pattern in [
            r"regarding\s+(?:the\s+)?(?:user'?s?\s+)?(.{10,60}?)(?:\.|,|;|$)",
            r"(?:about|around|concerning)\s+(?:the\s+)?(?:user'?s?\s+)?(.{10,60}?)(?:\.|,|;|$)",
            r"user'?s?\s+(feelings?|thoughts?|experiences?|struggles?|progress|goals?|plans?|work|family|health|sleep|mood|exercise)\s",
        ]:
            m = re.search(pattern, cleaned, re.IGNORECASE)
            if m:
                topic = m.group(1).strip().rstrip(".,;")
                if len(topic) > 5:
                    topics.append(topic)
                    break

    if topics:
        # Use the most specific topic found
        best = max(topics, key=len)
        return f"How have I been doing with {best}?"

    # Fallback: find any line with concrete content (not boilerplate)
    for line in lines:
        cleaned = line.strip().lstrip("•*-# ")
        if len(cleaned) < 30:
            continue
        if any(cleaned.lower().startswith(skip) for skip in _REFLECTION_SKIP):
            continue
        # This line has non-boilerplate content — use it
        return cleaned[:120]

    return None


def _make_marker(content: str, collection: str = "") -> str:
    """Extract a unique-ish marker substring from content."""
    text = content

    # For reflections/summaries, skip boilerplate headers to find unique content
    if collection in ("reflections", "summaries"):
        # Strip markdown headers, bullet prefixes, boilerplate openings
        lines = text.split("\n")
        for line in lines:
            stripped = line.strip().lstrip("#•*- ")
            if len(stripped) < 20:
                continue
            skip_phrases = {
                "what went well", "what could improve", "here are",
                "based on", "factual bullets", "the assistant",
            }
            if any(sp in stripped.lower() for sp in skip_phrases):
                continue
            text = stripped
            break

    # Find a distinctive multi-word phrase
    words = text.split()
    common = {
        "about", "there", "their", "would", "could", "should",
        "really", "which", "where", "these", "those", "being",
        "effectively", "assistant", "provided", "acknowledged",
    }
    for i in range(min(len(words), 30)):
        w = words[i].strip(".,!?\"'()[]{}:;-")
        if len(w) > 4 and w.lower() not in common:
            # Return 3-word phrase for better uniqueness
            end = min(i + 3, len(words))
            phrase = " ".join(w.strip(".,!?") for w in words[i:end])
            return phrase
    return text[:40]


def _sanitize_yaml_string(s: str) -> str:
    """Make a string safe for YAML output."""
    # Replace problematic characters
    s = s.replace("\n", " ").replace("\r", "")
    s = s.replace('"', '\\"')
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _is_degenerate(mem: Dict, collection: str) -> bool:
    """Filter out memories that would produce bad benchmark cases."""
    content = mem.get("content", "")
    md = mem.get("metadata", {})

    # Too short
    if len(content) < 25:
        return True

    # Error messages / API failures
    if "[API Error]" in content or "Error code:" in content:
        return True
    if "[OpenAI unavailable]" in content or "[API unavailable]" in content:
        return True

    # Conversations: skip trivial greetings and very short queries
    if collection == "conversations":
        q = md.get("query", "")
        if q and len(q) < 13:
            return True
        # Skip if response is an error
        resp = md.get("response", "")
        if resp and ("[API" in resp or "Error" in resp[:20] or "unavailable" in resp):
            return True

    # Facts: skip degenerate triples
    if collection == "facts":
        parts = content.split("|")
        if len(parts) < 3:
            return True
        obj = parts[2].strip()
        pred = parts[1].strip()
        # Skip if object is trivially short or generic
        if len(obj) < 3 or obj.lower() in {"true", "false", "yes", "no", "none", "working"}:
            return True
        # Skip predicates that are too vague
        if pred.strip() in {"is", "was", "has", "had", "does"}:
            return True

    # Summaries: skip if mostly boilerplate
    if collection == "summaries":
        lower = content.lower()
        if lower.startswith("[api error]") or "error code:" in lower[:50]:
            return True

    return False


def sample_memories(
    store: MultiCollectionChromaStore,
    n_per_collection: Dict[str, int],
    seed: int,
) -> List[Dict]:
    """
    Randomly sample memories from real ChromaDB collections.

    Uses a broad diverse query set to pull candidates, filters degenerates,
    then random-samples from the union.
    """
    rng = random.Random(seed)
    sampled = []

    # Wide-net queries to maximize candidate pool diversity
    diverse_queries = [
        # Topics
        "family", "work", "health", "sleep", "exercise", "food", "cooking",
        "music", "travel", "hobby", "friend", "pet", "school", "study",
        "relationship", "dating", "money", "stress", "anxiety", "happy",
        "sad", "angry", "tired", "motivation", "goal", "plan", "project",
        "code", "programming", "game", "movie", "book", "weather",
        # Activities
        "went to", "talked about", "feeling", "working on", "thinking about",
        "worried about", "excited about", "struggling with", "learning",
        # Structural
        "tell me", "how", "what", "my", "remember", "believe", "want",
        "help", "problem", "idea", "recently", "yesterday", "tomorrow",
    ]

    for collection, n in n_per_collection.items():
        candidates = {}
        for q in diverse_queries:
            try:
                results = store.query_collection(collection, q, n_results=50)
                for r in results:
                    doc_id = r.get("id", "")
                    if doc_id and doc_id not in candidates:
                        candidates[doc_id] = r
            except Exception:
                continue

        pool = list(candidates.values())

        # Filter degenerates
        pool = [m for m in pool if not _is_degenerate(m, collection)]

        # Pre-filter: only keep items that produce a valid query
        viable = []
        for m in pool:
            q = _extract_query_from_memory(m, collection)
            if q and len(q) > 10:
                viable.append(m)
        pool = viable

        if not pool:
            print(f"  WARNING: No viable candidates for {collection}")
            continue

        # Oversample 2x, then deduplicate by query to avoid near-duplicates
        oversample = min(n * 3, len(pool))
        candidates_list = rng.sample(pool, oversample)

        # Deduplicate by generated query (avoid "What are my hobbies?" x5)
        seen_queries = set()
        chosen = []
        for m in candidates_list:
            if len(chosen) >= n:
                break
            q = _extract_query_from_memory(m, collection)
            if q and q not in seen_queries:
                seen_queries.add(q)
                chosen.append(m)

        for mem in chosen:
            mem["_source_collection"] = collection
            sampled.append(mem)

        print(f"  {collection}: sampled {len(chosen)} from {len(pool)} viable (of {len(candidates)} raw)")

    rng.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# Retrieval labeling
# ---------------------------------------------------------------------------

async def label_with_retrieval(
    sampled: List[Dict],
    store: MultiCollectionChromaStore,
    top_k: int = 15,
) -> List[Dict]:
    """
    For each sampled memory, generate a query and run retrieval to find
    what the system actually returns.

    Returns enriched dicts with query, retrieval results, and draft labels.
    """
    # Build retriever with real embeddings (no gate system for clean signal)
    # Use an empty CorpusManager so the retriever doesn't crash on
    # get_recent_memories calls.
    from memory.corpus_manager import CorpusManager
    import tempfile
    corpus_file = os.path.join(tempfile.mkdtemp(), "empty_corpus.json")
    corpus_mgr = CorpusManager(corpus_file=corpus_file)

    scorer = MemoryScorer(
        time_manager=None,
        conversation_context=deque(maxlen=50),
    )
    retriever = MemoryRetriever(
        corpus_manager=corpus_mgr,
        chroma_store=store,
        gate_system=None,
        scorer=scorer,
        hybrid_retriever=None,
        time_manager=None,
    )
    intent_classifier = IntentClassifier()

    labeled = []
    for i, mem in enumerate(sampled):
        collection = mem["_source_collection"]
        query = _extract_query_from_memory(mem, collection)
        if not query:
            print(f"  SKIP [{i}]: could not extract query from {collection} memory")
            continue

        # Classify intent
        intent_result = intent_classifier.classify(query)

        # Use the appropriate retriever method per collection
        scorer._intent_weight_overrides = intent_result.weight_overrides or None
        try:
            if collection == "conversations":
                results = await retriever.get_memories(query=query, limit=top_k)
            elif collection == "facts":
                results = await retriever.get_facts(query=query, limit=top_k)
            elif collection == "summaries":
                results = retriever.get_summaries_hybrid(query=query, limit=top_k)
            elif collection == "reflections":
                results = await retriever.get_reflections_hybrid(query=query, limit=top_k)
            else:
                results = await retriever.get_memories(query=query, limit=top_k)
        except (TypeError, ValueError) as e:
            # Handle timezone-aware vs naive datetime mismatches etc.
            print(f"  SKIP [{i}]: retrieval error for {collection}: {e}")
            continue
        finally:
            scorer._intent_weight_overrides = None

        # Check if the source memory was retrieved
        # Match by ID first, then by content substring (some retriever methods
        # don't preserve the original ChromaDB doc ID)
        source_id = mem.get("id", "")
        source_content = mem.get("content", "")[:80].strip()
        retrieved_ids = []
        source_rank = -1
        for rank, r in enumerate(results):
            rid = r.get("id", "")
            retrieved_ids.append(rid or f"_pos{rank}")
            if source_rank < 0:
                # Try ID match
                if rid and rid == source_id:
                    source_rank = rank + 1
                # Try content match (for summaries/reflections that lose ID)
                elif source_content and source_content in (r.get("content", "") or ""):
                    source_rank = rank + 1

        labeled.append({
            "source_memory": mem,
            "query": query,
            "collection": collection,
            "intent": intent_result.intent.value,
            "intent_confidence": intent_result.confidence,
            "source_rank": source_rank,
            "retrieved_ids": retrieved_ids[:10],
            "retrieved_previews": [
                r.get("content", "")[:80] for r in results[:5]
            ],
            "top_k": top_k,
        })

        status = f"rank={source_rank}" if source_rank > 0 else "NOT FOUND"
        print(f"  [{i+1}/{len(sampled)}] {collection}: {status} | intent={intent_result.intent.value} | q={query[:60]}")

    return labeled


# ---------------------------------------------------------------------------
# YAML export
# ---------------------------------------------------------------------------

def export_yaml(
    labeled: List[Dict],
    output_path: str,
    reference_time: datetime,
):
    """Export labeled data as a benchmark YAML file."""
    lines = []
    lines.append("# Real-Data Retrieval Benchmark")
    lines.append("# ==============================")
    lines.append(f"# Generated: {datetime.now().isoformat()}")
    lines.append(f"# Reference time: {reference_time.isoformat()}")
    lines.append(f"# Source: {CHROMA_DIR}")
    lines.append(f"# Total seed memories: {len(labeled)}")
    lines.append("#")
    lines.append("# REVIEW INSTRUCTIONS:")
    lines.append("#   1. Check each test case's must_retrieve — is the labeled memory")
    lines.append("#      actually the right answer for the query?")
    lines.append("#   2. Add must_not_retrieve IDs for memories that should NOT surface")
    lines.append("#   3. Adjust queries if they're too vague or too specific")
    lines.append("#   4. Remove cases where the query is trivial or uninteresting")
    lines.append("#   5. source_rank shows where the system currently ranks the target")
    lines.append("#      (-1 = not found in top K)")
    lines.append("")
    lines.append("seed_memories:")
    lines.append("")

    # Emit seed memories
    for i, item in enumerate(labeled):
        mem = item["source_memory"]
        md = mem.get("metadata", {})
        content = mem.get("content", "")
        collection = item["collection"]
        sid = f"real_{collection}_{i:03d}"

        ts_str = md.get("timestamp", "")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                offset_hours = (ts - reference_time).total_seconds() / 3600
            except (ValueError, TypeError):
                offset_hours = -(i + 1) * 24
        else:
            offset_hours = -(i + 1) * 24

        marker = _make_marker(content, collection)
        truth = float(md.get("truth_score", 0.6))
        importance = float(md.get("importance_score", 0.5))
        tags = md.get("tags", "")

        # Build the exact content that will be seeded into ChromaDB
        # (mirrors conftest.py seeding logic, including reflection cleaning)
        if collection == "conversations":
            query_field = md.get("query", "")
            resp_field = md.get("response", content[:200])
            if query_field:
                seeded_content = f"User: {query_field}\nAssistant: {resp_field}"
            else:
                seeded_content = content[:300]
        elif collection == "reflections":
            # Reflections: hash must match what conftest actually embeds.
            # Conftest reads the YAML content (sanitized/single-line), then cleans it.
            # So we must: sanitize first (collapse newlines), then clean.
            from memory.memory_storage import _clean_reflection_for_embedding
            yaml_content = _sanitize_yaml_string(content[:800])
            seeded_content = _clean_reflection_for_embedding(yaml_content)
        else:
            seeded_content = content[:300]

        # Compute stable content hash from what will actually be seeded
        import hashlib
        norm = re.sub(r'\s+', ' ', seeded_content.lower().strip())
        content_hash = hashlib.sha256(norm.encode()).hexdigest()[:16]

        lines.append(f"  - id: {sid}")
        lines.append(f"    content_hash: \"{content_hash}\"")
        lines.append(f"    marker: \"{_sanitize_yaml_string(marker)}\"")
        lines.append(f"    collection: {collection}")

        if collection == "conversations":
            if md.get("query", ""):
                lines.append(f"    query: \"{_sanitize_yaml_string(query_field)}\"")
                lines.append(f"    response: \"{_sanitize_yaml_string(resp_field)}\"")
                lines.append(f"    also_in_corpus: false")
            else:
                lines.append(f"    content: \"{_sanitize_yaml_string(content[:300])}\"")
        elif collection == "reflections":
            # Store more content for reflections so substantive bullets survive
            lines.append(f"    content: \"{_sanitize_yaml_string(content[:800])}\"")
        else:
            lines.append(f"    content: \"{_sanitize_yaml_string(content[:300])}\"")

        lines.append(f"    timestamp_offset_hours: {offset_hours:.1f}")
        lines.append(f"    truth_score: {truth}")
        lines.append(f"    importance_score: {importance}")
        lines.append(f"    tags: \"{_sanitize_yaml_string(tags)}\"")
        lines.append("")

    # Emit test cases
    lines.append("")
    lines.append("test_cases:")
    lines.append("")

    for i, item in enumerate(labeled):
        collection = item["collection"]
        sid = f"real_{collection}_{i:03d}"
        query = item["query"]
        intent = item["intent"]
        conf = item["intent_confidence"]
        source_rank = item["source_rank"]

        case_id = f"real_{intent}_{i:03d}"

        lines.append(f"  - id: {case_id}")
        lines.append(f"    query: \"{_sanitize_yaml_string(query)}\"")
        lines.append(f"    expected_intent: {intent}")
        lines.append(f"    expected_confidence_min: {max(0.0, conf - 0.15):.2f}")
        lines.append(f"    must_retrieve:")
        lines.append(f"      - {sid}")
        lines.append(f"    must_not_retrieve: []")
        lines.append(f"    top_k: {item['top_k']}")
        lines.append(f"    min_recall: 1.0")

        # Route to correct retriever method per collection
        method_map = {
            "conversations": "get_memories",
            "facts": "get_facts",
            "summaries": "get_summaries_hybrid",
            "reflections": "get_reflections_hybrid",
        }
        method = method_map.get(collection, "get_memories")
        lines.append(f"    retrieval_method: {method}")

        # Add review comments
        lines.append(f"    # SOURCE RANK: {source_rank}  (current system ranking, -1=not found)")
        lines.append(f"    # TOP 5 RETRIEVED:")
        for j, preview in enumerate(item.get("retrieved_previews", [])):
            lines.append(f"    #   {j+1}. {_sanitize_yaml_string(preview[:70])}")
        lines.append(f"    description: \"[REVIEW] {collection} memory — source rank {source_rank}\"")
        lines.append("")

    # Write
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nWrote {len(labeled)} cases to {output_path}")

    # Print summary
    found = sum(1 for item in labeled if item["source_rank"] > 0)
    in_top3 = sum(1 for item in labeled if 0 < item["source_rank"] <= 3)
    not_found = sum(1 for item in labeled if item["source_rank"] < 0)
    print(f"  Found: {found}/{len(labeled)} ({found/len(labeled)*100:.0f}%)")
    print(f"  In top 3: {in_top3}/{len(labeled)} ({in_top3/len(labeled)*100:.0f}%)")
    print(f"  Not found: {not_found}/{len(labeled)} ({not_found/len(labeled)*100:.0f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Sample real memories for benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n-conversations", type=int, default=25,
        help="Number of conversation memories to sample",
    )
    parser.add_argument(
        "--n-facts", type=int, default=20,
        help="Number of fact memories to sample",
    )
    parser.add_argument(
        "--n-summaries", type=int, default=10,
        help="Number of summary memories to sample",
    )
    parser.add_argument(
        "--n-reflections", type=int, default=10,
        help="Number of reflection memories to sample",
    )
    parser.add_argument(
        "--top-k", type=int, default=15,
        help="Top K for retrieval labeling",
    )
    parser.add_argument(
        "--out", type=str,
        default="tests/fixtures/retrieval_benchmarks_real.yaml",
        help="Output YAML path",
    )
    args = parser.parse_args()

    reference_time = datetime(2026, 5, 15, 12, 0, 0)

    print(f"Opening ChromaDB at {CHROMA_DIR}...")
    store = MultiCollectionChromaStore(persist_directory=CHROMA_DIR)

    # Show stats
    stats = store.get_collection_stats()
    for name, info in sorted(stats.items()):
        c = info.get("count", 0)
        if c > 0:
            print(f"  {name}: {c}")

    n_per_collection = {
        "conversations": args.n_conversations,
        "facts": args.n_facts,
        "summaries": args.n_summaries,
        "reflections": args.n_reflections,
    }

    print(f"\nSampling (seed={args.seed})...")
    sampled = sample_memories(store, n_per_collection, args.seed)
    print(f"Total sampled: {len(sampled)}")

    print(f"\nRunning retrieval labeling (top_k={args.top_k})...")
    labeled = await label_with_retrieval(sampled, store, top_k=args.top_k)

    print(f"\nExporting YAML...")
    export_yaml(labeled, args.out, reference_time)


if __name__ == "__main__":
    asyncio.run(main())
