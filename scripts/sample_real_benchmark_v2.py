#!/usr/bin/env python3
"""
Sample real memories from production ChromaDB to build an adversarial benchmark.

V2 improvements over v1:
  1. Larger sample: 200+ target memories (up from 64)
  2. Near-neighbor adversarial distractors: for each target, include its
     top-K nearest neighbors in the seed pool (forces disambiguation)
  3. Real user queries: conversations use actual user queries, not synthetic
  4. Harder reflection/summary cases with clustered similar items

Usage:
    python scripts/sample_real_benchmark_v2.py [--seed 42] [--out tests/fixtures/retrieval_benchmarks_real.yaml]

The output YAML has the same schema as v1 and works with the existing harness.
"""

import argparse
import asyncio
import hashlib
import random
import re
import sys
import os
import tempfile
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from memory.memory_retriever import MemoryRetriever
from memory.memory_scorer import MemoryScorer
from core.intent_classifier import IntentClassifier


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHROMA_DIR = "data/chroma_minilm_backup_20260515_174233"

# Target sample sizes per collection
SAMPLE_COUNTS = {
    "conversations": 60,
    "facts": 60,
    "summaries": 40,
    "reflections": 40,
}

# How many nearest neighbors to include as distractors per target
NEIGHBORS_PER_TARGET = 3

# Total extra distractor pool per collection (random diverse items)
RANDOM_DISTRACTORS_PER_COLLECTION = 30


# ---------------------------------------------------------------------------
# Query extraction (reused from v1 with improvements)
# ---------------------------------------------------------------------------

_FACT_TEMPLATES = {
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
    "family_member": "Tell me about my family",
    "family_relationship": "How's my relationship with my family?",
    "relationship": "Who are the people in my life?",
    "relationship_with": "Tell me about my relationships",
    "pet": "Do I have any pets?",
    "pet_name": "What's my pet's name?",
    "current_activity": "What am I up to?",
    "current_mood": "How am I feeling?",
    "working": "What am I working on?",
    "goal": "What are my goals?",
    "intention": "What am I planning to do?",
    "plans": "What are my plans?",
    "project_type": "What kind of project am I working on?",
    "medication": "What medication am I on?",
    "exercise": "What exercise have I been doing?",
}

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

_REFLECTION_SKIP = {
    "what went well", "what could improve", "what could be improved",
    "the assistant effectively", "the assistant provided",
    "the assistant demonstrated", "the assistant acknowledged",
    "the assistant recognized", "there was a lack",
    "clear acknowledgment", "the assistant successfully",
    "the assistant showed", "the assistant maintained",
    "overall", "in summary",
}


def _extract_query(mem: Dict, collection: str) -> Optional[str]:
    """Generate a natural retrieval query from a memory."""
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
    """Use the actual user query from the conversation."""
    q = md.get("query", "")
    if q and len(q) > 12:
        return q
    m = re.match(r"User:\s*(.+?)(?:\n|$)", content)
    if m and len(m.group(1).strip()) > 12:
        return m.group(1).strip()
    return None


def _query_from_fact(md: Dict, content: str) -> Optional[str]:
    """Generate natural query from a fact triple."""
    parts = content.split("|")
    if len(parts) < 3:
        return None

    subj = parts[0].strip()
    pred = parts[1].strip()
    obj = parts[2].strip()

    if not obj or len(obj) < 2:
        return None
    if obj.lower() in {"true", "false", "yes", "no", "none", "working"}:
        return None

    if subj.lower() not in {"user", "i", "my", "me"}:
        if len(subj) > 2 and len(obj) > 3:
            return f"What do I know about {subj}?"
        return None

    pred_clean = pred.strip().lower()
    if pred_clean in _FACT_TEMPLATES:
        return _FACT_TEMPLATES[pred_clean]

    pred_human = pred.replace("_", " ")
    if len(obj) > 5:
        return f"Tell me about {obj}"
    else:
        return f"What's my {pred_human}?"


def _query_from_summary(content: str) -> Optional[str]:
    """Extract a topic-based query from a summary."""
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

    first = substantive[0]
    m = re.match(
        r"(?:The )?[Uu]ser(?:'s)?\s+(.+?)(?:\.|,|—|;|$)",
        first,
    )
    if m:
        action = m.group(1).strip()
        if len(action) > 10:
            return f"What happened with {action[:80]}?"

    if len(first) > 30 and not first.lower().startswith(("the assistant", "there was")):
        return first[:100]

    return None


def _query_from_reflection(content: str) -> Optional[str]:
    """
    Generate a query from a reflection using its retrieval text.

    Strategy: extract the first distinctive substantive sentence from the
    retrieval text and use it directly as the query. This ensures the query
    semantically matches the embedded content.
    """
    from memory.memory_storage import _extract_reflection_retrieval_text

    retrieval_text = _extract_reflection_retrieval_text(content)
    if not retrieval_text or len(retrieval_text) < 20:
        return None

    # The retrieval text format is: "Entity1 Entity2 sentence1. sentence2."
    # Skip initial entity tokens to find the first real sentence.
    # Entity tokens are typically capitalized single words at the start.
    words = retrieval_text.split()
    sentence_start = 0
    for i, w in enumerate(words):
        clean = w.strip(".,!?\"'()")
        # Stop skipping once we hit a lowercase word or a long enough sequence
        if clean and (clean[0].islower() or len(clean) > 12):
            sentence_start = i
            break
        # Also stop if we've skipped more than 8 tokens
        if i > 8:
            sentence_start = i
            break

    # Extract from sentence start
    text_body = " ".join(words[sentence_start:])
    if len(text_body) < 20:
        text_body = retrieval_text  # Fall back to full text

    # Find first substantive sentence
    segments = re.split(r'[.!?]+', text_body)
    for seg in segments:
        seg = seg.strip()
        if len(seg) < 20:
            continue
        seg_words = seg.split()
        if len(seg_words) < 4:
            continue
        lower = seg.lower()
        if any(lower.startswith(s) for s in (
            "the assistant", "the conversation", "the user",
            "this session", "in future", "overall",
        )):
            continue
        return seg[:100]

    # Fallback: use body text directly
    if len(text_body) > 25:
        return text_body[:80]

    return None


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _is_degenerate(mem: Dict, collection: str) -> bool:
    """Filter out memories that would produce bad benchmark cases."""
    content = mem.get("content", "")
    md = mem.get("metadata", {})

    if len(content) < 25:
        return True
    if "[API Error]" in content or "Error code:" in content:
        return True
    if "[OpenAI unavailable]" in content or "[API unavailable]" in content:
        return True

    if collection == "conversations":
        q = md.get("query", "")
        if q and len(q) < 13:
            return True
        resp = md.get("response", "")
        if resp and ("[API" in resp or "Error" in resp[:20] or "unavailable" in resp):
            return True

    if collection == "facts":
        parts = content.split("|")
        if len(parts) < 3:
            return True
        obj = parts[2].strip()
        pred = parts[1].strip()
        if len(obj) < 3 or obj.lower() in {"true", "false", "yes", "no", "none", "working"}:
            return True
        if pred.strip() in {"is", "was", "has", "had", "does"}:
            return True

    if collection == "summaries":
        lower = content.lower()
        if lower.startswith("[api error]") or "error code:" in lower[:50]:
            return True

    return False


# ---------------------------------------------------------------------------
# Near-neighbor adversarial sampling
# ---------------------------------------------------------------------------

def get_nearest_neighbors(
    store: MultiCollectionChromaStore,
    target_content: str,
    collection: str,
    n_neighbors: int = 3,
    exclude_ids: set = None,
) -> List[Dict]:
    """
    Find the N nearest neighbors to a target memory in the same collection.
    These are the adversarial distractors — items that are semantically close
    and could confuse the retriever.
    """
    if exclude_ids is None:
        exclude_ids = set()

    # Query ChromaDB with the target's content to find similar items
    try:
        results = store.query_collection(
            collection, target_content[:500], n_results=n_neighbors + 5
        )
    except Exception:
        return []

    neighbors = []
    for r in results:
        rid = r.get("id", "")
        if rid in exclude_ids:
            continue
        # Skip if it's basically identical content (self-match)
        r_content = r.get("content", "")
        if r_content and target_content[:100] in r_content[:100]:
            continue
        neighbors.append(r)
        if len(neighbors) >= n_neighbors:
            break

    return neighbors


# ---------------------------------------------------------------------------
# Main sampling logic
# ---------------------------------------------------------------------------

def sample_targets(
    store: MultiCollectionChromaStore,
    n_per_collection: Dict[str, int],
    seed: int,
) -> List[Dict]:
    """Sample target memories using diverse queries."""
    rng = random.Random(seed)
    sampled = []

    diverse_queries = [
        "family", "work", "health", "sleep", "exercise", "food", "cooking",
        "music", "travel", "hobby", "friend", "pet", "school", "study",
        "relationship", "dating", "money", "stress", "anxiety", "happy",
        "sad", "angry", "tired", "motivation", "goal", "plan", "project",
        "code", "programming", "game", "movie", "book", "weather",
        "went to", "talked about", "feeling", "working on", "thinking about",
        "worried about", "excited about", "struggling with", "learning",
        "tell me", "how", "what", "my", "remember", "believe", "want",
        "help", "problem", "idea", "recently", "yesterday", "tomorrow",
        # Additional queries for broader coverage
        "dog", "cat", "run", "morning", "therapy", "medication", "doctor",
        "brother", "sister", "mom", "dad", "partner", "birthday", "vacation",
        "apartment", "car", "phone", "laptop", "python", "javascript",
        "meeting", "deadline", "promotion", "interview", "resume",
        "dinner", "lunch", "coffee", "gym", "yoga", "meditation",
        "dream", "nightmare", "insomnia", "headache", "pain", "surgery",
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
        pool = [m for m in pool if not _is_degenerate(m, collection)]

        # Only keep items that produce a valid query
        viable = []
        for m in pool:
            q = _extract_query(m, collection)
            if q and len(q) > 10:
                viable.append(m)
        pool = viable

        if not pool:
            print(f"  WARNING: No viable candidates for {collection}")
            continue

        # Oversample, deduplicate by query
        oversample = min(n * 3, len(pool))
        candidates_list = rng.sample(pool, oversample)

        seen_queries = set()
        chosen = []
        for m in candidates_list:
            if len(chosen) >= n:
                break
            q = _extract_query(m, collection)
            if q and q not in seen_queries:
                seen_queries.add(q)
                chosen.append(m)

        for mem in chosen:
            mem["_source_collection"] = collection
            mem["_is_target"] = True
            sampled.append(mem)

        print(f"  {collection}: sampled {len(chosen)} targets from {len(pool)} viable")

    rng.shuffle(sampled)
    return sampled


def add_adversarial_distractors(
    store: MultiCollectionChromaStore,
    targets: List[Dict],
    neighbors_per_target: int = 3,
    random_distractors: int = 30,
    seed: int = 42,
) -> List[Dict]:
    """
    For each target memory, find its nearest neighbors and add them as
    adversarial distractors to the seed pool.
    """
    rng = random.Random(seed + 1)
    distractors = []
    target_ids = {m.get("id", "") for m in targets}

    # Group targets by collection
    by_collection: Dict[str, List[Dict]] = {}
    for t in targets:
        coll = t["_source_collection"]
        by_collection.setdefault(coll, []).append(t)

    print(f"\nAdding adversarial distractors...")

    for collection, coll_targets in by_collection.items():
        coll_distractor_ids = set()
        nn_count = 0

        for t in coll_targets:
            content = t.get("content", "")
            if not content:
                continue

            neighbors = get_nearest_neighbors(
                store, content, collection,
                n_neighbors=neighbors_per_target,
                exclude_ids=target_ids | coll_distractor_ids,
            )

            for n in neighbors:
                n["_source_collection"] = collection
                n["_is_target"] = False
                n["_distractor_for"] = t.get("id", "")
                distractors.append(n)
                nid = n.get("id", "")
                if nid:
                    coll_distractor_ids.add(nid)
                nn_count += 1

        # Also add random diverse distractors (items unrelated to any target)
        diverse_queries = ["random thought", "something else entirely", "unrelated topic",
                         "weather today", "what time is it", "general knowledge"]
        random_pool = {}
        for q in diverse_queries:
            try:
                results = store.query_collection(collection, q, n_results=30)
                for r in results:
                    rid = r.get("id", "")
                    if rid and rid not in target_ids and rid not in coll_distractor_ids:
                        random_pool[rid] = r
            except Exception:
                continue

        random_choices = rng.sample(
            list(random_pool.values()),
            min(random_distractors, len(random_pool))
        )
        for r in random_choices:
            r["_source_collection"] = collection
            r["_is_target"] = False
            r["_distractor_for"] = "_random"
            distractors.append(r)

        print(f"  {collection}: {nn_count} near-neighbor + {len(random_choices)} random distractors")

    return distractors


# ---------------------------------------------------------------------------
# Retrieval labeling
# ---------------------------------------------------------------------------

async def label_with_retrieval(
    targets: List[Dict],
    store: MultiCollectionChromaStore,
    top_k: int = 15,
) -> List[Dict]:
    """Run retrieval for each target to get baseline ranking."""
    from memory.corpus_manager import CorpusManager

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
    for i, mem in enumerate(targets):
        collection = mem["_source_collection"]
        query = _extract_query(mem, collection)
        if not query:
            continue

        intent_result = intent_classifier.classify(query)
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
            print(f"  SKIP [{i}]: retrieval error for {collection}: {e}")
            continue
        finally:
            scorer._intent_weight_overrides = None

        source_id = mem.get("id", "")
        source_content = mem.get("content", "")[:80].strip()
        source_rank = -1
        for rank, r in enumerate(results):
            if source_rank < 0:
                rid = r.get("id", "")
                if rid and rid == source_id:
                    source_rank = rank + 1
                elif source_content and source_content in (r.get("content", "") or ""):
                    source_rank = rank + 1

        labeled.append({
            "source_memory": mem,
            "query": query,
            "collection": collection,
            "intent": intent_result.intent.value,
            "intent_confidence": intent_result.confidence,
            "source_rank": source_rank,
            "retrieved_previews": [r.get("content", "")[:80] for r in results[:5]],
            "top_k": top_k,
        })

        status = f"rank={source_rank}" if source_rank > 0 else "NOT FOUND"
        if (i + 1) % 10 == 0 or source_rank < 0:
            print(f"  [{i+1}/{len(targets)}] {collection}: {status} | q={query[:50]}")

    return labeled


# ---------------------------------------------------------------------------
# YAML export
# ---------------------------------------------------------------------------

def _sanitize_yaml(s: str) -> str:
    """Make a string safe for YAML output."""
    s = s.replace("\n", " ").replace("\r", "")
    s = s.replace('"', '\\"')
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _make_marker(content: str, collection: str = "") -> str:
    """Extract a distinctive marker from content."""
    text = content
    if collection in ("reflections", "summaries"):
        lines = text.split("\n")
        for line in lines:
            stripped = line.strip().lstrip("#•*- ")
            if len(stripped) < 20:
                continue
            skip_phrases = {"what went well", "what could improve", "here are",
                          "based on", "factual bullets", "the assistant"}
            if any(sp in stripped.lower() for sp in skip_phrases):
                continue
            text = stripped
            break

    words = text.split()
    common = {"about", "there", "their", "would", "could", "should",
              "really", "which", "where", "these", "those", "being",
              "effectively", "assistant", "provided", "acknowledged"}
    for i in range(min(len(words), 30)):
        w = words[i].strip(".,!?\"'()[]{}:;-")
        if len(w) > 4 and w.lower() not in common:
            end = min(i + 3, len(words))
            phrase = " ".join(w.strip(".,!?") for w in words[i:end])
            return phrase
    return text[:40]


def export_yaml(
    labeled: List[Dict],
    distractors: List[Dict],
    output_path: str,
    reference_time: datetime,
):
    """Export labeled data + distractors as a benchmark YAML file."""
    from memory.memory_storage import _extract_reflection_retrieval_text

    lines = []
    lines.append("# Real-Data Retrieval Benchmark (v2 — adversarial)")
    lines.append("# =================================================")
    lines.append(f"# Generated: {datetime.now().isoformat()}")
    lines.append(f"# Reference time: {reference_time.isoformat()}")
    lines.append(f"# Source: {CHROMA_DIR}")
    lines.append(f"# Target memories: {len(labeled)}")
    lines.append(f"# Distractor memories: {len(distractors)}")
    lines.append(f"# Total seed pool: {len(labeled) + len(distractors)}")
    lines.append("#")
    lines.append("# V2 improvements:")
    lines.append("#   - Near-neighbor adversarial distractors per target")
    lines.append("#   - Larger seed pool for more retrieval pressure")
    lines.append("#   - Real user queries from conversation history")
    lines.append("")
    lines.append("seed_memories:")
    lines.append("")

    # Combine targets + distractors into one seed pool
    all_seeds = []

    # Targets first (these are what test cases point to)
    for i, item in enumerate(labeled):
        mem = item["source_memory"]
        md = mem.get("metadata", {})
        content = mem.get("content", "")
        collection = item["collection"]
        sid = f"real_{collection}_{i:03d}"
        item["_seed_id"] = sid
        all_seeds.append((sid, mem, collection, md, content))

    # Distractors (no test cases point to these — they exist as confusers)
    distractor_start_idx = len(labeled)
    for j, d in enumerate(distractors):
        collection = d["_source_collection"]
        content = d.get("content", "")
        md = d.get("metadata", {})
        sid = f"real_{collection}_d{j:03d}"
        all_seeds.append((sid, d, collection, md, content))

    # Emit all seed memories
    for sid, mem, collection, md, content in all_seeds:
        ts_str = md.get("timestamp", "")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                offset_hours = (ts - reference_time).total_seconds() / 3600
            except (ValueError, TypeError):
                offset_hours = -random.randint(24, 5000)
        else:
            offset_hours = -random.randint(24, 5000)

        marker = _make_marker(content, collection)
        truth = float(md.get("truth_score", 0.6))
        importance = float(md.get("importance_score", 0.5))
        tags = md.get("tags", "")

        # Compute content hash (mirrors conftest.py seeding logic)
        if collection == "conversations":
            query_field = md.get("query", "")
            resp_field = md.get("response", content[:200])
            if query_field:
                seeded_content = f"User: {query_field}\nAssistant: {resp_field}"
            else:
                seeded_content = content[:300]
        elif collection == "reflections":
            yaml_content = _sanitize_yaml(content[:800])
            seeded_content = _extract_reflection_retrieval_text(yaml_content)
        else:
            seeded_content = content[:300]

        norm = re.sub(r'\s+', ' ', seeded_content.lower().strip())
        content_hash = hashlib.sha256(norm.encode()).hexdigest()[:16]

        lines.append(f"  - id: {sid}")
        lines.append(f"    content_hash: \"{content_hash}\"")
        lines.append(f"    marker: \"{_sanitize_yaml(marker)}\"")
        lines.append(f"    collection: {collection}")

        if collection == "conversations":
            if md.get("query", ""):
                lines.append(f"    query: \"{_sanitize_yaml(query_field)}\"")
                lines.append(f"    response: \"{_sanitize_yaml(resp_field[:300])}\"")
                lines.append(f"    also_in_corpus: false")
            else:
                lines.append(f"    content: \"{_sanitize_yaml(content[:300])}\"")
        elif collection == "reflections":
            lines.append(f"    content: \"{_sanitize_yaml(content[:800])}\"")
        else:
            lines.append(f"    content: \"{_sanitize_yaml(content[:300])}\"")

        lines.append(f"    timestamp_offset_hours: {offset_hours:.1f}")
        lines.append(f"    truth_score: {truth}")
        lines.append(f"    importance_score: {importance}")
        lines.append(f"    tags: \"{_sanitize_yaml(tags)}\"")
        lines.append("")

    # Emit test cases (only for targets, not distractors)
    lines.append("")
    lines.append("test_cases:")
    lines.append("")

    for i, item in enumerate(labeled):
        collection = item["collection"]
        sid = item["_seed_id"]
        query = item["query"]
        intent = item["intent"]
        conf = item["intent_confidence"]
        source_rank = item["source_rank"]

        case_id = f"real_{intent}_{i:03d}"

        method_map = {
            "conversations": "get_memories",
            "facts": "get_facts",
            "summaries": "get_summaries_hybrid",
            "reflections": "get_reflections_hybrid",
        }
        method = method_map.get(collection, "get_memories")

        lines.append(f"  - id: {case_id}")
        lines.append(f"    query: \"{_sanitize_yaml(query)}\"")
        lines.append(f"    expected_intent: {intent}")
        lines.append(f"    expected_confidence_min: {max(0.0, conf - 0.15):.2f}")
        lines.append(f"    must_retrieve:")
        lines.append(f"      - {sid}")
        lines.append(f"    must_not_retrieve: []")
        lines.append(f"    top_k: 15")
        lines.append(f"    min_recall: 1.0")
        lines.append(f"    retrieval_method: {method}")
        lines.append(f"    description: \"[v2] {collection} — source rank {source_rank}\"")
        lines.append("")

    # Write
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    # Summary
    total_seeds = len(labeled) + len(distractors)
    found = sum(1 for item in labeled if item["source_rank"] > 0)
    in_top3 = sum(1 for item in labeled if 0 < item["source_rank"] <= 3)
    not_found = sum(1 for item in labeled if item["source_rank"] < 0)

    print(f"\nWrote to {output_path}")
    print(f"  Total seed pool: {total_seeds} ({len(labeled)} targets + {len(distractors)} distractors)")
    print(f"  Test cases: {len(labeled)}")
    print(f"  Found in production retrieval: {found}/{len(labeled)} ({found/len(labeled)*100:.0f}%)")
    print(f"  In top 3: {in_top3}/{len(labeled)} ({in_top3/len(labeled)*100:.0f}%)")
    print(f"  Not found (top-15): {not_found}/{len(labeled)} ({not_found/len(labeled)*100:.0f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Sample adversarial real-data benchmark (v2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-conversations", type=int, default=60)
    parser.add_argument("--n-facts", type=int, default=60)
    parser.add_argument("--n-summaries", type=int, default=40)
    parser.add_argument("--n-reflections", type=int, default=40)
    parser.add_argument("--neighbors", type=int, default=3,
                       help="Near-neighbors per target as distractors")
    parser.add_argument("--random-distractors", type=int, default=30,
                       help="Random distractors per collection")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument(
        "--out", type=str,
        default="tests/fixtures/retrieval_benchmarks_real.yaml",
        help="Output YAML path",
    )
    args = parser.parse_args()

    reference_time = datetime(2026, 5, 15, 12, 0, 0)

    print(f"Opening ChromaDB at {CHROMA_DIR}...")
    store = MultiCollectionChromaStore(persist_directory=CHROMA_DIR)

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

    print(f"\nSampling targets (seed={args.seed})...")
    targets = sample_targets(store, n_per_collection, args.seed)
    print(f"Total targets: {len(targets)}")

    print(f"\nAdding adversarial distractors (neighbors={args.neighbors})...")
    distractors = add_adversarial_distractors(
        store, targets,
        neighbors_per_target=args.neighbors,
        random_distractors=args.random_distractors,
        seed=args.seed,
    )
    print(f"Total distractors: {len(distractors)}")

    print(f"\nRunning retrieval labeling (top_k={args.top_k})...")
    labeled = await label_with_retrieval(targets, store, top_k=args.top_k)

    print(f"\nExporting YAML...")
    export_yaml(labeled, distractors, args.out, reference_time)


if __name__ == "__main__":
    asyncio.run(main())
