#!/usr/bin/env python3
"""
Bridge density projection (Phase 2.5B).

Tests how many personal graph entities can be mapped to Wikidata entities
via exact alias match and embedding similarity.

1. Fetches a sample of Wikidata entities via SPARQL (100-200 per domain)
2. Runs them against the existing personal graph's alias index
3. Tests embedding similarity for non-exact matches
4. Projects total bridge count at 50K scale

Go/no-go: If projected bridges < 50, the walk generator won't produce
meaningfully different results than random pairing.

Usage:
    python scripts/test_bridge_density.py
    python scripts/test_bridge_density.py --sample-size 200
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging_utils import get_logger

logger = get_logger("test_bridge_density")

# Wikidata SPARQL endpoint
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

# Domain queries — each fetches entities from a relevant Wikidata class
DOMAIN_QUERIES = {
    "health_medical": {
        "query": """SELECT ?item ?label WHERE {{
            ?item wdt:P31/wdt:P279* wd:Q12136 .
            ?item rdfs:label ?label . FILTER(LANG(?label) = "en")
        }} LIMIT {limit}""",
        "description": "Diseases and medical conditions",
    },
    "computer_science": {
        "query": """SELECT ?item ?label WHERE {{
            ?item wdt:P31/wdt:P279* wd:Q21198 .
            ?item rdfs:label ?label . FILTER(LANG(?label) = "en")
        }} LIMIT {limit}""",
        "description": "Computer science concepts",
    },
    "hobbies_interests": {
        "query": """SELECT ?item ?label WHERE {{
            ?item wdt:P31/wdt:P279* wd:Q11404 .
            ?item rdfs:label ?label . FILTER(LANG(?label) = "en")
        }} LIMIT {limit}""",
        "description": "Sports and hobbies",
    },
    "psychology": {
        "query": """SELECT ?item ?label WHERE {{
            ?item wdt:P31/wdt:P279* wd:Q9418 .
            ?item rdfs:label ?label . FILTER(LANG(?label) = "en")
        }} LIMIT {limit}""",
        "description": "Psychology concepts",
    },
    "philosophy": {
        "query": """SELECT ?item ?label WHERE {{
            ?item wdt:P31/wdt:P279* wd:Q5891 .
            ?item rdfs:label ?label . FILTER(LANG(?label) = "en")
        }} LIMIT {limit}""",
        "description": "Philosophy concepts",
    },
    "chemistry": {
        "query": """SELECT ?item ?label WHERE {{
            ?item wdt:P31/wdt:P279* wd:Q11173 .
            ?item rdfs:label ?label . FILTER(LANG(?label) = "en")
        }} LIMIT {limit}""",
        "description": "Chemical compounds",
    },
}


def fetch_wikidata_sample(domain: str, query_template: str, limit: int) -> list[dict]:
    """Fetch a sample of Wikidata entities via SPARQL."""
    import requests

    query = query_template.format(limit=limit)
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "DaemonBridgeDensityTest/1.0 (research)",
    }

    try:
        resp = requests.get(WIKIDATA_SPARQL, params={"query": query}, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        entities = []
        for binding in data.get("results", {}).get("bindings", []):
            qid = binding.get("item", {}).get("value", "").split("/")[-1]
            label = binding.get("label", {}).get("value", "")
            if qid and label:
                entities.append({"qid": qid, "label": label, "domain": domain})
        return entities
    except Exception as e:
        print(f"  WARNING: SPARQL query for {domain} failed: {e}")
        return []


def test_exact_matches(entities: list[dict], graph_memory, entity_resolver) -> list[dict]:
    """Test exact alias matches between Wikidata entities and personal graph."""
    matches = []
    for ent in entities:
        label = ent["label"]
        # Try exact resolution via alias index
        resolved = entity_resolver.resolve(label)
        if resolved:
            matches.append({
                "wikidata_qid": ent["qid"],
                "wikidata_label": label,
                "personal_entity": resolved,
                "match_type": "exact_alias",
                "domain": ent["domain"],
            })
            continue

        # Try slugified match
        slug = label.lower().strip().replace(" ", "_")
        if graph_memory.graph.has_node(slug):
            matches.append({
                "wikidata_qid": ent["qid"],
                "wikidata_label": label,
                "personal_entity": slug,
                "match_type": "slug_match",
                "domain": ent["domain"],
            })
    return matches


def test_embedding_matches(
    entities: list[dict],
    graph_memory,
    entity_resolver,
    threshold: float = 0.60,
    already_matched: set[str] | None = None,
) -> list[dict]:
    """Test embedding similarity matches using the MiniLM embedder."""
    if already_matched is None:
        already_matched = set()

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("  WARNING: sentence-transformers not available, skipping embedding matches")
        return []

    # Get personal entity labels
    personal_entities = []
    for nid, data in graph_memory.graph.nodes(data=True):
        if nid in already_matched:
            continue
        source = data.get("metadata", {}).get("source", "personal")
        if source == "personal":
            display = data.get("display_name", nid)
            personal_entities.append((nid, display))

    if not personal_entities:
        return []

    # Filter unmatched Wikidata entities
    unmatched = [e for e in entities if e["qid"] not in already_matched]
    if not unmatched:
        return []

    print(f"  Embedding comparison: {len(unmatched)} Wikidata × {len(personal_entities)} personal entities...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode
    wiki_labels = [e["label"] for e in unmatched]
    personal_labels = [pe[1] for pe in personal_entities]

    wiki_embeddings = model.encode(wiki_labels, normalize_embeddings=True)
    personal_embeddings = model.encode(personal_labels, normalize_embeddings=True)

    # Compute similarity matrix
    sim_matrix = np.dot(wiki_embeddings, personal_embeddings.T)

    matches = []
    for i, ent in enumerate(unmatched):
        best_j = int(np.argmax(sim_matrix[i]))
        best_sim = float(sim_matrix[i, best_j])
        if best_sim >= threshold:
            matches.append({
                "wikidata_qid": ent["qid"],
                "wikidata_label": ent["label"],
                "personal_entity": personal_entities[best_j][0],
                "personal_display": personal_entities[best_j][1],
                "match_type": "embedding",
                "similarity": round(best_sim, 3),
                "domain": ent["domain"],
            })

    return matches


def run_test(sample_per_domain: int = 100):
    from config.app_config import (
        CHROMA_PATH,
        KNOWLEDGE_GRAPH_PERSIST_PATH,
        KNOWLEDGE_GRAPH_ALIASES_PATH,
        WIKIDATA_EMBEDDING_MATCH_THRESHOLD,
    )
    from memory.graph_memory import GraphMemory
    from memory.entity_resolver import EntityResolver

    print("=" * 70)
    print("BRIDGE DENSITY PROJECTION")
    print("=" * 70)

    # Load personal graph
    print("\nLoading personal knowledge graph...")
    graph = GraphMemory(persist_path=KNOWLEDGE_GRAPH_PERSIST_PATH)
    resolver = EntityResolver(graph, aliases_path=KNOWLEDGE_GRAPH_ALIASES_PATH)

    personal_count = sum(
        1 for _, d in graph.graph.nodes(data=True)
        if d.get("metadata", {}).get("source", "personal") == "personal"
    )
    print(f"  Personal entities: {personal_count}")
    print(f"  Total nodes: {graph.node_count()}")
    print(f"  Total edges: {graph.edge_count()}")
    print(f"  Embedding threshold: {WIKIDATA_EMBEDDING_MATCH_THRESHOLD}")

    # Fetch Wikidata samples
    print(f"\nFetching Wikidata samples ({sample_per_domain} per domain)...")
    all_entities = []
    for domain, info in DOMAIN_QUERIES.items():
        t0 = time.time()
        entities = fetch_wikidata_sample(domain, info["query"], sample_per_domain)
        elapsed = time.time() - t0
        print(f"  {domain}: {len(entities)} entities ({elapsed:.1f}s)")
        all_entities.extend(entities)
        # Rate limiting — be respectful
        time.sleep(1)

    if not all_entities:
        print("\nERROR: No Wikidata entities fetched. Check network/SPARQL availability.")
        return

    total_sampled = len(all_entities)
    print(f"\n  Total sampled: {total_sampled}")

    # Test exact matches
    print("\nTesting exact alias matches...")
    exact_matches = test_exact_matches(all_entities, graph, resolver)
    print(f"  Exact matches: {len(exact_matches)}")
    for m in exact_matches:
        print(f"    {m['wikidata_label']} ({m['wikidata_qid']}) → {m['personal_entity']} [{m['match_type']}]")

    # Test embedding matches
    already = {m["wikidata_qid"] for m in exact_matches}
    print("\nTesting embedding similarity matches...")
    embedding_matches = test_embedding_matches(
        all_entities, graph, resolver,
        threshold=WIKIDATA_EMBEDDING_MATCH_THRESHOLD,
        already_matched=already,
    )
    print(f"  Embedding matches: {len(embedding_matches)}")
    for m in embedding_matches:
        print(f"    {m['wikidata_label']} → {m['personal_display']} (sim={m['similarity']}) [{m['domain']}]")

    all_matches = exact_matches + embedding_matches
    total_matches = len(all_matches)

    # Projection
    print(f"\n{'='*70}")
    print(f"PROJECTION")
    print(f"{'='*70}")
    match_rate = total_matches / total_sampled if total_sampled > 0 else 0
    projected_at_5k = int(match_rate * 5000)
    projected_at_50k = int(match_rate * 50000)

    print(f"  Sample size: {total_sampled}")
    print(f"  Total matches: {total_matches}")
    print(f"  Match rate: {match_rate:.1%}")
    print(f"  Projected bridges at 5K entities/domain: {projected_at_5k}")
    print(f"  Projected bridges at 50K total entities: {projected_at_50k}")

    print(f"\n--- GO/NO-GO GATE ---")
    if projected_at_50k >= 50:
        print(f"  PASS: Projected {projected_at_50k} bridges >= 50 threshold")
        print(f"  Graph walk generator should produce meaningful boundary-crossing walks.")
    elif projected_at_50k >= 20:
        print(f"  MARGINAL: Projected {projected_at_50k} bridges. May work with wiki enrichment growing bridges over time.")
        print(f"  Consider lowering embedding threshold or adding manual seed mappings.")
    else:
        print(f"  FAIL: Projected {projected_at_50k} bridges < 20.")
        print(f"  Walk generator unlikely to produce meaningful results.")
        print(f"  Options: lower threshold, add manual mappings, or rely on wiki enrichment to grow bridges first.")

    # Unique domains with matches
    matched_domains = set(m["domain"] for m in all_matches)
    print(f"\n  Domains with matches: {matched_domains or 'none'}")

    # Save results
    output = {
        "sample_size": total_sampled,
        "personal_entities": personal_count,
        "exact_matches": len(exact_matches),
        "embedding_matches": len(embedding_matches),
        "total_matches": total_matches,
        "match_rate": round(match_rate, 4),
        "projected_at_50k": projected_at_50k,
        "threshold": WIKIDATA_EMBEDDING_MATCH_THRESHOLD,
        "matches": all_matches,
    }
    output_path = Path("data/bridge_density_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Bridge density projection test")
    parser.add_argument("--sample-size", type=int, default=100,
                       help="Entities per domain to sample from Wikidata (default 100)")
    args = parser.parse_args()

    run_test(sample_per_domain=args.sample_size)


if __name__ == "__main__":
    main()
