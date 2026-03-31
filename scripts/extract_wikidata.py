#!/usr/bin/env python3
"""
Wikidata subgraph extraction (Phase 3).

Queries the Wikidata SPARQL endpoint to build a structured knowledge
subgraph across 10 domain categories relevant to the user's personal
graph.  Fetches entity labels, descriptions, aliases, and inter-entity
relations.

Output: data/wikidata_cache.json

Improvements over naive extraction:
  1. Alias expansion — fetches skos:altLabel for informal name variants
  2. Description fields — enables description-enriched embedding matching
  3. Domain tags — enables category-aware matching (only compare personal
     entities against Wikidata entities in plausible domains)

Usage:
    python scripts/extract_wikidata.py
    python scripts/extract_wikidata.py --entities-per-domain 200 --dry-run
    python scripts/extract_wikidata.py --domains health_medical,hobbies_interests
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging_utils import get_logger

logger = get_logger("extract_wikidata")

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

# Domain definitions: Wikidata root classes + seed items.
# Each domain has multiple entry points to maximize coverage of concepts
# that overlap with personal graph entities.
DOMAIN_MAP = {
    "health_medical": {
        "description": "Diseases, drugs, conditions, neurotransmitters",
        "queries": [
            # Diseases
            ('SELECT ?item ?label ?desc WHERE {{ '
             '?item wdt:P31/wdt:P279* wd:Q12136 . '
             '?item rdfs:label ?label . FILTER(LANG(?label) = "en") '
             'OPTIONAL {{ ?item schema:description ?desc . FILTER(LANG(?desc) = "en") }} '
             '}} LIMIT {limit}'),
            # Chemical compounds (drugs, neurotransmitters)
            ('SELECT ?item ?label ?desc WHERE {{ '
             '?item wdt:P31/wdt:P279* wd:Q11173 . '
             '?item rdfs:label ?label . FILTER(LANG(?label) = "en") '
             'OPTIONAL {{ ?item schema:description ?desc . FILTER(LANG(?desc) = "en") }} '
             '}} LIMIT {limit}'),
        ],
    },
    "fitness_exercise": {
        "description": "Sports, exercises, physical activities",
        "queries": [
            # Sports
            ('SELECT ?item ?label ?desc WHERE {{ '
             '?item wdt:P31/wdt:P279* wd:Q349 . '
             '?item rdfs:label ?label . FILTER(LANG(?label) = "en") '
             'OPTIONAL {{ ?item schema:description ?desc . FILTER(LANG(?desc) = "en") }} '
             '}} LIMIT {limit}'),
            # Physical exercises
            ('SELECT ?item ?label ?desc WHERE {{ '
             '?item wdt:P31/wdt:P279* wd:Q61065 . '
             '?item rdfs:label ?label . FILTER(LANG(?label) = "en") '
             'OPTIONAL {{ ?item schema:description ?desc . FILTER(LANG(?desc) = "en") }} '
             '}} LIMIT {limit}'),
        ],
    },
    "education_statistics": {
        "description": "Academic disciplines, mathematical concepts",
        "queries": [
            # Academic disciplines
            ('SELECT ?item ?label ?desc WHERE {{ '
             '?item wdt:P31/wdt:P279* wd:Q11862829 . '
             '?item rdfs:label ?label . FILTER(LANG(?label) = "en") '
             'OPTIONAL {{ ?item schema:description ?desc . FILTER(LANG(?desc) = "en") }} '
             '}} LIMIT {limit}'),
            # Statistical concepts
            ('SELECT ?item ?label ?desc WHERE {{ '
             '?item wdt:P31/wdt:P279* wd:Q12483 . '
             '?item rdfs:label ?label . FILTER(LANG(?label) = "en") '
             'OPTIONAL {{ ?item schema:description ?desc . FILTER(LANG(?desc) = "en") }} '
             '}} LIMIT {limit}'),
        ],
    },
    "computer_science": {
        "description": "CS concepts, programming, algorithms",
        "queries": [
            ('SELECT ?item ?label ?desc WHERE {{ '
             '?item wdt:P31/wdt:P279* wd:Q21198 . '
             '?item rdfs:label ?label . FILTER(LANG(?label) = "en") '
             'OPTIONAL {{ ?item schema:description ?desc . FILTER(LANG(?desc) = "en") }} '
             '}} LIMIT {limit}'),
        ],
    },
    "philosophy_mind": {
        "description": "Philosophy, consciousness, ethics",
        "queries": [
            ('SELECT ?item ?label ?desc WHERE {{ '
             '?item wdt:P31/wdt:P279* wd:Q5891 . '
             '?item rdfs:label ?label . FILTER(LANG(?label) = "en") '
             'OPTIONAL {{ ?item schema:description ?desc . FILTER(LANG(?desc) = "en") }} '
             '}} LIMIT {limit}'),
        ],
    },
    "hobbies_interests": {
        "description": "Board games, beer, music, video games, crafts",
        "queries": [
            # Board games
            ('SELECT ?item ?label ?desc WHERE {{ '
             '?item wdt:P31/wdt:P279* wd:Q131436 . '
             '?item rdfs:label ?label . FILTER(LANG(?label) = "en") '
             'OPTIONAL {{ ?item schema:description ?desc . FILTER(LANG(?desc) = "en") }} '
             '}} LIMIT {limit}'),
            # Beer styles
            ('SELECT ?item ?label ?desc WHERE {{ '
             '?item wdt:P31/wdt:P279* wd:Q44 . '
             '?item rdfs:label ?label . FILTER(LANG(?label) = "en") '
             'OPTIONAL {{ ?item schema:description ?desc . FILTER(LANG(?desc) = "en") }} '
             '}} LIMIT {limit}'),
            # Video games
            ('SELECT ?item ?label ?desc WHERE {{ '
             '?item wdt:P31/wdt:P279* wd:Q7889 . '
             '?item rdfs:label ?label . FILTER(LANG(?label) = "en") '
             'OPTIONAL {{ ?item schema:description ?desc . FILTER(LANG(?desc) = "en") }} '
             '}} LIMIT {limit}'),
        ],
    },
    "psychology_relationships": {
        "description": "Psychology concepts, cognitive science",
        "queries": [
            ('SELECT ?item ?label ?desc WHERE {{ '
             '?item wdt:P31/wdt:P279* wd:Q9418 . '
             '?item rdfs:label ?label . FILTER(LANG(?label) = "en") '
             'OPTIONAL {{ ?item schema:description ?desc . FILTER(LANG(?desc) = "en") }} '
             '}} LIMIT {limit}'),
        ],
    },
    "geography_places": {
        "description": "Cities, regions, countries",
        "queries": [
            # Cities > 100K population
            ('SELECT ?item ?label ?desc WHERE {{ '
             '?item wdt:P31/wdt:P279* wd:Q515 . '
             '?item wdt:P1082 ?pop . FILTER(?pop > 100000) '
             '?item rdfs:label ?label . FILTER(LANG(?label) = "en") '
             'OPTIONAL {{ ?item schema:description ?desc . FILTER(LANG(?desc) = "en") }} '
             '}} LIMIT {limit}'),
        ],
    },
    "history_politics": {
        "description": "Historical events, political concepts",
        "queries": [
            ('SELECT ?item ?label ?desc WHERE {{ '
             '?item wdt:P31/wdt:P279* wd:Q1190554 . '
             '?item rdfs:label ?label . FILTER(LANG(?label) = "en") '
             'OPTIONAL {{ ?item schema:description ?desc . FILTER(LANG(?desc) = "en") }} '
             '}} LIMIT {limit}'),
        ],
    },
    "cross_domain_science": {
        "description": "Systems theory, biology, physics concepts",
        "queries": [
            # Biological processes
            ('SELECT ?item ?label ?desc WHERE {{ '
             '?item wdt:P31/wdt:P279* wd:Q2996394 . '
             '?item rdfs:label ?label . FILTER(LANG(?label) = "en") '
             'OPTIONAL {{ ?item schema:description ?desc . FILTER(LANG(?desc) = "en") }} '
             '}} LIMIT {limit}'),
        ],
    },
}

# Wikidata properties to fetch for inter-entity relations
RELATION_PROPERTIES = {
    "P31": "instance_of",
    "P279": "subclass_of",
    "P361": "part_of",
    "P527": "has_part",
    "P1542": "has_effect",
    "P1536": "has_cause",
    "P366": "has_use",
    "P460": "said_to_be_same_as",
    "P921": "main_subject",
    "P2283": "uses",
}


def sparql_query(query: str, timeout: int = 60) -> list[dict]:
    """Execute a SPARQL query against the Wikidata endpoint."""
    import requests

    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "DaemonWikidataExtractor/1.0 (research; lukeh)",
    }

    try:
        resp = requests.get(
            WIKIDATA_SPARQL,
            params={"query": query},
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("results", {}).get("bindings", [])
    except Exception as e:
        logger.warning(f"SPARQL query failed: {e}")
        return []


def fetch_aliases(qids: list[str], batch_size: int = 200) -> dict[str, list[str]]:
    """Fetch alternative labels (aliases) for a batch of Wikidata entities."""
    aliases = {}
    for i in range(0, len(qids), batch_size):
        batch = qids[i:i + batch_size]
        values = " ".join(f"wd:{qid}" for qid in batch)
        query = f"""
        SELECT ?item ?altLabel WHERE {{
            VALUES ?item {{ {values} }}
            ?item skos:altLabel ?altLabel .
            FILTER(LANG(?altLabel) = "en")
        }}
        """
        results = sparql_query(query)
        for binding in results:
            qid = binding.get("item", {}).get("value", "").split("/")[-1]
            alt = binding.get("altLabel", {}).get("value", "")
            if qid and alt:
                aliases.setdefault(qid, []).append(alt)
        time.sleep(1)  # rate limiting
    return aliases


def fetch_relations(qids: list[str], batch_size: int = 200) -> list[dict]:
    """Fetch inter-entity relations for entities within our subgraph."""
    qid_set = set(qids)
    relations = []
    prop_values = " ".join(f"wdt:{pid}" for pid in RELATION_PROPERTIES)

    for i in range(0, len(qids), batch_size):
        batch = qids[i:i + batch_size]
        values = " ".join(f"wd:{qid}" for qid in batch)
        query = f"""
        SELECT ?source ?prop ?target WHERE {{
            VALUES ?source {{ {values} }}
            VALUES ?prop {{ {prop_values} }}
            ?source ?prop ?target .
            FILTER(STRSTARTS(STR(?target), "http://www.wikidata.org/entity/Q"))
        }} LIMIT 5000
        """
        results = sparql_query(query, timeout=90)
        for binding in results:
            src = binding.get("source", {}).get("value", "").split("/")[-1]
            prop_uri = binding.get("prop", {}).get("value", "")
            tgt = binding.get("target", {}).get("value", "").split("/")[-1]

            # Extract property ID from URI
            pid = prop_uri.split("/")[-1] if "/" in prop_uri else ""

            # Only keep relations where target is in our subgraph
            if src and tgt and pid in RELATION_PROPERTIES and tgt in qid_set:
                relations.append({
                    "source_qid": src,
                    "property_id": pid,
                    "target_qid": tgt,
                    "relation_label": RELATION_PROPERTIES[pid],
                })
        time.sleep(1)  # rate limiting

    return relations


def extract(entities_per_domain: int = 500, domains: list[str] | None = None, dry_run: bool = False):
    """Main extraction pipeline."""
    from datetime import datetime

    target_domains = domains or list(DOMAIN_MAP.keys())

    print("=" * 70)
    print("WIKIDATA SUBGRAPH EXTRACTION")
    print(f"Domains: {len(target_domains)}")
    print(f"Entities per domain: {entities_per_domain}")
    print(f"Dry run: {dry_run}")
    print("=" * 70)

    all_entities = {}  # qid -> entity dict
    total_start = time.time()

    # Phase 1: Fetch entities per domain
    for domain in target_domains:
        if domain not in DOMAIN_MAP:
            print(f"\n  WARNING: Unknown domain '{domain}', skipping")
            continue

        info = DOMAIN_MAP[domain]
        print(f"\n--- {domain}: {info['description']} ---")

        domain_entities = {}
        per_query_limit = entities_per_domain // len(info["queries"]) + 1

        for qi, query_template in enumerate(info["queries"]):
            query = query_template.format(limit=per_query_limit)
            t0 = time.time()
            results = sparql_query(query)
            elapsed = time.time() - t0

            for binding in results:
                qid = binding.get("item", {}).get("value", "").split("/")[-1]
                label = binding.get("label", {}).get("value", "")
                desc = binding.get("desc", {}).get("value", "")

                if qid and label and qid not in domain_entities:
                    domain_entities[qid] = {
                        "qid": qid,
                        "label": label,
                        "description": desc,
                        "aliases": [],
                        "domain_category": domain,
                    }

            print(f"  Query {qi+1}/{len(info['queries'])}: {len(results)} results ({elapsed:.1f}s)")
            time.sleep(1.5)  # rate limiting between queries

        # Cap per domain
        items = list(domain_entities.values())[:entities_per_domain]
        for item in items:
            all_entities[item["qid"]] = item

        print(f"  Total for {domain}: {len(items)}")

    print(f"\n{'='*70}")
    print(f"Phase 1 complete: {len(all_entities)} entities across {len(target_domains)} domains")

    if dry_run:
        print("\nDry run — skipping alias fetch and relation extraction.")
        # Show sample
        for domain in target_domains[:3]:
            samples = [e for e in all_entities.values() if e["domain_category"] == domain][:5]
            print(f"\n  {domain} samples:")
            for s in samples:
                print(f"    {s['qid']:10s} {s['label'][:40]:40s} {s['description'][:50]}")
        return

    # Phase 2: Fetch aliases
    qids = list(all_entities.keys())
    print(f"\nPhase 2: Fetching aliases for {len(qids)} entities...")
    t0 = time.time()
    aliases = fetch_aliases(qids, batch_size=200)
    alias_elapsed = time.time() - t0

    alias_count = 0
    for qid, alt_labels in aliases.items():
        if qid in all_entities:
            # Deduplicate and limit
            existing = set(a.lower() for a in all_entities[qid].get("aliases", []))
            for alt in alt_labels:
                if alt.lower() not in existing and len(existing) < 20:
                    all_entities[qid]["aliases"].append(alt)
                    existing.add(alt.lower())
                    alias_count += 1

    print(f"  Aliases added: {alias_count} across {len(aliases)} entities ({alias_elapsed:.1f}s)")

    # Phase 3: Fetch inter-entity relations
    print(f"\nPhase 3: Fetching relations...")
    t0 = time.time()
    relations = fetch_relations(qids, batch_size=200)
    rel_elapsed = time.time() - t0
    print(f"  Relations found: {len(relations)} ({rel_elapsed:.1f}s)")

    # Save
    from config.app_config import WIKIDATA_PERSIST_PATH
    output_path = Path(WIKIDATA_PERSIST_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": "1.0",
        "extracted_at": datetime.now().isoformat(),
        "entity_count": len(all_entities),
        "relation_count": len(relations),
        "domains": target_domains,
        "entities_per_domain": entities_per_domain,
        "entities": all_entities,
        "relations": relations,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    total_elapsed = time.time() - total_start

    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"  Entities: {len(all_entities)}")
    print(f"  Aliases:  {alias_count}")
    print(f"  Relations: {len(relations)}")
    print(f"  Elapsed: {total_elapsed:.0f}s")
    print(f"  Output: {output_path}")

    # Domain breakdown
    from collections import Counter
    domain_counts = Counter(e["domain_category"] for e in all_entities.values())
    print(f"\n  Per domain:")
    for d, c in domain_counts.most_common():
        print(f"    {d:30s} {c:5d}")


def main():
    parser = argparse.ArgumentParser(description="Wikidata subgraph extraction")
    parser.add_argument("--entities-per-domain", type=int, default=500,
                       help="Entities per domain (default 500)")
    parser.add_argument("--domains", type=str, default=None,
                       help="Comma-separated domain list (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Fetch entities only, skip aliases and relations")
    args = parser.parse_args()

    domains = args.domains.split(",") if args.domains else None
    extract(entities_per_domain=args.entities_per_domain, domains=domains, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
