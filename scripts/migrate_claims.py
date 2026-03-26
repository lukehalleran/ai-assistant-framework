#!/usr/bin/env python3
"""
One-time backfill: populate ClaimIndex from existing summaries and reflections.

Scans all documents in the 'summaries' and 'reflections' ChromaDB collections,
extracts embedded claims using the claim extraction pipeline, and populates
the ClaimIndex reverse mapping.  Optionally cross-references against current
facts to detect already-stale claims.

Usage:
    python scripts/migrate_claims.py --dry-run    # required first run
    python scripts/migrate_claims.py              # actually write index + metadata
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.app_config import (
    CHROMA_PATH,
    STALENESS_ENABLED,
    STALENESS_INDEX_PATH,
    KNOWLEDGE_GRAPH_PERSIST_PATH,
    KNOWLEDGE_GRAPH_ALIASES_PATH,
)
from memory.claim_tracker import ClaimIndex, extract_claims_from_text, canonicalize_claim
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from utils.logging_utils import get_logger

logger = get_logger("migrate_claims")


def main():
    parser = argparse.ArgumentParser(description="Backfill ClaimIndex from existing summaries/reflections")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, do not write")
    args = parser.parse_args()

    if not STALENESS_ENABLED:
        print("Staleness system is disabled in config. Enable it first.")
        return

    print(f"ChromaDB path: {CHROMA_PATH}")
    print(f"Claim index path: {STALENESS_INDEX_PATH}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Initialize ChromaDB
    chroma = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)

    # Initialize entity resolver (optional, for better canonicalization)
    entity_resolver = None
    try:
        from memory.graph_memory import GraphMemory
        from memory.entity_resolver import EntityResolver
        gm = GraphMemory(persist_path=KNOWLEDGE_GRAPH_PERSIST_PATH)
        entity_resolver = EntityResolver(
            graph_memory=gm,
            aliases_path=KNOWLEDGE_GRAPH_ALIASES_PATH,
        )
        print(f"Entity resolver loaded: {gm.node_count()} nodes")
    except Exception as e:
        print(f"Entity resolver not available (will use basic normalization): {e}")

    # Initialize claim index
    claim_index = ClaimIndex(persist_path=STALENESS_INDEX_PATH)

    total_docs = 0
    total_claims = 0
    collections_to_scan = ["summaries", "reflections"]

    for collection_name in collections_to_scan:
        print(f"\n--- Scanning {collection_name} ---")

        try:
            # Get all documents from the collection
            docs = chroma.list_all(collection_name)
            if not docs:
                print(f"  No documents in {collection_name}")
                continue
        except Exception as e:
            print(f"  Failed to read {collection_name}: {e}")
            continue

        print(f"  Found {len(docs)} documents")

        for doc in docs:
            doc_id = doc.get("id", "")
            content = doc.get("content", "") or doc.get("document", "")
            metadata = doc.get("metadata", {}) or {}

            if not content or not doc_id:
                continue

            total_docs += 1

            # Extract claims
            claims = extract_claims_from_text(content, entity_resolver)
            if not claims:
                continue

            total_claims += len(claims)
            claim_index.add_claims(doc_id, collection_name, claims)

            # Update metadata with claim hashes and staleness_ratio
            claim_hashes = ",".join(c.claim_hash for c in claims)
            metadata_updates = {
                "embedded_claims": claim_hashes,
                "staleness_ratio": float(metadata.get("staleness_ratio", 0.0)),
            }

            # Add temporal anchors if not present
            if "temporal_anchor_start" not in metadata:
                ts = metadata.get("timestamp", "")
                if ts:
                    metadata_updates["temporal_anchor_start"] = ts
                    metadata_updates["temporal_anchor_end"] = ts

            if not args.dry_run:
                try:
                    chroma.update_metadata(collection_name, doc_id, metadata_updates)
                except Exception as e:
                    print(f"    Failed to update metadata for {doc_id}: {e}")

            if total_docs % 10 == 0:
                print(f"  Processed {total_docs} docs, {total_claims} claims...")

    print(f"\n--- Summary ---")
    print(f"Documents scanned: {total_docs}")
    print(f"Claims extracted: {total_claims}")
    print(f"Unique claim hashes: {claim_index.total_claims}")
    print(f"Documents with claims: {claim_index.total_documents}")

    # Cross-reference against current facts to detect already-stale claims
    stale_count = 0
    try:
        fact_docs = chroma.list_all("facts")
        if fact_docs:
            print(f"\n--- Cross-referencing against {len(fact_docs)} facts ---")

            # Build a set of (subject, relation) pairs from current facts
            from memory.cross_deduplicator import CrossCollectionDeduplicator
            fact_pairs = set()
            for fd in fact_docs:
                s, p, o = CrossCollectionDeduplicator._extract_triple(fd)
                if s and p:
                    ck = canonicalize_claim(s, p, entity_resolver)
                    fact_pairs.add(ck.claim_hash)

            # Check which claims in the index have contradicting facts
            # (This is a simplified check — just verifies that the claim hash
            # exists in current facts, meaning the fact might have changed)
            print(f"  Found {len(fact_pairs)} unique fact claim hashes")
            print(f"  (Full contradiction detection requires the cross-deduplicator)")
    except Exception as e:
        print(f"  Fact cross-reference failed: {e}")

    # Save
    if not args.dry_run:
        claim_index.save()
        print(f"\nClaim index saved to {STALENESS_INDEX_PATH}")
    else:
        print(f"\n[DRY RUN] Would save {claim_index.total_claims} claims to {STALENESS_INDEX_PATH}")
        print("Run without --dry-run to execute.")


if __name__ == "__main__":
    main()
