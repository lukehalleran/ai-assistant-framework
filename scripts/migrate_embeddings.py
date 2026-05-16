#!/usr/bin/env python3
"""
Side-by-side embedding model migration for ChromaDB collections.

Reads all documents from the current MiniLM store, re-embeds them with
BGE-small-en-v1.5 into a separate temporary store, verifies counts and
IDs match, then prints swap instructions.

Does NOT modify the original store. Does NOT swap directories automatically.

Usage:
    python scripts/migrate_embeddings.py --dry-run          # report counts only
    python scripts/migrate_embeddings.py                    # build new BGE store
    python scripts/migrate_embeddings.py --source PATH      # custom source dir

The script:
  1. Backs up source → data/chroma_minilm_backup_YYYYMMDD_HHMMSS/
  2. Builds new store at data/chroma_bge_tmp/
  3. For each collection: reads ids + documents + metadatas, inserts with same IDs
  4. Verifies old count == new count, samples IDs match
  5. Writes migration manifest to data/embedding_migration_manifest.json
  6. Writes model marker to new store: data/chroma_bge_tmp/EMBEDDING_MODEL.json
  7. Prints manual swap instructions
"""

import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLD_MODEL = "all-MiniLM-L6-v2"
NEW_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
BATCH_SIZE = 500  # ChromaDB add batch size

DEFAULT_SOURCE = "data/chroma_db_v4"
DEFAULT_TARGET = "data/chroma_bge_tmp"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_all_collection_names(client):
    """Get all collection names from a ChromaDB client."""
    collections = client.list_collections()
    # Chroma 0.4+ returns Collection objects; 0.5+ may return names
    names = []
    for c in collections:
        if hasattr(c, 'name'):
            names.append(c.name)
        elif isinstance(c, str):
            names.append(c)
    return sorted(names)


def export_collection(client, name):
    """Export all data from a collection: ids, documents, metadatas."""
    coll = client.get_collection(name=name)
    count = coll.count()
    if count == 0:
        return {"ids": [], "documents": [], "metadatas": [], "count": 0}

    # ChromaDB get() returns all documents
    result = coll.get(include=["documents", "metadatas"])
    ids = result.get("ids", []) or []
    docs = result.get("documents", []) or []
    metas = result.get("metadatas", []) or []

    return {
        "ids": ids,
        "documents": docs,
        "metadatas": metas,
        "count": count,
    }


def import_collection(client, name, data, embedding_fn):
    """Import data into a new collection, preserving original IDs."""
    coll = client.get_or_create_collection(
        name=name, embedding_function=embedding_fn
    )

    ids = data["ids"]
    docs = data["documents"]
    metas = data["metadatas"]

    if not ids:
        return 0

    # Insert in batches to avoid memory spikes
    total = len(ids)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_ids = ids[start:end]
        batch_docs = [d or "" for d in docs[start:end]]
        batch_metas = [m or {} for m in metas[start:end]]
        coll.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)

    return coll.count()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Migrate ChromaDB embeddings to a new model")
    parser.add_argument("--dry-run", action="store_true", help="Report counts only, don't build new store")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help=f"Source ChromaDB dir (default: {DEFAULT_SOURCE})")
    parser.add_argument("--target", default=DEFAULT_TARGET, help=f"Target ChromaDB dir (default: {DEFAULT_TARGET})")
    parser.add_argument("--skip-backup", action="store_true", help="Skip backup step (if already backed up)")
    args = parser.parse_args()

    source_dir = args.source
    target_dir = args.target
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not os.path.exists(source_dir):
        print(f"ERROR: Source directory not found: {source_dir}")
        sys.exit(1)

    # Open source store (read-only, no embedding fn needed for get())
    print(f"Opening source store: {source_dir}")
    src_client = chromadb.PersistentClient(
        path=source_dir,
        settings=Settings(anonymized_telemetry=False),
    )

    collection_names = get_all_collection_names(src_client)
    print(f"Found {len(collection_names)} collections: {', '.join(collection_names)}")

    # Export all data
    exports = {}
    total_docs = 0
    print(f"\n{'Collection':<25} {'Count':>8}")
    print("-" * 35)
    for name in collection_names:
        data = export_collection(src_client, name)
        exports[name] = data
        total_docs += data["count"]
        print(f"  {name:<23} {data['count']:>8}")
    print("-" * 35)
    print(f"  {'TOTAL':<23} {total_docs:>8}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would migrate {total_docs} documents across {len(collection_names)} collections")
        print(f"[DRY RUN] Old model: {OLD_MODEL}")
        print(f"[DRY RUN] New model: {NEW_MODEL}")
        print(f"[DRY RUN] Target dir: {target_dir}")
        return

    # Step 1: Backup
    if not args.skip_backup:
        backup_dir = f"data/chroma_minilm_backup_{timestamp}"
        print(f"\nStep 1: Backing up {source_dir} → {backup_dir}")
        shutil.copytree(source_dir, backup_dir)
        print(f"  Backup complete ({shutil.disk_usage(backup_dir).used // (1024*1024)}MB)")
    else:
        backup_dir = "(skipped)"
        print("\nStep 1: Backup skipped (--skip-backup)")

    # Step 2: Build new BGE store
    if os.path.exists(target_dir):
        print(f"\nWARNING: Target directory exists: {target_dir}")
        print("  Removing old target to start fresh...")
        shutil.rmtree(target_dir)

    print(f"\nStep 2: Building new BGE store at {target_dir}")
    print(f"  Loading {NEW_MODEL}...")
    t0 = time.perf_counter()

    tgt_client = chromadb.PersistentClient(
        path=target_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    bge_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=NEW_MODEL,
        device=os.getenv("CHROMA_DEVICE", "cpu"),
    )
    model_load_s = time.perf_counter() - t0
    print(f"  Model loaded in {model_load_s:.1f}s")

    # Step 3: Migrate each collection
    print(f"\nStep 3: Migrating collections...")
    results = {}
    for name in collection_names:
        data = exports[name]
        t0 = time.perf_counter()
        new_count = import_collection(tgt_client, name, data, bge_fn)
        elapsed = time.perf_counter() - t0
        old_count = data["count"]
        match = "OK" if new_count == old_count else "MISMATCH"
        results[name] = {
            "old_count": old_count,
            "new_count": new_count,
            "match": new_count == old_count,
            "time_s": round(elapsed, 1),
        }
        print(f"  {name:<23} {old_count:>6} → {new_count:>6}  [{match}]  ({elapsed:.1f}s)")

    # Step 4: Verify
    print(f"\nStep 4: Verification")
    all_match = all(r["match"] for r in results.values())

    # Sample ID verification
    id_checks_passed = 0
    id_checks_total = 0
    for name in collection_names:
        data = exports[name]
        if not data["ids"]:
            continue
        # Check first and last ID exist in new store
        new_coll = tgt_client.get_collection(name=name)
        for sample_id in [data["ids"][0], data["ids"][-1]]:
            id_checks_total += 1
            try:
                result = new_coll.get(ids=[sample_id])
                if result["ids"] and result["ids"][0] == sample_id:
                    id_checks_passed += 1
            except Exception:
                pass

    print(f"  Count verification: {'PASS' if all_match else 'FAIL'}")
    print(f"  ID spot-check: {id_checks_passed}/{id_checks_total} passed")

    if not all_match:
        print("\n  ERROR: Count mismatch detected. Do NOT swap directories.")
        print("  Investigate before proceeding.")

    # Step 5: Write model marker
    model_marker = {
        "embedding_model": NEW_MODEL,
        "embedding_dimension": EMBEDDING_DIM,
        "migrated_from": OLD_MODEL,
        "migrated_at": datetime.now().isoformat(),
        "collections": {n: r["old_count"] for n, r in results.items()},
    }
    marker_path = os.path.join(target_dir, "EMBEDDING_MODEL.json")
    with open(marker_path, "w") as f:
        json.dump(model_marker, f, indent=2)
    print(f"  Model marker written: {marker_path}")

    # Step 6: Write manifest
    manifest = {
        "old_model": OLD_MODEL,
        "new_model": NEW_MODEL,
        "embedding_dimension": EMBEDDING_DIM,
        "migrated_at": datetime.now().isoformat(),
        "backup_path": backup_dir,
        "source_path": source_dir,
        "target_path": target_dir,
        "collections": {n: r["old_count"] for n, r in results.items()},
        "verification": {
            "counts_match": all_match,
            "id_spot_checks": f"{id_checks_passed}/{id_checks_total}",
        },
        "total_documents": total_docs,
    }
    manifest_path = "data/embedding_migration_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest written: {manifest_path}")

    # Step 7: Print swap instructions
    print(f"\n{'='*60}")
    if all_match and id_checks_passed == id_checks_total:
        print("  MIGRATION COMPLETE — VERIFICATION PASSED")
        print(f"{'='*60}")
        print(f"""
  To swap to BGE embeddings:

  1. Stop Daemon

  2. Swap directories:
     mv {source_dir} data/chroma_minilm_pre_swap_{timestamp}
     mv {target_dir} {source_dir}

  3. Change default model in multi_collection_chroma_store.py:
     CHROMA_ST_MODEL default: "{OLD_MODEL}" → "{NEW_MODEL}"

  4. Start Daemon

  5. Run benchmarks:
     python -m pytest tests/benchmarks/ -m benchmark -q

  To rollback:
     mv {source_dir} data/chroma_bge_failed_{timestamp}
     mv data/chroma_minilm_pre_swap_{timestamp} {source_dir}
     (revert model default)
""")
    else:
        print("  MIGRATION COMPLETE — VERIFICATION FAILED")
        print(f"{'='*60}")
        print("  Do NOT swap directories. Investigate mismatches first.")


if __name__ == "__main__":
    main()
