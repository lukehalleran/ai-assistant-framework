#!/usr/bin/env python3
"""
Load embedded Wikipedia parquet files into the wiki_knowledge ChromaDB collection.

Usage:
    python scripts/load_parquet_to_chromadb.py --parquet-dir /path/to/embedded_parquet
    python scripts/load_parquet_to_chromadb.py --parquet-dir /path/to/embedded_parquet --dry-run
    python scripts/load_parquet_to_chromadb.py --parquet-dir /path/to/embedded_parquet --batch-size 500

Reads each .parquet file produced by embed_wiki_chunks_to_parquet.py and calls
add_wiki_chunk() on MultiCollectionChromaStore for each chunk row.

Supports:
    - Checkpoint/resume (skips already-loaded parquet files)
    - Batch upsert for performance
    - Dry-run mode for previewing without writing
    - Progress reporting
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pyarrow.parquet as pq
import pandas as pd

from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from utils.logging_utils import get_logger

logger = get_logger("load_parquet_to_chromadb")

DEFAULT_BATCH_SIZE = 200
CHECKPOINT_FILE = Path("data/wiki_chromadb_checkpoint.json")


def load_checkpoint(checkpoint_path: Path) -> set:
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            return set(json.load(f))
    return set()


def save_checkpoint(checkpoint_path: Path, done_files: set):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump(sorted(done_files), f)


def load_parquet_to_chromadb(
    parquet_dir: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dry_run: bool = False,
    checkpoint_path: Path = CHECKPOINT_FILE,
):
    parquet_path = Path(parquet_dir)
    if not parquet_path.exists():
        print(f"Parquet directory not found: {parquet_path}")
        sys.exit(1)

    parquet_files = sorted(parquet_path.glob("*.parquet"))
    if not parquet_files:
        print(f"No .parquet files found in {parquet_path}")
        sys.exit(1)

    done_files = load_checkpoint(checkpoint_path)
    remaining = [f for f in parquet_files if f.name not in done_files]

    print(f"Found {len(parquet_files)} parquet files, {len(done_files)} already loaded, {len(remaining)} remaining")

    if dry_run:
        total_rows = 0
        for pf in remaining[:10]:
            n = pq.ParquetFile(pf).metadata.num_rows
            total_rows += n
            print(f"  [dry-run] {pf.name}: {n} chunks")
        if len(remaining) > 10:
            print(f"  ... and {len(remaining) - 10} more files")
            # Estimate total from sample
            sample_avg = total_rows / min(len(remaining), 10)
            est_total = int(sample_avg * len(remaining))
            print(f"  Estimated total chunks to load: ~{est_total:,}")
        else:
            print(f"  Total chunks to load: {total_rows:,}")
        return

    # Init ChromaDB store
    store = MultiCollectionChromaStore()

    total_chunks = 0
    total_files = 0
    t_start = time.time()

    for i, pf in enumerate(remaining):
        try:
            df = pd.read_parquet(pf)
        except Exception as e:
            print(f"[ERROR] Failed to read {pf.name}: {e}")
            continue

        file_chunks = 0
        batch_docs = []
        batch_metas = []
        batch_ids = []

        for _, row in df.iterrows():
            text = row.get("text", "")
            title = row.get("title", "")
            if not text:
                continue

            content = f"{title} {text}"
            chunk_id = f"wiki_{row.get('id', '')}_{row.get('chunk_index', 0)}"

            metadata = {
                "title": title,
                "article_id": str(row.get("id", "")),
                "chunk_index": int(row.get("chunk_index", 0)),
                "total_chunks": int(row.get("total_chunks", 1)),
                "section": str(row.get("section", "")),
                "section_level": int(row.get("section_level", 0)),
                "source": "wikipedia",
                "type": "wiki",
            }

            batch_docs.append(content)
            batch_metas.append(metadata)
            batch_ids.append(chunk_id)
            file_chunks += 1

            if len(batch_docs) >= batch_size:
                _upsert_batch(store, batch_docs, batch_metas, batch_ids)
                batch_docs, batch_metas, batch_ids = [], [], []

        # Flush remaining
        if batch_docs:
            _upsert_batch(store, batch_docs, batch_metas, batch_ids)

        done_files.add(pf.name)
        total_chunks += file_chunks
        total_files += 1

        if total_files % 100 == 0:
            save_checkpoint(checkpoint_path, done_files)
            elapsed = time.time() - t_start
            rate = total_chunks / elapsed if elapsed > 0 else 0
            print(
                f"[PROGRESS] {total_files}/{len(remaining)} files, "
                f"{total_chunks:,} chunks, "
                f"{rate:.0f} chunks/sec, "
                f"{elapsed:.0f}s elapsed"
            )

    save_checkpoint(checkpoint_path, done_files)
    elapsed = time.time() - t_start
    print(
        f"Done: loaded {total_chunks:,} chunks from {total_files} files "
        f"in {elapsed:.1f}s"
    )


def _upsert_batch(store, docs, metas, ids):
    """Batch upsert directly to the wiki_knowledge collection."""
    collection = store.collections.get("wiki_knowledge")
    if collection is None:
        raise RuntimeError("wiki_knowledge collection not initialized")
    collection.upsert(
        documents=docs,
        metadatas=metas,
        ids=ids,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Load embedded Wikipedia parquets into ChromaDB wiki_knowledge"
    )
    parser.add_argument(
        "--parquet-dir",
        type=str,
        required=True,
        help="Directory containing .parquet files from the embedding pipeline",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Chunks per ChromaDB upsert batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be loaded without writing",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_FILE),
        help=f"Checkpoint file path (default: {CHECKPOINT_FILE})",
    )
    args = parser.parse_args()

    load_parquet_to_chromadb(
        parquet_dir=args.parquet_dir,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        checkpoint_path=Path(args.checkpoint),
    )


if __name__ == "__main__":
    main()
