#!/usr/bin/env python3
"""
Build FAISS IVFPQ index from Wikipedia embeddings.

Two input modes:
  1. Stream from tar.zst (no extraction needed — recommended):
     python scripts/build_faiss_index.py --tar /run/media/lukeh/T9/wiki_embeddings.tar.zst

  2. From extracted parquet directory (legacy):
     python scripts/build_faiss_index.py

Phases can be run separately:
  --phase 1     Extract embeddings + metadata only
  --phase 2     Build FAISS IVFPQ from existing embeddings (if phase 1 done)
  --phase both  Full pipeline (default)

IVFPQ compression: 48 subquantizers × 8 bits = 48 bytes/vector (~32x reduction
from 1536-byte float32). Full 41M-vector index fits in ~2 GB RAM.

Supports resume: if interrupted, rerun same command to continue.
"""

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import time

import faiss
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
_DATA_ROOT       = os.environ.get("WIKI_DATA_ROOT", "/run/media/lukeh/T9")
_WIKI_OUT        = os.path.join(_DATA_ROOT, "wiki_data")
EMBED_FILE       = os.path.join(_WIKI_OUT, "embeddings_mmap.dat")
METADATA_FILE    = os.path.join(_WIKI_OUT, "metadata.parquet")
FAISS_FILE       = os.path.join(_WIKI_OUT, "vector_index_ivf.faiss")
ONDISK_FILE      = os.path.join(_WIKI_OUT, "vector_index_ivf.faiss.ondisk")
META_PARTS_DIR   = os.path.join(_WIKI_OUT, "_meta_parts")
CHECKPOINT_FILE  = os.path.join(_WIKI_OUT, "_checkpoint.json")
INDEX_META_FILE  = os.path.join(_WIKI_OUT, "index_meta.json")

# Legacy extracted-dir path
PARQUET_DIR      = os.environ.get("PARQUET_DIR",
                                  os.path.join(_DATA_ROOT, "embedded_parquet"))

# ── Constants ──────────────────────────────────────────────────────
DIM           = 384                    # all-MiniLM-L6-v2
FLUSH_EVERY   = 5_000                  # files per checkpoint
ADD_BATCH     = 500_000                # vectors per FAISS add call
META_COLS     = [
    "id", "title", "text", "chunk_index", "total_chunks",
    "prev_snippet", "next_snippet", "char_start", "char_end",
    "section", "section_level",
]


# ── Streaming helpers ──────────────────────────────────────────────

def stream_from_tar(archive_path: str, skip: int = 0):
    """Yield (file_number, DataFrame) from tar.zst without extracting."""
    proc = subprocess.Popen(
        ["zstd", "-d", "-c", archive_path],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        bufsize=4 * 1024 * 1024,
    )
    n = 0
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".parquet"):
                    continue
                n += 1
                if n <= skip:
                    if n % 100_000 == 0:
                        print(f"  skipping {n:,}/{skip:,} ...")
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                try:
                    yield n, pd.read_parquet(io.BytesIO(f.read()))
                except Exception as exc:
                    print(f"  WARN skip {member.name}: {exc}", file=sys.stderr)
    finally:
        proc.terminate()
        proc.wait()


def stream_from_dir(parquet_dir: str):
    """Yield (file_number, DataFrame) from extracted directory."""
    files = sorted(Path(parquet_dir).glob("*.parquet"))
    for i, pf in enumerate(files, 1):
        yield i, pd.read_parquet(pf)


# ── Checkpoint / metadata helpers ──────────────────────────────────

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return None


def save_checkpoint(file_count, idx_offset, part_num):
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"file_count": file_count, "idx_offset": idx_offset,
                    "part_num": part_num}, f)
    os.replace(tmp, CHECKPOINT_FILE)


def flush_meta(batch, part_num):
    """Write a batch of DataFrames as a numbered parquet part file."""
    os.makedirs(META_PARTS_DIR, exist_ok=True)
    combined = pd.concat(batch, ignore_index=True)
    table = pa.Table.from_pandas(combined, preserve_index=False)
    out = os.path.join(META_PARTS_DIR, f"part_{part_num:05d}.parquet")
    pq.write_table(table, out, compression="zstd")
    return part_num + 1


def merge_meta_parts():
    """Merge part files into final metadata.parquet (streaming, low RAM)."""
    parts = sorted(Path(META_PARTS_DIR).glob("part_*.parquet"))
    if not parts:
        print("  No metadata parts to merge.")
        return
    print(f"  Merging {len(parts)} metadata parts ...")
    schema = pq.read_schema(parts[0])
    writer = pq.ParquetWriter(METADATA_FILE, schema, compression="zstd")
    for p in parts:
        writer.write_table(pq.read_table(p))
    writer.close()
    # Clean up parts
    for p in parts:
        p.unlink()
    Path(META_PARTS_DIR).rmdir()
    print(f"  Metadata saved: {METADATA_FILE}")


# ── Phase 1: extract embeddings + metadata ─────────────────────────

def phase1(source_iter):
    """Stream parquets → binary embeddings file + parquet metadata parts."""
    os.makedirs(_WIKI_OUT, exist_ok=True)

    # Resume logic
    ckpt = load_checkpoint()
    if ckpt:
        skip       = ckpt["file_count"]
        idx_offset = ckpt["idx_offset"]
        part_num   = ckpt["part_num"]
        expected_bytes = idx_offset * DIM * 4
        actual_bytes   = os.path.getsize(EMBED_FILE) if os.path.exists(EMBED_FILE) else 0
        if actual_bytes != expected_bytes:
            print(f"  WARN: embedding file {actual_bytes} bytes vs expected "
                  f"{expected_bytes}, truncating")
            with open(EMBED_FILE, "r+b") as f:
                f.truncate(expected_bytes)
        print(f"  Resuming from file {skip:,}, vector {idx_offset:,}, part {part_num}")
    else:
        skip = 0
        idx_offset = 0
        part_num   = 0

    emb_f = open(EMBED_FILE, "ab" if skip > 0 else "wb")
    meta_batch = []
    file_count = skip
    t0 = time.time()

    try:
        for file_num, df in source_iter:
            n = len(df)

            # ── embeddings → raw binary ──
            emb = np.vstack(df["embedding"].values).astype("float32")
            emb_f.write(emb.tobytes())

            # ── metadata ──
            cols = [c for c in META_COLS if c in df.columns]
            bdf = df[cols].copy()
            bdf.insert(0, "idx", range(idx_offset, idx_offset + n))
            bdf["source"] = "wikipedia"
            meta_batch.append(bdf)

            idx_offset += n
            file_count = file_num

            # ── periodic flush ──
            if file_count % FLUSH_EVERY == 0:
                emb_f.flush()
                part_num = flush_meta(meta_batch, part_num)
                meta_batch.clear()
                save_checkpoint(file_count, idx_offset, part_num)

                elapsed = time.time() - t0
                processed = file_count - skip
                rate = processed / elapsed if elapsed > 0 else 0
                emb_gb = os.path.getsize(EMBED_FILE) / 1e9
                print(f"  [{file_count:,} files | {idx_offset:,} vectors | "
                      f"{emb_gb:.1f} GB | {elapsed/60:.1f}min | {rate:.0f} files/s]")

    except KeyboardInterrupt:
        print("\n  Interrupted — saving checkpoint ...")

    # Final flush
    if meta_batch:
        part_num = flush_meta(meta_batch, part_num)
    emb_f.close()
    save_checkpoint(file_count, idx_offset, part_num)

    # Merge metadata parts into single file
    merge_meta_parts()

    # Save index metadata
    with open(INDEX_META_FILE, "w") as f:
        json.dump({"total_vectors": idx_offset, "embedding_dim": DIM,
                    "files_processed": file_count}, f)

    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    elapsed = time.time() - t0
    print(f"\nPhase 1 complete: {file_count:,} files, {idx_offset:,} vectors, "
          f"{elapsed/60:.1f} min")
    return idx_offset


# ── Phase 2: build FAISS index ─────────────────────────────────────

def phase2(total_vectors=None, max_vectors=None, build_dir=None):
    """Build FAISS IVFPQ index from binary embeddings file.

    Uses Product Quantization to compress vectors (~48 bytes each),
    so the full 41M-vector index fits in ~2 GB RAM. No ondisk needed.

    Args:
        total_vectors: Total vectors in embeddings file (auto-detected).
        max_vectors:   Optional cap — build index from first N vectors only.
        build_dir:     Unused (kept for CLI compat). PQ index fits in RAM.
    """
    if total_vectors is None:
        if os.path.exists(INDEX_META_FILE):
            with open(INDEX_META_FILE) as f:
                total_vectors = json.load(f)["total_vectors"]
        else:
            total_vectors = os.path.getsize(EMBED_FILE) // (DIM * 4)

    use_vectors = total_vectors
    if max_vectors and max_vectors < total_vectors:
        use_vectors = max_vectors
        print(f"\nPhase 2: building FAISS IVFPQ for {use_vectors:,} / "
              f"{total_vectors:,} vectors (subset mode) ...")
    else:
        print(f"\nPhase 2: building FAISS IVFPQ for {use_vectors:,} vectors ...")

    emb = np.memmap(EMBED_FILE, dtype="float32", mode="r",
                    shape=(total_vectors, DIM))

    # PQ parameters: 48 subquantizers × 8 bits = 48 bytes/vector
    # 384 dims / 48 subquantizers = 8 dims per sub — divides evenly
    m = 48                                     # subquantizers
    nlist = int(2 * np.sqrt(use_vectors))      # IVF centroids
    train_need = max(nlist * 39, 256 * m)      # PQ needs ≥256 per sub
    train_size = min(use_vectors, max(train_need, 100_000), 500_000)

    index_size_gb = use_vectors * m / 1e9
    print(f"  IVFPQ: {nlist:,} centroids, {m} subquantizers, "
          f"~{index_size_gb:.1f} GB in RAM")
    print(f"  Training on {train_size:,} samples")

    quantizer = faiss.IndexFlatL2(DIM)
    index = faiss.IndexIVFPQ(quantizer, DIM, nlist, m, 8)

    rng = np.random.default_rng(42)
    train_idx = rng.choice(use_vectors, train_size, replace=False)
    train_idx.sort()  # sequential mmap reads
    train_data = np.ascontiguousarray(emb[train_idx])

    mem_gb = train_data.nbytes / 1e9
    print(f"  Training ({mem_gb:.1f} GB training data) ...")
    sys.stdout.flush()
    t0 = time.time()
    index.train(train_data)
    print(f"  Trained in {time.time() - t0:.0f}s")
    sys.stdout.flush()
    del train_data

    # ── Add vectors in batches (all in RAM — PQ is compact) ──
    print("  Adding vectors ...")
    t_add = time.time()
    for start in range(0, use_vectors, ADD_BATCH):
        end = min(start + ADD_BATCH, use_vectors)
        index.add(np.array(emb[start:end]))
        elapsed = time.time() - t_add
        print(f"    {end:,} / {use_vectors:,}  ({elapsed:.0f}s)", flush=True)

    index.nprobe = 32
    faiss.write_index(index, FAISS_FILE)

    faiss_gb = os.path.getsize(FAISS_FILE) / 1e9
    print(f"\n  FAISS saved: {FAISS_FILE}")
    print(f"  {index.ntotal:,} vectors, {faiss_gb:.1f} GB, "
          f"{time.time() - t0:.0f}s total")
    print(f"  PQ compression: {DIM*4} → {m} bytes/vector "
          f"({DIM*4/m:.0f}x reduction)")


# ── Legacy: load from extracted directory (original behavior) ──────

def load_merged_parquet_mmap(parquet_dir: str):
    """Original function — reads extracted parquet directory."""
    from datetime import datetime

    parquet_files = list(Path(parquet_dir).glob("*.parquet"))
    total_rows = sum(pq.ParquetFile(pf).metadata.num_rows for pf in parquet_files)

    embeddings_mmap = np.memmap(EMBED_FILE, dtype="float32", mode="w+",
                                shape=(total_rows, DIM))
    metadata_rows = []
    idx_offset = 0
    ts = datetime.now().isoformat()

    for pf in parquet_files:
        table = pq.read_table(pf)
        df = table.to_pandas()
        n = len(df)
        emb_block = np.vstack(df["embedding"].values).astype("float32")
        embeddings_mmap[idx_offset:idx_offset + n] = emb_block

        for _, row in df.iterrows():
            meta = {"idx": idx_offset, "id": row.get("id", ""),
                    "title": row.get("title", ""), "text": row.get("text", ""),
                    "source": "wikipedia", "namespace": "wikipedia",
                    "timestamp": ts}
            for col in df.columns:
                if col not in {"embedding", "idx", "id", "title", "text",
                               "source", "namespace", "timestamp"}:
                    meta[col] = row.get(col, "")
            metadata_rows.append(meta)
            idx_offset += 1

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_parquet(METADATA_FILE)
    return embeddings_mmap, metadata_df


def build_production_faiss_index(embeddings: np.ndarray):
    """Original FAISS builder for small-ish datasets."""
    d, n = embeddings.shape[1], embeddings.shape[0]
    nlist = int(4 * np.sqrt(n))
    train_size = min(n, max(nlist * 39, 100_000))

    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    index.train(embeddings[np.random.choice(n, train_size, replace=False)])
    if hasattr(index, "nprobe"):
        index.nprobe = 32
    index.add(embeddings)
    return index


# ── Main ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tar", metavar="PATH",
                   default=os.path.join(_DATA_ROOT, "wiki_embeddings.tar.zst"),
                   help="Path to wiki_embeddings.tar.zst")
    p.add_argument("--phase", choices=["1", "2", "both"], default="both",
                   help="1=extract only, 2=FAISS only, both=full (default)")
    p.add_argument("--max-vectors", type=int, metavar="N", default=None,
                   help="Phase 2: only index first N vectors (subset mode)")
    p.add_argument("--build-dir", metavar="DIR", default=None,
                   help="Phase 2: build ondisk index here (fast local SSD), "
                        "then copy to final T9 location")
    p.add_argument("--legacy", action="store_true",
                   help="Use legacy mode (read from extracted parquet dir)")
    args = p.parse_args()

    os.makedirs(_WIKI_OUT, exist_ok=True)

    if args.legacy:
        # Original behavior
        print(f"Legacy mode: reading from {PARQUET_DIR}")
        embeddings, metadata = load_merged_parquet_mmap(PARQUET_DIR)
        if os.path.exists(FAISS_FILE):
            idx = faiss.read_index(FAISS_FILE)
        else:
            idx = build_production_faiss_index(embeddings)
            faiss.write_index(idx, FAISS_FILE)
        print(f"Done ({idx.ntotal} vectors, {len(metadata)} rows).")
        sys.exit(0)

    # Streaming mode
    total = None

    if args.phase in ("1", "both"):
        if not os.path.exists(args.tar):
            sys.exit(f"ERROR: tar not found: {args.tar}")
        ckpt = load_checkpoint()
        skip = ckpt["file_count"] if ckpt else 0
        source = stream_from_tar(args.tar, skip=skip)
        total = phase1(source)

    if args.phase in ("2", "both"):
        phase2(total, max_vectors=args.max_vectors, build_dir=args.build_dir)

    print("\nAll done!")
