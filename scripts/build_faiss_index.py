#!/usr/bin/env python3
# scripts/build_faiss_index.py

import os
import faiss
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pathlib import Path

PARQUET_DIR       = "/run/media/lukeh/T9/embedded_parquet"  # Updated to use semantic chunks
EMBED_MMAP_FILE   = "/run/media/lukeh/T9/wiki_data/embeddings_mmap.dat"
METADATA_FILE     = "/run/media/lukeh/T9/wiki_data/metadata.parquet"
FAISS_INDEX_FILE  = "/run/media/lukeh/T9/wiki_data/vector_index_ivf.faiss"
MODEL_NAME        = "all-MiniLM-L6-v2"

def load_merged_parquet_mmap(parquet_dir: str):
    """Load parquet metadata only, create memory-mapped embeddings."""
    from datetime import datetime

    # 1) Count total rows across all .parquet files
    parquet_files = list(Path(parquet_dir).glob("*.parquet"))
    total_rows = sum(pq.ParquetFile(pf).metadata.num_rows for pf in parquet_files)

    # 2) Memory-map an array of shape (total_rows, embedding_dim)
    embedding_dim = 384  # change if your embeddings have a different size
    embeddings_mmap = np.memmap(
        EMBED_MMAP_FILE,
        dtype="float32",
        mode="w+",
        shape=(total_rows, embedding_dim)
    )

    # 3) Iterate again and fill the mmap
    metadata_rows = []
    idx_offset = 0
    index_timestamp = datetime.now().isoformat()

    for pf in parquet_files:
        # Read all columns (not just embedding, id, title, text)
        table = pq.read_table(pf)
        df = table.to_pandas()
        n = len(df)

        # assume each row's "embedding" is a list or array of length embedding_dim
        emb_block = np.vstack(df["embedding"].values).astype("float32")
        embeddings_mmap[idx_offset:idx_offset + n] = emb_block

        # collect metadata for each row - preserve ALL fields
        for i, row in df.iterrows():
            metadata_row = {
                "idx": idx_offset + i,
                "id": row.get("id", ""),
                "title": row.get("title", ""),
                "text": row.get("text", ""),
                "source": "wikipedia",  # Add source field
                "namespace": "wikipedia",  # Add namespace field
                "timestamp": index_timestamp,  # Add indexing timestamp
            }

            # Preserve all other fields from the source parquet
            for col in df.columns:
                if col not in ["embedding", "idx", "id", "title", "text", "source", "namespace", "timestamp"]:
                    metadata_row[col] = row.get(col, "")

            metadata_rows.append(metadata_row)
        idx_offset += n

    # 4) Build a DataFrame & save it
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_parquet(METADATA_FILE)

    # 5) Return both artifacts
    return embeddings_mmap, metadata_df


def build_production_faiss_index(embeddings: np.ndarray):
    d = embeddings.shape[1]
    n = embeddings.shape[0]

    # number of IVF centroids
    nlist = int(4 * np.sqrt(n))

    # FAISS recommends ~39×nlist training points
    required_train = nlist * 39
    train_size = min(n, max(required_train, 100_000))

    quantizer = faiss.IndexFlatL2(d)
    index     = faiss.IndexIVFFlat(quantizer, d, nlist)

    # sample enough points
    train_data = embeddings[np.random.choice(n, train_size, replace=False)]
    index.train(train_data)

    if hasattr(index, "nprobe"):
        index.nprobe = 32

    index.add(embeddings)
    return index



if __name__ == "__main__":
    print("🔨 Building FAISS index…")
    # This is where the NameError was happening
    embeddings, metadata = load_merged_parquet_mmap(PARQUET_DIR)

    if os.path.exists(FAISS_INDEX_FILE):
        idx = faiss.read_index(FAISS_INDEX_FILE)
    else:
        idx = build_production_faiss_index(embeddings)
        faiss.write_index(idx, FAISS_INDEX_FILE)

    print(f"✅ FAISS index built ({idx.ntotal} vectors) and metadata ({len(metadata)} rows).")
