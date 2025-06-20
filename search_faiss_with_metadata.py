import os
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from config import DEFAULT_TOP_K


# === CONFIG ===
PARQUET_DIR = "/run/media/lukeh/T9/test_parquet"
MERGED_PARQUET = "merged_embeddings.parquet"
FAISS_INDEX_FILE = "vector_index.faiss"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = DEFAULT_TOP_K



# === BUILD OR LOAD FAISS INDEX ===
def build_production_faiss_index(embeddings: np.ndarray, index_type="IVF"):
    d = embeddings.shape[1]
    n = embeddings.shape[0]

    if index_type == "IVF":
        nlist = int(4 * np.sqrt(n))
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist)

        train_size = min(n, 100000)
        train_data = embeddings[np.random.choice(n, train_size, replace=False)]
        index.train(train_data)

    elif index_type == "HNSW":
        M = 32
        index = faiss.IndexHNSWFlat(d, M)

    elif index_type == "IVF_PQ":
        nlist = int(4 * np.sqrt(n))
        m = 16
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)

        train_size = min(n, 100000)
        train_data = embeddings[np.random.choice(n, train_size, replace=False)]
        index.train(train_data)

    # Add vectors in batches
    batch_size = 100000
    for i in range(0, n, batch_size):
        index.add(embeddings[i:i + batch_size])

    if hasattr(index, "nprobe"):
        index.nprobe = 32

    return index

def load_faiss_index():
    print("[INFO] Loading existing FAISS index...")
    return faiss.read_index(FAISS_INDEX_FILE)
import pyarrow.parquet as pq
from pathlib import Path

def load_merged_parquet_mmap(parquet_dir):
    """Load parquet metadata only, create memory-mapped embeddings."""
    total_vectors = 0
    parquet_files = list(Path(parquet_dir).glob("*.parquet"))

    for pf in parquet_files:
        pq_file = pq.ParquetFile(pf)
        total_vectors += pq_file.metadata.num_rows

    embeddings_mmap = np.memmap(
        "embeddings_mmap.dat",
        dtype="float32",
        mode="w+",
        shape=(total_vectors, 384)
    )

    metadata_rows = []
    current_idx = 0

    for pf in parquet_files:
        table = pq.read_table(pf, columns=["embedding", "id", "title", "text"])
        df = table.to_pandas()

        n_rows = len(df)
        embeddings_mmap[current_idx:current_idx + n_rows] = np.vstack(df["embedding"].values)

        for i, row in df.iterrows():
            metadata_rows.append({
                "idx": current_idx + i,
                "id": row["id"],
                "title": row["title"],
                "text": row["text"]
            })

        current_idx += n_rows

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_parquet("metadata.parquet")

    return embeddings_mmap, metadata_df

# === MAIN SETUP ===
print("✅ Loading model and data...")
model = SentenceTransformer(MODEL_NAME)

embeddings_mmap, metadata_df = load_merged_parquet_mmap(PARQUET_DIR)

if os.path.exists("vector_index_ivf.faiss"):
    index = faiss.read_index("vector_index_ivf.faiss")
else:
    index = build_production_faiss_index(embeddings_mmap, index_type="IVF")
    faiss.write_index(index, "vector_index_ivf.faiss")
# === PUBLIC FUNCTION ===
def semantic_search_optimized(query, top_k=TOP_K, index=None, metadata_df=None):
    """Optimized search with proper index"""
    if index is None or metadata_df is None:
        raise ValueError("Index and metadata must be pre-loaded")

    query_embedding = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding.astype("float32"), top_k)

    results = []
    for i, (dist, idx) in enumerate(zip(D[0], I[0])):
        if idx == -1:
            continue

        meta = metadata_df.iloc[idx]
        results.append({
            "rank": i + 1,
            "score": float(dist),
            "similarity": 1.0 / (1.0 + dist),
            "id": meta["id"],
            "title": meta["title"],
            "text": meta["text"][:500]
        })

    return results
## wrapper for rebuilt function
def semantic_search(query, top_k=TOP_K):
    return semantic_search_optimized(query, top_k=top_k, index=index, metadata_df=metadata_df)




