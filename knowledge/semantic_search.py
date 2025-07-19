# daemon_7_11_25_refactor/daemon_7_11_25_refactor/knowledge/semantic_search.py
import os
import logging
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from config.config import DEFAULT_TOP_K
from utils.logging_utils import log_and_time

logger = logging.getLogger(__name__)
logger.debug("knowledge.semantic_search is alive")

# === CONFIG (tweak these to your setup) ===
PARQUET_DIR       = "/run/media/lukeh/T9/test_parquet"
MERGED_PARQUET    = os.path.join(PARQUET_DIR, "merged_embeddings.parquet")
METADATA_FILE     = "metadata.parquet"
FAISS_INDEX_FILE  = "vector_index_ivf.faiss"
MODEL_NAME        = "all-MiniLM-L6-v2"
TOP_K             = DEFAULT_TOP_K

# — load or build your index & metadata once on import —
@log_and_time("Load embeddings + metadata")
def _load_resources():
    global model, index, metadata_df

    # 1) load embedding model
    model = SentenceTransformer(MODEL_NAME)

    # 2) load FAISS index
    try:
        if os.path.exists(FAISS_INDEX_FILE):
            index = faiss.read_index(FAISS_INDEX_FILE)
        else:
            logger.warning(f"FAISS index not found: {FAISS_INDEX_FILE}. Semantic search will be disabled.")
            index = None
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        index = None

    # 3) load metadata parquet
    if os.path.exists(METADATA_FILE):
        metadata_df = pd.read_parquet(METADATA_FILE)
    else:
        raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")

_load_resources()

@log_and_time("Semantic search")
def semantic_search(query: str, top_k: int = TOP_K) -> list:
    """
    Perform a FAISS-based semantic search over your pre-loaded index.
    Returns a list of dicts with keys: rank, score, similarity, id, title, text.
    """
    # encode query
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    # search
    D, I = index.search(q_emb, top_k)

    results = []
    for rank, (dist, idx) in enumerate(zip(D[0], I[0]), start=1):
        if idx < 0:
            continue
        row = metadata_df.iloc[idx]
        results.append({
            "rank": rank,
            "score": float(dist),
            "similarity": 1.0 / (1.0 + dist),
            "id":    row["id"],
            "title": row["title"],
            "text":  row["text"][:500]
        })
    return results
