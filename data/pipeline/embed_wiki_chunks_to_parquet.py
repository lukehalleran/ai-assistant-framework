# embed_wiki_chunks_to_parquet.py
print("UPGRADED PARQUET EMBEDDER LAUNCHED", flush=True)
# === API Keys and Config ===
import os
import logging

# Use the root logger or create a child logger that will inherit handlers
logger = logging.getLogger(__name__)
logger.debug("embed_wiki_chunks_to_parquet.py is alive")
import os
import json
import sys
import shutil
import time
import queue
import threading
from pathlib import Path
from datetime import datetime
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from more_itertools import chunked
import logging
# from Injection_Protection_Feed_to_Embeder import sanitize_text # Commented out if not available

# ==== TEST MODE CONFIG ====
TEST_MODE = os.environ.get("TEST_MODE") == "1" # Ensure TEST_MODE is read directly here

# ==== CONFIGURATION ====
CHUNK_SIZE = 512 # These seem to be internal to semantic_chunk_text, not used externally for producer
OVERLAP = 256
ENCODE_BATCH_SIZE = 8
SAFE_BATCH_SIZE = 50
HEARTBEAT_INTERVAL = 100
MIN_FREE_GB = 10
MAX_QUEUE_SIZE = 100
NUM_CONSUMER_THREADS = int(os.environ.get("NUM_CONSUMER_THREADS", "1"))

# WIKI_PATH refers to the input directory for basic chunks.
# We need to adjust this to point to semantic_chunks if using semantic chunking.
# However, the producer function will handle globbing, so it's less critical as a global path.
# For clarity, let's make the producer aware of where to look.
# WIKI_PATH = Path("/run/media/lukeh/T9/quarentine_chunks") # Original, potentially unused or misnamed
SEMANTIC_CHUNKS_SOURCE_DIR = Path("semantic_chunks") # New variable to point to semantic chunks
BASIC_CHUNKS_SOURCE_DIR = Path("wiki_chunks") # New variable to point to basic chunks

PARQUET_PATH = Path("embedded_parquet")
CHECKPOINT_FILE = Path("embed_checkpoint.json")
PARQUET_PATH.mkdir(parents=True, exist_ok=True)

# ==== DISK CHECK ====
def check_disk_space(path="/"):
    total, used, free = shutil.disk_usage(path)
    free_gb = free // (2**30)
    if free_gb < MIN_FREE_GB:
        print(f"🛑 LOW DISK SPACE: Only {free_gb} GB free. Exiting.")
        sys.exit(1)
    print(f"[INFO] Disk check passed: {free_gb} GB free")

check_disk_space("/run/media/lukeh/T9")

# ==== INIT ====
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}", flush=True)
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
model.max_seq_length = 384

# ==== CHECKPOINT ====
def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_checkpoint(done_chunks):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(sorted(list(done_chunks)), f)

done_chunks = load_checkpoint()
logger.debug(f" Total files already done: {len(done_chunks)}")

# ==== CHUNKING ====
# This semantic_chunk_text function is redundant if semantic_chunker.py is used.
# The embedder should *not* be re-chunking. It should consume already chunked data.
# Let's assume it's kept for the 'basic' pipeline path for now, but for semantic, it's skipped.
def semantic_chunk_text(text, target_size=200, max_size=250, min_size=90):
    import re
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > max_size:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 < target_size:
                    current_chunk += sentence + " "
                else:
                    if len(current_chunk) >= min_size:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
                    else:
                        current_chunk += sentence + " "
        else:
            if len(current_chunk) + len(para) + 2 < target_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

    if current_chunk and len(current_chunk) >= min_size:
        chunks.append(current_chunk.strip())

    return chunks

def chunk_text_with_metadata(text, source_name):
    # This function is also for basic chunking.
    # When semantic chunking is used, this function will not be called for the embedder.
    chunks_data = []
    chunk_texts = semantic_chunk_text(text) # Uses the internal semantic_chunk_text
    current_pos = 0

    for i, chunk_text in enumerate(chunk_texts):
        char_start = text.find(chunk_text, current_pos)
        if char_start == -1:
            char_start = current_pos
        char_end = char_start + len(chunk_text)
        current_pos = char_end

        chunks_data.append({
            'text': chunk_text,
            'chunk_index': i,
            'total_chunks': len(chunk_texts),
            'prev_chunk': chunk_texts[i-1][:100] if i > 0 else "",
            'next_chunk': chunk_texts[i+1][:100] if i < len(chunk_texts)-1 else "",
            'char_start': char_start,
            'char_end': char_end
        })

    return chunks_data

# ==== EMBEDDER ====
def embedder(file_queue, done_chunks):
    processed = 0
    total_time = 0
    total_chunks = 0
    done_articles_count = len(done_chunks)

    while True:
        item = file_queue.get()
        if item is None:
            break

        source_name, title, chunks_data = item

        if source_name in done_chunks:
            print(f"[SKIP EMBEDDER] {source_name} already done.", flush=True)
            file_queue.task_done()
            continue

        logger.debug(f" Embedding chunk for {title} (source: {source_name})...", flush=True)

        start_time = time.time()
        try:
            texts = [c['text'][:model.max_seq_length] for c in chunks_data]
            texts = [t.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore") for t in texts]

            embeddings = model.encode(
                texts,
                batch_size=ENCODE_BATCH_SIZE,
                show_progress_bar=False,
                device=device,
                normalize_embeddings=True
            )

            # Call the REVISED store_to_parquet here
            store_to_parquet(source_name, title, chunks_data, embeddings)
            done_chunks.add(source_name)

            if device == "cuda":
                torch.cuda.empty_cache()

            end_time = time.time()
            elapsed = end_time - start_time
            # Correct output_parquet_path for console message
            stem = source_name.replace(" ", "_").replace("/", "_").replace(":", "_").replace("__", "_")
            output_parquet_path = PARQUET_PATH / f"{stem}.parquet"
            file_size = os.path.getsize(output_parquet_path) / (1024 * 1024)

            print(f"✅ Finished {source_name}: {len(chunks_data)} chunks, {file_size:.2f} MB, {elapsed:.2f} sec", flush=True)

            total_time += elapsed
            total_chunks += len(chunks_data)
            processed += 1

            if processed % HEARTBEAT_INTERVAL == 0:
                save_checkpoint(done_chunks)

        except Exception as e:
            print(f"[ERROR] Failed to embed {source_name}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            file_queue.task_done()
            continue
        file_queue.task_done()

    save_checkpoint(done_chunks)
    print(f"🎉 All done: {processed} files (articles), {total_chunks} total chunks, {total_time:.2f} sec total", flush=True)
    print(f"🧠 Average time per file: {total_time / max(1, processed):.2f} sec")
    print(f"📊 Average time per chunk: {total_time / max(1, total_chunks):.2f} sec")


# REVISED PRODUCER TO GROUP CHUNKS BY ARTICLE (for semantic chunks)
def producer(file_queue, done_chunks):
    import glob
    jsonl_files = sorted(glob.glob(str(SEMANTIC_CHUNKS_SOURCE_DIR / "*.jsonl")))

    article_chunks_buffer = {}
    test_article_count = 0

    for jsonl_path in jsonl_files:
        if TEST_MODE and test_article_count >= 10:
            print(f"🧪 Test mode: queued {test_article_count} articles (or parts) — exiting producer loop.", flush=True)
            break

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                chunk_text = data['text']
                metadata = data['metadata']

                article_title = metadata.get('title', 'Unknown Title')
                page_id = metadata.get('page_id', 'Unknown Page ID')
                article_key = (article_title, page_id)
                article_identifier = f"{article_title}_{page_id}"

                if article_identifier in done_chunks:
                    continue

                if TEST_MODE and test_article_count >= 10 and article_key not in article_chunks_buffer:
                    break

                if article_key not in article_chunks_buffer:
                    article_chunks_buffer[article_key] = []
                    if TEST_MODE:
                        test_article_count += 1

                article_chunks_buffer[article_key].append({
                    'text': chunk_text,
                    'chunk_index': metadata.get('chunk_idx', 0),
                    'total_chunks': metadata.get('total_chunks', 1),
                    'prev_snippet': '',
                    'next_snippet': '',
                    'char_start': 0,
                    'char_end': len(chunk_text),
                    'section': metadata.get('section', ''),
                    'section_level': metadata.get('section_level', 0)
                })

        for article_key, chunks_list in list(article_chunks_buffer.items()):
            article_title, page_id = article_key
            article_identifier = f"{article_title}_{page_id}"

            if article_identifier in done_chunks or (TEST_MODE and test_article_count >= 10):
                pass

            print(f"[QUEUEING] Article {article_identifier} with {len(chunks_list)} chunks", flush=True)
            file_queue.put((article_identifier, article_title, chunks_list))
            del article_chunks_buffer[article_key]

    for article_key, chunks_list in article_chunks_buffer.items():
        article_title, page_id = article_key
        article_identifier = f"{article_title}_{page_id}"
        if article_identifier in done_chunks:
            continue
        print(f"[QUEUEING] Remaining Article {article_identifier} with {len(chunks_list)} chunks", flush=True)
        file_queue.put((article_identifier, article_title, chunks_list))

    for _ in range(NUM_CONSUMER_THREADS):
        file_queue.put(None)


# REVISED store_to_parquet to handle source_name as article identifier
def store_to_parquet(source_name, title, chunks_data, embeddings):
    safe_filename = source_name.replace(" ", "_").replace("/", "_").replace(":", "_").replace("__", "_")
    output_path = PARQUET_PATH / f"{safe_filename}.parquet"

    df = pd.DataFrame({
        "id": [f"{title}_{c.get('chunk_index', i)}" for i, c in enumerate(chunks_data)],
        "title": [title] * len(chunks_data),
        "text": [c['text'] for c in chunks_data],
        "embedding": [emb.tolist() for emb in embeddings],
        "chunk_index": [c.get('chunk_index', i) for i, c in enumerate(chunks_data)],
        "total_chunks": [c.get('total_chunks', len(chunks_data)) for c in chunks_data],
        "prev_snippet": [c.get('prev_snippet', '') for c in chunks_data],
        "next_snippet": [c.get('next_snippet', '') for c in chunks_data],
        "char_start": [c.get('char_start', 0) for c in chunks_data],
        "char_end": [c.get('char_end', len(c['text'])) for c in chunks_data],
        "section": [c.get('section', '') for c in chunks_data],
        "section_level": [c.get('section_level', 0) for c in chunks_data]
    })
    df.to_parquet(output_path, index=False)

# ==== MAIN ====
def run_embedding_pipeline():
    print("✅ Starting upgraded embedding pipeline", flush=True)
    file_queue = queue.Queue(MAX_QUEUE_SIZE)

    prod_thread = threading.Thread(target=producer, args=(file_queue, done_chunks))
    prod_thread.start()

    consumer_threads = []
    for _ in range(NUM_CONSUMER_THREADS):
        cons_thread = threading.Thread(target=embedder, args=(file_queue, done_chunks))
        cons_thread.start()
        consumer_threads.append(cons_thread)

    prod_thread.join()
    for cons_thread in consumer_threads:
        cons_thread.join()

    print("🎉 Embedding pipeline finished.", flush=True)
