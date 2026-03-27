#!/bin/bash

# === CONFIG ===
PROJECT_DIR="${PROJECT_DIR:-$HOME/Daemon_v1}"
PIPELINE_SCRIPT_PATH="$PROJECT_DIR/data/pipeline/unified_pipeline.py"
INDEX_SCRIPT_PATH="$PROJECT_DIR/scripts/build_faiss_index.py"
LOADER_SCRIPT_PATH="$PROJECT_DIR/scripts/load_parquet_to_chromadb.py"
EXTRACTED_XML_PATH="${EXTRACTED_XML_PATH:-$PROJECT_DIR/data/pipeline/wiki_data/enwiki-latest-pages-articles.xml}"
LOG_PATH="$PROJECT_DIR/full_pipeline_run.log"

# Embedding config — tune for your hardware
export ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-128}"
export NUM_CONSUMER_THREADS="${NUM_CONSUMER_THREADS:-2}"
export DISK_CHECK_PATH="${DISK_CHECK_PATH:-.}"

# FAISS/parquet output paths
export WIKI_DATA_ROOT="${WIKI_DATA_ROOT:-$PROJECT_DIR/data/pipeline}"
export PARQUET_DIR="${PARQUET_DIR:-$WIKI_DATA_ROOT/embedded_parquet}"

# === COMMANDS ===
# 1. Chunk + Embed → parquet
PIPELINE_CMD="python3 $PIPELINE_SCRIPT_PATH --extracted $EXTRACTED_XML_PATH --semantic"

# 2. Build FAISS index from parquets
INDEX_CMD="python3 $INDEX_SCRIPT_PATH"

# 3. Load parquets into ChromaDB wiki_knowledge collection
LOADER_CMD="python3 $LOADER_SCRIPT_PATH --parquet-dir $PARQUET_DIR"

# === EXECUTION ===
echo "Launching FULL Wikipedia embedding pipeline in the background."
echo "Three stages: (1) Chunk/Embed, (2) Build FAISS Index, (3) Load to ChromaDB"
echo ""
echo "  ENCODE_BATCH_SIZE=$ENCODE_BATCH_SIZE"
echo "  NUM_CONSUMER_THREADS=$NUM_CONSUMER_THREADS"
echo "  EXTRACTED_XML_PATH=$EXTRACTED_XML_PATH"
echo "  PARQUET_DIR=$PARQUET_DIR"
echo "  WIKI_DATA_ROOT=$WIKI_DATA_ROOT"
echo ""

nohup bash -c "systemd-inhibit --why='Embedding full Wikipedia' --mode=block nice -n 10 sh -c '$PIPELINE_CMD && $INDEX_CMD && $LOADER_CMD'" > "$LOG_PATH" 2>&1 &

echo "Pipeline queued and running in the background (PID: $!)."
echo "Logging to: $LOG_PATH"
echo "Run 'tail -f $LOG_PATH' to watch progress."
echo "To stop: kill $!"
