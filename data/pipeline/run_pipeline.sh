#!/bin/bash

# === CONFIG ===
# Adjust these paths to match your project structure
PROJECT_DIR="$HOME/Daemon_RAG_Agent_working"
PIPELINE_SCRIPT_PATH="$PROJECT_DIR/data/pipeline/unified_pipeline.py"
INDEX_SCRIPT_PATH="$PROJECT_DIR/scripts/build_faiss_index.py"
EXTRACTED_XML_PATH="$PROJECT_DIR/data/wiki/enwiki-latest-pages-articles.xml"
LOG_PATH="$PROJECT_DIR/full_pipeline_run.log"

# === COMMANDS ===
# 1. The pipeline command (Chunking + Embedding)
#    This runs first. We are NOT using --max-articles for the full run.
PIPELINE_CMD="python3 $PIPELINE_SCRIPT_PATH --extracted $EXTRACTED_XML_PATH --semantic"

# 2. The indexing command (Build FAISS)
#    This runs only after the pipeline command finishes successfully.
INDEX_CMD="python3 $INDEX_SCRIPT_PATH"

# === EXECUTION ===
echo "🚀 Launching FULL Wikipedia embedding pipeline in the background."
echo "This will run in two stages: (1) Chunk/Embed, (2) Build Index."
echo "Running with low priority and preventing system sleep."
echo "This will take a very, very long time."

# We chain the commands with '&&'. The indexer (INDEX_CMD) will only run
# after the pipeline (PIPELINE_CMD) finishes successfully.
# The entire chain is wrapped by inhibit, nice, and nohup.
nohup bash -c "systemd-inhibit --why='Embedding full Wikipedia' --mode=block nice -n 10 sh -c '$PIPELINE_CMD && $INDEX_CMD'" > "$LOG_PATH" 2>&1 &

echo "---"
echo "✅ Pipeline and Indexer are queued and running in the background."
echo "📄 Logging to: $LOG_PATH"
echo "🧠 Run 'tail -f $LOG_PATH' to watch progress."
echo "🛑 To stop, find the process ID with 'ps aux | grep systemd-inhibit' and use 'kill <PID>'."

