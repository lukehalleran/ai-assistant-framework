#!/bin/bash
# docker-entrypoint.sh
# Entrypoint script for Daemon RAG Agent container
# Provides initialization, health checks, and graceful shutdown

set -e

# Color output for logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}Daemon RAG Agent - Starting${NC}"
echo -e "${GREEN}=====================================${NC}"

# ============================================================================
# Pre-flight checks
# ============================================================================

echo -e "${YELLOW}Running pre-flight checks...${NC}"

# Check required environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}WARNING: OPENAI_API_KEY not set${NC}"
    echo "Some features may not work without an API key"
fi

# Check data directory
if [ ! -d "/app/data" ]; then
    echo -e "${YELLOW}Creating /app/data directory${NC}"
    mkdir -p /app/data
fi

# Check if corpus file exists, create empty if not
CORPUS_FILE="${CORPUS_FILE:-/app/data/corpus_v4.json}"
if [ ! -f "$CORPUS_FILE" ]; then
    echo -e "${YELLOW}Corpus file not found, creating empty corpus${NC}"
    echo '[]' > "$CORPUS_FILE"
fi

# Check ChromaDB directory
CHROMA_PATH="${CHROMA_PATH:-/app/data/chroma_db_v4_v2}"
if [ ! -d "$CHROMA_PATH" ]; then
    echo -e "${YELLOW}Creating ChromaDB directory${NC}"
    mkdir -p "$CHROMA_PATH"
fi

# Verify Python environment
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "User: $(whoami)"

echo -e "${GREEN}Pre-flight checks complete${NC}"

# ============================================================================
# Start application
# ============================================================================

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}Starting Daemon RAG Agent${NC}"
echo -e "${GREEN}=====================================${NC}"
echo "Mode: ${1:-gui}"
echo "Gradio server: ${GRADIO_SERVER_NAME}:${GRADIO_PORT}"
echo "Health check: http://${GRADIO_SERVER_NAME}:${GRADIO_PORT}/health"
echo -e "${GREEN}=====================================${NC}"

# Execute the main application
# Use exec to replace shell with Python process for proper signal handling
# This allows Python to directly receive SIGTERM/SIGINT for graceful shutdown
exec python main.py "$@"
