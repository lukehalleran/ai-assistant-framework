# Dockerfile for Daemon RAG Agent
# Multi-stage build for optimized production image with offline model support

# ============================================================================
# Stage 1: Builder - Install dependencies & Pre-download Models
# ============================================================================
FROM python:3.11-slim AS builder

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better layer caching
WORKDIR /tmp
COPY requirements.txt .

# Install Python dependencies
# Note: This may take 5-10 minutes due to torch, transformers, sentence-transformers
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Download spaCy language model (required for NLP)
RUN python -m spacy download en_core_web_sm

# Pre-download sentence-transformers embedding model
# This ensures offline mode (HF_HUB_OFFLINE=1) works at runtime
# The model weights will be cached and copied to the runtime stage
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ============================================================================
# Stage 2: Runtime - Create minimal production image
# ============================================================================
FROM python:3.11-slim

# Metadata
LABEL maintainer="Daemon RAG Agent"
LABEL description="Memory-augmented conversational AI with hierarchical memory and RAG"
LABEL version="v4"

# Create non-root user for security (use existing daemon group if present)
RUN groupadd -r daemon 2>/dev/null || true && \
    useradd -r -g daemon daemon 2>/dev/null || true

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    # Default to CPU for ChromaDB (override with docker-compose)
    CHROMA_DEVICE=cpu \
    # Hugging Face offline mode - models pre-downloaded in builder stage
    HF_HUB_OFFLINE=1 \
    # Point HF cache to app directory (writable by daemon user)
    HF_HOME=/app/data/cache/huggingface \
    # Gradio server settings
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_PORT=7860

# Install runtime system dependencies
# Note: Minimal set for spaCy, ChromaDB, and basic operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy cached models from builder to app cache
# This enables offline mode by providing pre-downloaded model weights
# sentence-transformers caches to ~/.cache/huggingface by default
COPY --from=builder --chown=daemon:daemon /root/.cache/huggingface /app/data/cache/huggingface

# Set working directory
WORKDIR /app

# Create directory structure with proper permissions
RUN mkdir -p /app/data /app/logs /app/conversation_logs /app/data/cache && \
    chown -R daemon:daemon /app

# Copy application code
COPY --chown=daemon:daemon config/ /app/config/
COPY --chown=daemon:daemon core/ /app/core/
COPY --chown=daemon:daemon memory/ /app/memory/
COPY --chown=daemon:daemon models/ /app/models/
COPY --chown=daemon:daemon utils/ /app/utils/
COPY --chown=daemon:daemon gui/ /app/gui/
COPY --chown=daemon:daemon processing/ /app/processing/
COPY --chown=daemon:daemon knowledge/ /app/knowledge/
COPY --chown=daemon:daemon integrations/ /app/integrations/
COPY --chown=daemon:daemon personality/ /app/personality/
COPY --chown=daemon:daemon main.py /app/
COPY --chown=daemon:daemon .env.example /app/
COPY --chown=daemon:daemon docker-entrypoint.sh /app/

# Make entrypoint executable
RUN chmod +x /app/docker-entrypoint.sh

# Switch to non-root user
USER daemon

# Expose Gradio port
EXPOSE 7860

# Health check
# Checks the /health endpoint every 30s, starts after 60s, max 3 failures
# Note: gui/launch.py registers /health endpoint on Gradio's FastAPI app
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Set entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command: Run GUI mode
# Override with docker-compose or docker run command
CMD ["gui"]
