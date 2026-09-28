# Dockerfile for Daemon RAG Agent
# Multi-stage build: React/Vite SPA + Python deps/offline models + runtime.
#
# 2026-09-27 (class: BC-71 doc/build drift, BC-82 validated-only-under-dev-config):
# this image predated the 2026-07-14 FastAPI migration and was never updated —
# `api/` wasn't copied (default "gui" mode is `api.app.create_app()`, main.py:1689,
# so the container 404'd at import time), no SPA build existed (React UI absent,
# API + /admin only), everything was wired to the legacy Gradio port 7860 instead
# of the FastAPI default (config/app_config.py API_HOST/API_PORT, 8000), and only
# one of the four models the runtime actually loads offline was pre-downloaded.
# Fixed below against the live app, not the pre-migration one.

# ============================================================================
# Stage 1: Frontend builder - build the React/Vite SPA (served from web/dist,
# api/app.py FRONTEND_DIST_DIR, resolved relative to WORKDIR /app at runtime)
# ============================================================================
FROM node:20-slim AS frontend-builder

WORKDIR /web

# Lockfile first for layer caching
COPY web/package.json web/package-lock.json ./
RUN npm ci

COPY web/ ./
# web/package.json "build": "tsc --noEmit && vite build" -> web/dist
RUN npm run build

# ============================================================================
# Stage 2: Python base - shared image + env for the builder and runtime stages
# (kept separate from `builder` so the base pull/parse can be validated on its
# own without paying for the pip/model download stage below)
# ============================================================================
FROM python:3.11-slim AS python-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ============================================================================
# Stage 3: Builder - install Python dependencies & pre-download offline models
# ============================================================================
FROM python-base AS builder

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

# Pre-download every offline model the runtime actually loads under
# HF_HUB_OFFLINE=1 (2026-09-27 — the prior image only pre-downloaded the
# first of these four, so every other load fell through to the network and
# failed offline): all-MiniLM-L6-v2 (tone/topic/web-trigger embeddings,
# gate wiki + semantic-chunk paths — ModelManager's shared SentenceTransformer),
# BAAI/bge-small-en-v1.5 (the Chroma store's embedder, also the memory gate's
# scoring model — memory/storage/multi_collection_chroma_store.py:186),
# cross-encoder/ms-marco-MiniLM-L-6-v2 (gate rerank — processing/gate_system.py:741),
# gpt2 (token-count fallback tokenizer — models/tokenizer_manager.py:97).
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2'); \
SentenceTransformer('BAAI/bge-small-en-v1.5')"
RUN python -c "\
from sentence_transformers import CrossEncoder; \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
RUN python -c "\
from transformers import AutoTokenizer; \
AutoTokenizer.from_pretrained('gpt2')"

# ============================================================================
# Stage 4: Runtime - minimal production image
# ============================================================================
FROM python-base

# Metadata
LABEL maintainer="Daemon RAG Agent"
LABEL description="Memory-augmented conversational AI with hierarchical memory and RAG"
LABEL version="v4"

# Create non-root user for security (use existing daemon group if present)
RUN groupadd -r daemon 2>/dev/null || true && \
    useradd -r -g daemon daemon 2>/dev/null || true

# Set environment variables
ENV PYTHONPATH=/app \
    # Default to CPU for ChromaDB (override with docker-compose)
    CHROMA_DEVICE=cpu \
    # Hugging Face offline mode - models pre-downloaded in builder stage
    HF_HUB_OFFLINE=1 \
    # Point HF cache to app directory (writable by daemon user)
    HF_HOME=/app/data/cache/huggingface \
    # FastAPI server (config/app_config.py API_HOST/API_PORT read
    # DAEMON_API_HOST/DAEMON_API_PORT). 0.0.0.0 is required here even though
    # the app's own default is loopback (127.0.0.1, config.yaml api.host) —
    # a process bound to the container's loopback is unreachable through
    # Docker's port mapping, which lands on the container's external
    # interface, not its loopback. Note this does NOT open up the
    # Host-header trust check (api/launch_auth.py): "0.0.0.0" is rejected
    # as an unspecified address by normalize_trusted_hostnames, so the
    # trusted set is still loopback-only unless api.allowed_hosts names a
    # real external hostname.
    DAEMON_API_HOST=0.0.0.0 \
    DAEMON_API_PORT=8000 \
    # Legacy/first-run Gradio path (gui/launch.py) — the default "gui" mode
    # only falls back to this (standalone Gradio) for the first-run wizard,
    # or when `--legacy-gui` is passed explicitly; kept reachable the same way.
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
# sentence-transformers/transformers cache to ~/.cache/huggingface by default
COPY --from=builder --chown=daemon:daemon /root/.cache/huggingface /app/data/cache/huggingface

# Set working directory
WORKDIR /app

# Create directory structure with proper permissions
RUN mkdir -p /app/data /app/logs /app/conversation_logs /app/data/cache && \
    chown -R daemon:daemon /app

# Copy application code
# (2026-09-27: api/ added — the FastAPI package `main.py`'s default "gui" mode
# imports; see `from api.app import create_app, mount_admin_and_frontend`.)
COPY --chown=daemon:daemon api/ /app/api/
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

# Built React SPA (2026-09-27: was never built into the prior image; served
# from FRONTEND_DIST_DIR="web/dist", relative to WORKDIR /app, by
# api/app.py::mount_admin_and_frontend when api.serve_frontend is true).
COPY --from=frontend-builder --chown=daemon:daemon /web/dist /app/web/dist

# Make entrypoint executable
RUN chmod +x /app/docker-entrypoint.sh

# Switch to non-root user
USER daemon

# FastAPI port (2026-09-27: was 7860/Gradio; the default "gui" mode serves
# FastAPI + the SPA on API_PORT, config/app_config.py, main.py:1702). 7860
# is also exposed for the first-run wizard / `--legacy-gui` fallback path.
EXPOSE 8000
EXPOSE 7860

# Health check
# Checks /health every 30s, starts after 60s, max 3 failures.
# api/app.py registers a minimal `{"status": "ok"}` /health on the FastAPI
# app (no orchestrator/API-key dependency); the legacy --legacy-gui path
# wires utils.health_check onto Gradio's own app instead.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command: Run GUI mode (FastAPI + SPA unless first-run/--legacy-gui)
# Override with docker-compose or docker run command
CMD ["gui"]
