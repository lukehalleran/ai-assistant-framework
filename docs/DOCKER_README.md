# Docker Deployment Guide

Complete guide for containerized deployment of Daemon RAG Agent.

> **2026-09-27:** this image was rebuilt against the live app (class: BC-71 documentation
> drift, BC-82 validated-only-under-dev-config). The 2026-07-14 FastAPI migration had never
> been carried into the Dockerfile/compose files: `api/` wasn't copied (the default `gui`
> mode is `api.app.create_app()` — the container 404'd at import time), no React SPA was
> built into the image, everything was wired to the legacy Gradio port 7860 instead of the
> FastAPI default (8000), and only one of the four models the runtime actually loads offline
> was pre-downloaded. All of that is fixed below. See `~/Daemon_v1/FOLLOWUPS.md` ("Docker NOT
> ready") for the original finding.

## Quick Start

### Prerequisites

- Docker 20.10+ and Docker Compose 2.0+ (or Podman + podman-compose — the Dockerfile and
  compose file are engine-agnostic)
- 8GB+ RAM recommended
- 10GB+ disk space for image and data
- Network access during the build: the image pulls Python + Node base layers, `pip install`s
  torch/transformers/sentence-transformers, runs `npm ci`, and pre-downloads four Hugging
  Face/spaCy models. None of this is optional — see "Offline Mode" below for *why* it all
  has to happen at build time.

### 1. Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API key
nano .env  # or vim, emacs, etc.

# Required: OPENAI_API_KEY=your_key_here (or another provider's key — see
# config/config.yaml `models:` and the FAQ below)
```

### 2. Build and Run

**Option A: Using docker-compose (recommended)**
```bash
# Build image
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f daemon-gui
```

**Option B: Using standalone Docker**
```bash
# Build image
docker build -t daemon-rag-agent:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -p 7860:7860 \
  --env-file .env \
  --name daemon-rag \
  -v daemon-data:/app/data \
  daemon-rag-agent:latest
```

### 3. Access Application

- **Web UI (React SPA)**: http://localhost:8000/
- **Admin UI (Gradio dev tabs)**: http://localhost:8000/admin
- **Health Check**: http://localhost:8000/health
- **First run only**: a fresh `/app/data` volume has no user profile yet, so the container
  launches the setup wizard as a *standalone Gradio app* on port 7860
  (`gui/launch.py::check_first_run`) instead of the FastAPI server — open
  http://localhost:7860/ to complete it. Once a profile exists, subsequent starts serve the
  FastAPI+SPA app on 8000 as above. `--legacy-gui` (see "Common Commands") also uses 7860.

## Architecture

### Multi-Stage Build

The Dockerfile is a four-stage build:

**Stage 1: `frontend-builder`** (Node 20)
- `npm ci` + `npm run build` in `web/` (Vite + React + TypeScript + Mantine)
- Produces `web/dist`, copied into the runtime stage; nothing from this stage ships in the
  final image except that one directory

**Stage 2: `python-base`**
- Shared `python:3.11-slim` + env vars for the builder and runtime stages below (kept as its
  own stage so it can be built/validated on its own without paying for the heavy stage 3 pip
  install — see "CI/CD Integration")

**Stage 3: `builder`** (~2GB, from `python-base`)
- Installs build dependencies (gcc, g++, git)
- Compiles Python packages (torch, transformers, sentence-transformers)
- Downloads the spaCy `en_core_web_sm` language model
- Pre-downloads all four models the runtime loads offline (see "Offline Mode")
- Creates a virtual environment at `/opt/venv`

**Stage 4: runtime** (~1.5GB, from `python-base`)
- Minimal Python 3.11-slim base
- Copies only the compiled venv from `builder`
- Copies the pre-downloaded model cache from `builder`
- Copies `web/dist` from `frontend-builder`
- Copies the application code, **including `api/`** (the FastAPI package — a prior version
  of this image omitted it, since it predated the FastAPI migration)
- Runs as non-root `daemon` user
- `HEALTHCHECK` against `/health` on port 8000

### Directory Structure (Container)

```
/app/
├── api/                    # FastAPI app (routers, launch_auth, app.py)
├── core/                   # Application code
├── memory/
├── models/
├── gui/                    # Gradio dev UI (mounted at /admin) + wizard
├── web/
│   └── dist/               # Built React SPA (served at /)
├── config/
├── data/                   # Persistent volume mount
│   ├── corpus_v4.json      # Conversation memory
│   ├── chroma_db_v4/       # Vector database
│   └── cache/
│       └── huggingface/    # Pre-downloaded models
├── conversation_logs/      # Optional volume mount
├── main.py
└── docker-entrypoint.sh
```

### Volumes

**daemon-data** (persistent)
- Corpus JSON files
- ChromaDB vector database
- Hugging Face model cache
- Wikipedia indices (if populated)

**daemon-logs** (optional)
- Timestamped conversation logs
- Useful for debugging and analysis

## Usage

### Common Commands

```bash
docker-compose up -d --build         # Build and start
docker-compose down                  # Stop services
docker-compose restart daemon-gui    # Restart services
docker-compose logs -f daemon-gui    # View logs
docker-compose ps                    # Check status
curl http://localhost:8000/health    # Test health endpoint
docker-compose run --rm daemon-gui cli  # Interactive CLI mode
docker-compose run --rm daemon-gui --legacy-gui  # Standalone Gradio on 7860
docker-compose down -v               # Remove all data (volumes)
```

### Common Tasks

**View real-time logs**
```bash
docker-compose logs -f daemon-gui
```

**Check health status**
```bash
curl http://localhost:8000/health
```

**Interactive CLI mode**
```bash
docker-compose run --rm daemon-gui cli
```

**Restart with fresh data**
```bash
docker-compose down -v  # Remove volumes
docker-compose up -d    # Start fresh (re-runs the first-run wizard on :7860)
```

**Access container shell**
```bash
docker-compose exec daemon-gui /bin/bash
```

**Inspect volumes**
```bash
docker volume ls
docker volume inspect daemon-rag-agent_daemon-data
```

## Configuration

### Environment Variables

Configuration is done via the `.env` file (a minimal template — most settings are controlled by `config/config.yaml`, with `config/config.local.yaml` for local overrides). See `.env.example` for the shipped template.

**Required:**
```env
OPENAI_API_KEY=sk-...
```

**Common overrides:**
```env
# Optional API keys
# ANTHROPIC_API_KEY=..., TAVILY_API_KEY=..., WOLFRAM_APP_ID=..., E2B_API_KEY=...

# Mode: "user" (streamlined) or "dev" (all features)
DAEMON_MODE=user

# Prompt token budget (default 10000; floor 8000, ceiling 16000 — config.yaml token_budget)
PROMPT_TOKEN_BUDGET=10000

# FastAPI server (compose already pins these; restate here for a plain `docker run`)
DAEMON_API_HOST=0.0.0.0
DAEMON_API_PORT=8000

# Paths (inside container — pinned in docker-compose.yml's `environment:` block so they
# agree with docker-entrypoint.sh's own directory pre-check; see that file's comments)
CORPUS_FILE=/app/data/corpus_v4.json
CHROMA_PATH=/app/data/chroma_db_v4

# ChromaDB device
CHROMA_DEVICE=cpu  # or 'cuda' for GPU
```

### Resource Limits

Default limits in `docker-compose.yml`:
- **CPU**: 4 cores max, 2 cores reserved
- **Memory**: 8GB max, 4GB reserved

**To adjust:**
```yaml
deploy:
  resources:
    limits:
      cpus: '8.0'
      memory: 16G
```

### Port Configuration

Default port mappings: `8000:8000` (FastAPI + SPA, primary) and `7860:7860` (first-run
wizard / `--legacy-gui` fallback only — see "Access Application" above).

**To change the external port:**
```yaml
ports:
  - "8080:8000"  # Access the SPA at http://localhost:8080
```

## GPU Support

### Requirements
- NVIDIA GPU with CUDA support
- nvidia-docker2 installed
- NVIDIA Container Toolkit

### Setup

1. **Install NVIDIA Container Toolkit**
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

2. **Modify docker-compose.yml**

Uncomment GPU configuration:
```yaml
daemon-gui:
  environment:
    CHROMA_DEVICE: cuda  # Enable GPU

  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

3. **Rebuild and restart**
```bash
docker-compose down
docker-compose build
docker-compose up -d
```

## Offline Mode

The image is built with **offline mode enabled** (`HF_HUB_OFFLINE=1`) for every model the
runtime loads:

| Model | Used for | Source |
|---|---|---|
| `all-MiniLM-L6-v2` | tone/topic/web-trigger embeddings; wiki + semantic-chunk gate paths | `models/model_manager.py` (shared `ModelManager` SentenceTransformer) |
| `BAAI/bge-small-en-v1.5` | ChromaDB store embedder + the memory gate's scoring model | `memory/storage/multi_collection_chroma_store.py:186` |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | gate reranker (top-K rerank after cosine filtering) | `processing/gate_system.py:741` |
| `gpt2` | token-count fallback tokenizer | `models/tokenizer_manager.py:97` |

All four are pre-downloaded in the `builder` stage and their weights are copied into
`/app/data/cache/huggingface` in the runtime stage — a prior version of this image only
pre-downloaded `all-MiniLM-L6-v2`, so every other model silently failed to load under
`HF_HUB_OFFLINE=1` at runtime (the container never surfaced this as a build error, only as
degraded behavior later — see the class: BC-71 note at the top of this file).

This ensures:
- No internet required for embeddings/reranking/tokenization at runtime
- Faster startup (no download wait)
- Reproducible deployments
- Air-gapped deployment support (once built — the *build* itself needs network)

## Health Checks

### Docker Health Check

Built into container, runs every 30s:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

`api/app.py` registers a minimal, dependency-free `{"status": "ok"}` `/health` route (no
orchestrator or API-key check — it answers as soon as the process is listening). The
detailed `utils.health_check.get_health_status()` payload (corpus/ChromaDB/orchestrator/API
key checks) is only wired onto the legacy `--legacy-gui` Gradio app, not the default path.

### Manual Health Check

```bash
curl http://localhost:8000/health
```

**Response (healthy):**
```json
{"status": "ok"}
```

## Troubleshooting

### Container Won't Start

**Check logs:**
```bash
docker-compose logs daemon-gui
```

**Common issues:**
- Missing API key: Add `OPENAI_API_KEY` to `.env`
- Port conflict: Change port mapping in `docker-compose.yml`
- Insufficient memory: Increase Docker memory limit

### Health Check Failing

**Verify endpoint:**
```bash
docker-compose exec daemon-gui curl http://localhost:8000/health
```

**Common causes:**
- Application still starting (wait 60s)
- First run: the container is on the setup wizard (port 7860), not the FastAPI app — see
  "Access Application" above; the 8000 healthcheck will keep failing until the wizard
  finishes and the container is restarted into normal `gui` mode
- FastAPI not binding correctly (check `DAEMON_API_HOST=0.0.0.0` — the app's own config
  default is loopback-only, which Docker's port mapping cannot reach)
- Port mismatch (verify `DAEMON_API_PORT=8000`)

### Out of Memory

**Symptoms:**
- Container killed unexpectedly
- `OOMKilled` in `docker ps -a`

**Solutions:**
```bash
# Increase Docker memory limit
# Docker Desktop: Settings → Resources → Memory

# Or reduce the prompt token budget in .env (floor is 8000):
PROMPT_TOKEN_BUDGET=8000
```

### Slow Performance

**CPU mode (default):**
- Expected: 2-5s response time
- ChromaDB uses CPU embeddings

**To improve:**
1. Enable GPU support (see GPU section)
2. Reduce retrieval/context settings in `config/config.yaml` (e.g. the `token_budget:` section, per-section retrieval counts)
3. Increase CPU allocation:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '8.0'
   ```

### Volume Permissions

**Symptoms:**
- Permission denied errors
- Can't write to corpus file

**Fix:**
```bash
# Container runs as daemon:daemon (system UID/GID — allocated by
# `useradd -r`/`groupadd -r`, not a fixed 999)
# Ensure host volumes have correct permissions

docker-compose down
docker volume rm daemon-rag-agent_daemon-data
docker-compose up -d  # Will recreate with correct permissions
```

### Model Download Issues

**If offline mode fails:**
```bash
# Rebuild with verbose output
docker-compose build --no-cache --progress=plain

# Check the builder stage completed all four model downloads — look for the
# four `RUN python -c "..."` steps in the Dockerfile succeeding, not just the
# first (all-MiniLM-L6-v2).
```

### Frontend Build Issues

**If the SPA doesn't load (`/` falls back to "API + /admin only" in the logs):**
```bash
# Rebuild the frontend-builder stage explicitly
docker-compose build --no-cache --progress=plain daemon-gui

# Confirm `npm run build` succeeded and web/dist/index.html exists inside the image:
docker-compose run --rm --entrypoint /bin/bash daemon-gui -c "ls -la /app/web/dist"
```

## Production Deployment

### Security Checklist

- [ ] Run as non-root user (default: `daemon`)
- [ ] Use `.env` file, not hardcoded secrets
- [ ] Enable TLS/HTTPS via reverse proxy
- [ ] Restrict network access (firewall rules)
- [ ] Set `api.allowed_hosts` (config.yaml) to any real external hostname you expose
      the app under — the Host-header trust check otherwise only accepts loopback names
      (`localhost`/`127.0.0.1`/`::1`); see `api/launch_auth.py`
- [ ] Regular security updates (`docker-compose pull`)
- [ ] Monitor logs for anomalies
- [ ] Backup volumes regularly

### Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name daemon.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Streaming (SSE) support for the chat endpoint
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
    }
}
```

### Monitoring

**Prometheus metrics** (optional):
```bash
# Install cAdvisor for container metrics
docker run -d \
  --name=cadvisor \
  -p 8080:8080 \
  -v /:/rootfs:ro \
  -v /var/run:/var/run:ro \
  -v /sys:/sys:ro \
  -v /var/lib/docker/:/var/lib/docker:ro \
  gcr.io/cadvisor/cadvisor:latest
```

### Backup Strategy

**Automated backup script:**
```bash
#!/bin/bash
# backup.sh - Backup daemon volumes

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/daemon-rag"

# Stop container (optional, for consistency)
docker-compose stop daemon-gui

# Backup data volume
docker run --rm \
  -v daemon-rag-agent_daemon-data:/data \
  -v $BACKUP_DIR:/backup \
  alpine tar czf /backup/daemon-data-$DATE.tar.gz -C /data .

# Backup logs volume
docker run --rm \
  -v daemon-rag-agent_daemon-logs:/data \
  -v $BACKUP_DIR:/backup \
  alpine tar czf /backup/daemon-logs-$DATE.tar.gz -C /data .

# Restart container
docker-compose start daemon-gui

echo "Backup complete: $BACKUP_DIR"
```

### Restore from Backup

```bash
#!/bin/bash
# restore.sh - Restore daemon volumes

BACKUP_FILE=$1

# Stop and remove containers
docker-compose down

# Remove old volume
docker volume rm daemon-rag-agent_daemon-data

# Recreate volume
docker volume create daemon-rag-agent_daemon-data

# Restore data
docker run --rm \
  -v daemon-rag-agent_daemon-data:/data \
  -v $(dirname $BACKUP_FILE):/backup \
  alpine tar xzf /backup/$(basename $BACKUP_FILE) -C /data

# Restart
docker-compose up -d
```

## Development

### Local Development with Docker

**Mount source code for live editing:**
```yaml
# docker-compose.override.yml
services:
  daemon-gui:
    volumes:
      - ./core:/app/core
      - ./memory:/app/memory
      - ./models:/app/models
      - ./gui:/app/gui
      - ./api:/app/api
    environment:
      PYTHONUNBUFFERED: 1
```

Note: this mounts *Python* source only. The SPA under `/app/web/dist` is baked in at build
time by the `frontend-builder` stage — for live frontend editing, run `cd web && npm run
dev` on the host instead (it proxies `/api` to `:8000`, see the top-level `CLAUDE.md`
Commands section) rather than trying to hot-reload inside the container.

**Reload on change:**
```bash
docker-compose up -d
docker-compose restart daemon-gui  # After code changes
```

### Debugging

**Interactive shell:**
```bash
docker-compose run --rm --entrypoint /bin/bash daemon-gui
```

**Test health check:**
```bash
docker-compose exec daemon-gui curl -s http://localhost:8000/health
```

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/docker.yml
name: Build and Push Docker Image

on:
  push:
    branches: [ master ]
  release:
    types: [ published ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: docker build -t daemon-rag-agent:${{ github.sha }} .

      - name: Run health check test
        run: |
          docker run -d --name test -p 8000:8000 \
            -e OPENAI_API_KEY=${{ secrets.OPENAI_API_KEY }} \
            daemon-rag-agent:${{ github.sha }}

          sleep 60  # Wait for startup

          curl -f http://localhost:8000/health || exit 1

          docker stop test

      - name: Push to registry
        if: github.event_name == 'release'
        run: |
          echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
          docker tag daemon-rag-agent:${{ github.sha }} yourusername/daemon-rag-agent:latest
          docker push yourusername/daemon-rag-agent:latest
```

A cheap syntax-only smoke test that needs no more than the base image (useful for a
resource-capped CI runner, or before spending the ~10-20 minutes the full build takes):
```bash
docker build --target python-base -t daemon-rag-agent:base-check .
docker rmi daemon-rag-agent:base-check
```
This validates the Dockerfile parses and the base image pulls, without paying for the
`pip install`/`npm ci`/model-download stages.

## FAQ

### Q: Can I run without Docker?
**A:** Yes, use native Python setup: `pip install -r requirements.txt && python main.py`

### Q: How much disk space is needed?
**A:** ~10GB total:
- Image: ~2GB (Python deps + four pre-downloaded models + the built SPA)
- Data volumes: variable (starts ~100MB, grows with conversations)

### Q: Can I use other LLM providers?
**A:** Yes. Models/providers are configured in `config/config.yaml` (`models:` section — OpenRouter/OpenAI-compatible, Anthropic, DeepSeek, local). Set the matching API key in `.env`:
```env
ANTHROPIC_API_KEY=sk-ant-...
```

### Q: How do I upgrade to a new version?
**A:**
```bash
git pull
docker-compose build
docker-compose up -d
# Data volumes persist automatically
```

### Q: Can I run multiple instances?
**A:** Yes, use different ports and volume names:
```yaml
services:
  daemon-gui-1:
    ports: ["8000:8000"]
    volumes: ["daemon-data-1:/app/data"]

  daemon-gui-2:
    ports: ["8001:8000"]
    volumes: ["daemon-data-2:/app/data"]
```

## Support

- **Issues**: see the project's GitHub Issues page
- **Documentation**: See `README.md` and `CLAUDE.md`

## License

[Your License Here]
