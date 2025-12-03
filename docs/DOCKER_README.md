# Docker Deployment Guide

Complete guide for containerized deployment of Daemon RAG Agent.

## Quick Start

### Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- 8GB+ RAM recommended
- 10GB+ disk space for image and data

### 1. Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API key
nano .env  # or vim, emacs, etc.

# Required: OPENAI_API_KEY=your_key_here
```

### 2. Build and Run

**Option A: Using helper script (recommended)**
```bash
./docker-compose-helper.sh start
```

**Option B: Using docker-compose directly**
```bash
# Build image
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f daemon-gui
```

**Option C: Using standalone Docker**
```bash
# Build image
./build-docker.sh

# Run container
docker run -d \
  -p 7860:7860 \
  --env-file .env \
  --name daemon-rag \
  -v daemon-data:/app/data \
  daemon-rag-agent:latest
```

### 3. Access Application

- **Web GUI**: http://localhost:7860
- **Health Check**: http://localhost:7860/health

## Architecture

### Multi-Stage Build

The Dockerfile uses a two-stage build for optimal image size:

**Stage 1: Builder** (~2GB)
- Installs build dependencies (gcc, g++, git)
- Compiles Python packages (torch, transformers)
- Downloads spaCy language model
- Pre-downloads sentence-transformers embedding model
- Creates virtual environment

**Stage 2: Runtime** (~1.5GB)
- Minimal Python 3.11-slim base
- Copies only compiled dependencies
- Copies pre-downloaded models for offline mode
- Runs as non-root `daemon` user
- Includes health check endpoint

### Directory Structure (Container)

```
/app/
├── core/                   # Application code
├── memory/
├── models/
├── gui/
├── config/
├── data/                   # Persistent volume mount
│   ├── corpus_v4.json     # Conversation memory
│   ├── chroma_db_v4_v2/   # Vector database
│   └── cache/
│       └── huggingface/   # Pre-downloaded models
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

### Helper Script Commands

```bash
./docker-compose-helper.sh start     # Build and start
./docker-compose-helper.sh stop      # Stop services
./docker-compose-helper.sh restart   # Restart services
./docker-compose-helper.sh logs      # View logs
./docker-compose-helper.sh status    # Check status
./docker-compose-helper.sh health    # Test health endpoint
./docker-compose-helper.sh cli       # Interactive CLI mode
./docker-compose-helper.sh clean     # Remove all data
```

### Common Tasks

**View real-time logs**
```bash
docker-compose logs -f daemon-gui
```

**Check health status**
```bash
curl http://localhost:7860/health
```

**Interactive CLI mode**
```bash
docker-compose run --rm daemon-gui cli
```

**Restart with fresh data**
```bash
docker-compose down -v  # Remove volumes
docker-compose up -d    # Start fresh
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

All configuration is done via `.env` file. See `.env.example` for full list of 105+ variables.

**Required:**
```env
OPENAI_API_KEY=sk-...
```

**Common overrides:**
```env
# Model settings
MODEL_MAX_TOKENS=4096
PROMPT_TOKEN_BUDGET=2048
DEFAULT_MODEL=gpt-4o-mini

# Memory settings
SUMMARY_EVERY_N=20
MAX_CONVERSATIONS_BEFORE_SUMMARY=25

# Paths (inside container)
CORPUS_FILE=/app/data/corpus_v4.json
CHROMA_PATH=/app/data/chroma_db_v4_v2

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

Default port mapping: `7860:7860`

**To change external port:**
```yaml
ports:
  - "8080:7860"  # Access at http://localhost:8080
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

The image is built with **offline mode enabled** for Hugging Face models:
- `HF_HUB_OFFLINE=1` prevents network calls to Hugging Face Hub
- `all-MiniLM-L6-v2` embedding model pre-downloaded during build
- Model weights baked into image (~440MB)

This ensures:
- ✅ No internet required for embeddings
- ✅ Faster startup (no download wait)
- ✅ Reproducible deployments
- ✅ Air-gapped deployment support

## Health Checks

### Docker Health Check

Built into container, runs every 30s:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1
```

### Manual Health Check

```bash
curl http://localhost:7860/health
```

**Response (healthy):**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-29T12:34:56.789012",
  "checks": {
    "corpus_file": {
      "status": "ok",
      "exists": true,
      "path": "/app/data/corpus_v4.json"
    },
    "chromadb": {
      "status": "ok",
      "exists": true,
      "path": "/app/data/chroma_db_v4_v2"
    },
    "orchestrator": {
      "status": "ok",
      "initialized": true
    },
    "api_key": {
      "status": "ok",
      "configured": true
    }
  }
}
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
docker-compose exec daemon-gui curl http://localhost:7860/health
```

**Common causes:**
- Application still starting (wait 60s)
- Gradio not binding correctly (check `GRADIO_SERVER_NAME=0.0.0.0`)
- Port mismatch (verify `GRADIO_PORT=7860`)

### Out of Memory

**Symptoms:**
- Container killed unexpectedly
- `OOMKilled` in `docker ps -a`

**Solutions:**
```bash
# Increase Docker memory limit
# Docker Desktop: Settings → Resources → Memory

# Or reduce model token limits in .env:
MODEL_MAX_TOKENS=2048
PROMPT_TOKEN_BUDGET=1024
```

### Slow Performance

**CPU mode (default):**
- Expected: 2-5s response time
- ChromaDB uses CPU embeddings

**To improve:**
1. Enable GPU support (see GPU section)
2. Reduce context window:
   ```env
   PROMPT_MAX_MEMS=20
   PROMPT_MAX_SEMANTIC=10
   ```
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
# Container runs as daemon:daemon (UID 999)
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

# Check builder stage completed:
# Look for: "Downloading (…)MiniLM-L6-v2/.gitattributes"
```

## Production Deployment

### Security Checklist

- [ ] Run as non-root user (default: `daemon`)
- [ ] Use `.env` file, not hardcoded secrets
- [ ] Enable TLS/HTTPS via reverse proxy
- [ ] Restrict network access (firewall rules)
- [ ] Regular security updates (`docker-compose pull`)
- [ ] Monitor logs for anomalies
- [ ] Backup volumes regularly

### Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name daemon.example.com;

    location / {
        proxy_pass http://localhost:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support for Gradio
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
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
    environment:
      PYTHONUNBUFFERED: 1
```

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
docker-compose exec daemon-gui python -c "
from utils.health_check import get_health_status
import json
print(json.dumps(get_health_status(), indent=2))
"
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
          docker run -d --name test -p 7860:7860 \
            -e OPENAI_API_KEY=${{ secrets.OPENAI_API_KEY }} \
            daemon-rag-agent:${{ github.sha }}

          sleep 60  # Wait for startup

          curl -f http://localhost:7860/health || exit 1

          docker stop test

      - name: Push to registry
        if: github.event_name == 'release'
        run: |
          echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
          docker tag daemon-rag-agent:${{ github.sha }} yourusername/daemon-rag-agent:latest
          docker push yourusername/daemon-rag-agent:latest
```

## FAQ

### Q: Can I run without Docker?
**A:** Yes, use native Python setup: `pip install -r requirements.txt && python main.py`

### Q: How much disk space is needed?
**A:** ~10GB total:
- Image: ~1.5GB
- Models: ~600MB
- Data volumes: variable (starts ~100MB, grows with conversations)

### Q: Can I use other LLM providers?
**A:** Yes, set provider in `.env`:
```env
LLM_PROVIDER=anthropic  # or 'openai', 'openrouter'
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
    ports: ["7860:7860"]
    volumes: ["daemon-data-1:/app/data"]

  daemon-gui-2:
    ports: ["7861:7860"]
    volumes: ["daemon-data-2:/app/data"]
```

## Support

- **Issues**: https://github.com/yourusername/daemon-rag-agent/issues
- **Discussions**: https://github.com/yourusername/daemon-rag-agent/discussions
- **Documentation**: See `README.md` and `CLAUDE.md`

## License

[Your License Here]
