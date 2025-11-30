# Daemon: Memory-Augmented AI Assistant

A near production-grade conversational AI system with hierarchical memory, semantic search, and Wikipedia-scale knowledge retrieval - **built from first principles in 4 months**.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GPU](https://img.shields.io/badge/gpu-RTX_3060-green.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Screenshots 
<img width="1919" height="1071" alt="2025-07-28_20-48" src="https://github.com/user-attachments/assets/eac216c2-b9be-4e5b-a799-cc0a41d80266" />

This is assistant respoding to me on a fresh start up, demonstrating tone and memory peristence
# Daemon RAG Agent

<div align="center">

**A Production-Grade Conversational AI System with Hierarchical Memory and Wikipedia-Scale Knowledge Retrieval**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker Ready](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

*Built from first principles over 8 months • 54,000+ lines of code • 195 files*

[Features](#features) • [Quick Start](#quick-start) • [Architecture](#architecture) • [Usage](#usage) • [Deployment](#deployment)

</div>

---

## Overview

Daemon is a **full-stack RAG (Retrieval-Augmented Generation) system** implementing a complete conversational AI pipeline with cognitive-inspired memory architecture. Built entirely from first principles—before discovering LangChain or LlamaIndex—this project demonstrates deep understanding of every component in modern AI systems.

**What makes Daemon different:**

- **5-tier hierarchical memory** modeled on human cognition (episodic, semantic, procedural, summary, meta)
- **Multi-stage relevance filtering** using FAISS → cosine similarity → cross-encoder reranking
- **Wikipedia-scale knowledge** with 6.5M+ articles semantically indexed
- **Crisis-aware tone detection** that adapts response depth to emotional context
- **Chain-of-thought reasoning** via thinking blocks for transparent decision-making

This is a portfolio project representing 8 months of part-time development, demonstrating the journey from Python beginner to architecting production-scale AI systems.

---

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Hierarchical Memory** | 5-tier cognitive architecture with temporal decay and access reinforcement |
| **Hybrid Retrieval** | Combines recent context + semantic search for balanced recall |
| **Multi-Stage Gating** | FAISS → cosine → cross-encoder pipeline (~200ms total) |
| **STM Analyzer** | Short-term memory compression reduces redundant context |
| **Thinking Blocks** | Chain-of-thought reasoning with transparent decision logs |
| **Tone Detection** | Crisis-aware response adaptation (HIGH/MEDIUM/CONCERN/CONVERSATIONAL) |
| **Multi-Provider LLM** | OpenAI, Anthropic, DeepSeek, Google, and local models |
| **Wikipedia Integration** | 6.5M+ articles with FAISS semantic search |

### Technical Highlights

```
54,349  lines of code
  195   files
   65   test files
  861+  extracted facts in ChromaDB
6.5M+  Wikipedia articles indexed
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- OpenAI API key (or compatible provider)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/daemon-rag-agent.git
cd daemon-rag-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Set API key
export OPENAI_API_KEY=sk-your-key-here
# Or create .env file with: OPENAI_API_KEY=sk-your-key-here
```

### Launch

```bash
# GUI mode (recommended)
python main.py
# Opens at http://localhost:7860

# CLI mode
python main.py cli
```

### Docker Quick Start

```bash
# Build and run
./build-docker.sh
docker-compose up -d

# Access at http://localhost:7860
```

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER QUERY                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR                                                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Topic     │  │    Tone     │  │   Heavy     │  │   File Processing   │ │
│  │ Extraction  │  │  Detection  │  │Topic Check  │  │  (PDF/DOCX/CSV)     │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                │                     │           │
│         └────────────────┴────────────────┴─────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STM ANALYZER (Short-Term Memory)                                            │
│  Analyzes last 8 turns → extracts topic, intent, tone, open_threads         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  MEMORY RETRIEVAL (Parallel)                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  MemoryCoordinator V2                                                 │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────────┐  │   │
│  │  │  Retriever │  │   Scorer   │  │  Storage   │  │ ThreadManager  │  │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │  Recent    │  │  Semantic  │  │   Facts    │  │ Summaries  │            │
│  │ (15 max)   │  │  (15 max)  │  │  (30 max)  │  │  (10 max)  │            │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘            │
│                                                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                             │
│  │Reflections │  │   Wiki     │  │  Semantic  │                             │
│  │  (10 max)  │  │  (3 max)   │  │  Chunks    │                             │
│  └────────────┘  └────────────┘  └────────────┘                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  MULTI-STAGE GATING                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐   │
│  │  Stage 1: FAISS  │──│ Stage 2: Cosine  │──│ Stage 3: Cross-Encoder   │   │
│  │    (top 50)      │  │   (threshold)    │  │    (MS-MARCO rerank)     │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PROMPT BUILDER                                                              │
│  Token Budget: 15,000 │ Middle-Out Compression │ Cross-Section Deduplication│
│                                                                              │
│  [TIME] → [RECENT] → [MEMORIES] → [FACTS] → [SUMMARIES] → [STM] → [QUERY]  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LLM GENERATION                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  <thinking>Step-by-step reasoning...</thinking>                        │ │
│  │  Final response to user                                                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  Modes: Standard │ Best-of-N │ Duel │ Ensemble                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  MEMORY STORAGE                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────────┐ │
│  │ CorpusManager  │  │   ChromaDB     │  │     Fact Extractor            │ │
│  │    (JSON)      │  │   (Vectors)    │  │  (spaCy + LLM triples)        │ │
│  └────────────────┘  └────────────────┘  └────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                           RESPONSE TO USER
```

### Memory System (5 Tiers)

The memory architecture is inspired by cognitive science models of human memory:

```
┌─────────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL MEMORY                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  EPISODIC (conversations)                                │   │
│  │  Raw Q/A pairs with thread context                       │   │
│  │  Storage: JSON + ChromaDB                                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  SEMANTIC (facts)                                        │   │
│  │  Extracted triples: (subject, relation, object)          │   │
│  │  Example: ("user", "prefers", "Python")                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  PROCEDURAL (patterns)                                   │   │
│  │  Learned behaviors and user preferences                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  SUMMARY (compressed blocks)                             │   │
│  │  LLM-generated summaries every N conversations           │   │
│  │  Preserves key information, reduces token usage          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  META (reflections)                                      │   │
│  │  Session-end quality assessments                         │   │
│  │  What went well, areas for improvement                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Scoring Algorithm

Memories are ranked using a composite scoring formula:

```python
final_score = (
    0.60 * relevance      +    # Semantic similarity to query
    0.15 * recency_decay  +    # Time-based decay (active hours)
    0.20 * truth_score    +    # Access reinforcement (0.5 base + 0.02 per access)
    0.02 * importance     +    # Keyword/entity presence
    0.02 * continuity     +    # Token overlap with recent context
    0.01 * structure      +    # Numeric/pattern matching
    anchor_bonus - penalties   # Thread continuity, topic match, penalties
)
```

### Retrieval Pipeline Performance

| Stage | Operation | Latency | Output |
|-------|-----------|---------|--------|
| 1 | FAISS Search | ~50ms | Top 50 candidates |
| 2 | Cosine Filter | ~50ms | ~20-30 results |
| 3 | Cross-Encoder | ~100ms | Final top K |
| **Total** | | **~200ms** | |

---

## Usage

### GUI Interface

The Gradio interface provides a full-featured chat experience:

```bash
python main.py
# Opens at http://localhost:7860
```

**Features:**
- Real-time streaming responses
- Model selection dropdown
- Personality switching
- Raw mode toggle (bypass RAG)
- Debug trace panel
- File upload (PDF, DOCX, CSV)

### CLI Mode

For terminal-based interaction:

```bash
python main.py cli

# Example session:
> What is quantum computing?
[Daemon generates response with full RAG pipeline]

> /topic quantum mechanics
[Sets conversation topic for better retrieval]

> Tell me about superposition
[Continues with topic-aware context]

> /clear_topic
[Clears topic constraint]
```

### Commands

| Command | Description |
|---------|-------------|
| `/topic <name>` | Set conversation topic |
| `/clear_topic` | Clear topic constraint |
| `/model <name>` | Switch LLM model |
| `/raw` | Toggle raw mode |
| `/debug` | Show debug trace |
| `/quit` | Exit CLI |

### Programmatic Access

```python
from gradio_client import Client

client = Client("http://localhost:7860")

result = client.predict(
    message="Explain how RAG systems work",
    history=[],
    personality="default",
    model_name="sonnet-4.5",
    raw_mode=False,
    api_name="/chat"
)

print(result)
```

### cURL

```bash
curl -X POST http://localhost:7860/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      "What is quantum computing?",
      [],
      "default",
      "sonnet-4.5",
      false
    ],
    "fn_index": 0
  }'
```

---

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...              # API key for LLM providers

# Paths
CORPUS_FILE=./data/corpus_v4.json  # JSON storage location
CHROMA_PATH=./data/chroma_db_v4    # Vector store location
CHROMA_DEVICE=cpu                  # Or "cuda" for GPU

# Memory Limits
CORPUS_MAX_ENTRIES=2000            # Max conversations stored
PROMPT_TOKEN_BUDGET=15000          # Total prompt size limit
PROMPT_MAX_RECENT=15               # Recent conversations in prompt
PROMPT_MAX_MEMS=15                 # Semantic memories in prompt
PROMPT_MAX_FACTS=30                # Facts in prompt
PROMPT_MAX_WIKI=3                  # Wikipedia articles in prompt

# Feature Toggles
USE_STM_PASS=1                     # Enable STM analyzer
ENABLE_TONE_DETECTION=1            # Crisis-aware responses
LLM_FACTS_ENABLED=1                # LLM fact extraction
REFLECTIONS_ENABLED=1              # Session-end reflections

# GUI
GRADIO_SERVER_NAME=127.0.0.1       # Bind address
GRADIO_PORT=7860                   # Web interface port
```

### config.yaml

```yaml
daemon:
  version: v4
  data_dir: ./data
  debug_mode: false

memory:
  corpus_max_entries: 2000
  prompt_max_recent: 15
  prompt_max_mems: 15
  prompt_max_facts: 30
  summary_interval: 20

models:
  default_model: sonnet-4.5
  default_max_tokens: 1850
  enable_thinking_blocks: true

gating:
  cosine_threshold: 0.15
  xenc_threshold: 0.55

features:
  use_stm_pass: true
  enable_tone_detection: true
  enable_fact_extraction: true
```

### Performance Profiles

```bash
# Fast (CPU, reduced context)
make -f Makefile.fast run

# Balanced (moderate quality)
make -f Makefile.balanced run

# Maximum quality (GPU)
make -f Makefile.max run
```

---

## Tone Detection

Daemon implements crisis-aware tone detection that adapts response depth:

| Level | Trigger Examples | Response Style |
|-------|------------------|----------------|
| **HIGH** | "I want to die", "can't go on" | Full therapeutic presence, multi-paragraph |
| **MEDIUM** | "panic attack", "falling apart" | 2-3 paragraphs, supportive |
| **CONCERN** | "really anxious", "scared" | 2-4 sentences, brief validation |
| **CONVERSATIONAL** | Status updates, technical questions | 1-3 sentences, direct |

**Key Innovation:** The system distinguishes personal crisis from world event observation:
- "I'm suffering and can't go on" → HIGH (personal crisis)
- "People are suffering due to deportations" → CONVERSATIONAL (world observation)

---

## Deployment

### Docker

**Build:**
```bash
./build-docker.sh
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  daemon-rag-agent:
    image: daemon-rag-agent:latest
    container_name: daemon-rag
    restart: unless-stopped
    ports:
      - "7860:7860"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GRADIO_SERVER_NAME=0.0.0.0
      - CHROMA_DEVICE=cpu
    volumes:
      - daemon-data:/app/data
      - daemon-logs:/app/conversation_logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7860/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

volumes:
  daemon-data:
  daemon-logs:
```

**Run:**
```bash
docker-compose up -d
docker-compose logs -f daemon-rag-agent
```

**Backup:**
```bash
docker run --rm -v daemon-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/daemon-backup-$(date +%Y%m%d).tar.gz /data
```

### Kubernetes

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: daemon-rag-agent
spec:
  replicas: 1
  selector:
    matchLabels:
      app: daemon-rag
  template:
    metadata:
      labels:
        app: daemon-rag
    spec:
      containers:
      - name: daemon-rag
        image: daemon-rag-agent:latest
        ports:
        - containerPort: 7860
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: daemon-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 7860
          initialDelaySeconds: 60
        readinessProbe:
          httpGet:
            path: /health
            port: 7860
          initialDelaySeconds: 30
```

**Deploy:**
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

### Health Check

**GET /health**

```json
{
  "status": "healthy",
  "timestamp": "2025-11-29T21:00:00Z",
  "corpus_loaded": true,
  "chroma_connected": true,
  "api_key_present": true
}
```

---

## Wikipedia Pipeline

### Sample Mode (Quick Test)

```bash
make -f Makefile.fast pipeline
# Processes ~1000 articles
```

### Full Dump Processing

```bash
python data/pipeline/unified_pipeline.py --download --semantic

# Pipeline steps:
# 1. Download Wikipedia dump (enwiki-latest-pages-articles.xml.bz2)
# 2. Parse XML with section boundaries
# 3. Chunk articles (512 tokens, 50 overlap)
# 4. Embed with sentence-transformers (all-MiniLM-L6-v2)
# 5. Build FAISS IVF index
# 6. Store metadata (title, section, position)
```

**Requirements:** ~102GB storage for 6.5M+ articles

---

## Project Structure

```
daemon-rag-agent/
├── main.py                          # Entry point
├── config.yaml                      # Configuration
├── requirements.txt                 # Dependencies
├── Dockerfile                       # Container build
├── docker-compose.yml               # Service orchestration
│
├── core/
│   ├── orchestrator.py              # Main request handler (1000+ lines)
│   ├── response_generator.py        # LLM streaming
│   ├── stm_analyzer.py              # Short-term memory analyzer
│   ├── system_prompt.txt            # Base system prompt
│   │
│   └── prompt/                      # Modular prompt system
│       ├── builder.py               # UnifiedPromptBuilder (1400+ lines)
│       ├── context_gatherer.py      # Parallel retrieval (900+ lines)
│       ├── formatter.py             # Section assembly
│       └── token_manager.py         # Budget management
│
├── memory/
│   ├── corpus_manager.py            # JSON persistence
│   ├── coordinator.py               # V2 modular orchestration
│   ├── memory_retriever.py          # Retrieval operations
│   ├── memory_storage.py            # Storage operations
│   ├── memory_scorer.py             # Ranking algorithm
│   ├── thread_manager.py            # Thread tracking
│   ├── hybrid_retriever.py          # Advanced semantic search
│   ├── fact_extractor.py            # spaCy extraction
│   ├── memory_consolidator.py       # Summary generation
│   │
│   └── storage/
│       └── multi_collection_chroma_store.py  # ChromaDB wrapper
│
├── processing/
│   └── gate_system.py               # Multi-stage filtering
│
├── models/
│   ├── model_manager.py             # Multi-provider LLM client
│   └── tokenizer_manager.py         # Token counting
│
├── knowledge/
│   ├── WikiManager.py               # Wikipedia integration
│   └── semantic_search.py           # FAISS search
│
├── utils/
│   ├── tone_detector.py             # Crisis detection
│   ├── topic_manager.py             # Topic extraction
│   ├── query_checker.py             # Query analysis
│   └── file_processor.py            # PDF/DOCX/CSV
│
├── gui/
│   ├── launch.py                    # Gradio interface
│   └── handlers.py                  # UI event handlers
│
├── data/
│   ├── corpus_v4.json               # Conversation storage
│   ├── chroma_db_v4/                # Vector embeddings
│   └── wiki/                        # Wikipedia index (102GB)
│
└── tests/                           # 65 test files
    ├── test_basic_pipeline.py
    ├── test_memory_coordinator.py
    └── ...
```

---

## Troubleshooting

### Common Issues

**"No such file or directory: corpus_v4.json"**
```bash
mkdir -p data
echo '[]' > data/corpus_v4.json
```

**"ChromaDB dimension mismatch error"**
```bash
rm -rf data/chroma_db_v4
# Restart application (will rebuild)
```

**"CUDA out of memory"**
```bash
export CHROMA_DEVICE=cpu
```

**"Wikipedia fetch timeout"**
```bash
export WIKI_TIMEOUT=5.0  # Increase timeout
# Or disable: export PROMPT_MAX_WIKI=0
```

**"Token limit exceeded"**
```bash
export PROMPT_TOKEN_BUDGET=8000
```

### Debug Mode

```bash
export LOG_LEVEL=DEBUG
python main.py

# Check logs
tail -f conversation_logs/daemon_v4_*.json
```

---

## Resource Usage

| Resource | Idle | Active Query |
|----------|------|--------------|
| CPU | ~5% | 30-50% |
| RAM | ~500MB | ~1.5GB |
| GPU VRAM | - | 2-8GB (optional) |

**Storage:**
- Corpus JSON: ~10MB (2000 conversations)
- ChromaDB: ~50MB
- Wikipedia FAISS: 102GB (optional)
- Logs: ~1MB/day

---

## Development Context

This project was built from first principles as a learning exercise, developed **before discovering LangChain or LlamaIndex**. Key learnings include:

- **Semantic Chunking:** Initial fixed-size chunking broke context across sections; rebuilt with section-aware chunking
- **Memory Evolution:** Progressed from flat storage (V1) → episodic/semantic split (V2) → 5-tier hierarchy with decay (V3)
- **Gating Optimization:** Moved from LLM-based relevance (3s, expensive) → cosine only (fast, imprecise) → multi-stage pipeline (fast + accurate)
- **Token Management:** Learned through context overflow errors; implemented middle-out compression

---

## Roadmap

**Near-term:**
- [ ] Real-time web search integration
- [ ] Multi-user support with authentication
- [ ] Enhanced monitoring (Prometheus/Grafana)

**Mid-term:**
- [ ] Multimodal support (images via CLIP, audio via Whisper)
- [ ] Fine-tuning pipeline for domain adaptation
- [ ] Distributed memory (Qdrant, Weaviate)

**Long-term:**
- [ ] Multilingual support
- [ ] Advanced reasoning (tree-of-thought)
- [ ] Plugin system for extensibility

---

## Contributing

```bash
# Clone and setup
git clone https://github.com/yourusername/daemon-rag-agent.git
cd daemon-rag-agent
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Run tests
python -m pytest -q

# Search source directories (avoid data/)
# ✅ Use: core/**/*.py, memory/**/*.py
# ❌ Avoid: **/*.py (includes 35K+ wiki files)
```

**Git Workflow:**
- `master`: Stable release
- `feature/*`: New features
- `fix/*`: Bug fixes
- `refactor/*`: Code improvements

**Commit Style:** Use conventional commits (`feat:`, `fix:`, `refactor:`, `docs:`)

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This project represents 8 months of learning, experimentation, and iteration. Special thanks to:

- The open-source ML community for tools like sentence-transformers, FAISS, and ChromaDB
- Anthropic and OpenAI for accessible LLM APIs
- The countless Stack Overflow answers that helped debug async/await patterns

---

<div align="center">

**Built with curiosity and persistence**

*From zero Python knowledge to 54,000 lines of production code*

</div>


## Contact

**Luke Halleran**
- MS Analytics Student @ Georgia Tech (Starting Jan 2026)
- Former Actuarial Analyst @ Mercer
- GitHub: [@lukehalleran](https://github.com/lukehalleran)
- LinkedIn: [luke-halleran](https://linkedin.com/in/luke-halleran-a7605088/)
- Email: lukehalleran@gmail.com

---

