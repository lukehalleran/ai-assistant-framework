
# Daemon AI Assistant Framework

**An advanced Retrieval-Augmented Generation (RAG) system with hierarchical memory, cosine similarity gating, and dynamic personality management**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

**Daemon** is a production-ready conversational AI system that enhances language model responses through intelligent context retrieval, hierarchical memory management, and efficient cosine similarity-based gating. Built with a modular architecture, it integrates multiple knowledge sources while maintaining continuity across conversations and sessions.

---

##  Why Daemon over LangChain or Haystack?

Unlike higher-abstraction frameworks, Daemon offers:
- **Direct access to memory internals** (episodic, semantic, procedural, summary)
- **Custom gating logic** with cosine filtering and optional reranking
- **Persistent, modifiable memory across sessions** — suitable for long-term assistants
- **Minimal dependencies** for faster dev cycles and transparent behavior

Daemon trades some plug-and-play ease for **extensibility, control, and interpretability** — ideal for researchers and tinkerers building truly agentic systems.

---

## Key Features

### Intelligent Memory Management

- **Hierarchical Architecture**:
  - *Episodic*: Timestamped conversation turns
  - *Semantic*: Extracted facts and claims
  - *Procedural*: How-to tasks and step-by-step knowledge
  - *Summary*: Long-term memory compression
  - *Meta*: Memories about memory itself

- **Consolidation** every 20 interactions to prevent overflow
- **Temporal Decay** with relevance-weighted retrieval
- **Persistent Across Sessions** — memory lives forever

---

###  Efficient Cosine Similarity Gating

- Multi-stage pipeline:
  1. FAISS semantic search (top 50)
  2. Cosine filter (threshold = 0.65)
  3. Hierarchical expansion
  4. (Optional) Cross-encoder reranking

- Batch processing with NumPy
- Special handling for meta/system queries
- Graceful fallback when no high-similarity results found

---

###  Advanced Knowledge Integration

- **Wikipedia API** integration with summarization
- **Semantic search** over embedded documents
- **Topic extraction** using SpaCy entity recognition
- **Multi-source fusion** in the prompt builder

---

###  Dynamic Personality System

- Hot-swappable personalities define:
  - Prompt + directive files
  - Memory access patterns
  - Tone, verbosity, and source preferences

Built-ins:
- `Default`: Balanced helper
- `Therapy`: Long memory span, empathetic
- `Snarky`: Minimal memory, witty tone

---

###  Streaming Response Generation

- `async/await` support
- Token-by-token streaming output
- Local or API-based model backends
- Graceful recovery from stream failures

---

###  Production-Ready Tooling

- Full logging (DEBUG, INFO, WARNING)
- Execution time decorators
- Prompt truncation and token budget awareness
- File parsing: TXT, DOCX, CSV, Python
- Time awareness: Session timestamps and response duration

---

##  Dreams and Truth Evaluation (Experimental)

Daemon includes a unique **"dreaming" system**:
- Memories are recombined and reweighted during idle cycles to form *symbolic insight paths* (dreams)
- Each memory is scored with a **truth scalar** based on post-convo evaluation
- Inaccurate memories decay or mutate; consistent ones stabilize
- Enables long-term *adaptive learning without full RL*

This system helps avoid AI "hallucination lock-in" by reevaluating ideas over time.

---

## Interface Preview (Coming Soon)

<!-- Insert a GIF or screenshot here -->
> Screenshot of Gradio interface showing memory selection, personalities, and live chat stream.

---

##  Architecture Diagram
graph TD
    A[User Interface<br/>Gradio Web App] --> B[Core Orchestrator]
    B --> B1[Personality Manager]
    B --> B2[Response Generator]
    B --> B3[File Processor]
    
    B --> C[Prompt Building Pipeline]
    C --> C1[Unified Hierarchical]
    C --> C2[Gated Builder]
    C --> C3[Topic Manager]
    
    C --> D[Memory Systems]
    D --> D1[Memory Coordinator]
    D --> D2[Hierarchical Memory]
    D --> D3[Corpus Manager]
    
    D --> E[Knowledge & Storage]
    E --> E1[Multi-Collection ChromaDB]
    E --> E2[FAISS Search]
    E --> E3[Wikipedia API]
    
    E --> F[Gating System]
    F --> F1[Cosine Similarity Filter]
    F --> F2[Cross-Encoder Rerank]
##  Configuration

Edit values in `config/config.py`:

```python
GATE_REL_THRESHOLD = 0.65
MAX_FINAL_MEMORIES = 5
DEFAULT_MAX_TOKENS = 2048
CORPUS_FILE = "data/corpus.json"
CHROMA_PATH = "data/chroma_db"
```

## **Project Structure**
```
daemon-ai-assistant/
├── core/               # Core orchestrator and stream gen
├── memory/             # Episodic, semantic, procedural, etc.
├── knowledge/          # Wikipedia + semantic chunks
├── processing/         # Gating and filter systems
├── personality/        # Personalities + directives
├── models/             # LLM manager
├── gui/                # Gradio frontend
├── utils/              # Logging, timing, loaders
├── config/             # Constants and paths
└── tests/              # Unit and integration tests
```

##  Performance Benchmarks

| Task                 | Latency         |
|----------------------|-----------------|
| Memory Retrieval     | <100ms (10k)    |
| Cosine Gating        | ~50ms (50 items)|
| Wikipedia Search     | ~500ms          |
| Response Gen (GPT-4) | ~20–30 tok/s    |
| Memory Usage         | ~2GB + model    |

---

## 🧬 Advanced Usage

### Add New Personality

```python
"custom": {
    "system_prompt_file": "system_prompt_custom.txt",
    "directives_file": "directives_custom.txt",
    "num_memories": 10,
    "include_wiki": True,
    "include_semantic_search": True
}
```

### Query Memory System

```python
memories = await memory_system.hierarchical_memory.retrieve_relevant_memories(
    query="how does cosine similarity work?", max_memories=10
)

results = memory_system.chroma_store.search_conversations("cosine", n_results=5)
```

### Adjust Gating Sensitivity

```python
gate_system = MultiStageGateSystem(model_manager, cosine_threshold=0.70)
```

## FAQ

**Q: Can I run this fully locally?**  
A: Yes — you can use local LLMs (e.g., Mistral, GPT-Neo) and offline embeddings.

**Q: Is there a GUI?**  
A: Yes! Gradio-based web UI included. CLI mode also supported.

**Q: Does it work with open-weight models?**  
A: Yes. Local support works with models like LLaMA, Mixtral, Mistral, etc.

## Getting Started

### Prerequisites

- Python 3.8+
- 16GB+ RAM (recommended)
- 50GB+ disk space
- CUDA GPU (optional for speed)

### Installation (not yet tested outside Fedora 42)

```bash
git clone https://github.com/yourusername/daemon-ai-assistant
cd daemon-ai-assistant
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
export OPENAI_API_KEY="your-key-here"  # optional
```

### Quick Start

```bash
# Start web interface
python main.py

# CLI mode
python main.py cli
```

## Contributing

1. Fork this repo  
2. Create a feature branch  
3. Add relevant tests  
4. Open a pull request

## License

MIT License. See LICENSE for details.

## Acknowledgements

- Sentence Transformers
- ChromaDB
- FAISS
- Gradio
- spaCy
