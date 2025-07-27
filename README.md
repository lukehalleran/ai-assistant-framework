# Near Production-Grade Conversational AI with Biologically-Inspired Memory

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/framework-custom-orange.svg)]()
[![GPU](https://img.shields.io/badge/gpu-RTX_3060-green.svg)]()

> Built from scratch over four months as my first Python project, demonstrating a deep understanding of AI systems architecture, memory management, and software engineering best practices.

---

## Overview

This project is an advanced conversational AI framework featuring a **hierarchical memory architecture** inspired by human cognition, efficient semantic retrieval, and adaptive learning mechanisms. Built independently of high-level frameworks, it demonstrates how precise control over AI memory can shift from a stateless to a stateful architecture.

A planned experimental "dream" system aims to locate undiscovered semantic connections ("insights") through stochastic recombination of conversational memories and external knowledge sources, combined with post processing "truth score" evaluation and modification, approximating lightweight reinforcement learning.

---

## Quick Start

```bash
git clone https://github.com/yourusername/daemon-ai-assistant
cd daemon-ai-assistant
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY="your-key"
python -m spacy download en_core_web_sm

python main.py  # Web UI at http://localhost:7860
```

---

## Layout 
| Component       | Technology                           | Purpose                                     |
| --------------- | ------------------------------------ | ------------------------------------------- |
| Orchestrator    | Custom Async Python                  | Manages the end-to-end query lifecycle      |
| Memory System   | ChromaDB + Hierarchical Logic        | Persistent, multi-layered memory            |
| Vector Search   | FAISS                                | High-speed semantic retrieval               |
| Embedding Model | all-MiniLM-L6-v2                     | Text vectorization                          |
| Gating System   | Cosine Similarity + Cross-Encoder    | Filters irrelevant context before prompting |
| LLM Backend     | gpt-4-turbo (or any local/API model) | Response generation                         |
| User Interface  | Gradio                               | Web-based chat interface                    |




## Key Features

* **Hierarchical Memory**: Episodic, Semantic, Procedural, and Meta-memory
* **Real-time Semantic Retrieval**: Sub-100ms latency for 10,000+ items
* **Dream Generation (Experimental)**: Analogous to human REM sleep
* **Near Production-Ready**: Logging, async streaming, robust error handling, and atomic persistence

---

## Technical Architecture

### Design Philosophy

Custom-built to ensure:

* Transparency in memory management
* Debugging ease
* Experimentation flexibility
* Inspired by exisiting biological systems

### Core Components

**Hierarchical Memory System**
- Episodic Memory (ChromaDB + FAISS)
- Semantic Memory (Cross-encoder reranking)
- Procedural Memory (Pattern extraction)
- Meta-Memory (Self-reflection capabilities)

**Multi-Stage Gating Pipeline**
1. FAISS Semantic Search (Top-50 candidates)
2. Cosine Similarity Filtering (θ = 0.65)
3. Hierarchical Memory Expansion
4. Cross-Encoder Reranking (MS-MARCO MiniLM)

### Technical Highlights

* **Temporal Decay (Memory Forgetting)**

```python
decay_factor = 1.0 / (1.0 + memory.decay_rate * age_days)
final_score = relevance * importance * decay_factor * access_boost
```

* **Adaptive Truth Scoring**
  * Each memory maintains a "truth scalar" based on consistency
  * Contradictory information triggers re-evaluation
  * Continuous validation to prevent hallucination

* **Batch Processing**

  * NumPy optimizations
  * Asynchronous parallel retrieval
  * LRU caching

---

## Storage & Performance

| Component          | Technology           | Purpose          | Performance          |
| ------------------ | -------------------- | ---------------- | -------------------- |
| Vector Store       | FAISS IVF Index      | Semantic search  | \~50ms @ 50k vectors |
| Persistent Storage | ChromaDB             | Long-term memory | \~100ms retrieval    |
| Embedding Cache    | Memory-mapped arrays | Fast lookups     | <10ms access         |
| Session State      | Atomic JSON          | Crash recovery   | Instant              |

---

## Advanced Features

### Context Fusion

```python
context_sources = {
    "recent_conversations": weight=0.4,
    "semantic_memories": weight=0.3,
    "knowledge_base": weight=0.2,
    "summaries": weight=0.1
}
```

### Dynamic Personalities

* Modifiable modules adjusting behavior dynamically

### Semantic Chunk Optimization

* Structured, hierarchical Wikipedia embeddings tested on entire database

---

## Performance Metrics

| Operation          | Latency     | Throughput    | Resource Usage |
| ------------------ | ----------- | ------------- | -------------- |
| Memory Retrieval   | <100ms      | 1000 qps      | 2GB RAM        |
| Context Assembly   | \~200ms     | 500 qps       | 4GB RAM        |
| Response Streaming | 20-30 tok/s | -             | GPU-dependent  |
| Consolidation      | \~500ms     | Every 20 msgs | Minimal        |

---

## Engineering Quality

* **SOLID Principles**
* **Design Patterns**
* **Comprehensive Testing**
* **Robustness**

---

## Technical Journey

* **Month 1**: Python basics, API integration
* **Month 2**: Modular refactoring
* **Month 3**: Memory management optimization
* **Month 4**: Approaching production readiness, full testing, second refactor

### Technologies Learned

* **AI/ML**: Transformers, Sentence-BERT, FAISS
* **Databases**: ChromaDB, atomic storage
* **Backend**: AsyncIO, concurrency
* **DevOps**: Logging, monitoring, performance testing

---

## Roadmap

### Future Research

* Lightweight reinforcement learning
* Multimodal integration
* Distributed/federated memory

### Engineering Improvements

* Rust extensions
* GraphQL querying
* Kubernetes scaling
* Analytics dashboard

---

## Practical Applications

* **Personal AI Assistant**
* **Research Tool**
* **Adaptive Educational Assistant**
* **Mental Health Companion**

**Deployment Metrics**

* Operational 24/7
* 10k+ conversations processed
* 1M+ articles indexed

---

## Impact & Value

* Demonstrates ability to deliver robust, scalable AI systems
* Highlights self-directed learning and technical problem-solving
* Offers novel research contributions

---



## License & Contact

MIT License – Free for academic and commercial use.

> "The best way to predict the future is to build it."
