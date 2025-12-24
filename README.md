# Daemon: Memory-Augmented AI Assistant

A near production-grade conversational AI system with hierarchical memory, semantic search, and Wikipedia-scale knowledge retrieval - **built from first principles in 8 months**.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GPU](https://img.shields.io/badge/gpu-RTX_3060-green.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Screenshots 
<img width="1919" height="1071" alt="2025-07-28_20-48" src="https://github.com/user-attachments/assets/eac216c2-b9be-4e5b-a799-cc0a41d80266" />

This is assistant respoding to me on a fresh start up, demonstrating tone and memory peristence
## Project Architect 

**Context**:  Former actuarial analyst (Mercer, 2022-2024) transitioning to MS Analytics at Georgia Tech. Started learning LLM Assisted Python Development in March 2025, built this system by July 2025.

**Why This Matters**: This project demonstrates the ability to rapidly acquire complex technical skills and apply them to solve real problems. Rather than following tutorials, I architected a complete RAG system before discovering frameworks like LangChain existed - proving first-principles thinking and independent problem-solving ability.

## Technical Achievement

### What I Built
- **Full RAG Pipeline**: Query → Memory Retrieval → Context Building → LLM Generation → Memory Storage
- **Wikipedia-Scale Search**: Processed entire Wikipedia dump (6.5M+ articles), achieving sub-200ms retrieval
- **Hierarchical Memory System**: Episodic, semantic, procedural, and meta-memory layers with automatic consolidation
- **Production Infrastructure**: Async streaming, atomic persistence, comprehensive logging, error recovery

### Major Iterations

#### Semantic Chunking Discovery
- Started with simple 500-character splits (worked but suboptimal)  
- Retrieval quality was poor despite having all the data  
- Rebuilt pipeline to respect semantic boundaries — massive improvement  
- **Lesson:** How you chunk matters more than how much you store  

#### Memory Architecture Evolution
- **V1:** Flat storage in JSON (worked for <100 conversations)  
- **V2:** ChromaDB for persistence (better but no relationships)  
- **V3:** Hierarchical system, made descison to forgo parent-child relationships (current)  
- Added temporal decay after realizing not all memories are equally important  

#### Performance Optimization Journey
- **Initial:** LLM-based gating (2–3 seconds per query)  
- **Optimized:** Cosine similarity pre-filtering (~200 ms)  
- **Final:** Multi-stage pipeline with cross-encoder (best of both worlds)  
```
text
User Query
    ↓
┌─────────────────────────────────────────┐
│         ORCHESTRATOR                    │
│  • Topic extraction (spaCy NER)         │
│  • Query routing & planning              │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│      MEMORY COORDINATOR                  │
│  • ChromaDB collections (5 types)        │
│  • Hierarchical memory trees              │
│  • Temporal decay & importance scoring    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│      MULTI-STAGE GATE SYSTEM             │
│  1. FAISS semantic search (top 50)       │
│  2. Cosine similarity filter (θ=0.45)    │
│  3. Cross-encoder reranking (MS-MARCO)   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│      PROMPT BUILDER                      │
│  • Token budget management (4096)        │
│  • Context prioritization                 │
│  • Dynamic personality injection          │
└─────────────────────────────────────────┘
    ↓
Response Generation (GPT-4 / Claude / Local Models Once RTX 4090 Acquired)
```

## Key Innovations

### 1. Memory Architecture
```python
# Actual code from the project
class HierarchicalMemorySystem:
    def __init__(self):
        self.memories = {}  # id -> MemoryNode
        self.hierarchy = defaultdict(list)  # parent -> children
        self.type_index = defaultdict(list)  # MemoryType -> ids
        
    async def retrieve_relevant_memories(self, query, max_memories=10):
        # 1. Search all ChromaDB collections
        # 2. Apply temporal decay: 1.0 / (1.0 + decay_rate * age_days)
        # 3. Boost by importance and access recency
        # 4. Return top memories after scoring
```

**Why This Matters**: Most RAG systems treat memory as flat storage. This implements cognitive-inspired hierarchical organization with parent-child relationships and automatic summarization.

### 2. Iterative Pipeline Development
- **Version 1**: Successfully processed entire Wikipedia dump (~6.5M articles) with basic chunking
- **Problem Discovered**: Basic 500-char chunks lost semantic coherence and article structure
- **Version 2**: Built semantic chunking system that:
  - Preserves Wikipedia section hierarchy
  - Maintains semantic boundaries (paragraphs, sections)
  - Implements sliding window overlap for context preservation
  - Streams 50GB+ XML without memory overflow

```python
# Evolution from basic to semantic chunking
# V1: Simple splitting (worked but suboptimal)
chunks = text[i:i+500] for i in range(0, len(text), 500)

# V2: Semantic-aware chunking (current)
sections = parse_wiki_sections(article)
for section in sections:
    chunks = semantic_chunk_with_overlap(
        section, 
        chunk_size=1000,
        overlap=200,
        respect_boundaries=True
    )
```

**Learning Moment**: Discovered through testing that retrieval quality depends more on chunk semantics than raw coverage. Rebuilt the entire pipeline to fix this.

### 3. Intelligent Gating System
Instead of passing all retrieved context to the LLM (expensive and noisy), I implemented a multi-stage filter:
```python
Stage 1: FAISS retrieves 50 candidates (broad net)
Stage 2: Cosine similarity filters to ~20 (relevance threshold)
Stage 3: Cross-encoder reranks to final 10 (semantic understanding)
Stage 4: LLM evalaues both query and final 10 to determine relenvt data to pass to final LLM call (planned)
```

## Performance Metrics (Verified with Comprehensive Logging)

| Operation | Latency | Scale/Notes |
|-----------|---------|-------------|
| Memory Retrieval | ~100-200ms | ChromaDB + FAISS hybrid search |
| Context Building | ~200ms | Multi-stage filtering pipeline |
| Response Generation | 2-3s | Streaming to UI with GPT-4 |
| Memory Consolidation | ~500ms | Automatic every 20 conversations |
| Wikipedia Processing | ~1000 articles/min | Full dump processed (6.5M articles) |
| Semantic Chunking | ~50 articles/sec | With overlap and structure preservation |

## Code Quality & Engineering

### Testing Coverage (Comphensive)
- Unit tests for all core components 
- Integration tests for full pipeline
- Memory leak detection and monitoring
- Async operation verification

### Production Features
- **Atomic Operations**: Prevents data corruption during crashes
- **Graceful Degradation**: Falls back when services unavailable  
- **Comprehensive Logging**: Every operation tracked for debugging
- **Resource Management**: Automatic GPU memory clearing, connection pooling

## Real-World Application

This system has been actively developed and tested:
- **Wikipedia Processing**: Successfully processed entire Wikipedia dump (6.5M+ articles)
- **Iterative Improvement**: Rebuilt chunking pipeline after discovering semantic issues
- **Active Development**: Running on RTX 3060, handling real conversations
- **Multiple Personalities**: Configurable modes (default, therapy, analytical) which access memories differently, and can be given access to different data sources
- **File Support**: Processes PDF, DOCX, CSV inputs

**Current Status**: Preparing to re-process Wikipedia with improved semantic chunking for better retrieval quality.

## What This Demonstrates

### For Technical Roles
- **System Design**: Clean separation of concerns, modular architecture
- **ML Engineering**: Practical implementation of embeddings, vector search, RAG
- **Problem Solving**: Novel solutions like hierarchical memory and dream generation
- **Learning Velocity**: 0 to near production system in 4 months

### For Recruiters
- **Self-Directed Learning**: No formal CS background, learned through building
- **First-Principles Thinking**: Solved problems before discovering existing solutions
- **Production Mindset**: Built for reliability, not just functionality
- **Documentation**: Clear code, comprehensive logging, maintainable structure

## Technical Deep Dive

### Memory Types & Processing
```python
MemoryType.EPISODIC    # Raw conversations
MemoryType.SEMANTIC    # Extracted facts
MemoryType.PROCEDURAL  # Learned patterns
MemoryType.SUMMARY     # Compressed episodes
MemoryType.META        # Patterns about patterns
```

### Token Budget Management
The system intelligently allocates limited context window (4096 tokens):
1. Recent conversation: 40% 
2. Relevant memories: 30%
3. Wikipedia context: 20%
4. Summaries: 10%

## Future Enhancements (Roadmap)

- [ ] Multimodal support (images, audio)
- [ ] Real-time web search integration  
- [ ] Distributed processing across multiple GPUs
- [ ] Fine-tuning pipeline for domain adaptation
- [ ] Reinforcement learning from conversation feedback via post processing modfication of "truth scalar" meta data

## Installation & Usage

```bash
# Clone and setup
git clone https://github.com/lukehalleran/daemon-ai
cd daemon-ai
pip install -r requirements.txt

# For full Wikipedia (optional, 50GB download)
python unified_pipeline.py --download --semantic

# Launch web interface
python main.py
# Access at http://localhost:7860
```


## Contact

**Luke Halleran**
- MS Analytics Student @ Georgia Tech (Starting Jan 2026)
- Former Actuarial Analyst @ Mercer
- GitHub: [@lukehalleran](https://github.com/lukehalleran)
- LinkedIn: [luke-halleran](https://linkedin.com/in/luke-halleran-a7605088/)
- Email: lukehalleran@gmail.com

---

