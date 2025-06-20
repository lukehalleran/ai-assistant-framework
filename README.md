# AI Assistant Framework

**An Advanced Contextual AI System with Hierarchical Memory, Semantic Search, and Dynamic Knowledge Integration**

A sophisticated Retrieval-Augmented Generation (RAG) system that combines hierarchical memory management, LLM-gated context filtering, semantic search across Wikipedia, and multi-personality support for enhanced conversational AI capabilities.

## 🌟 Project Overview

This project implements an intelligent assistant framework that goes beyond simple chatbots by incorporating:

* **Hierarchical Memory System**: Multi-level memory organization with episodic, semantic, procedural, and meta-memory types
* **LLM-Gated Context Filtering**: Intelligent relevance gating using language models to filter context
* **Semantic Search**: FAISS vector indexing with memory-mapped embeddings for efficient similarity search
* **Wikipedia Knowledge Integration**: Full-scale processing and semantic chunking of Wikipedia dumps
* **Dynamic Personality System**: Switchable personalities with distinct behaviors and knowledge access patterns
* **Persistent Memory Management**: ChromaDB integration with automatic summarization and decay mechanisms
* **Multi-Model Architecture**: Support for both local models (HuggingFace) and API-based models (OpenAI)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
│                      (Gradio Web App)                           │
│  ┌────────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Text Input      │  │ File Upload  │  │ Debug Panel      │  │
│  │                 │  │ (.txt,.docx, │  │                  │  │
│  │                 │  │  .csv, .py)  │  │                  │  │
│  └────────────────┘  └──────────────┘  └──────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    Context Pipeline Layer                        │
│  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ Prompt Builder  │  │ Time Manager │  │ Topic Manager   │   │
│  │ (Hierarchical)  │  │              │  │ (spaCy NLP)     │   │
│  └────────┬────────┘  └──────────────┘  └─────────────────┘   │
│           │                                                      │
│  ┌────────▼────────────────────────────────────────────────┐   │
│  │           LLM Gating System (MultiStageGateSystem)       │   │
│  │  • Memory Filtering  • Wiki Relevance  • Chunk Selection │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    Knowledge Sources Layer                       │
│  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ Hierarchical    │  │ Semantic     │  │ Wikipedia       │   │
│  │ Memory System   │  │ Search       │  │ Manager         │   │
│  │ • Episodic      │  │ (FAISS)      │  │                 │   │
│  │ • Semantic      │  │              │  │                 │   │
│  │ • Procedural    │  └──────────────┘  └─────────────────┘   │
│  │ • Summary       │                                            │
│  │ • Meta          │  ┌────────────────────────────────────┐   │
│  └─────────────────┘  │        ChromaDB Storage           │   │
│                       │  (Persistent Vector Database)      │   │
│                       └────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                      Model Layer                                 │
│  ┌─────────────────────┐  ┌─────────────────────────────────┐  │
│  │   Local Models      │  │      API Models                 │  │
│  │  • GPT-Neo 1.3B     │  │  • GPT-4 Turbo                 │  │
│  │  • Custom Models    │  │  • GPT-3.5 Turbo               │  │
│  │  (HuggingFace)      │  │  (OpenAI)                      │  │
│  └─────────────────────┘  └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Key Features

### 1. **Hierarchical Memory Management**
- **Multi-Type Memory System**: Episodic, semantic, procedural, summary, and meta memories
- **Automatic Consolidation**: Summarizes every 20 interactions
- **Temporal Decay**: Importance-weighted memory retrieval with time decay
- **Parent-Child Relationships**: Maintains memory hierarchies for context expansion

### 2. **LLM-Gated Context Filtering**
- **Intelligent Relevance Assessment**: Uses fast LLM to evaluate context relevance
- **Multi-Stage Filtering**: Title filtering → Content filtering → Final selection
- **Parallel Processing**: Asynchronous batch processing for efficiency
- **Configurable Thresholds**: Adjustable confidence levels for filtering

### 3. **Advanced Semantic Search**
- **FAISS Integration**: IVF (Inverted File) indexing for scalability
- **Memory-Mapped Embeddings**: Efficient handling of large vector datasets
- **Metadata Preservation**: Maintains rich metadata for each embedded chunk
- **Production-Ready Indexing**: Support for IVF, HNSW, and IVF-PQ index types

### 4. **Wikipedia Knowledge Processing**
- **Streaming XML Processing**: Handles full Wikipedia dumps (50GB+)
- **Semantic Chunking**: Intelligent text segmentation preserving context
- **Section-Aware Processing**: Maintains Wikipedia article structure
- **Multi-Threaded Extraction**: Parallel processing for speed

### 5. **Dynamic Personality System**
- **Multiple Personalities**: Default, therapy, and snarky modes
- **Configurable Behaviors**: Different memory access, wiki usage, and response styles
- **Hot-Swappable**: Change personalities at runtime via UI

### 6. **Security & Safety Features**
- **Injection Protection**: Multi-language pattern detection for prompt injection
- **Content Sanitization**: Automatic redaction of suspicious content
- **Harm Prevention**: Safety checks and content filtering
- **Audit Logging**: Tracks flagged content and security events

## 💻 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for local models)
- 50GB+ disk space (for full Wikipedia processing)
- 16GB+ RAM (32GB recommended for large-scale processing)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-assistant-framework
cd ai-assistant-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm

# Set up environment variables
export OPENAI_API_KEY="your-api-key-here"
```

### Process Wikipedia Data (Optional)

```bash
# Full Wikipedia processing with semantic chunking
python unified_pipeline.py --download --semantic

# Test mode with limited articles
python unified_pipeline.py --extracted sample_data.xml --test --semantic --max-articles 100

# Custom chunking parameters
python unified_pipeline.py --download --semantic --chunk-size 1000 --chunk-overlap 200
```

## 🎯 Usage

### Web Interface

```bash
python gui.py
```

Access the Gradio interface at `http://localhost:7860`

**Features:**
- Text input with multi-line support
- File upload (`.txt`, `.docx`, `.csv`, `.py`)
- Personality switching
- Debug panel showing:
  - Topics detected
  - Wikipedia content retrieved
  - Semantic search results
  - Memory retrieval details
  - Hierarchical memory statistics

### Command Line Interface

```bash
python runtime.py
```

Interactive REPL with full feature access.

### API Usage

```python
from tokenizer_manager import TokenizerManager
from prompt_builder import PromptBuilder
from hierarchical_memory import HierarchicalMemorySystem
from models import model_manager

# Initialize components
tokenizer_manager = TokenizerManager()
prompt_builder = PromptBuilder(tokenizer_manager)
memory_system = HierarchicalMemorySystem(model_manager)

# Store an interaction
memory_id = await memory_system.store_interaction(
    query="What is machine learning?",
    response="Machine learning is...",
    tags=["AI", "education"]
)

# Retrieve relevant memories
memories = await memory_system.retrieve_relevant_memories(
    "Tell me about neural networks",
    max_memories=10
)
```

## ⚙️ Configuration

### Core Settings (`config.py`)

```python
# Model Configuration
DEFAULT_MODEL_NAME = "gpt-4-turbo"  # or "gpt-neo" for local
API_MODEL_CONTEXT_LIMIT = 128000
LOCAL_MODEL_CONTEXT_LIMIT = 4096

# Memory Configuration
CORPUS_FILE = "data/corpus_v4.json"
CHROMA_PATH = "./chroma_db_v4"

# Search Configuration
CONFIDENCE_THRESHOLD = 1.5
DEFAULT_TOP_K = 5

# Hierarchical Memory
GATE_REL_THRESHOLD = 0.6
MAX_FINAL_MEMORIES = 15
MEM_IMPORTANCE_SCORE = 0.7
```

### Personality Configuration

Create custom personalities by adding to `personality_manager.py`:

```python
"custom": {
    "system_prompt_file": "system_prompt_custom.txt",
    "directives_file": "structured_directives_custom.txt",
    "num_memories": 10,
    "include_wiki": True,
    "include_semantic_search": True
}
```

## 🔧 Technical Implementation

### Memory Architecture

The hierarchical memory system implements a sophisticated multi-level approach:

1. **Memory Types**:
   - **Episodic**: Individual user-assistant interactions
   - **Semantic**: Extracted facts and knowledge from conversations
   - **Procedural**: How-to knowledge and patterns
   - **Summary**: Compressed episodic memories (every 20 interactions)
   - **Meta**: Memories about memory patterns and system behavior

2. **Storage & Retrieval**:
   - ChromaDB for vector similarity search
   - JSON-based corpus with atomic writes
   - Parent-child relationships for context expansion
   - Temporal decay and importance weighting

### Semantic Search Implementation

```python
# Vector index types supported
- IVF (Inverted File): Default, good balance of speed/accuracy
- HNSW (Hierarchical Navigable Small World): Better recall, more memory
- IVF-PQ (Product Quantization): Compressed, for very large datasets
```

### LLM Gating Pipeline

1. **Quick Semantic Search**: Get top-k candidates from FAISS
2. **Title Relevance Check**: Fast yes/no filtering on titles
3. **Content Relevance Scoring**: Detailed relevance assessment
4. **Hierarchical Expansion**: Add parent/child memories if relevant
5. **Final Scoring**: Apply decay, importance, and access recency

### Wikipedia Processing Pipeline

1. **Streaming XML Parser**: Uses lxml for memory-efficient processing
2. **Semantic Chunking**: 
   - Preserves section boundaries
   - Target chunk size: 1000 tokens
   - Overlap: 200 tokens
   - Minimum chunk size: 100 tokens
3. **Parallel Embedding**: Multi-threaded with configurable batch sizes
4. **Parquet Storage**: Efficient columnar format with metadata

## 📊 Performance Considerations

### Memory Usage
- **Base System**: ~2GB RAM
- **With Local Model**: ~4-8GB RAM (model-dependent)
- **Full Wikipedia Index**: ~15GB RAM for FAISS index
- **Memory-Mapped Embeddings**: Reduces RAM usage significantly

### Processing Speed
- **Wikipedia Extraction**: ~1000 articles/minute (8 threads)
- **Embedding Generation**: ~50-100 chunks/second (GPU)
- **Query Response**: <2 seconds typical (with all features)
- **Memory Retrieval**: <100ms for hierarchical search

### Optimization Tips
1. Use GPU for embedding generation (`CUDA` device)
2. Adjust `NUM_CONSUMER_THREADS` based on CPU cores
3. Use IVF index for datasets >1M vectors
4. Enable memory mapping for large embedding files
5. Configure batch sizes based on available VRAM

## 🛡️ Security Features

### Injection Protection
- Multi-language pattern detection (7 languages)
- Automatic content sanitization
- Suspicious content logging
- Real-time threat detection

### Privacy Considerations
- All data stored locally (except API calls)
- No telemetry or external tracking
- Configurable content filtering
- Audit logs for security events

## 🧪 Testing

### Run Tests
```bash
# Unit tests
python -m pytest tests/

# Integration tests
python -m pytest tests/integration/

# Test with sample data
python unified_pipeline.py --extracted test_data.xml --test --max-articles 10
```

### Debug Mode
Enable detailed logging in `config.py`:
```python
DEBUG_MODE = True
```

## 📚 Project Structure

```
ai-assistant-framework/
├── core/
│   ├── config.py                 # Configuration settings
│   ├── models.py                 # Model management and inference
│   ├── ModelManager.py           # Model abstraction layer
│   └── tokenizer_manager.py      # Tokenizer management
├── memory/
│   ├── memory.py                 # Basic memory management
│   ├── hierarchical_memory.py    # Hierarchical memory system
│   ├── memory_integration.py     # Memory system integration
│   └── persistence.py            # ChromaDB persistence
├── context/
│   ├── prompt_builder.py         # Prompt construction
│   ├── llm_gates.py             # LLM-based context filtering
│   ├── topic_manager.py         # Topic extraction and tracking
│   └── time_manager.py          # Temporal context management
├── knowledge/
│   ├── WikiManager.py           # Wikipedia integration
│   ├── semantic_chunker.py      # Semantic text chunking
│   ├── search_faiss_with_metadata.py  # FAISS search
│   └── unified_pipeline.py      # Wikipedia processing pipeline
├── interface/
│   ├── gui.py                   # Gradio web interface
│   └── runtime.py               # CLI interface
├── processing/
│   ├── extract_wikipedia_articles.py  # Wikipedia extraction
│   ├── embed_wiki_chunks_to_parquet.py # Embedding generation
│   └── Injection_Protection_Feed_to_Embeder.py # Security
├── personality/
│   └── personality_manager.py    # Personality system
├── data/                        # Data storage
├── chroma_db/                   # Vector database
├── wiki_chunks/                 # Processed Wikipedia chunks
├── semantic_chunks/             # Semantic chunks
└── embedded_parquet/            # Embedded vectors
```

## 🤝 Contributing

While this project is part of my portfolio, I welcome feedback and suggestions:

1. **Report Issues**: Use GitHub Issues for bugs or feature requests
2. **Suggest Improvements**: Open a discussion for ideas
3. **Code Reviews**: Comments on implementation are appreciated

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Wikipedia** for providing open knowledge dumps
- **HuggingFace** for transformer models and libraries
- **OpenAI** for GPT API access
- **Meta AI** for FAISS vector search library
- **Anthropic** for research on AI safety and alignment
- **ChromaDB** for vector database functionality
- **Gradio** for the web interface framework

## 📈 Future Enhancements

- [ ] Multi-modal support (images, audio)
- [ ] Distributed processing for larger datasets
- [ ] Real-time knowledge updates
- [ ] Custom model fine-tuning interface
- [ ] Advanced visualization of memory hierarchies
- [ ] Plugin system for extending functionality
- [ ] Export/import memory snapshots

## 📞 Contact

**Luke Halleran**
- GitHub: [@lukehalleran](https://github.com/lukehalleran)
- Email: lukehalleran@gmail.com
- LinkedIn: [Your LinkedIn]

---

**This project demonstrates advanced competencies in:**
- *Natural Language Processing & Information Retrieval*
- *Distributed Systems & Parallel Processing*
- *Machine Learning Engineering & Model Deployment*
- *System Architecture & Design Patterns*
- *Python Programming & Software Engineering*
- *AI Safety & Security Considerations*
