# AI Assistant Framework

**A sophisticated Retrieval-Augmented Generation (RAG) system with semantic search, Wikipedia integration, and intelligent memory management**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

This AI Assistant Framework is a production-ready conversational AI system that enhances language model responses through intelligent context retrieval and memory management. Built from scratch over 4 months, it demonstrates advanced software architecture and machine learning engineering principles.

### Key Achievements
- **Semantic Search Engine**: Custom-built vector search using FAISS with sub-second query times
- **Memory System**: Dual-layer memory architecture with automatic summarization for long-term retention
- **Scalable Data Pipeline**: Processes 50GB+ Wikipedia dumps with streaming XML parsing
- **Multi-Model Support**: Seamless switching between local (HuggingFace) and cloud (OpenAI) models
- **Security-First Design**: Built-in prompt injection protection and content filtering

##  Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                       │
│                   (Gradio Web App)                      │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                    Core Pipeline                        │
│ ┌─────────────┐ ┌──────────────┐ ┌───────────────┐    │
│ │Topic Manager│ │Prompt Builder│ │ Time Manager  │     │
│ └─────────────┘ └──────────────┘ └───────────────┘    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                 Knowledge Sources                       │
│ ┌─────────────┐ ┌──────────────┐ ┌───────────────┐    │
│ │Wiki Manager │ │FAISS Search  │ │ Memory Store  │     │
│ │(Wikipedia)  │ │(Semantic)    │ │ (ChromaDB)    │     │
│ └─────────────┘ └──────────────┘ └───────────────┘    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                    Model Layer                          │
│ ┌─────────────────────┐ ┌─────────────────────────┐    │
│ │   Local Models      │ │      API Models       │     │
│ │  (HuggingFace)     │ │    (OpenAI GPT-4)    │     │
│ └─────────────────────┘ └─────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

##  Features

###  Intelligent Information Retrieval
- **Multi-Stage Filtering Pipeline**: 
  - Initial broad search (n=100) to cast a wide net
  - Cosine similarity threshold filtering to remove low-relevance results
  - Semantic search on filtered candidates for intent matching
  - Final reranking done by call to local LLM model to optimize context order
- **Semantic Search**: Uses sentence transformers to understand query intent, not just keywords
- **Dynamic Context Building**: Automatically selects the most relevant information for each query
- **Wikipedia Integration**: Access to millions of articles with intelligent chunking and indexing
- **Section-Aware Processing**: Maintains Wikipedia article structure during chunking
- **Temporal Awareness**: Built-in time and date handling for context-aware responses

###  Advanced Memory Management
- **Short-term Memory**: Maintains conversation context for coherent multi-turn interactions
- **Long-term Memory**: Automatically summarizes conversations every 20 interactions
- **Vector Storage**: ChromaDB integration for persistent, searchable memory
- **Memory Prioritization**: Intelligent selection of which memories to retain based on relevance
- **Cross-Session Persistence**: Memories survive between sessions for continuity

###  Performance & Scalability
- **Streaming Processing**: Handles 50GB+ datasets without memory overflow
- **Multi-threaded Extraction**: Parallel processing for faster indexing
- **Memory-mapped Embeddings**: Efficient loading of large vector databases
- **IVF Indexing**: Scales to millions of documents while maintaining fast search
- **Batch Processing**: Optimized embedding generation for large document sets

### 🛡 Security & Reliability
- **Prompt Injection Protection**: Multi-layer input sanitization to prevent manipulation
- **Multi-Language Pattern Detection**: Security filters work across languages to catch injection attempts
- **Content Sanitization**: Automatic redaction of suspicious content
- **Harm Prevention**: Safety checks and content filtering for user protection
- **Atomic Operations**: Ensures data integrity during writes
- **Comprehensive Logging**: Detailed activity logs for debugging and monitoring
- **Audit Logging**: Tracks flagged content and security events for review
- **Error Recovery**: Graceful handling of API failures and system errors

###  Dynamic Personality System
- **Multiple Personalities**: Built-in personalities including default, therapy, and snarky modes
- **Configurable Behaviors**: Each personality has different:
  - Memory access patterns
  - Wikipedia usage preferences
  - Response styles and tones
- **Hot-Swappable**: Change personalities at runtime via UI without restarting
- **Custom Personality Creation**: Framework supports adding new personalities

###  User Experience
- **Responsive Web Interface**: Gradio UI that works seamlessly on desktop and mobile
- **Real-time Response Streaming**: Immediate feedback with token-by-token generation

##  Getting Started

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for local models)
- 50GB+ disk space (for full Wikipedia processing)

### Installation

```bash
# Clone the repository
git clone https://github.com/lukehalleran/ai-assistant-framework
cd ai-assistant-framework

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Set up environment variables
export OPENAI_API_KEY="your-api-key-here"  # Optional: for GPT-4 support
```

### Quick Start

1. **Run with sample data (recommended for first-time users)**
   ```bash
   python unified_pipeline.py --extracted sample_data.xml --test --semantic
   ```

2. **Launch the web interface**
   ```bash
   python gui.py
   ```
   Access at: http://localhost:7860

3. **Or use the command-line interface**
   ```bash
   python runtime.py
   ```

### Full Wikipedia Processing

For the complete knowledge base:
```bash
python unified_pipeline.py --download --semantic --chunk-size 1000
```
*Note: This downloads and processes the entire Wikipedia dump (~50GB)*

## 🔧 Configuration

Key settings in `config.py`:

| Setting | Description | Default |
|---------|-------------|---------|
| `DEFAULT_MODEL` | Primary language model | `gpt-3.5-turbo` |
| `CONFIDENCE_THRESHOLD` | Semantic search sensitivity | `0.7` |
| `CHUNK_SIZE` | Text chunk size for embeddings | `1000` |
| `DEBUG_MODE` | Enable detailed logging | `False` |
| `SEMANTIC_ONLY_MODE` | Disable memory/wiki features | `False` |

##  Performance Metrics

Based on testing with Wikipedia dataset:
- **Indexing Speed**: ~1,000 articles/minute (multi-threaded)
- **Search Latency**: <100ms for semantic search
- **Memory Usage**: ~4GB for 1M document index
- **Model Switching**: <2s between local/API models

##  Future Enhancements

- [ ] Multi-language support
- [ ] Real-time web search integration
- [ ] Voice input/output capabilities
- [ ] Custom knowledge base uploading
- [ ] Distributed processing support
- [ ] Fine-tuning pipeline for domain adaptation

##  Contributing

While this is primarily a portfolio project, feedback and suggestions are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests for improvements
- Share your use cases and experiences

## Use Cases

This framework has been designed for:
- **Research Assistants**: Quickly find and synthesize information
- **Educational Tools**: Interactive learning with persistent memory
- **Customer Support**: Knowledge-base powered support systems
- **Content Generation**: Context-aware writing assistance
- **Data Analysis**: Natural language queries over structured data

##  Technical Highlights

For recruiters and technical evaluators, this project demonstrates:

- **System Design**: Clean architecture with separation of concerns
- **ML Engineering**: Practical implementation of embeddings, vector search, and RAG
- **Data Engineering**: Efficient processing of large-scale datasets
- **Software Engineering**: Modular, extensible codebase with proper abstractions
- **Problem Solving**: Novel solutions for memory management and context building

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [HuggingFace](https://huggingface.co/) for transformer models and libraries
- [Meta/Facebook](https://github.com/facebookresearch/faiss) for the FAISS vector search library
- [OpenAI](https://openai.com/) for GPT API access
- [Wikipedia](https://www.wikipedia.org/) for open knowledge dumps

##  Contact

**Luke Halleran**
- GitHub: [@lukehalleran](https://github.com/lukehalleran)
- Email: [lukehalleran@gmail.com](mailto:lukehalleran@gmail.com)
- LinkedIn: [luke-halleran-a7605088/]

---

*Built with passion for AI and a commitment to continuous learning. This project represents 4 months of part self-study and implementation, showcasing the transition from AI enthusiast to capable ML engineer.*
