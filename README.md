
# AI Assistant Framework

**Small Project I Am Working On**  
_Contextual AI Assistant with Semantic Search and Memory_

A sophisticated Retrieval-Augmented Generation (RAG) system that combines semantic search, Wikipedia knowledge integration, and persistent memory management for enhanced conversational AI capabilities.

## Project Overview

This project is an intelligent assistant framework that enhances language model responses by:

- **Semantic Search**: Leveraging FAISS vector indexing for efficient similarity search across Wikipedia content  
- **Dynamic Context Building**: Intelligently selecting relevant information based on user queries  
- **Persistent Memory**: Maintaining conversation history and learning from interactions  
- **Multi-Model Support**: Supporting both local models (via HuggingFace) and API-based models (OpenAI)  

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│                   (Gradio Web App)                       │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                  Core Pipeline                           │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐    │
│  │Topic Manager│  │Prompt Builder│  │ Time Manager  │    │
│  └─────────────┘  └──────────────┘  └───────────────┘    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              Knowledge Sources                           │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐    │
│  │Wiki Manager │  │FAISS Search  │  │ Memory Store  │    │
│  │(Wikipedia)  │  │(Semantic)    │  │ (ChromaDB)    │    │
│  └─────────────┘  └──────────────┘  └───────────────┘    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                Model Layer                              │
│  ┌─────────────────────┐  ┌─────────────────────────┐   │
│  │  Local Models       │  │  API Models              │   │
│  │  (HuggingFace)      │  │  (OpenAI GPT-4)          │   │
│  └─────────────────────┘  └─────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Intelligent Context Retrieval

- Semantic search using sentence transformers and FAISS indexing  
- Wikipedia article retrieval with fallback mechanisms  
- Dynamic topic extraction using spaCy NLP  

### 2. Memory Management

- Short-term conversation memory  
- Long-term summarization (every 20 interactions)  
- ChromaDB for persistent vector storage  

### 3. Scalable Processing

- Streaming XML processing for large Wikipedia dumps  
- Semantic chunking for optimal embedding size  
- Multi-threaded article extraction  

### 4. Flexible Model Support

- Local model support via HuggingFace Transformers  
- OpenAI API integration (GPT-4)  
- Dynamic model switching at runtime  

## Installation

### Prerequisites

- Python 3.8+  
- CUDA-capable GPU (recommended for local models)  
- 50GB+ disk space for Wikipedia dump processing  

### Setup

```bash
# Clone the repository
git clone https://github.com/lukehalleran/ai-assistant-framework
cd ai-assistant-framework

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Set up environment variables
export OPENAI_API_KEY="your-api-key-here"
```

### Process Wikipedia data (optional, for full functionality)

```bash
# Download and process Wikipedia dump
python unified_pipeline.py --download --semantic

# Or use a smaller test dataset
python unified_pipeline.py --extracted sample_data.xml --test --semantic
```

## Usage

### Quick Start - Web Interface

```bash
python gui.py
```

Access the Gradio interface at: [http://localhost:7860](http://localhost:7860)

### Command Line Interface

```bash
python runtime.py
```

### Processing Wikipedia Data

```bash
# Full pipeline with semantic chunking
python unified_pipeline.py --download --semantic --chunk-size 1000

# Test mode with limited articles
python unified_pipeline.py --extracted data.xml --test --max-articles 100
```

## Configuration

Key configuration options in `config.py`:

- `DEFAULT_MODEL`: Set default language model  
- `CONFIDENCE_THRESHOLD`: Adjust semantic search sensitivity  
- `DEBUG_MODE`: Enable detailed logging  
- `SEMANTIC_ONLY_MODE`: Use only semantic search (no memory/wiki)  

## Technical Implementation

### Semantic Chunking Algorithm

The system implements intelligent text chunking that:

- Preserves section boundaries  
- Maintains semantic coherence  
- Optimizes for embedding model token limits  

### Vector Search Optimization

- IVF (Inverted File) indexing for scalability  
- Memory-mapped embeddings for efficient loading  
- Configurable similarity thresholds  

### Memory Persistence

- JSON-based corpus storage with atomic writes  
- ChromaDB for vector similarity search  
- Automatic summarization for long-term retention  

### Security Features

- Input sanitization for prompt injection protection  
- Multi-language pattern detection  
- Suspicious content logging and redaction  

## Contributing

This project is part of my portfolio for OMSA applications.  
While not currently accepting contributions, feedback and suggestions are welcome!

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.

## Acknowledgments

- Wikipedia for providing open knowledge dumps  
- HuggingFace for transformer models and libraries  
- OpenAI for GPT API access  
- FAISS team at Meta for the vector search library  

## Contact

- GitHub: [lukehalleran](https://github.com/lukehalleran)  
- Email: lukehalleran@gmail.com  

---

**This project demonstrates competencies in:**  
_Natural Language Processing, Information Retrieval, System Design, Python Programming, and Machine Learning Engineering_
