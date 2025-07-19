# memory/storage/chroma_store_multi.py
# Multi-collection ChromaDB storage system

import chromadb
from chromadb.config import Settings
import hashlib
from datetime import datetime
import json
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class MultiCollectionChromaStore:
    """ChromaDB store with separate collections for different memory types"""

    # Define collection names and their purposes
    COLLECTIONS = {
        'conversations': {
            'name': 'conversation-memory',
            'description': 'User-assistant conversation history',
            'embedding_model': 'all-MiniLM-L6-v2'
        },
        'wiki': {
            'name': 'wiki-knowledge',
            'description': 'Wikipedia article chunks',
            'embedding_model': 'all-MiniLM-L6-v2'
        },
        'semantic': {
            'name': 'semantic-chunks',
            'description': 'Semantically chunked long-form content',
            'embedding_model': 'all-MiniLM-L6-v2'
        },
        'summaries': {
            'name': 'conversation-summaries',
            'description': 'Periodic summaries of conversations',
            'embedding_model': 'all-MiniLM-L6-v2'
        },
        'facts': {
            'name': 'extracted-facts',
            'description': 'Important facts and information extracted from conversations',
            'embedding_model': 'all-MiniLM-L6-v2'
        }
    }

    def __init__(self, persist_directory: str = "data/chroma_multi"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Initialize embedding models (could use different ones per collection)
        self.embedders = {}
        self.collections = {}

        # Initialize all collections
        self._initialize_collections()

    def _initialize_collections(self):
        """Initialize all collection types"""
        for key, config in self.COLLECTIONS.items():
            try:
                # Try to get existing collection
                collection = self.client.get_collection(config['name'])
                logger.info(f"Loaded existing collection: {config['name']} ({collection.count()} documents)")
            except:
                # Create new collection
                collection = self.client.create_collection(
                    name=config['name'],
                    metadata={
                        "description": config['description'],
                        "embedding_model": config['embedding_model'],
                        "created_at": datetime.now().isoformat()
                    }
                )
                logger.info(f"Created new collection: {config['name']}")

            self.collections[key] = collection

            # Initialize embedder for this collection (lazy load)
            if config['embedding_model'] not in self.embedders:
                self.embedders[config['embedding_model']] = SentenceTransformer(config['embedding_model'])

    def _generate_id(self, content: str, collection_type: str) -> str:
        """Generate unique ID for a memory"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
        return f"{collection_type}_{content_hash}_{timestamp}"

    def add_conversation_memory(self, query: str, response: str, metadata: Dict = None) -> str:
        """Add a conversation turn to memory"""
        content = f"User: {query}\nAssistant: {response}"
        memory_id = self._generate_id(content, "conv")

        metadata = metadata or {}
        metadata.update({
            "timestamp": datetime.now().isoformat(),
            "query": query[:200],  # Store preview
            "response": response[:200],
            "type": "conversation",
            "turn_length": len(query) + len(response)
        })

        self.collections['conversations'].add(
            ids=[memory_id],
            documents=[content],
            metadatas=[metadata]
        )

        logger.debug(f"Added conversation memory: {memory_id}")
        return memory_id

    def add_wiki_chunk(self, chunk: Dict) -> str:
        """Add a Wikipedia chunk"""
        content = f"{chunk.get('title', '')} {chunk.get('text', '')}"
        memory_id = self._generate_id(content, "wiki")

        metadata = {
            "title": chunk.get('title', 'Unknown'),
            "article_id": chunk.get('id', 'unknown'),
            "chunk_index": chunk.get('chunk_index', 0),
            "timestamp": datetime.now().isoformat(),
            "type": "wiki"
        }

        self.collections['wiki'].add(
            ids=[memory_id],
            documents=[content],
            metadatas=[metadata]
        )

        return memory_id

    def add_semantic_chunk(self, chunk: Dict) -> str:
        """Add a semantic chunk from long-form content"""
        content = chunk.get('content', chunk.get('text', ''))
        memory_id = self._generate_id(content, "sem")

        metadata = {
            "source": chunk.get('source', 'unknown'),
            "chunk_type": chunk.get('type', 'paragraph'),
            "importance": chunk.get('importance', 0.5),
            "timestamp": datetime.now().isoformat(),
            "type": "semantic"
        }

        self.collections['semantic'].add(
            ids=[memory_id],
            documents=[content],
            metadatas=[metadata]
        )

        return memory_id

    def add_summary(self, summary: str, period: str, metadata: Dict = None) -> str:
        """Add a conversation summary"""
        memory_id = self._generate_id(summary, "summ")

        metadata = metadata or {}
        metadata.update({
            "period": period,  # e.g., "2024-01-15_afternoon"
            "timestamp": datetime.now().isoformat(),
            "type": "summary",
            "length": len(summary)
        })

        self.collections['summaries'].add(
            ids=[memory_id],
            documents=[summary],
            metadatas=[metadata]
        )

        return memory_id

    def add_fact(self, fact: str, source: str, confidence: float = 1.0) -> str:
        """Add an extracted fact"""
        memory_id = self._generate_id(fact, "fact")

        metadata = {
            "source": source,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "type": "fact"
        }

        self.collections['facts'].add(
            ids=[memory_id],
            documents=[fact],
            metadatas=[metadata]
        )

        return memory_id

    def search_conversations(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search conversation memories"""
        results = self.collections['conversations'].query(
            query_texts=[query],
            n_results=n_results
        )

        return self._format_results(results, 'conversations')

    def search_wiki(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search Wikipedia chunks"""
        results = self.collections['wiki'].query(
            query_texts=[query],
            n_results=n_results
        )

        return self._format_results(results, 'wiki')

    def search_all(self, query: str, n_results_per_type: int = 3) -> Dict[str, List[Dict]]:
        """Search across all collections"""
        all_results = {}

        for key, collection in self.collections.items():
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=n_results_per_type
                )
                all_results[key] = self._format_results(results, key)
            except Exception as e:
                logger.error(f"Error searching {key}: {e}")
                all_results[key] = []

        return all_results

    def _format_results(self, results: Dict, collection_type: str) -> List[Dict]:
        """Format ChromaDB results into consistent structure"""
        formatted = []

        if not results['ids'] or not results['ids'][0]:
            return formatted

        for i, (doc_id, doc, meta, distance) in enumerate(zip(
            results['ids'][0],
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            formatted.append({
                'id': doc_id,
                'content': doc,
                'metadata': meta,
                'relevance_score': 1.0 / (1.0 + distance),  # Convert distance to similarity
                'collection': collection_type,
                'rank': i + 1
            })

        return formatted

    def get_collection_stats(self) -> Dict[str, Dict]:
        """Get statistics for all collections"""
        stats = {}

        for key, collection in self.collections.items():
            stats[key] = {
                'name': self.COLLECTIONS[key]['name'],
                'description': self.COLLECTIONS[key]['description'],
                'count': collection.count(),
                'embedding_model': self.COLLECTIONS[key]['embedding_model']
            }

        return stats

    def clear_collection(self, collection_key: str):
        """Clear a specific collection"""
        if collection_key in self.collections:
            # Delete and recreate the collection
            self.client.delete_collection(self.COLLECTIONS[collection_key]['name'])
            logger.info(f"Cleared collection: {collection_key}")

            # Reinitialize
            self._initialize_collections()

    def migrate_from_single_collection(self, old_collection_name: str = "assistant-memory"):
        """Migrate from single collection to multi-collection setup"""
        try:
            old_collection = self.client.get_collection(old_collection_name)
            all_data = old_collection.get()

            if not all_data['ids']:
                logger.info("No data to migrate")
                return

            migrated = {
                'conversations': 0,
                'wiki': 0,
                'summaries': 0,
                'facts': 0,
                'semantic': 0
            }

            for doc_id, doc, meta in zip(all_data['ids'], all_data['documents'], all_data['metadatas']):
                # Determine type based on content or metadata
                if 'User:' in doc and 'Assistant:' in doc:
                    self.add_conversation_memory(
                        query=doc.split('Assistant:')[0].replace('User:', '').strip(),
                        response=doc.split('Assistant:')[1].strip(),
                        metadata=meta
                    )
                    migrated['conversations'] += 1
                elif meta.get('type') == 'wiki' or 'title' in meta:
                    self.collections['wiki'].add(
                        ids=[doc_id],
                        documents=[doc],
                        metadatas=[meta]
                    )
                    migrated['wiki'] += 1
                elif '@summary' in meta.get('tags', []):
                    self.add_summary(doc, period="migrated", metadata=meta)
                    migrated['summaries'] += 1
                else:
                    # Default to semantic chunks
                    self.collections['semantic'].add(
                        ids=[doc_id],
                        documents=[doc],
                        metadatas=[meta]
                    )
                    migrated['semantic'] += 1

            logger.info(f"Migration complete: {migrated}")
            return migrated

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return None
