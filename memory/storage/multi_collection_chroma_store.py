import chromadb
from chromadb.config import Settings
import hashlib
from datetime import datetime
import json
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import logging
from threading import Lock

logger = logging.getLogger(__name__)

class MultiCollectionChromaStore:
    """ChromaDB store with separate collections for different memory types"""

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

        self.embedders = {}
        self.collections = {}
        self.embedder_lock = Lock()

        self._initialize_collections()
        logger.debug(f"[CHROMA INIT] Available collections: {list(self.collections.keys())}")

    def _initialize_collections(self):
        for key, config in self.COLLECTIONS.items():
            try:
                collection = self.client.get_collection(config['name'])
                logger.info(f"Loaded existing collection: {config['name']} ({collection.count()} documents)")
            except:
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

            with self.embedder_lock:
                if config['embedding_model'] not in self.embedders:
                    self.embedders[config['embedding_model']] = SentenceTransformer(config['embedding_model'])

    def _generate_id(self, content: str, collection_type: str) -> str:
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
        return f"{collection_type}_{content_hash}_{timestamp}"

    def _safe_add(self, collection_key: str, ids, documents, metadatas):
        try:
            self.collections[collection_key].add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            # Persist is deprecated and handled automatically by the client now.
            # self.client.persist()
            logger.debug(f"Added to {collection_key} memory: {ids[0]}")
        except Exception as e:
            logger.error(f"Failed to add to {collection_key}: {e}")

    def add_conversation_memory(self, query: str, response: str, metadata: Dict = None) -> str:
        content = f"User: {query}\nAssistant: {response}"
        memory_id = self._generate_id(content, "conv")

        metadata = metadata or {}
        metadata.update({
            "timestamp": datetime.now().isoformat(),
            "query": query[:200],
            "response": response[:200],
            "type": "conversation",
            "turn_length": len(query) + len(response)
        })

        self._safe_add('conversations', [memory_id], [content], [metadata])

        # Confirm it was indexed
        try:
            result = self.collections['conversations'].get(ids=[memory_id])
            if result and result['ids']:
                logger.debug(f"✅ Memory {memory_id} successfully indexed and retrieved from ChromaDB.")
            else:
                logger.warning(f"⚠️ Memory {memory_id} was added but could not be retrieved for confirmation.")
        except Exception as e:
            logger.error(f"❌ Error confirming memory {memory_id} in ChromaDB: {e}")

        return memory_id


    def add_wiki_chunk(self, chunk: Dict) -> str:
        content = f"{chunk.get('title', '')} {chunk.get('text', '')}"
        memory_id = self._generate_id(content, "wiki")

        metadata = {
            "title": chunk.get('title', 'Unknown'),
            "article_id": chunk.get('id', 'unknown'),
            "chunk_index": chunk.get('chunk_index', 0),
            "timestamp": datetime.now().isoformat(),
            "type": "wiki"
        }

        self._safe_add('wiki', [memory_id], [content], [metadata])
        return memory_id

    def add_semantic_chunk(self, chunk: Dict) -> str:
        content = chunk.get('content', chunk.get('text', ''))
        memory_id = self._generate_id(content, "sem")

        metadata = {
            "source": chunk.get('source', 'unknown'),
            "chunk_type": chunk.get('type', 'paragraph'),
            "importance": chunk.get('importance', 0.5),
            "timestamp": datetime.now().isoformat(),
            "type": "semantic"
        }

        self._safe_add('semantic', [memory_id], [content], [metadata])
        return memory_id

    def add_summary(self, summary: str, period: str, metadata: Dict = None) -> str:
        memory_id = self._generate_id(summary, "summ")

        metadata = metadata or {}
        metadata.update({
            "period": period,
            "timestamp": datetime.now().isoformat(),
            "type": "summary",
            "length": len(summary)
        })

        self._safe_add('summaries', [memory_id], [summary], [metadata])
        return memory_id

    def add_fact(self, fact: str, source: str, confidence: float = 1.0) -> str:
        memory_id = self._generate_id(fact, "fact")

        metadata = {
            "source": source,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "type": "fact"
        }

        self._safe_add('facts', [memory_id], [fact], [metadata])
        return memory_id

    def get_all_from_collection(self, collection_key: str) -> List[Dict]:
        """Retrieve all documents from a specific collection."""
        if collection_key not in self.collections:
            return []
        try:
            collection = self.collections[collection_key]
            count = collection.count()
            if count == 0:
                return []

            # Retrieve all documents by specifying the limit
            results = collection.get(limit=count, include=["metadatas", "documents"])

            formatted = []
            if not results.get('ids'):
                return formatted

            for i, doc_id in enumerate(results['ids']):
                formatted.append({
                    'id': doc_id,
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i],
                    'collection': collection_key
                })
            return formatted
        except Exception as e:
            logger.error(f"Error getting all documents from {collection_key}: {e}")
            return []

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
                # Ensure we don't query an empty collection, which can cause errors
                if collection.count() > 0:
                    results = collection.query(
                        query_texts=[query],
                        n_results=min(n_results_per_type, collection.count()) # Cannot request more results than exist
                    )
                    all_results[key] = self._format_results(results, key)
                else:
                    all_results[key] = []
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
