"""
# memory/storage/multi_collection_chroma_store.py

Module Contract
- Purpose: Wrapper over ChromaDB with 12 separate collections (conversations, summaries,
  wiki_knowledge, facts, reflections, obsidian_notes, reference_docs, procedural,
  procedural_skills, proposals, threads, synthesis_results). Provides unified
  add/query/update helpers with robust metadata flattening.
- Class: MultiCollectionChromaStore(persist_directory)
- Key methods:
  - add_to_collection(name, text, metadata) -> str  [generic add, returns doc_id]
  - add_conversation_memory(query, response, metadata) -> str
  - add_summary(summary, period, metadata) -> str
  - add_wiki_chunk(chunk) -> str
  - add_fact(fact_text, source, confidence, extra_metadata) -> str  [with dedup check]
  - add_reflection(reflection, source_ids, reflection_type) -> str
  - query_collection(name, query_text, n_results, where_filter) -> List[Dict]
    Returns flat list of {id, content, metadata, relevance_score, collection, rank}.
  - query_multiple_collections(collection_names, query_text, n_results) -> Dict[str, List[Dict]]  [async]
  - search_all(query, n_results_per_type) -> Dict[str, List[Dict]]  [sync across all collections]
  - get_by_id(collection_name, doc_id) -> Optional[Dict]  [{id, content, metadata} or None]
  - get_recent(collection_name, limit) -> List[Dict]  [sorted by timestamp desc]
  - list_all(collection_name) -> List[Dict]  [all docs in collection]
  - update_metadata(collection_name, doc_id, metadata_updates) -> bool  [merge updates into existing]
  - delete_fact(fact_id) -> bool
  - create_collection(name) -> None  [dynamic collection creation]
  - get_collection_stats() -> Dict[str, Dict]  [count + sample per collection]
- Key behaviors:
  - SentenceTransformer embedding function configured once, shared across all collections
  - _flatten_for_chroma() ensures all metadata values are primitives or JSON strings
  - Fact deduplication via cosine similarity check before insertion
  - Results un-nested from ChromaDB's nested format into flat dicts
- Side effects:
  - Persists to CHROMA_PATH directory on disk
  - Telemetry disabled via ANONYMIZED_TELEMETRY env var
"""
import os
import logging

# Disable ChromaDB telemetry BEFORE importing chromadb
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Suppress chromadb telemetry errors (known bug with posthog compatibility)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import hashlib
import json
import uuid
import asyncio
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)


def _flatten_for_chroma(md: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Ensure all metadata values are primitives that Chroma accepts."""
    flat = {}
    for k, v in (md or {}).items():
        if isinstance(v, (str, int, float, bool)):
            flat[k] = v
        elif v is None:
            continue
        elif isinstance(v, (list, tuple, set)):
            flat[k] = ",".join(map(str, v))
        else:
            # dicts or custom objects -> JSON string
            try:
                flat[k] = json.dumps(v, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"[ChromaStore] Could not JSON-serialize metadata key '{k}': {e}, falling back to str()")
                flat[k] = str(v)
    return flat

class MultiCollectionChromaStore:
    """ChromaDB store with separate collections for different memory types"""

    def __init__(self, persist_directory: str = "data/chroma_multi"):
        self.persist_directory = persist_directory

        # Initialize ChromaDB with telemetry disabled to avoid posthog version conflicts
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Single, shared embedder for this store instance
        model_name = os.getenv("CHROMA_ST_MODEL", "all-MiniLM-L6-v2")
        device = os.getenv("CHROMA_DEVICE", "cpu")  # set to "cuda" if desired
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name, device=device
        )

        # Keep your existing dict of collections (None placeholders are fine)
        self.collections = {
            'conversations': None,
            'summaries': None,
            'wiki_knowledge': None,
            'facts': None,
            'reflections': None,
            'obsidian_notes': None,  # User's personal notes from Obsidian vault
            'reference_docs': None,  # User uploaded reference documents
            'procedural': None,      # Git commits and how-to knowledge
            'procedural_skills': None,  # Reusable problem-solving patterns
            'proposals': None,           # Goal-directed code change proposals
            'threads': None,             # Open threads (commitments, deadlines, questions)
            'synthesis_results': None,   # Cross-domain synthesis insights from graph walks
        }

        self._initialize_collections()



    def _collection_embedder_name(self, coll) -> str:
        """
        Best-effort peek at the embedder name Chroma stored for a collection.
        Some versions expose it on the collection metadata; if not, return 'unknown'.
        """
        try:
            # Chroma 0.4/0.5 stores this in 'metadata' -> 'embedding_function'
            md = getattr(coll, "metadata", None) or {}
            name = md.get("embedding_function") or md.get("embedding_function_name")
            if name:
                return str(name)
        except Exception:
            pass
        # Fallback: try the client API list for a match
        try:
            for c in self.client.list_collections():
                if getattr(c, "name", None) == getattr(coll, "name", None):
                    md = getattr(c, "metadata", None) or {}
                    name = md.get("embedding_function") or md.get("embedding_function_name")
                    if name:
                        return str(name)
        except Exception:
            pass
        return "unknown"
    def _initialize_collections(self):
        """Initialize/open collections; if embedder mismatches, delete & recreate."""
        logger.info(f"[Chroma] Using persistent dir: {self.persist_directory}")

        def _create(name: str):
            self.collections[name] = self.client.get_or_create_collection(
                name=name, embedding_function=self.embedding_fn
            )
            logger.debug(f"[Chroma] Initialized collection: {name}")

        for name in self.collections.keys():
            try:
                # Happy path: create/open with our embedder
                _create(name)
            except ValueError as e:
                msg = str(e)
                if "Embedding function name mismatch" not in msg:
                    logger.error(f"[Chroma] Failed to init '{name}': {e}")
                    raise

                # Inspect what Chroma thinks it has, then fix it in-place
                try:
                    existing = self.client.get_collection(name=name)  # may succeed even with mismatch
                except Exception:
                    existing = None

                existing_fn = self._collection_embedder_name(existing) if existing else "default"
                logger.warning(f"[Chroma] '{name}' uses embedder '{existing_fn}', expected 'sentence_transformer'. Recreating…")

                # Delete then recreate with our embedder
                try:
                    self.client.delete_collection(name=name)
                except Exception as de:
                    logger.error(f"[Chroma] CRITICAL: Could not delete stale collection '{name}': {de} - data corruption risk")

                _create(name)
            except Exception as e:
                logger.error(f"[Chroma] Failed to init '{name}': {e}")
                raise



    # memory/storage/multi_collection_chroma_store.py

    def list_all(self, collection_name: str) -> List[Dict]:
        if collection_name not in self.collections:
            return []
        coll = self.collections[collection_name]

        # ❌ don't ask for "ids" here; not supported in your Chroma build
        results = coll.get(include=["documents", "metadatas"])  # <- only these

        docs = results.get("documents", []) or []
        metas = results.get("metadatas", []) or []

        # Some builds still return ids even if not requested; handle if present
        ids   = results.get("ids", []) or []

        out = []
        n = max(len(docs), len(metas), len(ids))
        for i in range(n):
            _id  = ids[i]  if i < len(ids)  else None
            doc  = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) else {}
            out.append({
                "id": _id,
                "content": doc or "",
                "metadata": meta or {},
            })
        return out

    # Generic helpers expected by coordinators
    def create_collection(self, name: str):
        """Create/open a collection with the configured embedder."""
        self.collections[name] = self.client.get_or_create_collection(
            name=name, embedding_function=self.embedding_fn
        )
        return self.collections[name]

    def add_to_collection(self, name: str, text: str, metadata: Dict[str, Any]) -> str:
        """Add a single text item to the specified collection with metadata."""
        if name not in self.collections or self.collections[name] is None:
            self.create_collection(name)
        coll = self.collections[name]
        clean_md = _flatten_for_chroma(dict(metadata or {}))
        # ChromaDB requires non-empty metadata - add timestamp if empty
        if not clean_md:
            clean_md["timestamp"] = datetime.now().isoformat()
        doc_id = str(uuid.uuid4())
        coll.add(ids=[doc_id], documents=[text or ""], metadatas=[clean_md])
        return doc_id

    def get_recent(self, collection_name: str, limit: int = 8) -> List[Dict]:
        all_items = self.list_all(collection_name)

        def _ts(x):
            ts = (x.get("metadata") or {}).get("timestamp")
            try:
                from datetime import datetime
                if isinstance(ts, str):
                    return datetime.fromisoformat(ts)
            except Exception as e:
                logger.debug(f"[ChromaStore] Could not parse timestamp '{ts}': {e}, using minimum date")
            # fallback ensures items without timestamp don't crash
            from datetime import datetime as _dt
            return _dt.min

        all_items.sort(key=_ts, reverse=True)
        return all_items[:limit]


    def get_by_id(self, collection_name: str, doc_id: str) -> Optional[Dict]:
        """Fetch a single document by its ID.

        Returns:
            Dict with {id, content, metadata} or None if not found.
        """
        if collection_name not in self.collections:
            return None
        coll = self.collections[collection_name]
        try:
            results = coll.get(ids=[doc_id], include=["documents", "metadatas"])
            docs = results.get("documents", []) or []
            metas = results.get("metadatas", []) or []
            if not docs:
                return None
            return {
                "id": doc_id,
                "content": docs[0] or "",
                "metadata": metas[0] if metas else {},
            }
        except Exception:
            return None

    def _generate_id(self, content: str, collection_type: str) -> str:
        """Generate a unique ID for a document"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
        return f"{collection_type}_{content_hash}_{timestamp}"

    # Methods for conversations
    # In memory/storage/multi_collection_chroma_store.py, check the add_conversation_memory method
    def add_conversation_memory(self, query: str, response: str, metadata: Dict[str, Any]) -> str:
        """Add a conversation memory to the conversations collection"""
        try:
            # Ensure metadata is clean before passing to ChromaDB
            if metadata is None:
                metadata = {}

            # Log the metadata before adding
            logger.debug(f"[ChromaStore] Adding conversation memory with metadata: {metadata}")

            # Double-check metadata values
            for key, value in list(metadata.items()):
                if value is None:
                    logger.warning(f"[ChromaStore] Metadata key '{key}' has None value, removing it")
                    metadata.pop(key)
                elif not isinstance(value, (str, int, float, bool)):
                    logger.warning(f"[ChromaStore] Converting metadata key '{key}' from {type(value)} to string")
                    metadata[key] = str(value)

            # ChromaDB requires non-empty metadata - add timestamp if empty
            if not metadata:
                metadata["timestamp"] = datetime.now().isoformat()

            # Create the document ID
            doc_id = str(uuid.uuid4())

            # Add to collection
            self.collections['conversations'].add(
                ids=[doc_id],
                documents=[f"User: {query}\nAssistant: {response}"],
                metadatas=[metadata]
            )

            logger.debug(f"[ChromaStore] Successfully added conversation memory with ID: {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"Error adding conversation memory: {e}")
            import traceback
            traceback.print_exc()
            return None


    # Methods for summaries
    def add_summary(self, summary: str, period: str, metadata: Dict = None) -> str:
        memory_id = self._generate_id(summary, "summ")

        metadata = metadata or {}
        metadata.update({
            "period": period,
            "timestamp": datetime.now().isoformat(),
            "type": "summary"
        })

        self.collections['summaries'].add(
            documents=[summary],
            metadatas=[metadata],
            ids=[memory_id]
        )
        return memory_id

    # Methods for wiki knowledge
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

        self.collections['wiki_knowledge'].add(
            documents=[content],
            metadatas=[metadata],
            ids=[memory_id]
        )
        return memory_id

    # Methods for facts (new)
    # --- replace your add_fact with this version (keeps your logic, adds compat/flatten) ---
    # In multi_collection_chroma_store.py, update the add_fact method:

    # ⚠️ WARNING: Parent-child relationships (parent_id, child_ids) are DISABLED
    # This feature caused retrieval issues and has been INTENTIONALLY DEACTIVATED.
    # The parameters remain for API compatibility but are NOT used by any retrieval logic.
    # DO NOT re-enable without addressing the original solved issues:
    #   - Circular reference bugs in memory traversal
    #   - Cascade deletion failures
    #   - Query performance degradation with deep hierarchies
    # These fields are stored but intentionally ignored to maintain system stability.

    def add_fact(
        self,
        fact: str,
        source,  # str OR dict (legacy compat)
        confidence: float = 1.0,
        parent_id: Optional[str] = None,
        child_ids: Optional[List[str]] = None
    ) -> str:
        """
        Preserves existing signature, but also supports legacy calls:
        add_fact("a | b | c", {"source":"manual", "confidence":0.9, ...})
        """
        # Back-compat: if 'source' is actually a metadata dict
        extra_md = {}
        if isinstance(source, dict):
            md_in = dict(source)  # shallow copy
            source = md_in.pop("source", "unknown")
            confidence = float(md_in.pop("confidence", confidence))
            parent_id = md_in.pop("parent_id", parent_id)
            child_ids = md_in.pop("child_ids", child_ids)
            # anything else stays as extra metadata
            extra_md = md_in

        # Check for duplicates before adding
        if self._is_duplicate_fact(fact):
            logger.info(f"[ChromaStore] Skipping duplicate fact: {fact}")
            return None  # Return None to indicate fact was not added

        memory_id = self._generate_id(fact, "fact")
        logger.debug(f"[ChromaStore] Adding fact with ID {memory_id}: {fact}")

        metadata = {
            "source": source,
            "confidence": float(confidence),
            "timestamp": datetime.now().isoformat(),
            "type": "fact",
        }
        if parent_id is not None:
            metadata["parent_id"] = parent_id
        if child_ids:
            metadata["child_ids"] = child_ids  # will be flattened

        # merge any extra fields gathered from legacy dict
        metadata.update(extra_md)

        # flatten to Chroma-acceptable primitives
        metadata = _flatten_for_chroma(metadata)
        logger.debug(f"[ChromaStore] Fact metadata: {metadata}")

        try:
            self.collections["facts"].add(
                documents=[fact],
                metadatas=[metadata],
                ids=[memory_id],
            )
            logger.debug(f"[ChromaStore] Successfully added fact to collection")
            # Verify the fact was added
            count = self.collections["facts"].count()
            logger.debug(f"[ChromaStore] Facts collection now has {count} documents")
        except Exception as e:
            logger.error(f"[ChromaStore] Error adding fact: {e}")
            raise

        return memory_id

    def delete_fact(self, fact_id: str) -> bool:
        """Delete a fact by ID from the facts collection."""
        try:
            if "facts" not in self.collections:
                logger.warning("[ChromaStore] Facts collection not found")
                return False

            # ChromaDB delete by ID
            self.collections["facts"].delete(ids=[fact_id])
            logger.debug(f"[ChromaStore] Deleted fact with ID: {fact_id}")
            return True

        except Exception as e:
            logger.error(f"[ChromaStore] Error deleting fact {fact_id}: {e}")
            return False

    def _is_duplicate_fact(self, fact: str, similarity_threshold: float = 0.90) -> bool:
        """Check if a fact is duplicate or very similar to existing facts."""
        try:
            # Quick exact match check first
            existing_facts = self.list_all('facts')
            for existing_fact in existing_facts:
                existing_content = existing_fact.get('content', '')

                # Exact match
                if fact.strip().lower() == existing_content.strip().lower():
                    return True

                # High similarity check (simple token overlap)
                fact_tokens = set(fact.lower().split())
                existing_tokens = set(existing_content.lower().split())

                if fact_tokens and existing_tokens:
                    intersection = fact_tokens.intersection(existing_tokens)
                    union = fact_tokens.union(existing_tokens)
                    similarity = len(intersection) / len(union)

                    if similarity >= similarity_threshold:
                        logger.debug(f"[ChromaStore] Found similar fact (similarity: {similarity:.2f}): '{fact}' ~ '{existing_content}'")
                        return True

            return False

        except Exception as e:
            logger.warning(f"[ChromaStore] Error checking for duplicate fact: {e}")
            return False  # If check fails, allow the fact to be added

    # Methods for reflections (new)
    def add_reflection(self, reflection: str, source_ids: List[str], reflection_type: str) -> str:
        memory_id = self._generate_id(reflection, "refl")
        metadata = _flatten_for_chroma({
            "source_ids": source_ids,                # list -> "a,b,c"
            "reflection_type": reflection_type,
            "timestamp": datetime.now().isoformat(),
            "type": "reflection",
        })
        self.collections["reflections"].add(
            documents=[reflection],
            metadatas=[metadata],
            ids=[memory_id],
        )
        return memory_id


    # Query methods




    def query_collection(self, collection_name: str, query_text: str,
                     n_results: int = 5, **kwargs) -> List[Dict]:
        # Accept alias kwargs defensively
        if "n" in kwargs and isinstance(kwargs["n"], int):
            n_results = kwargs["n"]
        if "k" in kwargs and isinstance(kwargs["k"], int):
            n_results = kwargs["k"]

        if collection_name not in self.collections:
            raise ValueError(f"Unknown collection: {collection_name}")

        # DO NOT include "ids" (not supported by your Chroma build)
        results = self.collections[collection_name].query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]  # no "ids"
        )

        # Some builds still return ids even if not requested; handle both cases
        ids_list  = (results.get("ids") or [[]])[0] or []
        docs_list = (results.get("documents") or [[]])[0] or []
        metas     = (results.get("metadatas") or [[]])[0] or []
        dists     = (results.get("distances") or [[]])[0] or []

        # Iterate by the longest list so we don't drop rows if one list is shorter
        n = max(len(docs_list), len(metas), len(dists), len(ids_list))

        formatted = []
        for i in range(n):
            _id  = ids_list[i] if i < len(ids_list) else None
            doc  = docs_list[i] if i < len(docs_list) else ""
            meta = metas[i]     if i < len(metas)     else {}
            dist = dists[i]     if i < len(dists)     else None
            score = (1.0 / (1.0 + dist)) if isinstance(dist, (int, float)) else None

            formatted.append({
                "id": _id,
                "content": doc,
                "metadata": meta or {},
                "relevance_score": score,
                "collection": collection_name,
                "rank": i + 1,
            })

        return formatted





    def search_all(self, query: str, n_results_per_type: int = 3) -> Dict[str, List[Dict]]:
        """Search across all collections"""
        all_results = {}

        for name, collection in self.collections.items():
            try:
                results = self.query_collection(name, query, n_results_per_type)
                all_results[name] = results
            except Exception as e:
                logger.error(f"Error searching {name}: {e}")
                all_results[name] = []

        return all_results

    async def query_multiple_collections(self, collection_names: List[str], query_text: str,
                                        n_results: int = 10) -> Dict[str, List[Dict]]:
        """Query multiple collections in parallel for better performance"""
        async def query_single_collection(collection_name: str) -> tuple[str, List[Dict]]:
            try:
                if collection_name not in self.collections:
                    logger.debug(f"[BatchQuery] Collection {collection_name} not found")
                    return collection_name, []

                # Check if collection is empty
                collection = self.collections[collection_name]
                count = collection.count()

                if count == 0:
                    logger.debug(f"[BatchQuery] Collection {collection_name} is empty")
                    return collection_name, []

                # Run query in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    self.query_collection,
                    collection_name,
                    query_text,
                    min(n_results, count)
                )

                return collection_name, results or []

            except Exception as e:
                logger.error(f"[BatchQuery] Error querying {collection_name}: {e}")
                return collection_name, []

        # Run all queries in parallel
        tasks = [query_single_collection(name) for name in collection_names]
        results_dict = {}

        for collection_name, results in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(results, Exception):
                logger.error(f"[BatchQuery] Exception in {collection_name}: {results}")
                results_dict[collection_name] = []
            else:
                results_dict[collection_name] = results

        return results_dict

    def update_metadata(self, collection_name: str, doc_id: str, metadata_updates: dict) -> bool:
        """Update metadata fields on an existing document.

        Merges *metadata_updates* into the document's current metadata
        and writes the result back.  Returns True on success.
        """
        coll = self.collections.get(collection_name)
        if not coll:
            logger.warning("[ChromaStore] update_metadata: unknown collection '%s'", collection_name)
            return False
        try:
            existing = coll.get(ids=[doc_id], include=["metadatas"])
            if not existing or not existing.get("metadatas"):
                logger.warning("[ChromaStore] update_metadata: doc '%s' not found in '%s'", doc_id, collection_name)
                return False
            merged = {**(existing["metadatas"][0] or {}), **metadata_updates}
            coll.update(ids=[doc_id], metadatas=[_flatten_for_chroma(merged)])
            return True
        except Exception as e:
            logger.error("[ChromaStore] update_metadata failed for %s/%s: %s", collection_name, doc_id, e)
            return False

    def get_collection_stats(self) -> Dict[str, Dict]:
        """Get statistics for all collections"""
        stats = {}

        for name, collection in self.collections.items():
            try:
                count = collection.count()
                stats[name] = {
                    'name': name,
                    'count': count
                }
            except Exception as e:
                logger.error(f"Error getting stats for {name}: {e}")
                stats[name] = {
                    'name': name,
                    'count': 0,
                    'error': str(e)
                }

        return stats
