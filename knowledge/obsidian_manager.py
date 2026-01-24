# /knowledge/obsidian_manager.py
"""
ObsidianManager - Parse and embed Obsidian vault markdown files into ChromaDB.

Module Contract:
- Purpose: Integrate user's personal notes from Obsidian vault into the RAG pipeline
- Inputs:
  - embed_vault(force_reindex: bool) -> EmbedResult: Index vault to ChromaDB
  - get_notes(query: str, limit: int) -> List[Dict]: Retrieve relevant notes
- Outputs:
  - Embedded notes in obsidian_notes ChromaDB collection
  - Retrieved notes with metadata (tags, links, title, path)
- Side effects:
  - Writes to obsidian_notes ChromaDB collection
  - Logging of indexing progress
- Error handling:
  - Graceful degradation if vault doesn't exist
  - Per-file error handling (one bad file doesn't stop indexing)

Features:
- Smart hybrid chunking: whole notes if <1500 chars, else chunk by ## headers
- Preserves #tags as metadata for filtering
- Extracts [[wiki links]] as related_notes references
- Strips YAML frontmatter while keeping content clean
"""

import os
import re
import logging
import hashlib
from pathlib import Path

from utils.text_chunking import chunk_by_headers
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time

logger = logging.getLogger(__name__)


@dataclass
class EmbedResult:
    """Result of vault embedding operation."""
    total_files: int = 0
    embedded_files: int = 0
    total_chunks: int = 0
    skipped_files: int = 0
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class ObsidianManager:
    """Manages Obsidian vault parsing, embedding, and retrieval."""

    def __init__(self, chroma_store=None, vault_path: str = None):
        """
        Initialize ObsidianManager.

        Args:
            chroma_store: Optional ChromaDB store instance (lazy-loaded if None)
            vault_path: Path to Obsidian vault (uses config default if None)
        """
        self._chroma_store = chroma_store

        # Load config values
        try:
            from config.app_config import (
                OBSIDIAN_VAULT_PATH,
                OBSIDIAN_CHUNK_THRESHOLD,
                OBSIDIAN_MAX_NOTES_PROMPT,
            )
            self.vault_path = vault_path or OBSIDIAN_VAULT_PATH
            self.chunk_threshold = OBSIDIAN_CHUNK_THRESHOLD
            self.max_notes = OBSIDIAN_MAX_NOTES_PROMPT
        except ImportError:
            # Fallback defaults if config not available
            self.vault_path = vault_path or os.path.expanduser("~/Documents/Luke Notes")
            self.chunk_threshold = 1500
            self.max_notes = 5

        logger.debug(f"[Obsidian] Initialized with vault_path={self.vault_path}")

    @property
    def chroma_store(self):
        """Lazy-load ChromaDB store."""
        if self._chroma_store is None:
            try:
                from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
                from config.app_config import CHROMA_PATH
                self._chroma_store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
                logger.debug("[Obsidian] ChromaDB store lazy-loaded")
            except Exception as e:
                logger.error(f"[Obsidian] Failed to load ChromaDB store: {e}")
                raise
        return self._chroma_store

    def _strip_frontmatter(self, content: str) -> str:
        """
        Strip YAML frontmatter from markdown content.

        Frontmatter is delimited by --- at the start of the file.
        Example:
            ---
            title: My Note
            tags: [tag1, tag2]
            ---
            Actual content here...
        """
        if not content.startswith('---'):
            return content

        # Find the closing ---
        end_match = re.search(r'\n---\s*\n', content[3:])
        if end_match:
            return content[end_match.end() + 3:].strip()
        return content

    def _extract_tags(self, content: str) -> List[str]:
        """
        Extract #tags from content (not ## headers).

        Matches #tag patterns where tag starts with a letter and contains
        alphanumeric chars, underscores, or hyphens.
        """
        # Negative lookbehind to avoid matching ## headers
        tags = re.findall(r'(?<![#\w])#([a-zA-Z][a-zA-Z0-9_/-]*)', content)
        return list(set(tags))

    def _extract_wiki_links(self, content: str) -> List[str]:
        """
        Extract [[wiki links]] from content.

        Handles:
            [[Note Name]]
            [[Note Name|Display Text]]
        """
        # Match [[link]] or [[link|alias]]
        links = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', content)
        return list(set(links))

    def _chunk_by_headers(self, content: str, note_title: str) -> List[Dict[str, Any]]:
        """
        Smart hybrid chunking: whole note if small, else chunk by ## headers.

        Args:
            content: Note content (frontmatter already stripped)
            note_title: Title of the note for section naming

        Returns:
            List of chunk dicts with text, section, chunk_index, total_chunks
        """
        return chunk_by_headers(
            content=content,
            title=note_title,
            chunk_threshold=self.chunk_threshold,
            include_header_in_text=False,
            min_chunk_size=0
        )

    def _generate_id(self, content: str, prefix: str = "obs") -> str:
        """Generate a unique ID for a document."""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
        return f"{prefix}_{content_hash}_{timestamp}"

    def embed_vault(self, force_reindex: bool = False) -> EmbedResult:
        """
        Embed all markdown files from the Obsidian vault.

        Args:
            force_reindex: If True, re-embed all files even if already indexed

        Returns:
            EmbedResult with statistics about the operation
        """
        start_time = time.time()
        result = EmbedResult()

        # Expand and validate vault path
        vault = Path(self.vault_path).expanduser()
        if not vault.exists():
            result.errors.append(f"Vault path does not exist: {self.vault_path}")
            logger.error(result.errors[-1])
            result.duration_seconds = time.time() - start_time
            return result

        if not vault.is_dir():
            result.errors.append(f"Vault path is not a directory: {self.vault_path}")
            logger.error(result.errors[-1])
            result.duration_seconds = time.time() - start_time
            return result

        # Find all markdown files
        md_files = list(vault.rglob("*.md"))
        result.total_files = len(md_files)
        logger.info(f"[Obsidian] Found {result.total_files} markdown files in {vault}")

        # Get existing file paths to skip if not force_reindex
        existing_paths = set()
        if not force_reindex:
            try:
                # Use collection.get() to retrieve all existing documents
                # query_collection with empty string doesn't reliably return all docs
                collection = self.chroma_store.collections.get('obsidian_notes')
                if collection:
                    all_docs = collection.get(include=['metadatas'])
                    metadatas = all_docs.get('metadatas', []) or []
                    for meta in metadatas:
                        if meta and isinstance(meta, dict):
                            path = meta.get('file_path', '')
                            if path:
                                existing_paths.add(path)
                    logger.info(f"[Obsidian] Found {len(existing_paths)} already indexed file paths")
            except Exception as e:
                logger.warning(f"[Obsidian] Could not check existing files: {e}")

        # Process each file
        for md_file in md_files:
            try:
                rel_path = str(md_file.relative_to(vault))

                # Skip if already indexed (unless force_reindex)
                if rel_path in existing_paths:
                    result.skipped_files += 1
                    continue

                # Read file content
                try:
                    content = md_file.read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    # Try with latin-1 as fallback
                    content = md_file.read_text(encoding='latin-1')

                # Strip frontmatter
                content = self._strip_frontmatter(content)

                # Skip empty files
                if not content.strip():
                    result.skipped_files += 1
                    continue

                # Extract metadata
                tags = self._extract_tags(content)
                wiki_links = self._extract_wiki_links(content)
                note_title = md_file.stem

                # Chunk the content
                chunks = self._chunk_by_headers(content, note_title)

                # Add each chunk to ChromaDB
                for chunk in chunks:
                    metadata = {
                        'type': 'obsidian_note',
                        'title': note_title,
                        'file_path': rel_path,
                        'tags': ','.join(tags) if tags else '',
                        'related_notes': ','.join(wiki_links) if wiki_links else '',
                        'section': chunk.get('section') or '',
                        'chunk_index': chunk['chunk_index'],
                        'total_chunks': chunk['total_chunks'],
                        'timestamp': datetime.now().isoformat(),
                        'truth_score': 0.9,  # High confidence for personal notes
                    }

                    self.chroma_store.add_to_collection(
                        'obsidian_notes',
                        chunk['text'],
                        metadata
                    )
                    result.total_chunks += 1

                result.embedded_files += 1

                # Progress logging
                if result.embedded_files % 50 == 0:
                    logger.info(f"[Obsidian] Embedded {result.embedded_files}/{result.total_files} files...")

            except Exception as e:
                error_msg = f"Error processing {md_file}: {str(e)}"
                result.errors.append(error_msg)
                logger.warning(f"[Obsidian] {error_msg}")

        result.duration_seconds = time.time() - start_time
        logger.info(
            f"[Obsidian] Embedding complete: {result.embedded_files} files, "
            f"{result.total_chunks} chunks, {result.skipped_files} skipped, "
            f"{len(result.errors)} errors, {result.duration_seconds:.1f}s"
        )
        return result

    async def get_notes(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant notes using hybrid search: 1/3 keyword + 2/3 semantic.

        Args:
            query: Search query
            limit: Maximum notes to return

        Returns:
            List of note dicts with content, metadata, and relevance_score
        """
        try:
            # Calculate split: 1/3 keyword, 2/3 semantic
            keyword_limit = max(1, limit // 3)
            semantic_limit = limit - keyword_limit

            # 1. KEYWORD SEARCH - match on title, tags, content
            keyword_results = self._keyword_search(query, keyword_limit * 3)  # Get extra for filtering

            # 2. SEMANTIC SEARCH - vector similarity
            semantic_results = self.chroma_store.query_collection(
                'obsidian_notes',
                query,
                n_results=semantic_limit * 2  # Get extra for dedup
            )

            # Format semantic results
            semantic_formatted = []
            for r in semantic_results:
                meta = r.get('metadata', {})
                semantic_formatted.append({
                    'content': r.get('content', ''),
                    'metadata': {
                        'type': 'obsidian_note',
                        'title': meta.get('title', ''),
                        'tags': meta.get('tags', ''),
                        'related_notes': meta.get('related_notes', ''),
                        'section': meta.get('section', ''),
                        'file_path': meta.get('file_path', ''),
                        'truth_score': float(meta.get('truth_score', 0.9)),
                        'timestamp': meta.get('timestamp', ''),
                    },
                    'relevance_score': r.get('relevance_score', 0.0),
                    'match_type': 'semantic',
                })

            # 3. COMBINE with deduplication (keyword first for priority)
            seen_titles = set()
            combined = []

            # Add keyword matches first (1/3)
            for note in keyword_results[:keyword_limit]:
                title = note.get('metadata', {}).get('title', '')
                if title not in seen_titles:
                    seen_titles.add(title)
                    combined.append(note)

            # Add semantic matches (2/3)
            for note in semantic_formatted:
                if len(combined) >= limit:
                    break
                title = note.get('metadata', {}).get('title', '')
                if title not in seen_titles:
                    seen_titles.add(title)
                    combined.append(note)

            logger.debug(f"[Obsidian] Hybrid retrieval: {len(keyword_results[:keyword_limit])} keyword + "
                        f"{len(combined) - len(keyword_results[:keyword_limit])} semantic = {len(combined)} total")
            return combined[:limit]

        except Exception as e:
            logger.warning(f"[Obsidian] Failed to retrieve notes: {e}")
            return []

    def _keyword_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search notes by keyword matching on title, tags, and content.

        Scores based on:
        - Exact title match: 1.0
        - Partial title match: 0.8
        - Tag match: 0.6
        - Content keyword match: 0.4
        """
        try:
            collection = self.chroma_store.collections.get('obsidian_notes')
            if not collection:
                return []

            # Get all documents for keyword search
            all_docs = collection.get(include=['documents', 'metadatas'])
            documents = all_docs.get('documents', []) or []
            metadatas = all_docs.get('metadatas', []) or []
            ids = all_docs.get('ids', []) or []

            # Normalize query for matching
            query_lower = query.lower()
            query_words = set(query_lower.split())

            # Score each document
            scored = []
            for i, (doc, meta) in enumerate(zip(documents, metadatas)):
                if not meta:
                    continue

                title = str(meta.get('title', '')).lower()
                tags = str(meta.get('tags', '')).lower()
                content = str(doc or '').lower()

                score = 0.0

                # Exact title match (highest priority)
                if query_lower in title or title in query_lower:
                    score = 1.0
                # Partial title match (query words in title)
                elif query_words & set(title.split()):
                    matching_words = len(query_words & set(title.split()))
                    score = 0.6 + (0.2 * min(matching_words / len(query_words), 1.0))
                # Tag match
                elif query_words & set(tags.replace(',', ' ').split()):
                    score = 0.5
                # Content keyword match
                elif query_words & set(content.split()):
                    matching_words = len(query_words & set(content.split()))
                    score = 0.2 + (0.2 * min(matching_words / max(len(query_words), 1), 1.0))

                if score > 0:
                    scored.append({
                        'content': doc,
                        'metadata': {
                            'type': 'obsidian_note',
                            'title': meta.get('title', ''),
                            'tags': meta.get('tags', ''),
                            'related_notes': meta.get('related_notes', ''),
                            'section': meta.get('section', ''),
                            'file_path': meta.get('file_path', ''),
                            'truth_score': float(meta.get('truth_score', 0.9)),
                            'timestamp': meta.get('timestamp', ''),
                        },
                        'relevance_score': score,
                        'match_type': 'keyword',
                    })

            # Sort by score descending
            scored.sort(key=lambda x: x['relevance_score'], reverse=True)
            return scored[:limit]

        except Exception as e:
            logger.warning(f"[Obsidian] Keyword search failed: {e}")
            return []

    def get_vault_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed vault.

        Returns:
            Dict with indexed_chunks, vault_path, vault_exists
        """
        try:
            # Try to get collection count
            try:
                results = self.chroma_store.query_collection(
                    'obsidian_notes',
                    query_text="",
                    n_results=1
                )
                # ChromaDB doesn't have a direct count, so we estimate
                # by checking if collection exists and has data
                count = len(results) if results else 0

                # Try to get actual count via collection
                collection = self.chroma_store.collections.get('obsidian_notes')
                if collection:
                    count = collection.count()
            except Exception:
                count = 0

            vault = Path(self.vault_path).expanduser()
            return {
                'indexed_chunks': count,
                'vault_path': str(vault),
                'vault_exists': vault.exists() and vault.is_dir(),
            }
        except Exception as e:
            logger.warning(f"[Obsidian] Failed to get stats: {e}")
            return {
                'indexed_chunks': 0,
                'vault_path': self.vault_path,
                'vault_exists': False,
            }

    def clear_index(self) -> bool:
        """
        Clear all indexed notes from ChromaDB.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete and recreate the collection
            self.chroma_store.client.delete_collection('obsidian_notes')
            self.chroma_store.collections['obsidian_notes'] = \
                self.chroma_store.client.get_or_create_collection(
                    name='obsidian_notes',
                    embedding_function=self.chroma_store.embedding_fn
                )
            logger.info("[Obsidian] Index cleared successfully")
            return True
        except Exception as e:
            logger.error(f"[Obsidian] Failed to clear index: {e}")
            return False
