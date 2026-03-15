# /knowledge/reference_docs_manager.py
"""
ReferenceDocsManager - Embed reference documents into ChromaDB for [DAEMON DOCUMENTATION] prompt section.

Module Contract:
- Purpose: Provide Daemon with self-knowledge by storing architecture docs, PROJECT_SKELETON,
           and other technical documentation that Daemon can retrieve to answer meta-questions
           about its own systems (e.g., "how does your memory scoring work?")
- Inputs:
  - upload_document(file_path: str, title: str) -> UploadResult: Index document to ChromaDB
  - get_documents(query: str, limit: int) -> List[Dict]: Retrieve relevant document chunks
  - list_documents() -> List[Dict]: List all uploaded documents
  - delete_document(title: str) -> bool: Remove a document from the index
  - sync_file(file_path: str, title: str) -> str: Sync single file using mtime ('uploaded'/'skipped'/'failed')
  - sync_directory(directory: str, file_patterns: List[str]) -> Dict: Batch sync directory with summary counts
- Outputs:
  - Embedded document chunks in reference_docs ChromaDB collection
  - Retrieved chunks with metadata (title, section, file_type, chunk_index, file_mtime)
  - Appears in prompt as [DAEMON DOCUMENTATION] section
- Side effects:
  - Writes to reference_docs ChromaDB collection
  - Logging of indexing progress
- Error handling:
  - Graceful handling of unsupported file types
  - Per-chunk error handling (one bad chunk doesn't stop indexing)

Features:
- Smart hybrid chunking: whole document if <2000 chars, else chunk by ## headers
- Supports .md, .txt files (expandable to PDF, DOCX later)
- Hybrid retrieval: 1/3 keyword + 2/3 semantic (like Obsidian notes)
- Higher priority than wiki but lower than personal notes
- Auto-seed on GUI startup: docs/ directory synced via mtime-based idempotency
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


# Configuration defaults (can be overridden via app_config)
DEFAULT_CHUNK_THRESHOLD = 2000  # Slightly larger than Obsidian notes
DEFAULT_MAX_DOCS_PROMPT = 5


@dataclass
class UploadResult:
    """Result of document upload operation."""
    success: bool = False
    title: str = ""
    total_chunks: int = 0
    file_type: str = ""
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class ReferenceDocsManager:
    """Manages reference document upload, embedding, and retrieval."""

    def __init__(self, chroma_store=None):
        """
        Initialize ReferenceDocsManager.

        Args:
            chroma_store: Optional ChromaDB store instance (lazy-loaded if None)
        """
        self._chroma_store = chroma_store

        # Load config values (with fallback defaults)
        try:
            from config.app_config import (
                REFERENCE_DOCS_CHUNK_THRESHOLD,
                REFERENCE_DOCS_MAX_PROMPT,
            )
            self.chunk_threshold = REFERENCE_DOCS_CHUNK_THRESHOLD
            self.max_docs = REFERENCE_DOCS_MAX_PROMPT
        except (ImportError, AttributeError):
            self.chunk_threshold = DEFAULT_CHUNK_THRESHOLD
            self.max_docs = DEFAULT_MAX_DOCS_PROMPT

        logger.debug(f"[RefDocs] Initialized with chunk_threshold={self.chunk_threshold}")

    @property
    def chroma_store(self):
        """Lazy-load ChromaDB store."""
        if self._chroma_store is None:
            try:
                from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
                from config.app_config import CHROMA_PATH
                self._chroma_store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
                logger.debug("[RefDocs] ChromaDB store lazy-loaded")
            except Exception as e:
                logger.error(f"[RefDocs] Failed to load ChromaDB store: {e}")
                raise
        return self._chroma_store

    def _strip_frontmatter(self, content: str) -> str:
        """Strip YAML frontmatter from markdown content."""
        if not content.startswith('---'):
            return content

        end_match = re.search(r'\n---\s*\n', content[3:])
        if end_match:
            return content[end_match.end() + 3:].strip()
        return content

    def _extract_sections(self, content: str) -> List[str]:
        """Extract section headers from content for metadata."""
        headers = re.findall(r'^##+ (.+)$', content, re.MULTILINE)
        return headers[:10]  # Cap at 10 sections for metadata

    def _chunk_by_headers(self, content: str, doc_title: str) -> List[Dict[str, Any]]:
        """
        Smart hybrid chunking: whole document if small, else chunk by ## headers.

        Args:
            content: Document content (frontmatter already stripped)
            doc_title: Title of the document for section naming

        Returns:
            List of chunk dicts with text, section, chunk_index, total_chunks
        """
        return chunk_by_headers(
            content=content,
            title=doc_title,
            chunk_threshold=self.chunk_threshold,
            include_header_in_text=True,
            min_chunk_size=50
        )

    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type from extension."""
        ext = Path(file_path).suffix.lower()
        type_map = {
            '.md': 'markdown',
            '.txt': 'text',
            '.py': 'python',
            '.js': 'javascript',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.rst': 'restructuredtext',
        }
        return type_map.get(ext, 'text')

    def upload_document(self, file_path: str, title: str = None) -> UploadResult:
        """
        Upload and embed a reference document.

        Args:
            file_path: Path to the document file
            title: Optional title (defaults to filename without extension)

        Returns:
            UploadResult with statistics about the operation
        """
        start_time = time.time()
        result = UploadResult()

        # Validate file
        path = Path(file_path).expanduser()
        if not path.exists():
            result.errors.append(f"File does not exist: {file_path}")
            logger.error(result.errors[-1])
            result.duration_seconds = time.time() - start_time
            return result

        if not path.is_file():
            result.errors.append(f"Path is not a file: {file_path}")
            logger.error(result.errors[-1])
            result.duration_seconds = time.time() - start_time
            return result

        # Set title (default to filename)
        result.title = title or path.stem
        result.file_type = self._detect_file_type(str(path))

        # Read content
        try:
            content = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                content = path.read_text(encoding='latin-1')
            except Exception as e:
                result.errors.append(f"Failed to read file: {e}")
                logger.error(result.errors[-1])
                result.duration_seconds = time.time() - start_time
                return result

        # Strip frontmatter for markdown
        if result.file_type == 'markdown':
            content = self._strip_frontmatter(content)

        if not content.strip():
            result.errors.append("File is empty after processing")
            logger.error(result.errors[-1])
            result.duration_seconds = time.time() - start_time
            return result

        # Check if document with same title already exists
        existing = self._get_document_chunks(result.title)
        if existing:
            # Delete existing to replace
            self.delete_document(result.title)
            logger.info(f"[RefDocs] Replacing existing document: {result.title}")

        # Extract section headers for metadata
        sections = self._extract_sections(content)

        # Chunk the content
        chunks = self._chunk_by_headers(content, result.title)

        # Add each chunk to ChromaDB
        for chunk in chunks:
            try:
                metadata = {
                    'type': 'reference_doc',
                    'title': result.title,
                    'file_path': str(path),
                    'file_type': result.file_type,
                    'section': chunk.get('section') or '',
                    'sections_overview': ','.join(sections[:5]) if sections else '',
                    'chunk_index': chunk['chunk_index'],
                    'total_chunks': chunk['total_chunks'],
                    'timestamp': datetime.now().isoformat(),
                    'truth_score': 0.85,  # High confidence for uploaded docs
                    'file_mtime': os.path.getmtime(str(path)),
                }

                self.chroma_store.add_to_collection(
                    'reference_docs',
                    chunk['text'],
                    metadata
                )
                result.total_chunks += 1

            except Exception as e:
                error_msg = f"Error adding chunk {chunk['chunk_index']}: {str(e)}"
                result.errors.append(error_msg)
                logger.warning(f"[RefDocs] {error_msg}")

        result.success = result.total_chunks > 0
        result.duration_seconds = time.time() - start_time

        if result.success:
            logger.info(
                f"[RefDocs] Upload complete: '{result.title}' - "
                f"{result.total_chunks} chunks, {result.file_type}, "
                f"{result.duration_seconds:.1f}s"
            )
        else:
            logger.error(f"[RefDocs] Upload failed for '{result.title}': {result.errors}")

        return result

    def upload_text(self, content: str, title: str, metadata_overrides: dict = None) -> UploadResult:
        """
        Upload text content directly (e.g., from GUI paste or conversation).

        Args:
            content: Text content to upload
            title: Title for the document
            metadata_overrides: Optional dict of metadata fields to merge into each chunk's metadata
                               (e.g., {'type': 'user_upload', 'is_image': True, 'image_path': '...'})

        Returns:
            UploadResult with statistics about the operation
        """
        start_time = time.time()
        result = UploadResult(title=title, file_type='text')

        if not content.strip():
            result.errors.append("Content is empty")
            result.duration_seconds = time.time() - start_time
            return result

        # Check if document with same title already exists
        existing = self._get_document_chunks(title)
        if existing:
            self.delete_document(title)
            logger.info(f"[RefDocs] Replacing existing document: {title}")

        # Extract section headers for metadata
        sections = self._extract_sections(content)

        # Chunk the content
        chunks = self._chunk_by_headers(content, title)

        # Add each chunk to ChromaDB
        for chunk in chunks:
            try:
                metadata = {
                    'type': 'reference_doc',
                    'title': title,
                    'file_path': '',  # No file for text upload
                    'file_type': 'text',
                    'section': chunk.get('section') or '',
                    'sections_overview': ','.join(sections[:5]) if sections else '',
                    'chunk_index': chunk['chunk_index'],
                    'total_chunks': chunk['total_chunks'],
                    'timestamp': datetime.now().isoformat(),
                    'truth_score': 0.85,
                }

                # Apply caller-provided metadata overrides (e.g., type='user_upload')
                if metadata_overrides:
                    metadata.update(metadata_overrides)

                self.chroma_store.add_to_collection(
                    'reference_docs',
                    chunk['text'],
                    metadata
                )
                result.total_chunks += 1

            except Exception as e:
                error_msg = f"Error adding chunk {chunk['chunk_index']}: {str(e)}"
                result.errors.append(error_msg)
                logger.warning(f"[RefDocs] {error_msg}")

        result.success = result.total_chunks > 0
        result.duration_seconds = time.time() - start_time

        if result.success:
            logger.info(
                f"[RefDocs] Text upload complete: '{title}' - "
                f"{result.total_chunks} chunks, {result.duration_seconds:.1f}s"
            )

        return result

    def _get_document_chunks(self, title: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document by title."""
        try:
            collection = self.chroma_store.collections.get('reference_docs')
            if not collection:
                return []

            all_docs = collection.get(include=['documents', 'metadatas'])
            documents = all_docs.get('documents', []) or []
            metadatas = all_docs.get('metadatas', []) or []
            ids = all_docs.get('ids', []) or []

            chunks = []
            for i, (doc, meta, doc_id) in enumerate(zip(documents, metadatas, ids)):
                if meta and meta.get('title') == title:
                    chunks.append({
                        'id': doc_id,
                        'content': doc,
                        'metadata': meta,
                    })

            return chunks

        except Exception as e:
            logger.warning(f"[RefDocs] Failed to get document chunks: {e}")
            return []

    def _get_document_mtime(self, title: str) -> Optional[float]:
        """Get stored file_mtime for a document by title. Returns None if not found."""
        chunks = self._get_document_chunks(title)
        if chunks:
            return chunks[0].get('metadata', {}).get('file_mtime')
        return None

    def sync_file(self, file_path: str, title: str = None) -> str:
        """
        Sync a single file. Returns 'uploaded', 'skipped', or 'failed'.
        Uses mtime comparison to skip unchanged files.
        """
        path = Path(file_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            return 'failed'
        title = title or path.stem
        current_mtime = os.path.getmtime(str(path))
        stored_mtime = self._get_document_mtime(title)
        if stored_mtime and current_mtime <= stored_mtime:
            return 'skipped'
        result = self.upload_document(str(path), title)
        return 'uploaded' if result.success else 'failed'

    def sync_directory(self, directory: str, file_patterns: List[str] = None) -> Dict[str, Any]:
        """
        Sync all matching files from a directory into reference_docs.
        Uses file mtime for idempotency — skips unchanged files.

        Args:
            directory: Path to directory to scan
            file_patterns: Glob patterns to match (default: ['*.md'])

        Returns:
            dict with 'uploaded', 'skipped', 'failed' counts and 'details' list
        """
        file_patterns = file_patterns or ['*.md']
        dir_path = Path(directory).expanduser().resolve()
        summary = {'uploaded': 0, 'skipped': 0, 'failed': 0, 'details': []}

        if not dir_path.is_dir():
            logger.warning(f"[RefDocs] sync_directory: not a directory: {directory}")
            return summary

        matched_files = []
        for pattern in file_patterns:
            matched_files.extend(dir_path.glob(pattern))

        for fpath in sorted(set(matched_files)):
            status = self.sync_file(str(fpath))
            summary[status] += 1
            summary['details'].append({'file': fpath.name, 'status': status})

        logger.info(
            f"[RefDocs] sync_directory '{directory}': "
            f"{summary['uploaded']} uploaded, {summary['skipped']} unchanged, "
            f"{summary['failed']} failed"
        )
        return summary

    async def get_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks using hybrid search: 1/3 keyword + 2/3 semantic.

        Args:
            query: Search query
            limit: Maximum chunks to return

        Returns:
            List of chunk dicts with content, metadata, and relevance_score
        """
        try:
            # Calculate split: 1/3 keyword, 2/3 semantic
            keyword_limit = max(1, limit // 3)
            semantic_limit = limit - keyword_limit

            # 1. KEYWORD SEARCH
            keyword_results = self._keyword_search(query, keyword_limit * 3)

            # 2. SEMANTIC SEARCH
            semantic_results = self.chroma_store.query_collection(
                'reference_docs',
                query,
                n_results=semantic_limit * 2
            )

            # Format semantic results
            semantic_formatted = []
            for r in semantic_results:
                meta = r.get('metadata', {})
                semantic_formatted.append({
                    'content': r.get('content', ''),
                    'metadata': {
                        'type': meta.get('type', 'reference_doc'),
                        'title': meta.get('title', ''),
                        'section': meta.get('section', ''),
                        'file_type': meta.get('file_type', ''),
                        'file_path': meta.get('file_path', ''),
                        'truth_score': float(meta.get('truth_score', 0.85)),
                        'timestamp': meta.get('timestamp', ''),
                    },
                    'relevance_score': r.get('relevance_score', 0.0),
                    'match_type': 'semantic',
                })

            # 3. COMBINE with deduplication
            seen = set()
            combined = []

            # Add keyword matches first (1/3)
            for doc in keyword_results[:keyword_limit]:
                title = doc.get('metadata', {}).get('title', '')
                section = doc.get('metadata', {}).get('section', '')
                key = f"{title}|{section}"
                if key not in seen:
                    seen.add(key)
                    combined.append(doc)

            # Add semantic matches (2/3)
            for doc in semantic_formatted:
                if len(combined) >= limit:
                    break
                title = doc.get('metadata', {}).get('title', '')
                section = doc.get('metadata', {}).get('section', '')
                key = f"{title}|{section}"
                if key not in seen:
                    seen.add(key)
                    combined.append(doc)

            logger.debug(f"[RefDocs] Hybrid retrieval: {len(keyword_results[:keyword_limit])} keyword + "
                        f"{len(combined) - len(keyword_results[:keyword_limit])} semantic = {len(combined)} total")
            return combined[:limit]

        except Exception as e:
            logger.warning(f"[RefDocs] Failed to retrieve documents: {e}")
            return []

    def _keyword_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search documents by keyword matching on title, section, content."""
        try:
            collection = self.chroma_store.collections.get('reference_docs')
            if not collection:
                return []

            all_docs = collection.get(include=['documents', 'metadatas'])
            documents = all_docs.get('documents', []) or []
            metadatas = all_docs.get('metadatas', []) or []

            query_lower = query.lower()
            query_words = set(query_lower.split())

            scored = []
            for doc, meta in zip(documents, metadatas):
                if not meta:
                    continue

                title = str(meta.get('title', '')).lower()
                section = str(meta.get('section', '')).lower()
                content = str(doc or '').lower()

                score = 0.0

                # Title match (highest priority)
                if query_lower in title or title in query_lower:
                    score = 1.0
                # Section header match
                elif query_lower in section or section in query_lower:
                    score = 0.9
                # Partial title match
                elif query_words & set(title.split()):
                    matching = len(query_words & set(title.split()))
                    score = 0.6 + (0.2 * min(matching / len(query_words), 1.0))
                # Section keyword match
                elif query_words & set(section.split()):
                    score = 0.5
                # Content keyword match
                elif query_words & set(content.split()):
                    matching = len(query_words & set(content.split()))
                    score = 0.2 + (0.2 * min(matching / max(len(query_words), 1), 1.0))

                if score > 0:
                    scored.append({
                        'content': doc,
                        'metadata': {
                            'type': meta.get('type', 'reference_doc'),
                            'title': meta.get('title', ''),
                            'section': meta.get('section', ''),
                            'file_type': meta.get('file_type', ''),
                            'file_path': meta.get('file_path', ''),
                            'truth_score': float(meta.get('truth_score', 0.85)),
                            'timestamp': meta.get('timestamp', ''),
                        },
                        'relevance_score': score,
                        'match_type': 'keyword',
                    })

            scored.sort(key=lambda x: x['relevance_score'], reverse=True)
            return scored[:limit]

        except Exception as e:
            logger.warning(f"[RefDocs] Keyword search failed: {e}")
            return []

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all uploaded documents.

        Returns:
            List of dicts with title, file_type, chunk_count, upload_time
        """
        try:
            collection = self.chroma_store.collections.get('reference_docs')
            if not collection:
                return []

            all_docs = collection.get(include=['metadatas'])
            metadatas = all_docs.get('metadatas', []) or []

            # Group by title
            docs_by_title = {}
            for meta in metadatas:
                if not meta:
                    continue
                title = meta.get('title', 'Untitled')
                if title not in docs_by_title:
                    docs_by_title[title] = {
                        'title': title,
                        'file_type': meta.get('file_type', 'unknown'),
                        'file_path': meta.get('file_path', ''),
                        'chunk_count': 0,
                        'upload_time': meta.get('timestamp', ''),
                    }
                docs_by_title[title]['chunk_count'] += 1

            return list(docs_by_title.values())

        except Exception as e:
            logger.warning(f"[RefDocs] Failed to list documents: {e}")
            return []

    def delete_document(self, title: str) -> bool:
        """
        Delete a document and all its chunks from the index.

        Args:
            title: Title of the document to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            collection = self.chroma_store.collections.get('reference_docs')
            if not collection:
                return False

            # Find all chunks with this title
            chunks = self._get_document_chunks(title)
            if not chunks:
                logger.warning(f"[RefDocs] No document found with title: {title}")
                return False

            # Delete each chunk by ID
            ids_to_delete = [c['id'] for c in chunks]
            collection.delete(ids=ids_to_delete)

            logger.info(f"[RefDocs] Deleted document '{title}' ({len(ids_to_delete)} chunks)")
            return True

        except Exception as e:
            logger.error(f"[RefDocs] Failed to delete document '{title}': {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the reference docs collection."""
        try:
            collection = self.chroma_store.collections.get('reference_docs')
            count = collection.count() if collection else 0
            docs = self.list_documents()

            return {
                'total_chunks': count,
                'document_count': len(docs),
                'documents': docs,
            }

        except Exception as e:
            logger.warning(f"[RefDocs] Failed to get stats: {e}")
            return {
                'total_chunks': 0,
                'document_count': 0,
                'documents': [],
            }

    def clear_all(self) -> bool:
        """Clear all reference documents from the index."""
        try:
            self.chroma_store.client.delete_collection('reference_docs')
            self.chroma_store.collections['reference_docs'] = \
                self.chroma_store.client.get_or_create_collection(
                    name='reference_docs',
                    embedding_function=self.chroma_store.embedding_fn
                )
            logger.info("[RefDocs] All documents cleared")
            return True
        except Exception as e:
            logger.error(f"[RefDocs] Failed to clear documents: {e}")
            return False
