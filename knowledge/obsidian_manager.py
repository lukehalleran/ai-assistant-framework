# /knowledge/obsidian_manager.py
"""
ObsidianManager - Parse and embed Obsidian vault markdown files into ChromaDB.

Module Contract:
- Purpose: Integrate user's personal notes from Obsidian vault into the RAG pipeline
  with multimodal image support for vision-capable models.
- Inputs:
  - embed_vault(force_reindex: bool) -> EmbedResult: Index vault to ChromaDB
  - get_notes(query: str, limit: int, include_images: bool, max_images_per_note: int) -> List[Dict]:
    Retrieve relevant notes with optional image loading for multimodal models
- Outputs:
  - Embedded notes in obsidian_notes ChromaDB collection
  - Retrieved notes with metadata (tags, links, title, path, images)
  - When include_images=True: notes include 'image_data' with base64-encoded images
- Side effects:
  - Writes to obsidian_notes ChromaDB collection
  - Reads image files from vault when include_images=True
  - Logging of indexing progress
- Error handling:
  - Graceful degradation if vault doesn't exist
  - Per-file error handling (one bad file doesn't stop indexing)
  - Missing images logged but don't fail retrieval

Features:
- Smart hybrid chunking: whole notes if <1500 chars, else chunk by ## headers
- Preserves #tags as metadata for filtering
- Extracts [[wiki links]] as related_notes references
- Extracts ![[images]] per-chunk for multimodal support [NEW 2026-01-30]
- Strips YAML frontmatter while keeping content clean
- Keyword search includes file_path for folder-based topic matching [NEW 2026-01-30]
- Rare-proper-noun floor [NEW 2026-08-26]: a word-boundary hit on a name-shaped query
  token (utils.query_checker.extract_rare_proper_nouns) floors the keyword score at
  0.75 — whole-query word-set scoring weighed "Morgan" the same as "not", so the
  "Advisor: Morgan Reeves" note lost to date-titled daily notes on generic overlap
- Image loading with resolution: same folder → parent → attachments → vault root → global search
"""

import os
import re
import logging
import hashlib
import base64
from pathlib import Path

from utils.text_chunking import chunk_by_headers
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time

logger = logging.getLogger(__name__)


@dataclass
class EmbedResult:
    """Result of vault embedding operation."""
    total_files: int = 0
    embedded_files: int = 0
    updated_files: int = 0
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
            self.vault_path = vault_path or os.path.expanduser("~/Documents/Notes")
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

        Handles edge cases:
        - Windows line endings (\\r\\n)
        - UTF-8 BOM at start of file
        - Malformed frontmatter (returns original content)
        """
        # Strip UTF-8 BOM if present
        if content.startswith('\ufeff'):
            content = content[1:]

        # Normalize line endings to Unix style for consistent parsing
        content = content.replace('\r\n', '\n').replace('\r', '\n')

        # Check for frontmatter start
        if not content.startswith('---'):
            return content

        # Find the closing ---
        # Use more flexible regex that handles various spacing
        end_match = re.search(r'\n---[ \t]*\n', content[3:])
        if end_match:
            return content[end_match.end() + 3:].strip()

        # Alternative: try to find closing --- at line start
        lines = content.split('\n')
        for i, line in enumerate(lines[1:], start=1):  # Skip first ---
            if line.strip() == '---':
                # Found closing ---, return content after it
                return '\n'.join(lines[i+1:]).strip()

        return content

    def _extract_frontmatter_metadata(self, content: str) -> Dict[str, Any]:
        """Read a small, dependency-free subset of YAML frontmatter.

        Provenance must survive embedding: daily notes generated by Daemon are
        summaries, not first-person user records.  Keep this deliberately
        conservative rather than attempting to become a general YAML parser.
        """
        if content.startswith('\ufeff'):
            content = content[1:]
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        if not content.startswith('---'):
            return {}
        end = content.find('\n---', 3)
        if end < 0:
            return {}
        metadata: Dict[str, Any] = {}
        for line in content[3:end].splitlines():
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            key, value = key.strip(), value.strip().strip('"')
            if key in {'author', 'source_type', 'derived_from', 'date', 'generated'}:
                metadata[key] = value
        return metadata

    _DATE_LINE_RE = re.compile(r'^date:\s*"?(\d{4}-\d{2}-\d{2})"?\s*$', re.MULTILINE)
    _ISO_IN_NAME_RE = re.compile(r'(\d{4})-(\d{2})-(\d{2})')
    _ISO_IN_HEADING_RE = re.compile(r'^#{1,6}[^\n]*?(\d{4})-(\d{2})-(\d{2})', re.MULTILINE)
    _SHORT_DATE_NAME_RE = re.compile(r'\b(\d{2})[ ._-](\d{2})[ ._-](\d{2})\b')

    def _infer_note_date(self, content: str, file_stem: str,
                         frontmatter: Dict[str, Any]) -> str:
        """Infer a note's content date from standard daily-note conventions.

        First-block frontmatter is authoritative, but legacy daily notes often
        carry a dual-frontmatter template (prose block first, `date:` in a
        LATER block) or encode the date only in the filename/heading — a
        note-date left empty makes the note invisible to windowed longitudinal
        scans (2026-08-31: every pre-2026 daily lost its date this way).
        Ladder, most→least authoritative; every rung validates month/day.
        """
        def _valid(year: str, month: str, day: str) -> str:
            try:
                return datetime(int(year), int(month), int(day)).date().isoformat()
            except ValueError:
                return ''

        declared = str(frontmatter.get('date') or '').strip()
        if declared:
            return declared
        # Line-anchored `date: YYYY-MM-DD` anywhere (dual-frontmatter shape)
        match = self._DATE_LINE_RE.search(content)
        if match:
            parts = match.group(1).split('-')
            date = _valid(*parts)
            if date:
                return date
        # ISO date in the filename stem ("2024-11-19.md")
        match = self._ISO_IN_NAME_RE.search(file_stem)
        if match:
            date = _valid(*match.groups())
            if date:
                return date
        # ISO date in a heading line ("# 📅 2025-03-01")
        match = self._ISO_IN_HEADING_RE.search(content)
        if match:
            date = _valid(*match.groups())
            if date:
                return date
        # Short-date filename ("02 19 25 Daily Note"): US MM DD YY, accepted
        # only when it validates as a real calendar date
        match = self._SHORT_DATE_NAME_RE.search(file_stem)
        if match:
            month, day, year = match.groups()
            date = _valid(f'20{year}', month, day)
            if date:
                return date
        return ''

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

    def _extract_images(self, content: str) -> List[str]:
        """
        Extract ![[image.png]] references from content.

        Handles:
            ![[Screenshot.png]]
            ![[image.png|alt text]]
        """
        # Match ![[image]] or ![[image|alias]]
        images = re.findall(r'!\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', content)
        return list(set(images))

    def _strip_inline_tags(self, content: str) -> str:
        """
        Strip inline #tags from content for cleaner embedding.

        Preserves ## headers while removing standalone #tag patterns.
        Tags are extracted separately as metadata.
        """
        # Remove #tag patterns (but not ## headers)
        # First pass: properly spaced tags (preceded by whitespace or start of line)
        cleaned = re.sub(r'(?<![#])(?:^|\s)#([a-zA-Z][a-zA-Z0-9_/-]*)\s*', ' ', content, flags=re.MULTILINE)

        # Second pass: tags directly attached to words (like "how#tag")
        # This is technically invalid Obsidian syntax but common in notes
        cleaned = re.sub(r'(?<![#])#([a-zA-Z][a-zA-Z0-9_/-]*)(?=\s|$|[^\w])', '', cleaned)

        # Clean up any resulting double/triple spaces
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        # Clean up empty lines
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        return cleaned.strip()

    def _resolve_image_path(self, image_name: str, note_path: str) -> Optional[Path]:
        """
        Resolve an image reference to its actual file path in the vault.

        Obsidian images can be in:
        1. Same folder as the note
        2. A subfolder relative to the note
        3. An attachments folder (common patterns: attachments/, assets/, images/)
        4. The vault root
        5. Anywhere in the vault (Obsidian searches globally)

        Args:
            image_name: The image filename from ![[image.png]]
            note_path: The relative path of the note containing the reference

        Returns:
            Path to the image file if found, None otherwise
        """
        vault = Path(self.vault_path).expanduser()
        note_dir = vault / Path(note_path).parent

        # Common attachment folder names
        attachment_folders = ['attachments', 'assets', 'images', 'media', 'files', 'Attachments', 'Assets', 'Images']

        # Search locations in order of likelihood
        search_paths = [
            # 1. Same folder as note
            note_dir / image_name,
            # 2. Note's parent folder (for nested structures)
            note_dir.parent / image_name,
        ]

        # 3. Attachment folders relative to note
        for folder in attachment_folders:
            search_paths.append(note_dir / folder / image_name)

        # 4. Attachment folders at vault root
        for folder in attachment_folders:
            search_paths.append(vault / folder / image_name)

        # 5. Vault root
        search_paths.append(vault / image_name)

        # Check each location
        for path in search_paths:
            if path.exists() and path.is_file():
                return path

        # 6. Last resort: search entire vault (expensive but thorough)
        try:
            matches = list(vault.rglob(image_name))
            if matches:
                return matches[0]  # Return first match
        except Exception as e:
            logger.debug(f"[Obsidian] Vault search for {image_name} failed: {e}")

        return None

    def _load_image_as_base64(self, image_path: Path, max_size_mb: float = 5.0) -> Optional[Dict[str, str]]:
        """
        Load an image file and convert to base64 for multimodal models.

        Args:
            image_path: Path to the image file
            max_size_mb: Maximum file size to load (default 5MB)

        Returns:
            Dict with 'data' (base64), 'media_type', and 'filename', or None if failed
        """
        try:
            # Check file size
            size_mb = image_path.stat().st_size / (1024 * 1024)
            if size_mb > max_size_mb:
                logger.warning(f"[Obsidian] Image too large ({size_mb:.1f}MB > {max_size_mb}MB): {image_path.name}")
                return None

            # Determine media type from extension
            ext = image_path.suffix.lower()
            media_types = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
                '.svg': 'image/svg+xml',
            }
            media_type = media_types.get(ext)
            if not media_type:
                logger.debug(f"[Obsidian] Unsupported image format: {ext}")
                return None

            # Read and encode
            with open(image_path, 'rb') as f:
                image_data = f.read()

            return {
                'data': base64.b64encode(image_data).decode('utf-8'),
                'media_type': media_type,
                'filename': image_path.name,
                'size_bytes': len(image_data),
            }

        except Exception as e:
            logger.warning(f"[Obsidian] Failed to load image {image_path}: {e}")
            return None

    def load_images_for_chunk(
        self,
        image_names: List[str],
        note_path: str,
        max_images: int = 5,
        max_total_mb: float = 10.0
    ) -> List[Dict[str, str]]:
        """
        Load actual image data for a list of image references.

        Args:
            image_names: List of image filenames from the chunk
            note_path: Relative path of the source note
            max_images: Maximum number of images to load
            max_total_mb: Maximum total size of all images

        Returns:
            List of image dicts with base64 data, ready for multimodal models
        """
        loaded_images = []
        total_size = 0
        max_total_bytes = max_total_mb * 1024 * 1024

        for image_name in image_names[:max_images]:
            # Resolve path
            image_path = self._resolve_image_path(image_name, note_path)
            if not image_path:
                logger.debug(f"[Obsidian] Could not find image: {image_name}")
                continue

            # Load image
            image_data = self._load_image_as_base64(image_path)
            if not image_data:
                continue

            # Check total size limit
            if total_size + image_data['size_bytes'] > max_total_bytes:
                logger.debug(f"[Obsidian] Total image size limit reached, skipping remaining images")
                break

            total_size += image_data['size_bytes']
            loaded_images.append(image_data)

        if loaded_images:
            logger.info(f"[Obsidian] Loaded {len(loaded_images)}/{len(image_names)} images ({total_size/1024:.1f}KB)")

        return loaded_images

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

    def _delete_file_chunks(self, file_path: str) -> int:
        """
        Delete all chunks for a given file_path from ChromaDB.

        Args:
            file_path: The relative file path stored in chunk metadata.

        Returns:
            Number of chunks deleted.
        """
        try:
            collection = self.chroma_store._get_collection('obsidian_notes')
            if not collection:
                return 0
            docs = collection.get(where={"file_path": file_path}, include=[])
            ids = docs.get('ids', []) or []
            if ids:
                collection.delete(ids=ids)
                logger.debug(f"[Obsidian] Deleted {len(ids)} old chunks for {file_path}")
            return len(ids)
        except Exception as e:
            logger.warning(f"[Obsidian] Failed to delete chunks for {file_path}: {e}")
            return 0

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

        # Force reindex: clear collection first to avoid duplicate chunks
        if force_reindex:
            self.clear_index()

        # Build mtime map from existing chunks: {file_path: stored_mtime_or_None}
        existing_mtimes: Dict[str, Optional[float]] = {}
        if not force_reindex:
            try:
                collection = self.chroma_store._get_collection('obsidian_notes')
                if collection:
                    all_docs = collection.get(include=['metadatas'])
                    metadatas = all_docs.get('metadatas', []) or []
                    for meta in metadatas:
                        if meta and isinstance(meta, dict):
                            path = meta.get('file_path', '')
                            if path and path not in existing_mtimes:
                                # Legacy chunks without file_mtime get None
                                existing_mtimes[path] = meta.get('file_mtime')
                    logger.info(f"[Obsidian] Found {len(existing_mtimes)} already indexed file paths")
            except Exception as e:
                logger.warning(f"[Obsidian] Could not check existing files: {e}")

        # Process each file
        for md_file in md_files:
            try:
                rel_path = str(md_file.relative_to(vault))

                # Check mtime for change detection
                current_mtime = os.path.getmtime(md_file)
                is_update = False

                if rel_path in existing_mtimes:
                    stored_mtime = existing_mtimes[rel_path]
                    if stored_mtime is not None and current_mtime <= stored_mtime:
                        # File unchanged since last embed
                        result.skipped_files += 1
                        continue
                    # File changed (or legacy chunk without mtime) — delete old chunks and re-embed
                    self._delete_file_chunks(rel_path)
                    is_update = True

                # Read file content
                try:
                    content = md_file.read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    # Try with latin-1 as fallback
                    content = md_file.read_text(encoding='latin-1')

                # Preserve provenance before stripping frontmatter.
                frontmatter = self._extract_frontmatter_metadata(content)
                # Strip frontmatter
                content = self._strip_frontmatter(content)

                # Skip empty files
                if not content.strip():
                    result.skipped_files += 1
                    continue

                # Extract metadata (before stripping tags)
                tags = self._extract_tags(content)
                wiki_links = self._extract_wiki_links(content)
                all_images = self._extract_images(content)  # All images in note (for reference)
                note_title = md_file.stem

                # Strip inline tags from content for cleaner embedding
                # Tags are preserved as metadata for filtering
                clean_content = self._strip_inline_tags(content)

                # Chunk the cleaned content
                chunks = self._chunk_by_headers(clean_content, note_title)

                # Batch add all chunks to ChromaDB (single embedding pass + disk write per note)
                now = datetime.now().isoformat()
                # Prefer the note's declared event date for longitudinal
                # retrieval.  Indexing time is not event time; using it would
                # move old daily notes into the current phase.
                note_date = self._infer_note_date(
                    content, os.path.splitext(os.path.basename(md_file))[0],
                    frontmatter,
                )
                event_timestamp = note_date or now
                batch_texts = []
                batch_metas = []
                for chunk in chunks:
                    chunk_images = self._extract_images(chunk['text'])
                    batch_texts.append(chunk['text'])
                    batch_metas.append({
                        'type': 'obsidian_note',
                        'title': note_title,
                        'file_path': rel_path,
                        'file_mtime': current_mtime,
                        'tags': ','.join(tags) if tags else '',
                        'related_notes': ','.join(wiki_links) if wiki_links else '',
                        'images': ','.join(chunk_images) if chunk_images else '',
                        'note_image_count': len(all_images),
                        'section': chunk.get('section') or '',
                        'chunk_index': chunk['chunk_index'],
                        'total_chunks': chunk['total_chunks'],
                        'timestamp': event_timestamp,
                        'truth_score': 0.9,
                        'author': frontmatter.get('author', ''),
                        'source_type': frontmatter.get('source_type', ''),
                        'derived_from': frontmatter.get('derived_from', ''),
                        'note_date': note_date,
                        'generated_at': frontmatter.get('generated', ''),
                    })

                if hasattr(self.chroma_store, 'add_batch_to_collection'):
                    self.chroma_store.add_batch_to_collection('obsidian_notes', batch_texts, batch_metas)
                else:
                    for t, m in zip(batch_texts, batch_metas):
                        self.chroma_store.add_to_collection('obsidian_notes', t, m)
                result.total_chunks += len(chunks)

                if is_update:
                    result.updated_files += 1
                else:
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
            f"[Obsidian] Embedding complete: {result.embedded_files} new, "
            f"{result.updated_files} updated, {result.total_chunks} chunks, "
            f"{result.skipped_files} skipped, {len(result.errors)} errors, "
            f"{result.duration_seconds:.1f}s"
        )
        return result

    async def get_notes(
        self,
        query: str,
        limit: int = 10,
        include_images: bool = False,
        max_images_per_note: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant notes using hybrid search: 1/3 keyword + 2/3 semantic.

        Args:
            query: Search query
            limit: Maximum notes to return
            include_images: If True, load actual image data for multimodal models
            max_images_per_note: Maximum images to load per note chunk

        Returns:
            List of note dicts with content, metadata, relevance_score, and optionally 'images' data
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
                        'images': meta.get('images', ''),  # Images in this chunk
                        'note_image_count': meta.get('note_image_count', 0),  # Total in note
                        'section': meta.get('section', ''),
                        'file_path': meta.get('file_path', ''),
                        'truth_score': float(meta.get('truth_score', 0.9)),
                        'timestamp': meta.get('timestamp', ''),
                        'note_date': meta.get('note_date', ''),
                        'generated_at': meta.get('generated_at', ''),
                        'date_basis': meta.get('date_basis', 'content' if meta.get('note_date') else 'index-time'),
                        'author': meta.get('author', ''),
                        'source_type': meta.get('source_type', ''),
                        'derived_from': meta.get('derived_from', ''),
                    },
                    'relevance_score': r.get('relevance_score', 0.0),
                    'match_type': 'semantic',
                })

            # 3. COMBINE with deduplication (keyword first for priority)
            # Use title + section as key to allow different sections of same file
            seen_keys = set()
            combined = []

            def get_dedup_key(note):
                meta = note.get('metadata', {})
                title = meta.get('title', '')
                section = meta.get('section', '')
                return f"{title}::{section}"

            # Add keyword matches first (1/3)
            for note in keyword_results[:keyword_limit]:
                key = get_dedup_key(note)
                if key not in seen_keys:
                    seen_keys.add(key)
                    combined.append(note)

            # Add semantic matches (2/3)
            for note in semantic_formatted:
                if len(combined) >= limit:
                    break
                key = get_dedup_key(note)
                if key not in seen_keys:
                    seen_keys.add(key)
                    combined.append(note)

            logger.debug(f"[Obsidian] Hybrid retrieval: {len(keyword_results[:keyword_limit])} keyword + "
                        f"{len(combined) - len(keyword_results[:keyword_limit])} semantic = {len(combined)} total")

            final_results = combined[:limit]

            # 4. LOAD IMAGES if requested (for multimodal models)
            if include_images:
                for note in final_results:
                    meta = note.get('metadata', {})
                    image_str = meta.get('images', '')
                    file_path = meta.get('file_path', '')

                    if image_str and file_path:
                        image_names = [img.strip() for img in image_str.split(',') if img.strip()]
                        if image_names:
                            loaded = self.load_images_for_chunk(
                                image_names,
                                file_path,
                                max_images=max_images_per_note
                            )
                            note['image_data'] = loaded  # Add actual image data
                        else:
                            note['image_data'] = []
                    else:
                        note['image_data'] = []

            return final_results

        except Exception as e:
            logger.warning(f"[Obsidian] Failed to retrieve notes: {e}")
            return []

    def _keyword_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search notes by keyword matching on title, tags, section, and content.

        Scores based on:
        - Exact phrase in title: 1.0
        - Title starts with query: 0.95
        - All query words in title: 0.9
        - Exact phrase in section: 0.85
        - Partial title/section match: 0.6-0.8
        - Tag match: 0.5
        - Content keyword match: 0.2-0.4
        """
        try:
            collection = self.chroma_store._get_collection('obsidian_notes')
            if not collection:
                return []

            # Get all documents for keyword search
            all_docs = collection.get(include=['documents', 'metadatas'])
            documents = all_docs.get('documents', []) or []
            metadatas = all_docs.get('metadatas', []) or []
            ids = all_docs.get('ids', []) or []

            # Normalize query for matching
            query_lower = query.lower().strip()
            query_words = set(query_lower.split())

            # Rare-proper-noun floor (2026-08-26): the word-set scoring below
            # weighs every query word equally, so "Morgan" counted the same as
            # "not" and the "Advisor: Morgan Reeves" note lost to daily
            # notes on generic overlap. A word-boundary hit on a name-shaped
            # query token floors the score at 0.75 — above generic content
            # overlap, below exact-title matches.
            proper_noun_pats = []
            try:
                import re as _re
                from utils.query_checker import extract_rare_proper_nouns
                proper_noun_pats = [
                    _re.compile(r"\b" + _re.escape(t) + r"\b", _re.IGNORECASE)
                    for t in extract_rare_proper_nouns(query)
                ]
            except Exception:
                proper_noun_pats = []

            # Score each document
            scored = []
            for i, (doc, meta) in enumerate(zip(documents, metadatas)):
                if not meta:
                    continue

                title = str(meta.get('title', '')).lower()
                section = str(meta.get('section', '')).lower()
                tags = str(meta.get('tags', '')).lower()
                file_path = str(meta.get('file_path', '')).lower()
                content = str(doc or '').lower()

                score = 0.0
                title_words = set(title.split())
                section_words = set(section.split())
                # Include file path components for matching (folder names often indicate topic)
                path_words = set(file_path.replace('/', ' ').replace('_', ' ').replace('-', ' ').split())

                # Combined title + path words for broader matching
                title_path_words = title_words | path_words

                # Exact phrase match in title (highest priority)
                if query_lower in title:
                    score = 1.0
                # Title starts with query phrase
                elif title.startswith(query_lower):
                    score = 0.95
                # All query words present in title
                elif query_words and query_words <= title_words:
                    score = 0.9
                # All query words present in title + file path (e.g., "ISYE 6501" in path + "Week 2" in title)
                elif query_words and query_words <= title_path_words:
                    score = 0.88
                # Exact phrase match in section
                elif query_lower in section:
                    score = 0.85
                # All query words present in section
                elif query_words and query_words <= section_words:
                    score = 0.8
                # Partial title+path match (some query words in title or path)
                elif query_words & title_path_words:
                    matching_words = len(query_words & title_path_words)
                    score = 0.6 + (0.25 * min(matching_words / len(query_words), 1.0))
                # Partial title match only (some query words in title)
                elif query_words & title_words:
                    matching_words = len(query_words & title_words)
                    score = 0.6 + (0.2 * min(matching_words / len(query_words), 1.0))
                # Partial section match
                elif query_words & section_words:
                    matching_words = len(query_words & section_words)
                    score = 0.5 + (0.2 * min(matching_words / len(query_words), 1.0))
                # Tag match
                elif query_words & set(tags.replace(',', ' ').split()):
                    score = 0.5
                # Content keyword match (check for exact phrase first)
                elif query_lower in content:
                    score = 0.45
                elif query_words & set(content.split()):
                    matching_words = len(query_words & set(content.split()))
                    score = 0.2 + (0.2 * min(matching_words / max(len(query_words), 1), 1.0))

                # Proper-noun floor: an exact rare-name hit anywhere in the
                # chunk outranks any generic word-overlap score.
                if proper_noun_pats:
                    haystack = f"{title}\n{section}\n{content}"
                    if any(p.search(haystack) for p in proper_noun_pats):
                        score = max(score, 0.75)

                if score > 0:
                    scored.append({
                        'content': doc,
                        'metadata': {
                            'type': 'obsidian_note',
                            'title': meta.get('title', ''),
                            'tags': meta.get('tags', ''),
                            'related_notes': meta.get('related_notes', ''),
                            'images': meta.get('images', ''),  # Images in this chunk
                            'note_image_count': meta.get('note_image_count', 0),  # Total in note
                            'section': meta.get('section', ''),
                            'file_path': meta.get('file_path', ''),
                            'truth_score': float(meta.get('truth_score', 0.9)),
                            'timestamp': meta.get('timestamp', ''),
                            'note_date': meta.get('note_date', ''),
                            'generated_at': meta.get('generated_at', ''),
                            'date_basis': meta.get('date_basis', 'content' if meta.get('note_date') else 'index-time'),
                            'author': meta.get('author', ''),
                            'source_type': meta.get('source_type', ''),
                            'derived_from': meta.get('derived_from', ''),
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
                collection = self.chroma_store._get_collection('obsidian_notes')
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
