"""
Shared text chunking utilities.

Module Contract:
- Purpose: Common text chunking for knowledge managers (Obsidian, Reference Docs, etc.)
- Functions:
  - chunk_by_headers(content, title, chunk_threshold, ...) -> List[Dict]: Split by ## headers,
    falls back to chunk_by_size() for large documents with no headers
  - chunk_by_size(content, title, chunk_size, overlap) -> List[Dict]: Fixed-size paragraph-boundary
    chunking with overlap, used as fallback when no ## headers exist
- Output: List of dicts with {text, section, chunk_index, total_chunks}
- Dependencies: re (stdlib only)
"""

import re
from typing import List, Dict, Any


def chunk_by_size(
    content: str,
    title: str,
    chunk_size: int = 2000,
    overlap: int = 200,
) -> List[Dict[str, Any]]:
    """
    Fixed-size chunking with paragraph-boundary splitting and overlap.

    Used as fallback when documents have no ## headers (e.g. plain-text PDFs).

    Args:
        content: Document text
        title: Document title for section naming
        chunk_size: Target characters per chunk
        overlap: Character overlap between chunks

    Returns:
        List of chunk dicts with text, section, chunk_index, total_chunks
    """
    if len(content) <= chunk_size:
        return [{
            'text': content,
            'section': None,
            'chunk_index': 0,
            'total_chunks': 1
        }]

    # Split into paragraphs (double-newline), fall back to single newlines
    paragraphs = re.split(r'\n\n+', content)
    if len(paragraphs) <= 1:
        paragraphs = content.split('\n')

    chunks = []
    current_parts: List[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        # If a single paragraph exceeds chunk_size, split it by sentences/words
        if para_len > chunk_size and not current_parts:
            # Just add it as its own chunk — don't lose content
            chunks.append(para)
            continue

        if current_len + para_len + 2 > chunk_size and current_parts:
            # Flush current chunk
            chunks.append('\n\n'.join(current_parts))
            # Keep last paragraph as overlap seed
            if overlap > 0 and current_parts:
                last = current_parts[-1]
                current_parts = [last] if len(last) <= overlap else []
                current_len = len(last) if current_parts else 0
            else:
                current_parts = []
                current_len = 0

        current_parts.append(para)
        current_len += para_len + 2  # +2 for \n\n join

    # Flush remaining
    if current_parts:
        chunks.append('\n\n'.join(current_parts))

    return [
        {
            'text': chunk_text,
            'section': f"{title} (part {i + 1})",
            'chunk_index': i,
            'total_chunks': len(chunks),
        }
        for i, chunk_text in enumerate(chunks)
    ]


def chunk_by_headers(
    content: str,
    title: str,
    chunk_threshold: int = 2000,
    include_header_in_text: bool = False,
    min_chunk_size: int = 0
) -> List[Dict[str, Any]]:
    """
    Smart hybrid chunking: whole document if small, else chunk by ## headers.

    Args:
        content: Document content (frontmatter already stripped)
        title: Title of the document for section naming
        chunk_threshold: Character threshold below which document is returned as single chunk
        include_header_in_text: If True, include the header line in chunk text
        min_chunk_size: Minimum characters for a chunk to be included (0 = no minimum)

    Returns:
        List of chunk dicts with text, section, chunk_index, total_chunks
    """
    # If under threshold, return as single chunk
    if len(content) < chunk_threshold:
        return [{
            'text': content,
            'section': None,
            'chunk_index': 0,
            'total_chunks': 1
        }]

    # Split by ## headers (keep the header with the content)
    # Use \n+ to allow blank lines before headers (common in markdown)
    sections = re.split(r'\n+(##+ .+)\n', content)
    chunks = []
    current_section = title
    current_text = []

    for part in sections:
        if re.match(r'^##+ ', part):
            # This is a header - save previous section and start new one
            if current_text:
                text = '\n'.join(current_text).strip()
                if text and len(text) > min_chunk_size:
                    chunks.append({
                        'text': text,
                        'section': current_section,
                        'chunk_index': len(chunks),
                        'total_chunks': -1  # Will be set after
                    })
            current_section = part.lstrip('#').strip()
            # Include header in next chunk's text if requested
            current_text = [part] if include_header_in_text else []
        else:
            current_text.append(part)

    # Don't forget the last section
    if current_text:
        text = '\n'.join(current_text).strip()
        if text and len(text) > min_chunk_size:
            chunks.append({
                'text': text,
                'section': current_section,
                'chunk_index': len(chunks),
                'total_chunks': -1
            })

    # Update total_chunks for all
    for chunk in chunks:
        chunk['total_chunks'] = len(chunks)

    # Fallback if no chunks were created
    if not chunks:
        return [{
            'text': content,
            'section': None,
            'chunk_index': 0,
            'total_chunks': 1
        }]

    # If header splitting produced only 1 chunk and content is large,
    # fall back to fixed-size chunking (document has no ## headers)
    if len(chunks) == 1 and len(content) >= chunk_threshold:
        return chunk_by_size(content, title, chunk_size=chunk_threshold)

    return chunks
