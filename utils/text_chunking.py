"""
Shared text chunking utilities.

This module provides common text chunking functions used across
different knowledge managers (Obsidian, Reference Docs, etc.).
"""

import re
from typing import List, Dict, Any


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

    return chunks
