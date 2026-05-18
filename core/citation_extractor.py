"""
# core/citation_extractor.py

Module Contract
- Purpose: Extract memory and web citations from LLM responses.
- Inputs:
  - CitationExtractor.extract(response, memory_map, web_source_map) -> (clean_response, citations)
- Outputs: Tuple of cleaned response text and list of citation dicts.
- Side effects: None (pure computation).
"""

import re
from typing import Dict, Any, List, Tuple

from utils.logging_utils import get_logger

logger = get_logger("citation_extractor")

# Pattern matches citation formats: MEM_RECENT_3, MEM_SEMANTIC_4-7, SUM_RECENT_1, REFL_SEMANTIC_2, PROFILE_CONTEXT, WEB_1
CITATION_PATTERN = re.compile(
    r'\[('
    r'WEB_\d+|'                   # WEB_1, WEB_2 (web search sources)
    r'MEM_\w+_\d+(?:-\d+)?|'      # MEM_RECENT_3, MEM_SEMANTIC_4-7
    r'SUM_\w+_\d+(?:-\d+)?|'      # SUM_RECENT_1, SUM_SEMANTIC_2-5
    r'REFL_\w+_\d+(?:-\d+)?|'     # REFL_RECENT_1, REFL_SEMANTIC_3
    r'FACT_\d+(?:-\d+)?|'         # FACT_3 (legacy)
    r'PROFILE_\w+'                # PROFILE_CONTEXT
    r')\]'
)


def expand_citation_range(mem_id: str) -> List[str]:
    """
    Expand a range citation like MEM_RECENT_4-7 into individual IDs.

    Args:
        mem_id: Citation ID (e.g., "MEM_RECENT_4-7" or "MEM_RECENT_3")

    Returns:
        List of individual citation IDs (e.g., ["MEM_RECENT_4", "MEM_RECENT_5", ...])
    """
    if '-' in mem_id and not mem_id.startswith('PROFILE'):
        match = re.match(r'([A-Z_]+_)(\d+)-(\d+)', mem_id)
        if match:
            prefix = match.group(1)
            start = int(match.group(2))
            end = int(match.group(3))
            return [f"{prefix}{i}" for i in range(start, end + 1)]
    return [mem_id]


def extract_citations(
    response: str,
    memory_map: Dict[str, Any],
    web_source_map: Dict[str, Dict[str, str]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extract memory and web citations from response.

    Handles: [MEM_RECENT_3], [MEM_SEMANTIC_4-7], [WEB_1], etc.
    Web citations validated against web_source_map; invalid IDs stripped.

    Returns:
        Tuple of (clean_response, citations_list)
    """
    cited_ids = set(CITATION_PATTERN.findall(response))

    citations = []
    seen_ids = set()
    invalid_cited_ids = []

    # Memory citations
    if memory_map:
        for mem_id in cited_ids:
            if mem_id.startswith("WEB_"):
                continue  # handle below
            expanded_ids = expand_citation_range(mem_id)
            for individual_id in expanded_ids:
                if individual_id in memory_map and individual_id not in seen_ids:
                    citations.append({
                        'memory_id': individual_id,
                        'type': memory_map[individual_id].get('type', 'unknown'),
                        'timestamp': memory_map[individual_id].get('timestamp', ''),
                        'content': memory_map[individual_id].get('content', '')[:200],
                        'relevance_score': memory_map[individual_id].get('relevance_score', 0.0),
                        'db_id': memory_map[individual_id].get('db_id', None)
                    })
                    seen_ids.add(individual_id)

    # Web citations -- validate against web_source_map
    for cid in cited_ids:
        if not cid.startswith("WEB_"):
            continue
        if cid in web_source_map and cid not in seen_ids:
            src = web_source_map[cid]
            citations.append({
                'memory_id': cid,
                'type': 'web_source',
                'title': src.get('title', ''),
                'url': src.get('url', ''),
                'domain': src.get('domain', ''),
            })
            seen_ids.add(cid)
        elif cid not in web_source_map:
            invalid_cited_ids.append(cid)

    # Strip invalid web citation markers from response
    clean_response = response
    for bad_id in invalid_cited_ids:
        clean_response = clean_response.replace(f"[{bad_id}]", "")

    # Remove all remaining citation tags for clean display
    clean_response = CITATION_PATTERN.sub('', clean_response)
    clean_response = re.sub(r'\s+', ' ', clean_response).strip()

    if invalid_cited_ids:
        logger.warning(f"[Citation] Invalid web citations stripped: {invalid_cited_ids}")
    logger.debug(f"[Citation] Extracted {len(citations)} citations (from {len(cited_ids)} tags, {len(invalid_cited_ids)} invalid)")

    return clean_response, citations
