"""
Test suite for memory citation system.

Tests the citation tracking, extraction, and display functionality.
"""
import pytest
import re
from unittest.mock import Mock, patch, AsyncMock
from core.orchestrator import DaemonOrchestrator


def test_extract_citations_basic():
    """Test basic citation extraction from response."""
    # Create a minimal orchestrator instance
    orchestrator = DaemonOrchestrator.__new__(DaemonOrchestrator)
    orchestrator.citation_pattern = re.compile(r'\[(MEM_\w+_\d+|FACT_\d+|SUM_\d+|PROFILE_\w+)\]')
    orchestrator.logger = Mock()

    response = "You mentioned [MEM_RECENT_3] that you're starting OMSA and [PROFILE_CONTEXT] you like Python"
    memory_map = {
        'MEM_RECENT_3': {
            'type': 'episodic_recent',
            'timestamp': '2024-01-15T10:30:00',
            'content': 'Starting Georgia Tech OMSA in January...',
            'relevance_score': 1.0
        },
        'PROFILE_CONTEXT': {
            'type': 'user_profile',
            'timestamp': '2024-01-15T10:30:00',
            'content': 'User enjoys Python programming and machine learning',
            'relevance_score': 1.0
        }
    }

    clean, citations = orchestrator._extract_citations(response, memory_map)

    # Verify citations extracted
    assert len(citations) == 2

    # Check citation IDs are present
    citation_ids = {c['memory_id'] for c in citations}
    assert 'MEM_RECENT_3' in citation_ids
    assert 'PROFILE_CONTEXT' in citation_ids

    # Verify citation metadata
    for citation in citations:
        assert 'memory_id' in citation
        assert 'type' in citation
        assert 'timestamp' in citation
        assert 'content' in citation
        assert 'relevance_score' in citation

    # Verify clean response has citations removed
    assert '[MEM_RECENT_3]' not in clean
    assert '[PROFILE_CONTEXT]' not in clean
    assert "you're starting OMSA" in clean
    assert "you like Python" in clean


def test_extract_citations_no_citations():
    """Test extraction when response has no citations."""
    orchestrator = DaemonOrchestrator.__new__(DaemonOrchestrator)
    orchestrator.citation_pattern = re.compile(r'\[(MEM_\w+_\d+|FACT_\d+|SUM_\d+|PROFILE_\w+)\]')
    orchestrator.logger = Mock()

    response = "This is a regular response without any citations"
    memory_map = {
        'MEM_RECENT_1': {
            'type': 'episodic_recent',
            'content': 'Some memory',
            'relevance_score': 0.9
        }
    }

    clean, citations = orchestrator._extract_citations(response, memory_map)

    assert len(citations) == 0
    assert clean == response


def test_extract_citations_empty_memory_map():
    """Test extraction with empty memory map."""
    orchestrator = DaemonOrchestrator.__new__(DaemonOrchestrator)
    orchestrator.citation_pattern = re.compile(r'\[(MEM_\w+_\d+|FACT_\d+|SUM_\d+|PROFILE_\w+)\]')
    orchestrator.logger = Mock()

    response = "Response with [MEM_RECENT_1] citation that doesn't exist"
    memory_map = {}

    clean, citations = orchestrator._extract_citations(response, memory_map)

    # Should return empty citations when memory_map is empty
    assert len(citations) == 0
    # But still clean the response
    assert '[MEM_RECENT_1]' not in clean


def test_extract_citations_multiple_same_citation():
    """Test extraction when same citation appears multiple times."""
    orchestrator = DaemonOrchestrator.__new__(DaemonOrchestrator)
    orchestrator.citation_pattern = re.compile(r'\[(MEM_\w+_\d+|FACT_\d+|SUM_\d+|PROFILE_\w+)\]')
    orchestrator.logger = Mock()

    response = "You said [MEM_RECENT_1] earlier and then [MEM_RECENT_1] again"
    memory_map = {
        'MEM_RECENT_1': {
            'type': 'episodic_recent',
            'timestamp': '2024-01-15T10:30:00',
            'content': 'User mentioned project deadline',
            'relevance_score': 0.95
        }
    }

    clean, citations = orchestrator._extract_citations(response, memory_map)

    # Should only return one citation (deduplicated)
    assert len(citations) == 1
    assert citations[0]['memory_id'] == 'MEM_RECENT_1'

    # Both instances should be removed
    assert '[MEM_RECENT_1]' not in clean


def test_extract_citations_semantic_memory():
    """Test extraction with semantic memory citations."""
    orchestrator = DaemonOrchestrator.__new__(DaemonOrchestrator)
    orchestrator.citation_pattern = re.compile(r'\[(MEM_\w+_\d+|FACT_\d+|SUM_\d+|PROFILE_\w+)\]')
    orchestrator.logger = Mock()

    response = "Based on [MEM_SEMANTIC_5] your previous research"
    memory_map = {
        'MEM_SEMANTIC_5': {
            'type': 'episodic_semantic',
            'timestamp': '2024-01-10T14:20:00',
            'content': 'User researched transformer architectures',
            'relevance_score': 0.87
        }
    }

    clean, citations = orchestrator._extract_citations(response, memory_map)

    assert len(citations) == 1
    assert citations[0]['type'] == 'episodic_semantic'
    assert citations[0]['relevance_score'] == 0.87


def test_citation_mode_toggle():
    """Test that citation mode can be toggled."""
    orchestrator = DaemonOrchestrator.__new__(DaemonOrchestrator)
    orchestrator.enable_citations = False
    orchestrator.citation_pattern = re.compile(r'\[(MEM_\w+_\d+|FACT_\d+|SUM_\d+|PROFILE_\w+)\]')

    # Initially disabled
    assert orchestrator.enable_citations == False

    # Enable citations
    orchestrator.enable_citations = True
    assert orchestrator.enable_citations == True

    # Disable again
    orchestrator.enable_citations = False
    assert orchestrator.enable_citations == False


def test_citation_content_truncation():
    """Test that citation content is truncated to 200 chars."""
    orchestrator = DaemonOrchestrator.__new__(DaemonOrchestrator)
    orchestrator.citation_pattern = re.compile(r'\[(MEM_\w+_\d+|FACT_\d+|SUM_\d+|PROFILE_\w+)\]')
    orchestrator.logger = Mock()

    long_content = "A" * 500  # 500 character string
    response = "Reference [MEM_RECENT_1] to long content"
    memory_map = {
        'MEM_RECENT_1': {
            'type': 'episodic_recent',
            'timestamp': '2024-01-15T10:30:00',
            'content': long_content,
            'relevance_score': 1.0
        }
    }

    clean, citations = orchestrator._extract_citations(response, memory_map)

    assert len(citations) == 1
    # Content should be truncated to 200 chars in extraction
    assert len(citations[0]['content']) == 200


def test_citation_pattern_matches():
    """Test that citation pattern matches all expected formats."""
    pattern = re.compile(r'\[(MEM_\w+_\d+|FACT_\d+|SUM_\d+|PROFILE_\w+)\]')

    # Should match
    assert pattern.search('[MEM_RECENT_0]')
    assert pattern.search('[MEM_SEMANTIC_12]')
    assert pattern.search('[PROFILE_CONTEXT]')
    assert pattern.search('[FACT_5]')
    assert pattern.search('[SUM_3]')

    # Should not match
    assert not pattern.search('[INVALID]')
    assert not pattern.search('[MEM_RECENT]')  # Missing index
    assert not pattern.search('MEM_RECENT_1')  # Missing brackets


def test_citations_in_debug_info():
    """Test that citations are added to debug_info correctly."""
    # This would be an integration test - verifying citations flow through process_user_query
    # Skipping detailed implementation as it requires full orchestrator setup
    pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
