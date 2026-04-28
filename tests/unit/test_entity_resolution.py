"""Tests for entity correction → summary resolution cascade.

Tests the _cascade_entity_resolution method on DaemonOrchestrator,
which annotates crisis-era summaries with resolution metadata when
the user corrects an entity-level assumption (e.g., "Flapjack did not die").
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import date

from core.correction_detector import EntityCorrectionEvent


@pytest.fixture
def mock_chroma_store():
    store = MagicMock()
    store.update_metadata = MagicMock(return_value=True)
    return store


@pytest.fixture
def mock_memory_system(mock_chroma_store):
    ms = MagicMock()
    ms.chroma_store = mock_chroma_store
    ms.entity_resolver = None
    ms.graph_memory = None
    return ms


@pytest.fixture
def orchestrator(mock_memory_system):
    """Minimal orchestrator with memory_system wired up."""
    from core.orchestrator import DaemonOrchestrator
    orch = DaemonOrchestrator.__new__(DaemonOrchestrator)
    orch.memory_system = mock_memory_system
    orch.logger = MagicMock()
    orch.correction_detector = None
    return orch


@pytest.fixture
def flapjack_event():
    return EntityCorrectionEvent(
        entity_name="Flapjack",
        correction_type="not_dead",
        correction_text="flapjack did not die hes still here",
        confidence=0.90,
    )


def _make_doc(doc_id, content, staleness_ratio=0.0, metadata_extra=None):
    """Helper to create a mock ChromaDB document dict."""
    md = {"staleness_ratio": staleness_ratio}
    if metadata_extra:
        md.update(metadata_extra)
    return {"id": doc_id, "content": content, "metadata": md}


class TestEntityResolutionCascade:
    """Test _cascade_entity_resolution on the orchestrator."""

    @patch("config.app_config.STALENESS_ENABLED", True)
    def test_crisis_summary_gets_resolution_note(self, orchestrator, mock_chroma_store, flapjack_event):
        """Summary with entity + crisis keyword should get resolution_note."""
        mock_chroma_store.query_collection.return_value = [
            _make_doc("s1", "Flapjack was in the ICU after eating Vyvanse"),
        ]

        orchestrator._cascade_entity_resolution([flapjack_event])

        mock_chroma_store.update_metadata.assert_called()
        call_args = mock_chroma_store.update_metadata.call_args_list[0]
        assert call_args[0][0] == "summaries"  # collection name
        assert call_args[0][1] == "s1"  # doc_id
        md_updates = call_args[0][2]
        assert "resolution_note" in md_updates
        assert "Flapjack" in md_updates["resolution_note"]
        assert "alive" in md_updates["resolution_note"]
        assert md_updates["staleness_ratio"] == 0.65

    @patch("config.app_config.STALENESS_ENABLED", True)
    def test_non_crisis_summary_not_modified(self, orchestrator, mock_chroma_store, flapjack_event):
        """Summary with entity but no crisis keyword should NOT be modified."""
        mock_chroma_store.query_collection.return_value = [
            _make_doc("s2", "Flapjack is a black cat who purrs intensely"),
        ]

        orchestrator._cascade_entity_resolution([flapjack_event])

        mock_chroma_store.update_metadata.assert_not_called()

    @patch("config.app_config.STALENESS_ENABLED", True)
    def test_entity_not_in_content_skipped(self, orchestrator, mock_chroma_store, flapjack_event):
        """Document returned by semantic search but not mentioning entity should be skipped."""
        mock_chroma_store.query_collection.return_value = [
            _make_doc("s3", "The cat was in the ICU after the accident"),
        ]

        orchestrator._cascade_entity_resolution([flapjack_event])

        mock_chroma_store.update_metadata.assert_not_called()

    @patch("config.app_config.STALENESS_ENABLED", True)
    def test_staleness_ratio_uses_max(self, orchestrator, mock_chroma_store, flapjack_event):
        """Should not overwrite a higher existing staleness_ratio."""
        mock_chroma_store.query_collection.return_value = [
            _make_doc("s4", "Flapjack almost died in the emergency", staleness_ratio=0.85),
        ]

        orchestrator._cascade_entity_resolution([flapjack_event])

        call_args = mock_chroma_store.update_metadata.call_args_list[0]
        md_updates = call_args[0][2]
        assert md_updates["staleness_ratio"] == 0.85  # max(0.85, 0.65) = 0.85

    @patch("config.app_config.STALENESS_ENABLED", True)
    def test_both_collections_queried(self, orchestrator, mock_chroma_store, flapjack_event):
        """Should query both summaries and reflections."""
        mock_chroma_store.query_collection.return_value = []

        orchestrator._cascade_entity_resolution([flapjack_event])

        collection_names = [call[0][0] for call in mock_chroma_store.query_collection.call_args_list]
        assert "summaries" in collection_names
        assert "reflections" in collection_names

    @patch("config.app_config.STALENESS_ENABLED", True)
    def test_multiple_matching_docs(self, orchestrator, mock_chroma_store, flapjack_event):
        """Multiple crisis documents should all get annotated."""
        crisis_docs = [
            _make_doc("s5", "Flapjack was critical in the ICU"),
            _make_doc("s6", "Flapjack DNR was signed at the emergency vet"),
            _make_doc("s7", "Flapjack purrs a lot"),  # no crisis keyword
        ]
        # summaries returns crisis docs, reflections returns empty
        mock_chroma_store.query_collection.side_effect = [crisis_docs, []]

        orchestrator._cascade_entity_resolution([flapjack_event])

        assert mock_chroma_store.update_metadata.call_count == 2  # s5 and s6, not s7

    @patch("config.app_config.STALENESS_ENABLED", True)
    def test_doc_without_id_skipped(self, orchestrator, mock_chroma_store, flapjack_event):
        """Documents without an ID should be skipped."""
        mock_chroma_store.query_collection.return_value = [
            {"content": "Flapjack was in the ICU", "metadata": {}},  # no id
        ]

        orchestrator._cascade_entity_resolution([flapjack_event])

        mock_chroma_store.update_metadata.assert_not_called()

    @patch("config.app_config.STALENESS_ENABLED", False)
    def test_disabled_when_staleness_off(self, orchestrator, mock_chroma_store, flapjack_event):
        """Should do nothing when STALENESS_ENABLED is False."""
        orchestrator._cascade_entity_resolution([flapjack_event])

        mock_chroma_store.query_collection.assert_not_called()

    @patch("config.app_config.STALENESS_ENABLED", True)
    def test_resolution_date_included(self, orchestrator, mock_chroma_store, flapjack_event):
        """Metadata should include resolution_date."""
        mock_chroma_store.query_collection.return_value = [
            _make_doc("s8", "Flapjack faced a serious health crisis"),
        ]

        orchestrator._cascade_entity_resolution([flapjack_event])

        md_updates = mock_chroma_store.update_metadata.call_args_list[0][0][2]
        assert md_updates["resolution_date"] == date.today().isoformat()

    @patch("config.app_config.STALENESS_ENABLED", True)
    def test_crisis_keyword_make_it(self, orchestrator, mock_chroma_store, flapjack_event):
        """'make it' should be detected as a crisis keyword."""
        # summaries returns the doc, reflections returns empty
        mock_chroma_store.query_collection.side_effect = [
            [_make_doc("s9", "in case Flapjack doesn't make it")],
            [],
        ]

        orchestrator._cascade_entity_resolution([flapjack_event])

        mock_chroma_store.update_metadata.assert_called_once()

    @patch("config.app_config.STALENESS_ENABLED", True)
    def test_entity_resolver_display_name(self, orchestrator, mock_chroma_store):
        """Should use entity resolver to get proper display name."""
        # Set up entity resolver mock
        mock_resolver = MagicMock()
        mock_resolver.resolve.return_value = "flapjack"
        orchestrator.memory_system.entity_resolver = mock_resolver

        mock_node = MagicMock()
        mock_node.display_name = "Flapjack"
        mock_graph = MagicMock()
        mock_graph.get_node.return_value = mock_node
        orchestrator.memory_system.graph_memory = mock_graph

        mock_chroma_store.query_collection.return_value = [
            _make_doc("s10", "flapjack was in the icu"),
        ]

        event = EntityCorrectionEvent(
            entity_name="flapjack",
            correction_type="not_dead",
            correction_text="flapjack did not die",
            confidence=0.90,
        )

        orchestrator._cascade_entity_resolution([event])

        md_updates = mock_chroma_store.update_metadata.call_args_list[0][0][2]
        assert "Flapjack" in md_updates["resolution_note"]  # Capitalized from resolver
        assert md_updates["resolution_entity"] == "Flapjack"

    @patch("config.app_config.STALENESS_ENABLED", True)
    def test_no_memory_system_is_noop(self, flapjack_event):
        """Should silently return if memory_system is None."""
        from core.orchestrator import DaemonOrchestrator
        orch = DaemonOrchestrator.__new__(DaemonOrchestrator)
        orch.memory_system = None
        orch.logger = MagicMock()
        orch._cascade_entity_resolution([flapjack_event])
        # Should not raise


class TestStalenessPrefix:
    """Test the enhanced _staleness_prefix with resolution_note support."""

    @patch("config.app_config.STALENESS_ENABLED", True)
    @patch("config.app_config.STALENESS_HISTORICAL_THRESHOLD", 0.6)
    def test_resolution_note_takes_priority(self):
        from core.prompt.builder import _staleness_prefix
        item = {
            "metadata": {
                "resolution_note": "Flapjack is alive (confirmed 2026-04-20)",
                "staleness_ratio": 0.65,
            }
        }
        prefix = _staleness_prefix(item)
        assert "[RESOLVED" in prefix
        assert "Flapjack is alive" in prefix

    @patch("config.app_config.STALENESS_ENABLED", True)
    @patch("config.app_config.STALENESS_HISTORICAL_THRESHOLD", 0.6)
    def test_historical_without_resolution(self):
        from core.prompt.builder import _staleness_prefix
        item = {"metadata": {"staleness_ratio": 0.7}}
        prefix = _staleness_prefix(item)
        assert "HISTORICAL" in prefix

    @patch("config.app_config.STALENESS_ENABLED", True)
    @patch("config.app_config.STALENESS_HISTORICAL_THRESHOLD", 0.6)
    def test_no_prefix_below_threshold(self):
        from core.prompt.builder import _staleness_prefix
        item = {"metadata": {"staleness_ratio": 0.3}}
        prefix = _staleness_prefix(item)
        assert prefix == ""

    @patch("config.app_config.STALENESS_ENABLED", True)
    @patch("config.app_config.STALENESS_HISTORICAL_THRESHOLD", 0.6)
    def test_resolution_note_on_item_level(self):
        """resolution_note can also be at item level (not just metadata)."""
        from core.prompt.builder import _staleness_prefix
        item = {
            "metadata": {},
            "resolution_note": "Rex survived the accident",
        }
        prefix = _staleness_prefix(item)
        assert "[RESOLVED" in prefix
        assert "Rex survived" in prefix

    @patch("config.app_config.STALENESS_ENABLED", False)
    def test_disabled_returns_empty(self):
        from core.prompt.builder import _staleness_prefix
        item = {
            "metadata": {
                "resolution_note": "Flapjack is alive",
                "staleness_ratio": 0.9,
            }
        }
        prefix = _staleness_prefix(item)
        assert prefix == ""
