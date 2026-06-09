"""
Tests for knowledge/document_generator.py.

Module Contract
- Purpose: Verify source gathering, dedup, outline/draft generation,
  citation validation, frontmatter, file writing, index management,
  path safety, and trigger detection.
- Inputs: Mock LLM, mock search providers, tmp_path fixtures.
- Outputs: Pass/fail assertions.
- Dependencies: pytest, pytest-asyncio, unittest.mock, standard library.
"""

import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from knowledge.document_generator import (
    DocumentGenerator,
    DocumentSource,
    GeneratedDocument,
    detect_document_intent,
    DOCUMENT_TRIGGER_PATTERN,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_model_manager():
    mm = AsyncMock()
    mm.default_model = "test-model"
    mm.generate_once = AsyncMock(return_value="## Introduction\nTest content [WEB_1].\n\n## Sources\n- [WEB_1] Test Source")
    return mm


@pytest.fixture
def mock_web_search_manager():
    wsm = MagicMock()
    wsm.is_available.return_value = True

    page = MagicMock()
    page.title = "Test Article"
    page.url = "https://example.com/article"
    page.snippet = "A snippet about the topic"
    page.content = "Full content of the article about the topic"
    page.score = 0.9

    result = MagicMock()
    result.pages = [page]

    wsm.search = AsyncMock(return_value=result)
    return wsm


@pytest.fixture
def mock_chroma_store():
    store = MagicMock()

    def mock_query(collection, query, n_results=5):
        if collection == "wiki_knowledge":
            return [
                {"content": "Wiki article about topic", "metadata": {"title": "Wiki Topic", "url": "https://en.wikipedia.org/wiki/Topic"}, "relevance_score": 0.8},
            ]
        elif collection == "obsidian_notes":
            return [
                {"content": "My notes on topic", "metadata": {"title": "Topic Notes"}, "relevance_score": 0.7},
            ]
        return []

    store.query_collection = MagicMock(side_effect=mock_query)
    return store


@pytest.fixture
def generator(mock_model_manager, mock_web_search_manager, mock_chroma_store, tmp_path):
    return DocumentGenerator(
        model_manager=mock_model_manager,
        web_search_manager=mock_web_search_manager,
        chroma_store=mock_chroma_store,
        output_dir=tmp_path / "documents",
        repo_root=tmp_path,
    )


@pytest.fixture
def generator_no_search(mock_model_manager, tmp_path):
    """Generator with no search providers — tests graceful degradation."""
    return DocumentGenerator(
        model_manager=mock_model_manager,
        output_dir=tmp_path / "documents",
        repo_root=tmp_path,
    )


# ############################################################################
#
#  Data models
#
# ############################################################################

class TestDocumentSource:

    def test_creation(self):
        src = DocumentSource(
            id="WEB_1", title="Test", url="https://example.com",
            source_type="web", snippet="A snippet",
        )
        assert src.id == "WEB_1"
        assert src.source_type == "web"

    def test_to_dict(self):
        src = DocumentSource(
            id="WIKI_1", title="Wiki", url=None,
            source_type="wikipedia", snippet="Content",
        )
        d = src.to_dict()
        assert d["id"] == "WIKI_1"
        assert d["url"] is None

    def test_to_frontmatter_entry(self):
        src = DocumentSource(
            id="WEB_1", title="Test", url="https://example.com",
            source_type="web", snippet="...",
        )
        entry = src.to_frontmatter_entry()
        assert entry["id"] == "WEB_1"
        assert entry["url"] == "https://example.com"
        assert entry["type"] == "web"

    def test_frontmatter_entry_no_url(self):
        src = DocumentSource(
            id="NOTE_1", title="Notes", url=None,
            source_type="notes", snippet="...",
        )
        entry = src.to_frontmatter_entry()
        assert "url" not in entry


class TestGeneratedDocument:

    def test_to_dict(self):
        src = DocumentSource(id="WEB_1", title="T", url=None, source_type="web", snippet="")
        doc = GeneratedDocument(
            path="/tmp/doc.md", title="Test Doc", doc_type="report",
            topic="test", focus=None, sources=[src], created_at="2026-01-01",
        )
        d = doc.to_dict()
        assert d["title"] == "Test Doc"
        assert len(d["sources"]) == 1
        assert d["sources"][0]["id"] == "WEB_1"


# ############################################################################
#
#  Source gathering
#
# ############################################################################

class TestGatherSources:

    @pytest.mark.asyncio
    async def test_gathers_from_all_providers(self, generator):
        sources = await generator._gather_sources("test topic")
        # Web + wiki + notes = at least 3
        assert len(sources) >= 3
        types = {s.source_type for s in sources}
        assert "web" in types
        assert "wikipedia" in types
        assert "notes" in types

    @pytest.mark.asyncio
    async def test_assigns_source_ids(self, generator):
        sources = await generator._gather_sources("test topic")
        ids = [s.id for s in sources]
        assert any(id.startswith("WEB_") for id in ids)

    @pytest.mark.asyncio
    async def test_graceful_degradation_no_providers(self, generator_no_search):
        sources = await generator_no_search._gather_sources("test topic")
        assert sources == []

    @pytest.mark.asyncio
    async def test_graceful_degradation_web_fails(self, generator):
        generator.web_search_manager.search = AsyncMock(side_effect=Exception("API down"))
        sources = await generator._gather_sources("test topic")
        # Should still have wiki + notes
        assert len(sources) >= 2
        assert all(s.source_type != "web" for s in sources)

    @pytest.mark.asyncio
    async def test_graceful_degradation_chroma_fails(self, generator):
        generator.chroma_store.query_collection = MagicMock(side_effect=Exception("DB error"))
        sources = await generator._gather_sources("test topic")
        # Should still have web
        assert len(sources) >= 1
        assert any(s.source_type == "web" for s in sources)

    @pytest.mark.asyncio
    async def test_focus_included_in_query(self, generator):
        await generator._gather_sources("climate change", focus="economic impact")
        call_args = generator.web_search_manager.search.call_args
        query = call_args.kwargs.get("query", call_args.args[0] if call_args.args else "")
        assert "economic impact" in query


# ############################################################################
#
#  Content-aware generation (user-provided primary material)
#
# ############################################################################

class TestContentAwareGeneration:
    """When the user pastes substantial material to evaluate, it becomes the
    PRIMARY source [INPUT_1] and web/encyclopedia search is suppressed; personal
    notes are still gathered for grounding. Regression: an "evaluate THIS
    proposal" request previously web-searched the bare topic and pulled
    irrelevant sources (Anarchism Wikipedia, a 1994 Unix-daemon PDF).
    """

    PROPOSAL = (
        "Proposal: Isolated Agent Branch Portfolio System. Spawn N isolated "
        "worker workspaces, each attempting a bounded improvement on its own "
        "branch. Most branches die cheaply; survivors become human-reviewable "
        "proposals; nothing ever auto-merges. The supervisor authors an "
        "immutable manifest; evaluation is trusted and external to the branch. "
    ) * 4  # comfortably above DOCUMENT_PROVIDED_MIN_CHARS

    @pytest.mark.asyncio
    async def test_provided_material_becomes_primary_source(self, generator):
        sources = await generator._gather_sources(
            "agent branch portfolio", source_material=self.PROPOSAL,
        )
        provided = [s for s in sources if s.source_type == "provided"]
        assert len(provided) == 1
        assert provided[0].id == "INPUT_1"
        assert "Isolated Agent Branch Portfolio" in provided[0].snippet

    @pytest.mark.asyncio
    async def test_web_and_wiki_suppressed_when_material_present(self, generator):
        sources = await generator._gather_sources(
            "agent branch portfolio", source_material=self.PROPOSAL,
        )
        types = {s.source_type for s in sources}
        assert "web" not in types, "web search must be suppressed with provided material"
        assert "wikipedia" not in types, "encyclopedia must be suppressed with provided material"
        assert "notes" in types, "personal notes still gathered for grounding"
        generator.web_search_manager.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_thin_material_does_not_suppress_web(self, generator):
        """Material below the min-chars floor is ignored — normal research runs."""
        sources = await generator._gather_sources(
            "agent branch portfolio", source_material="too short to anchor on",
        )
        types = {s.source_type for s in sources}
        assert "provided" not in types
        assert "web" in types

    @pytest.mark.asyncio
    async def test_no_material_preserves_prior_behavior(self, generator):
        sources = await generator._gather_sources("agent branch portfolio")
        types = {s.source_type for s in sources}
        assert "web" in types and "wikipedia" in types and "notes" in types
        assert "provided" not in types

    def test_provided_source_ranks_first_and_keeps_input_id(self, generator):
        from knowledge.document_generator import DocumentSource
        gathered = [
            DocumentSource(id="WEB_1", title="w", url="http://x", source_type="web",
                           snippet="s", relevance=0.9),
            DocumentSource(id="INPUT_1", title="User-provided material", url=None,
                           source_type="provided", snippet="the proposal", relevance=10.0),
        ]
        ranked = generator._dedupe_and_rank(gathered)
        assert ranked[0].source_type == "provided"
        assert ranked[0].id == "INPUT_1"

    def test_format_renders_provided_first_and_full(self, generator):
        from knowledge.document_generator import DocumentSource
        long_material = "UNIQUE_PROPOSAL_TOKEN " * 100  # > 300-char snippet cap
        sources = [
            DocumentSource(id="NOTE_1", title="a note", url=None, source_type="notes",
                           snippet="note body", relevance=0.5),
            DocumentSource(id="INPUT_1", title="User-provided material", url=None,
                           source_type="provided", snippet=long_material, relevance=10.0),
        ]
        text = generator._format_sources_for_prompt(sources)
        assert text.index("INPUT_1") < text.index("NOTE_1")  # provided first
        assert "PRIMARY MATERIAL" in text
        assert text.count("UNIQUE_PROPOSAL_TOKEN") > 50  # full, not 300-char capped

    def test_primary_instruction_only_when_provided_present(self, generator):
        from knowledge.document_generator import DocumentSource
        without = [DocumentSource(id="WEB_1", title="w", url=None, source_type="web",
                                  snippet="s", relevance=0.5)]
        assert generator._primary_material_instruction(without) == ""
        with_provided = without + [
            DocumentSource(id="INPUT_1", title="m", url=None, source_type="provided",
                           snippet="x", relevance=10.0)
        ]
        note = generator._primary_material_instruction(with_provided)
        assert "INPUT_1" in note and "PRIMARY" in note


# ############################################################################
#
#  Dedup and ranking
#
# ############################################################################

class TestDedupAndRank:

    def test_dedup_by_url(self, generator):
        sources = [
            DocumentSource(id="WEB_1", title="A", url="https://example.com/a", source_type="web", snippet=""),
            DocumentSource(id="WEB_2", title="B", url="https://example.com/a", source_type="web", snippet=""),
        ]
        result = generator._dedupe_and_rank(sources)
        assert len(result) == 1

    def test_dedup_by_title(self, generator):
        sources = [
            DocumentSource(id="WEB_1", title="Same Title", url="https://a.com", source_type="web", snippet=""),
            DocumentSource(id="WIKI_1", title="Same Title", url="https://b.com", source_type="wikipedia", snippet=""),
        ]
        result = generator._dedupe_and_rank(sources)
        assert len(result) == 1

    def test_caps_at_max_sources(self, generator):
        sources = [
            DocumentSource(id=f"WEB_{i}", title=f"Source {i}", url=f"https://example.com/{i}",
                          source_type="web", snippet="", relevance=1.0 - i * 0.01)
            for i in range(20)
        ]
        with patch("knowledge.document_generator.app_config") as mock_cfg:
            mock_cfg.DOCUMENT_MAX_SOURCES = 5
            result = generator._dedupe_and_rank(sources)
        assert len(result) == 5

    def test_reassigns_stable_ids(self, generator):
        sources = [
            DocumentSource(id="WEB_1", title="A", url="https://a.com", source_type="web", snippet="", relevance=0.9),
            DocumentSource(id="WIKI_1", title="B", url=None, source_type="wikipedia", snippet="", relevance=0.8),
        ]
        result = generator._dedupe_and_rank(sources)
        ids = {s.id for s in result}
        assert "WEB_1" in ids
        assert "WIKI_1" in ids

    def test_sorted_by_relevance(self, generator):
        sources = [
            DocumentSource(id="WEB_1", title="Low", url="https://low.com", source_type="web", snippet="", relevance=0.1),
            DocumentSource(id="WEB_2", title="High", url="https://high.com", source_type="web", snippet="", relevance=0.9),
        ]
        result = generator._dedupe_and_rank(sources)
        assert result[0].relevance > result[1].relevance


# ############################################################################
#
#  Citation validation
#
# ############################################################################

class TestCitationValidation:

    def test_valid_citations_preserved(self, generator):
        sources = [DocumentSource(id="WEB_1", title="T", url=None, source_type="web", snippet="")]
        md = "Some text [WEB_1] more text.\n\n## Sources\n- [WEB_1] T"
        result, count = generator._validate_and_clean(md, sources, 5)
        assert "[WEB_1]" in result

    def test_invalid_citations_stripped(self, generator):
        sources = [DocumentSource(id="WEB_1", title="T", url=None, source_type="web", snippet="")]
        md = "Text [WEB_1] and [WEB_99] and [FAKE_1].\n\n## Sources\n- [WEB_1] T"
        result, _ = generator._validate_and_clean(md, sources, 5)
        assert "[WEB_1]" in result
        assert "[WEB_99]" not in result
        assert "[FAKE_1]" not in result

    def test_sources_section_added_if_missing(self, generator):
        sources = [DocumentSource(id="WEB_1", title="T", url="https://t.com", source_type="web", snippet="")]
        md = "Some text [WEB_1]."
        result, _ = generator._validate_and_clean(md, sources, 5)
        assert "## Sources" in result

    def test_existing_sources_section_preserved(self, generator):
        sources = [DocumentSource(id="WEB_1", title="T", url=None, source_type="web", snippet="")]
        md = "Text [WEB_1].\n\n## Sources\n- [WEB_1] T"
        result, _ = generator._validate_and_clean(md, sources, 5)
        # Should not have duplicate Sources sections
        assert result.count("## Sources") == 1

    def test_section_count(self, generator):
        sources = []
        md = "## Introduction\nHello\n\n## Analysis\nStuff\n\n## Conclusion\nBye\n\n## Sources\n"
        _, count = generator._validate_and_clean(md, sources, 5)
        assert count == 3  # Introduction, Analysis, Conclusion (not Sources)


# ############################################################################
#
#  Frontmatter
#
# ############################################################################

class TestFrontmatter:

    def test_valid_yaml_structure(self, generator):
        sources = [DocumentSource(id="WEB_1", title="T", url="https://t.com", source_type="web", snippet="")]
        fm = generator._build_frontmatter("Test Title", "report", "test topic", None, sources, "2026-01-01T00:00:00Z")
        assert fm.startswith("---")
        assert fm.endswith("---")
        assert "title:" in fm
        assert "type: report" in fm
        assert "status: draft" in fm

    def test_includes_sources(self, generator):
        sources = [
            DocumentSource(id="WEB_1", title="Source A", url="https://a.com", source_type="web", snippet=""),
            DocumentSource(id="WIKI_1", title="Source B", url=None, source_type="wikipedia", snippet=""),
        ]
        fm = generator._build_frontmatter("Title", "summary", "topic", "focus", sources, "2026-01-01T00:00:00Z")
        assert "WEB_1" in fm
        assert "WIKI_1" in fm
        assert "type: summary" in fm
        assert "focus:" in fm

    def test_yaml_escape(self):
        assert DocumentGenerator._yaml_escape('He said "hello"') == 'He said \\"hello\\"'
        assert DocumentGenerator._yaml_escape("line\nbreak") == "line break"
        assert DocumentGenerator._yaml_escape(None) == ""

    def test_required_fields_present(self, generator):
        fm = generator._build_frontmatter("T", "report", "topic", None, [], "2026-01-01T00:00:00Z")
        for field in ["title:", "type:", "topic:", "created:", "status:", "sources:", "tags:"]:
            assert field in fm


# ############################################################################
#
#  File writing
#
# ############################################################################

class TestFileWriting:

    def test_creates_subdirectory(self, generator, tmp_path):
        path = generator._write_versioned_file("report", "test topic", "# Test\nContent")
        assert path.exists()
        assert "reports" in str(path)
        assert path.parent == tmp_path / "documents" / "reports"

    def test_summary_goes_to_summaries(self, generator, tmp_path):
        path = generator._write_versioned_file("summary", "test topic", "# Summary\nContent")
        assert "summaries" in str(path)

    def test_safe_slug(self):
        assert DocumentGenerator._safe_slug("Hello World!") == "hello-world"
        assert DocumentGenerator._safe_slug("résumé trends") == "r-sum-trends"
        assert DocumentGenerator._safe_slug("a" * 100) == "a" * 60
        assert DocumentGenerator._safe_slug("") == ""
        assert DocumentGenerator._safe_slug("   ") == ""

    def test_collision_suffix(self, generator, tmp_path):
        path1 = generator._write_versioned_file("report", "test", "Content 1")
        path2 = generator._write_versioned_file("report", "test", "Content 2")
        assert path1 != path2
        assert path1.exists()
        assert path2.exists()
        assert "-2" in path2.stem

    def test_does_not_overwrite(self, generator, tmp_path):
        path1 = generator._write_versioned_file("report", "test", "Original")
        path2 = generator._write_versioned_file("report", "test", "Different")
        assert path1.read_text() == "Original"
        assert path2.read_text() == "Different"

    def test_path_traversal_neutralized(self, generator, tmp_path):
        """Path traversal chars are stripped by _safe_slug, output stays inside documents/."""
        path = generator._write_versioned_file("report", "../../etc/passwd", "content")
        assert str(path.resolve()).startswith(str((tmp_path / "documents").resolve()))

    def test_content_written_correctly(self, generator, tmp_path):
        content = "# Test\n\nSome content here."
        path = generator._write_versioned_file("report", "test", content)
        assert path.read_text(encoding="utf-8") == content


# ############################################################################
#
#  Index management
#
# ############################################################################

class TestIndexManagement:

    def test_creates_index_if_missing(self, generator, tmp_path):
        index_path = tmp_path / "documents" / "index.json"
        assert not index_path.exists()

        generator._update_index({"id": "test", "title": "Test"})

        assert index_path.exists()
        data = json.loads(index_path.read_text())
        assert len(data) == 1
        assert data[0]["id"] == "test"

    def test_preserves_existing_entries(self, generator, tmp_path):
        index_path = tmp_path / "documents" / "index.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps([{"id": "old", "title": "Old"}]))

        generator._update_index({"id": "new", "title": "New"})

        data = json.loads(index_path.read_text())
        assert len(data) == 2
        assert data[0]["id"] == "old"
        assert data[1]["id"] == "new"

    def test_handles_corrupt_index(self, generator, tmp_path):
        index_path = tmp_path / "documents" / "index.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text("not valid json!")

        generator._update_index({"id": "new", "title": "New"})

        data = json.loads(index_path.read_text())
        assert len(data) == 1

    def test_entry_includes_required_fields(self, generator, tmp_path):
        entry = {
            "id": "test-2026-01-01",
            "path": "documents/reports/test-2026-01-01.md",
            "title": "Test Report",
            "type": "report",
            "topic": "test",
            "focus": None,
            "created": "2026-01-01T00:00:00Z",
            "status": "draft",
            "sources_count": 3,
            "source_types": ["web", "wikipedia"],
            "model": "test-model",
            "version": 1,
        }
        generator._update_index(entry)
        data = json.loads((tmp_path / "documents" / "index.json").read_text())
        stored = data[0]
        for key in ("id", "path", "title", "type", "topic", "created", "status", "sources_count"):
            assert key in stored


# ############################################################################
#
#  End-to-end generation
#
# ############################################################################

class TestGenerate:

    @pytest.mark.asyncio
    async def test_full_pipeline(self, generator, tmp_path):
        result = await generator.generate("test topic", doc_type="report")
        assert isinstance(result, GeneratedDocument)
        assert result.doc_type == "report"
        assert Path(result.path).exists()
        assert result.title  # should have extracted a title
        assert result.word_count > 0

        # Check file content
        content = Path(result.path).read_text()
        assert content.startswith("---")
        assert "type: report" in content

        # Check index was updated
        index = json.loads((tmp_path / "documents" / "index.json").read_text())
        assert len(index) == 1

    @pytest.mark.asyncio
    async def test_summary_pipeline(self, generator, tmp_path):
        result = await generator.generate("test topic", doc_type="summary")
        assert result.doc_type == "summary"
        assert "summaries" in result.path

    @pytest.mark.asyncio
    async def test_no_model_raises(self, tmp_path):
        gen = DocumentGenerator(output_dir=tmp_path)
        with pytest.raises(ValueError, match="model_manager"):
            await gen.generate("test")

    @pytest.mark.asyncio
    async def test_invalid_doc_type_raises(self, generator):
        with pytest.raises(ValueError, match="Invalid doc_type"):
            await generator.generate("test", doc_type="invalid")

    @pytest.mark.asyncio
    async def test_empty_llm_response_raises(self, generator):
        generator.model_manager.generate_once = AsyncMock(return_value="")
        with pytest.raises(RuntimeError, match="empty"):
            await generator.generate("test")

    @pytest.mark.asyncio
    async def test_api_error_sentinel_body_raises_and_writes_nothing(self, generator, tmp_path):
        """Regression: a 402 made generate_once return '[API Error] ...', which
        was written as the document body (frontmatter + error, title 'Sources').
        It must now abort and leave no file / index entry behind.
        """
        generator.model_manager.generate_once = AsyncMock(
            return_value="[API Error] Error code: 402 - requires more credits"
        )
        with pytest.raises(RuntimeError, match="LLM call failed"):
            await generator.generate("daemon implementation plan", doc_type="summary")

        # No corrupt artifacts written
        summaries = tmp_path / "documents" / "summaries"
        assert not summaries.exists() or not list(summaries.glob("*.md"))
        assert not (tmp_path / "documents" / "index.json").exists()

    @pytest.mark.asyncio
    async def test_credits_exhausted_sentinel_raises(self, generator):
        generator.model_manager.generate_once = AsyncMock(
            return_value="[CREDITS EXHAUSTED] You've run out of API credits."
        )
        with pytest.raises(RuntimeError, match="LLM call failed"):
            await generator.generate("test topic", doc_type="summary")

    @pytest.mark.asyncio
    async def test_report_outline_error_sentinel_raises(self, generator):
        """An error sentinel on the outline call aborts before drafting/writing."""
        generator.model_manager.generate_once = AsyncMock(
            return_value="[SERVER ERROR] The API provider is experiencing issues (HTTP 503)."
        )
        with pytest.raises(RuntimeError, match="LLM call failed"):
            await generator.generate("test topic", doc_type="report")

    @pytest.mark.asyncio
    async def test_refine_topic_ignores_error_sentinel(self, generator):
        """A sentinel from the topic-refine call must not become the topic."""
        generator.model_manager.generate_once = AsyncMock(
            return_value="[API Error] Error code: 402 - requires more credits"
        )
        # Noisy topic triggers refinement; sentinel result must be discarded,
        # falling back to the original topic string.
        refined = await generator._refine_topic("please write the plan again now", None)
        assert not refined.startswith("[API Error]")
        assert refined == "please write the plan again now"

    @pytest.mark.asyncio
    async def test_focus_passed_through(self, generator):
        result = await generator.generate("climate", focus="economic impact")
        content = Path(result.path).read_text()
        assert "economic impact" in content  # should be in frontmatter

    @pytest.mark.asyncio
    async def test_duplicate_runs_dont_overwrite(self, generator, tmp_path):
        r1 = await generator.generate("test topic")
        r2 = await generator.generate("test topic")
        assert r1.path != r2.path
        assert Path(r1.path).exists()
        assert Path(r2.path).exists()


# ############################################################################
#
#  Trigger detection
#
# ############################################################################

class TestTriggerDetection:

    @pytest.mark.parametrize("query", [
        "research climate change and save a report",
        "write a report about AI safety",
        "prepare a document about quantum computing",
        "make a markdown summary of machine learning",
        "save a summary on renewable energy",
        "create a research note about blockchain",
        "generate a report on economic trends and save it",
        "draft a report on supply chain issues",
        "produce a summary about climate policy",
    ])
    def test_strong_triggers(self, query):
        result = detect_document_intent(query)
        assert result is not None, f"Should trigger on: {query}"
        assert result["topic"]

    @pytest.mark.parametrize("query", [
        "research climate change",
        "look up quantum computing",
        "tell me about AI safety",
        "summarize the meeting",
        "what do we know about blockchain",
        "hello",
        "what's the weather like",
    ])
    def test_non_triggers(self, query):
        result = detect_document_intent(query)
        assert result is None, f"Should NOT trigger on: {query}"

    def test_detects_summary_type(self):
        result = detect_document_intent("write a summary about climate change")
        assert result is not None
        assert result["doc_type"] == "summary"

    def test_detects_report_type(self):
        result = detect_document_intent("write a report about climate change")
        assert result is not None
        assert result["doc_type"] == "report"

    def test_extracts_topic(self):
        result = detect_document_intent("create a report about quantum computing advances")
        assert result is not None
        assert "quantum computing" in result["topic"].lower()

    def test_case_insensitive(self):
        result = detect_document_intent("WRITE A REPORT about artificial intelligence")
        assert result is not None

    @pytest.mark.parametrize("query", [
        # Regression: a long, multi-sentence message where a save-verb and a
        # doc-noun merely CO-OCCUR (across sentences) must NOT trigger doc gen.
        # The old unbounded `.*` matched these and hijacked the turn with a
        # "Document saved" receipt, swallowing the real conversational reply.
        (
            "5c) Create a new dataframe from train_sales_data and call it "
            "train_sales_data_cook. This dataframe should exclude the outliers. "
            "Sort by Purchase_Amount and print the first 5 rows. 6b) create a new "
            "model and call it model_no_age. Print the model summary. 6c) Perform "
            "a partial F-test. What conclusion do you make regarding multicolinearity?"
        ),
        "create a new dataframe and then print the model summary",
        "generate the model object and read off the summary statistics",
        "I need to write code and later summarize the document I read",
    ])
    def test_incidental_cooccurrence_does_not_trigger(self, query):
        """Save-verb + doc-noun far apart in a long message must not fire."""
        assert detect_document_intent(query) is None, (
            f"Should NOT trigger (incidental co-occurrence): {query[:60]}"
        )

    @pytest.mark.parametrize("query", [
        # The doc-noun must still trigger when it is the (near) object of the
        # save-verb, even with an article + a modifier or two in between.
        "generate a markdown summary of the meeting",
        "save the research document on data centers",
        "can you write me a quick summary document",
        "create a brief report focusing on economic impact",
    ])
    def test_object_of_save_verb_still_triggers(self, query):
        assert detect_document_intent(query) is not None, (
            f"Should trigger (doc-noun is object of save-verb): {query}"
        )

    def test_buried_trigger_in_long_analytical_message_does_not_fire(self):
        """Regression: an analytical request fired doc-gen purely on a doc-phrase
        quoted DEEP inside a pasted proposal ("...a worker may write a final
        report..."). The actual ask has no doc-noun → should answer in chat.
        """
        filler = "Isolation has to be mechanical, not instruction-based. " * 30
        query = (
            "Evaluate this architecture proposal for Daemon and produce a "
            "concrete implementation plan, risks, and recommended first milestone. "
            + filler
            + "The branch may write a final report, but not mutate its manifest. "
            + filler
            + "whether this should integrate with the existing proposal system or "
            "remain external at first"
        )
        assert detect_document_intent(query) is None

    def test_head_request_with_pasted_material_still_fires(self):
        """A genuine doc request at the HEAD fires even in a long message with
        pasted material after it (content-aware path handles the material)."""
        filler = "Here is the full proposal text with lots of detail. " * 30
        query = "Write a report evaluating this proposal: " + filler
        assert detect_document_intent(query) is not None


# ############################################################################
#
#  Path safety
#
# ############################################################################

class TestPathSafety:

    def test_output_inside_documents(self, generator, tmp_path):
        path = generator._write_versioned_file("report", "normal topic", "content")
        assert str(path).startswith(str(tmp_path / "documents"))

    def test_traversal_in_topic_neutralized(self, generator, tmp_path):
        """Traversal chars in topic are stripped by slug, output stays safe."""
        path = generator._write_versioned_file("report", "../../../etc/passwd", "content")
        assert str(path.resolve()).startswith(str((tmp_path / "documents").resolve()))

    def test_traversal_in_doc_type_blocked(self, generator):
        # doc_type is used to build subdirectory name
        # _write_versioned_file prepends doc_type + "s", so "../../foo" becomes "../../foos"
        # The path safety check at the end should catch this
        with pytest.raises((PermissionError, OSError)):
            generator._write_versioned_file("../../etc", "test", "evil")

    def test_special_chars_in_topic_safe(self, generator):
        path = generator._write_versioned_file("report", "topic with <special> & chars!", "content")
        assert path.exists()
        assert "<" not in path.name
        assert ">" not in path.name
