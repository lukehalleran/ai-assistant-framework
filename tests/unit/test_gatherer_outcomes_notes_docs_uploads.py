"""Regression tests for CGR-20260913-007 #71-#74 (F7a,
docs/execution/generalization/failure_outcome_design.md, "F7 split and
gatherer outcome shape").

Root defect (verified against the deployed source before this batch):
`get_personal_notes`, `get_reference_docs`, `_fetch_upload_roster` and
`get_user_uploads` in core/prompt/gatherer_knowledge.py each collapse a
raised or producer-reported failure into a bare `[]` -- structurally
identical to a genuinely empty section (dm18_except_returns_empty). Their
producers already return a typed `OutcomeList` (knowledge/obsidian_manager.py
`get_notes`, F3b; knowledge/reference_docs_manager.py `get_documents`, F3a)
whose `.status`/`.reason` these gatherer methods never read before
flattening.

Drives the DEPLOYED `KnowledgeRetrievalMixin` methods directly (never a
re-derivation), the pattern tests/unit/test_obsidian_failure_outcomes.py and
tests/unit/test_upload_retrieval_pool.py already use:
`KnowledgeRetrievalMixin()` instantiated directly (no `__init__` override)
with a `MagicMock`/`AsyncMock` manager wired in. `get_personal_notes`,
`get_reference_docs` and `get_user_uploads` share an identical (query, limit)
call shape and outcome contract, so the shared cases below are parametrized
over all three instead of tripled. No real ChromaDB, embedder, vault or
network.

FAILING FIRST: run against the UNEDITED gatherer_knowledge.py (digest
77d094d6d6af46a5251296a30a5568889859a6407840945e23334cd8ec1885dc); failures
are recorded in batches/F7a.md.
"""
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.prompt.formatter import PromptFormatter
from core.prompt.gatherer_knowledge import KnowledgeRetrievalMixin
from tests.unit.test_independent_prompt_audit import full_builder, retrieval_limits
from utils.retrieval_outcome import OutcomeList, outcome_status

# Distinctive markers so a privacy assertion proves absence, not luck.
PRIVATE_EXC_TEXT = "leaked exception detail F7AQX9"
PRIVATE_QUERY = "sensitive question about TITLEF7AQX9 health"

# A query with none of _DOCUMENT_CONTEXT_RE's / _UPLOAD_FILENAME_TOKEN_RE's
# cue words -- isolates the notes/docs/uploads-leg tests from the roster.
NON_ROSTER_QUERY = "let's talk about something unrelated today"
# A query that DOES carry a document cue -- for the roster-specific tests.
ROSTER_QUERY = "what's in my homework document"

SECTIONS = ["personal_notes", "reference_docs", "user_uploads"]
_METHOD = {"personal_notes": "get_personal_notes", "reference_docs": "get_reference_docs",
           "user_uploads": "get_user_uploads"}
_MANAGER_ATTR = {"personal_notes": "obsidian_manager", "reference_docs": "reference_docs_manager",
                  "user_uploads": "reference_docs_manager"}
_PRODUCER_METHOD = {"personal_notes": "get_notes", "reference_docs": "get_documents",
                     "user_uploads": "get_documents"}
_EXTRA_ATTRS = {"personal_notes": {}, "reference_docs": {},
                 "user_uploads": {"_uploads_exist_cache": True}}


def _gatherer(**attrs):
    g = KnowledgeRetrievalMixin()
    g.memory_id_map = {}
    g._current_turn_upload_filenames = []
    for key, value in attrs.items():
        setattr(g, key, value)
    return g


def _note_item(title="Note A"):
    return {
        "content": (
            "This is a sufficiently long personal note chunk with real "
            f"prose content about {title} to clear the substance filter."
        ),
        "metadata": {"title": title, "section": "", "file_path": "vault/note.md"},
        "relevance_score": 0.7,
    }


def _refdoc_item(title="Doc A"):
    return {
        "content": f"Reference content about {title}.",
        "metadata": {"title": title, "section": "", "file_path": ""},
        "relevance_score": 0.8,
    }


def _upload_item(title="upload:Notes.txt", days_ago=0, score=0.9, match_type="semantic"):
    return {
        "content": f"Fresh upload content about {title}.",
        "metadata": {
            "type": "user_upload", "title": title,
            "timestamp": (datetime.now() - timedelta(days=days_ago)).isoformat(),
        },
        "relevance_score": score,
        "match_type": match_type,
    }


_ITEM = {"personal_notes": _note_item, "reference_docs": _refdoc_item,
          "user_uploads": _upload_item}


def _bound(section, producer_return=None, producer_side_effect=None, manager=None):
    """A gatherer with `section`'s manager wired to a fake producer; returns
    (gatherer, bound deployed method)."""
    mgr = manager if manager is not None else MagicMock()
    if manager is None:
        kwargs = ({"side_effect": producer_side_effect} if producer_side_effect is not None
                  else {"return_value": producer_return})
        setattr(mgr, _PRODUCER_METHOD[section], AsyncMock(**kwargs))
    attrs = {_MANAGER_ATTR[section]: mgr}
    attrs.update(_EXTRA_ATTRS[section])
    g = _gatherer(**attrs)
    return g, getattr(g, _METHOD[section])


# ===========================================================================
# #71/#72/#74 -- the shared leg contract (contract points 1, 2, 4, 6)
# ===========================================================================

class TestSectionOutcomes:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("section", SECTIONS)
    async def test_manager_raises_is_failed_and_empty(self, section):
        _, method = _bound(section, producer_side_effect=RuntimeError("boom"))
        result = await method(NON_ROSTER_QUERY, 10)

        assert outcome_status(result) == ("failed", "RuntimeError")
        assert result == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("section", SECTIONS)
    async def test_producer_failed_no_items_stays_failed(self, section):
        _, method = _bound(section, producer_return=OutcomeList.failed("StoreDown"))
        result = await method(NON_ROSTER_QUERY, 10)

        assert outcome_status(result) == ("failed", "StoreDown")
        assert result == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("section", SECTIONS)
    async def test_producer_unavailable_stays_unavailable(self, section):
        _, method = _bound(
            section, producer_return=OutcomeList.unavailable("collection_unavailable"))
        result = await method(NON_ROSTER_QUERY, 10)

        assert outcome_status(result) == ("unavailable", "collection_unavailable")
        assert result == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("section", SECTIONS)
    async def test_status_survives_the_transforms(self, section):
        """THE PROOF: items that pass every existing filter come back
        identically whether the leg is healthy or failed -- only the status
        differs. This is what dm18's flattening used to destroy."""
        make_item = _ITEM[section]
        items = [make_item("Item A"), make_item("Item B")]

        _, healthy_method = _bound(section, producer_return=OutcomeList(list(items)))
        healthy = await healthy_method(NON_ROSTER_QUERY, 10)

        _, failed_method = _bound(section, producer_return=OutcomeList(
            list(items), status="failed", reason="keyword:TimeoutError"))
        failed = await failed_method(NON_ROSTER_QUERY, 10)

        assert list(failed) == list(healthy)
        assert isinstance(failed, OutcomeList)
        assert failed.status == "failed"
        assert failed.reason == "keyword:TimeoutError"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("section", SECTIONS)
    async def test_control_healthy_nonempty_succeeds(self, section):
        _, method = _bound(section, producer_return=OutcomeList([_ITEM[section]()]))
        result = await method(NON_ROSTER_QUERY, 10)

        assert outcome_status(result) == ("succeeded", "")
        assert len(result) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("section", SECTIONS)
    async def test_control_healthy_empty_is_no_results(self, section):
        _, method = _bound(section, producer_return=OutcomeList([]))
        result = await method(NON_ROSTER_QUERY, 10)

        assert outcome_status(result) == ("no_results", "")
        assert result == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("section", SECTIONS)
    async def test_control_no_manager_is_no_results(self, section):
        g = _gatherer(**{_MANAGER_ATTR[section]: None})
        result = await getattr(g, _METHOD[section])(NON_ROSTER_QUERY, 10)

        assert result == []
        assert outcome_status(result) == ("no_results", "")


# ===========================================================================
# #73 -- _fetch_upload_roster's role inside get_user_uploads (contract pt. 3)
# ===========================================================================

class TestUploadRosterOutcomes:
    """A raising roster fetch marks the whole `user_uploads` section NOT
    CHECKED (F3b precedent), prefixed "roster:" so it is distinguishable
    from a documents-leg failure; a documents leg that already
    failed/unavailable wins over the roster."""

    @pytest.mark.asyncio
    async def test_raising_roster_fetch_marks_section_failed_items_kept(self):
        upload = _upload_item("upload:Kept.txt")
        mgr = MagicMock()
        mgr.get_documents = AsyncMock(return_value=OutcomeList([upload]))
        mgr.chroma_store._get_collection = MagicMock(side_effect=RuntimeError("boom"))
        g, _ = _bound("user_uploads", manager=mgr)

        result = await g.get_user_uploads(ROSTER_QUERY, 10)

        assert result.status == "failed"
        assert result.reason.startswith("roster:")
        assert "RuntimeError" in result.reason
        kept = [d for d in result if d.get("metadata", {}).get("type") == "user_upload"]
        assert len(kept) == 1
        assert kept[0]["metadata"]["title"] == "upload:Kept.txt"

    @pytest.mark.asyncio
    async def test_documents_leg_failure_wins_over_roster_failure(self):
        mgr = MagicMock()
        mgr.get_documents = AsyncMock(return_value=OutcomeList.failed("DocsDown"))
        mgr.chroma_store._get_collection = MagicMock(side_effect=RuntimeError("roster boom"))
        g, _ = _bound("user_uploads", manager=mgr)

        result = await g.get_user_uploads(ROSTER_QUERY, 10)

        assert result.status == "failed"
        assert result.reason == "DocsDown"
        assert not result.reason.startswith("roster:")

    @pytest.mark.asyncio
    async def test_non_roster_query_never_fetches_roster(self):
        mgr = MagicMock()
        mgr.get_documents = AsyncMock(return_value=OutcomeList([_upload_item()]))
        g, _ = _bound("user_uploads", manager=mgr)
        g._fetch_upload_roster = MagicMock(side_effect=AssertionError("must not be called"))

        result = await g.get_user_uploads(NON_ROSTER_QUERY, 10)

        assert outcome_status(result) == ("succeeded", "")
        assert g._last_upload_roster == []


# ===========================================================================
# Privacy: no reason string carries the query or exception text
# ===========================================================================

class TestPrivacyNoLeakedText:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("section", SECTIONS)
    async def test_reason_has_no_leaked_text(self, section):
        _, method = _bound(section, producer_side_effect=RuntimeError(PRIVATE_EXC_TEXT))
        result = await method(PRIVATE_QUERY, 10)

        assert PRIVATE_EXC_TEXT not in result.reason
        assert PRIVATE_QUERY not in result.reason

    @pytest.mark.asyncio
    async def test_roster_failure_reason_has_no_leaked_text(self):
        mgr = MagicMock()
        mgr.get_documents = AsyncMock(return_value=OutcomeList([]))
        mgr.chroma_store._get_collection = MagicMock(side_effect=RuntimeError(PRIVATE_EXC_TEXT))
        g, _ = _bound("user_uploads", manager=mgr)

        result = await g.get_user_uploads(f"{PRIVATE_QUERY} document", 10)

        assert PRIVATE_EXC_TEXT not in result.reason
        assert PRIVATE_QUERY not in result.reason


# ===========================================================================
# Through the deployed builder (F5 integrated) and formatter (F6a integrated)
# ===========================================================================

BUILDER_MARKER = "F7AQX9_sensitive_vault_detail_must_not_leak"


class TestThroughBuilderAndFormatter:
    @pytest.mark.asyncio
    async def test_failing_notes_manager_marks_section_failed_and_formatter_shows_could_not_check(
        self, monkeypatch
    ):
        builder = full_builder(
            monkeypatch, [{"query": "hi", "response": "hello"}], budget=10000)
        mgr = MagicMock()
        mgr.get_notes = AsyncMock(side_effect=RuntimeError(f"{BUILDER_MARKER} vault blew up"))
        notes_gatherer, _ = _bound("personal_notes", manager=mgr)
        builder.context_gatherer.get_personal_notes = notes_gatherer.get_personal_notes

        overrides = {**retrieval_limits(), "max_personal_notes": 5}
        result = await builder.build_prompt(
            "Synthetic question", retrieval_overrides=overrides)

        outcomes = result["_section_outcomes"]
        assert outcomes["personal_notes"] == {"status": "failed", "reason": "RuntimeError"}
        assert result["personal_notes"] == []
        assert BUILDER_MARKER not in str(result)

        monkeypatch.setattr("config.app_config.OBSIDIAN_ENABLED", True)
        fmt = PromptFormatter(token_manager=MagicMock(), time_manager=None)
        inventory = fmt._build_feature_inventory(result)

        assert "obsidian=ON(could not check)" in inventory
