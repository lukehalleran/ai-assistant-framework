"""CGR-20260913-007 (F7c): the remaining knowledge gatherers — git commits,
proposed features, procedural skills, graph context, unresolved threads,
narrative, daemon self-notes and relevant emails — return a typed failure
instead of an empty section (ANCHORS #75-#79, #81-#84).

Drives the DEPLOYED ``KnowledgeRetrievalMixin`` methods directly through a
minimal subclass, plus one builder-level narrative test. Fakes only
(``MagicMock``/``AsyncMock``/``SimpleNamespace``): no real ChromaDB,
embedder, graph, email service or network anywhere. Reuses the
``full_builder``/``retrieval_limits`` fake-builder pattern established by
``tests/unit/test_independent_prompt_audit.py`` (also used by F5/F7a/F7b)
and the ``_run_gatherer`` email-fake pattern from
``tests/unit/test_email_passive_context.py`` — both read-only precedent,
not imported (small local copies to avoid cross-test-module coupling).
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from core.email.provider import EmailMessage
from core.prompt.builder import UnifiedPromptBuilder
from core.prompt.gatherer_knowledge import KnowledgeRetrievalMixin
from core.prompt.token_manager import TokenManager
from utils.retrieval_outcome import outcome_status

MARKER = "F7CQX9"


class _G(KnowledgeRetrievalMixin):
    """Bare mixin host — only the attributes each method actually reads."""

    def __init__(self):
        self.memory_coordinator = None
        self.memory_id_map = {}
        self._chroma_store = None
        self._distress_active = False


# ---------------------------------------------------------------------------
# git_commits / proposed_features / procedural_skills / graph_context /
# unresolved_threads / daemon_self_notes — six direct-swap sites, uniform
# (query/kwargs, feature flag, raising producer) -> OutcomeList.failed(...)
# contract. Parametrized across sections to stay within the size cap.
# ---------------------------------------------------------------------------

GIT_QUERY = f"quasar telescope calibration notes {MARKER}"
PROPOSALS_QUERY = f"refactor the quasar caching pipeline {MARKER}"
SKILLS_QUERY = f"how do I chop onions for the quasar potluck {MARKER}"
GRAPH_QUERY = f"quasar research notes {MARKER}"
SELF_NOTES_QUERY = f"quasar deployment status {MARKER}"


def _git_case(monkeypatch, mode):
    monkeypatch.setattr("config.app_config.GIT_MEMORY_ENABLED", mode != "disabled")
    chroma = MagicMock()
    chroma.collections = {"procedural": object()}
    if mode == "raise":
        chroma.get_recent.side_effect = RuntimeError(f"boom {MARKER}")
    elif mode == "nonempty":
        chroma.get_recent.return_value = [
            {"id": "c1", "content": "fix: quasar bug", "metadata": {"timestamp": "t", "commit_hash": "abc"}}
        ]
        chroma.query_collection.return_value = []
    else:
        chroma.get_recent.return_value = []
        chroma.query_collection.return_value = []
    g = _G()
    g.memory_coordinator = SimpleNamespace(chroma_store=chroma)
    return g, lambda: g.get_git_commits(GIT_QUERY, limit=5)


def _proposals_case(monkeypatch, mode):
    monkeypatch.setattr("config.app_config.CODE_PROPOSALS_PROMPT_ENABLED", mode != "disabled")
    if mode == "raise":
        pf = SimpleNamespace(get_proposals=AsyncMock(side_effect=RuntimeError(f"boom {MARKER}")))
    elif mode == "nonempty":
        pf = SimpleNamespace(get_proposals=AsyncMock(return_value=[
            {"content": "Add quasar caching", "metadata": {
                "created_at": "t", "title": "Quasar cache", "priority": 3, "proposal_id": "p1"},
             "relevance_score": 0.9},
        ]))
    else:
        pf = SimpleNamespace(get_proposals=AsyncMock(return_value=[]))
    g = _G()
    g._proposal_filter = pf
    return g, lambda: g.get_proposed_features(PROPOSALS_QUERY, limit=3)


def _skills_case(monkeypatch, mode):
    monkeypatch.setattr("config.app_config.PROCEDURAL_SKILLS_ENABLED", mode != "disabled")
    if mode == "raise":
        get_skills = AsyncMock(side_effect=RuntimeError(f"boom {MARKER}"))
    elif mode == "nonempty":
        get_skills = AsyncMock(return_value=[
            {"metadata": {"created_at": "t", "category": "cooking"}, "relevance_score": 0.5, "id": "s1"}
        ])
    else:
        get_skills = AsyncMock(return_value=[])
    g = _G()
    g.memory_coordinator = SimpleNamespace(get_skills=get_skills)
    return g, lambda: g.get_procedural_skills(SKILLS_QUERY, limit=5)


def _graph_case(monkeypatch, mode):
    monkeypatch.setattr("config.app_config.KNOWLEDGE_GRAPH_ENABLED", mode != "disabled")
    monkeypatch.setattr("config.app_config.KNOWLEDGE_GRAPH_RETRIEVAL_DEPTH", 2)
    monkeypatch.setattr("config.app_config.ENABLE_GRAPH_ATTRIBUTION", False)
    graph = MagicMock()
    graph.node_count.return_value = 5
    graph.get_entity.return_value = None
    if mode == "raise":
        graph.get_context_sentences.side_effect = RuntimeError(f"boom {MARKER}")
    elif mode == "nonempty":
        graph.get_context_sentences.return_value = ["Quasar node relates to the research project."]
    else:
        graph.get_context_sentences.return_value = []
    resolver = MagicMock()
    resolver.resolve.side_effect = lambda phrase: "gqnode_quasar" if phrase == "quasar" else None
    g = _G()
    g.memory_coordinator = SimpleNamespace(graph_memory=graph, entity_resolver=resolver)
    return g, lambda: g.get_graph_context(GRAPH_QUERY, max_sentences=5)


def _threads_case(monkeypatch, mode):
    monkeypatch.setattr("config.app_config.THREAD_SURFACING_ENABLED", mode != "disabled")
    if mode == "raise":
        get_unresolved = MagicMock(side_effect=RuntimeError(f"boom {MARKER}"))
    elif mode == "nonempty":
        get_unresolved = MagicMock(return_value=[
            {"topic": "quasar review", "thread_type": "deadline", "urgency": 0.5}
        ])
    else:
        get_unresolved = MagicMock(return_value=[])
    g = _G()
    g.memory_coordinator = SimpleNamespace(get_unresolved_threads=get_unresolved)
    return g, lambda: g.get_unresolved_threads(max_results=3)


def _self_notes_case(monkeypatch, mode):
    chroma = MagicMock()
    if mode == "raise":
        chroma.query_collection.side_effect = RuntimeError(f"boom {MARKER}")
    elif mode == "nonempty":
        chroma.query_collection.return_value = [{"content": "Remember the quasar deadline.", "metadata": {}}]
    else:
        chroma.query_collection.return_value = []
    g = _G()
    g._chroma_store = chroma
    return g, lambda: g.get_daemon_self_notes(SELF_NOTES_QUERY, limit=3)


SECTION_BUILDERS = {
    "git_commits": (_git_case, "RuntimeError"),
    "proposed_features": (_proposals_case, "RuntimeError"),
    "procedural_skills": (_skills_case, "RuntimeError"),
    "graph_context": (_graph_case, "RuntimeError"),
    "unresolved_threads": (_threads_case, "RuntimeError"),
    "daemon_self_notes": (_self_notes_case, "RuntimeError"),
}
# daemon_self_notes has no feature flag gate — excluded from the disabled check.
DISABLED_SECTIONS = ["git_commits", "proposed_features", "procedural_skills",
                      "graph_context", "unresolved_threads"]


@pytest.mark.asyncio
@pytest.mark.parametrize("section", sorted(SECTION_BUILDERS))
async def test_raising_producer_is_failed(monkeypatch, section):
    """ANCHORS #75-#79, #82: a raising producer marks the section failed,
    keeps zero items, and the reason is only the exception class name —
    never the marker embedded in the query or the raised message."""
    build, exc_name = SECTION_BUILDERS[section]
    _g, call = build(monkeypatch, "raise")
    result = await call()
    status, reason = outcome_status(result)
    assert (status, reason) == ("failed", exc_name)
    assert result == []
    assert MARKER not in reason


@pytest.mark.asyncio
@pytest.mark.parametrize("section", sorted(SECTION_BUILDERS))
async def test_control_healthy_empty_is_no_results(monkeypatch, section):
    build, _exc = SECTION_BUILDERS[section]
    _g, call = build(monkeypatch, "empty")
    result = await call()
    assert outcome_status(result) == ("no_results", "")
    assert result == []


@pytest.mark.asyncio
@pytest.mark.parametrize("section", sorted(SECTION_BUILDERS))
async def test_control_healthy_nonempty_is_succeeded(monkeypatch, section):
    build, _exc = SECTION_BUILDERS[section]
    _g, call = build(monkeypatch, "nonempty")
    result = await call()
    status, _reason = outcome_status(result)
    assert status == "succeeded"
    assert len(result) >= 1


@pytest.mark.asyncio
@pytest.mark.parametrize("section", DISABLED_SECTIONS)
async def test_control_disabled_flag_returns_empty_unchanged(monkeypatch, section):
    """Contract point 4: the legit `if not X_ENABLED: return []` early
    returns stay untouched — still a plain no_results empty list."""
    build, _exc = SECTION_BUILDERS[section]
    _g, call = build(monkeypatch, "disabled")
    result = await call()
    assert result == []
    assert outcome_status(result) == ("no_results", "")


# ---------------------------------------------------------------------------
# get_relevant_emails — ANCHOR #83 (inner relevance-scoring except, fail
# closed to "relevance_unavailable") and ANCHOR #84 (outer except).
# ---------------------------------------------------------------------------

def _run_email_gatherer(monkeypatch, messages, query, embedder_factory=None,
                         service_search=None):
    """Fake service + fake contacts + deterministic embedder, driving the
    DEPLOYED KnowledgeRetrievalMixin.get_relevant_emails (returns a coroutine
    to be awaited by the caller)."""
    import core.actions.google_contacts as gc
    import core.email.service as svc
    from models.model_manager import ModelManager

    monkeypatch.setattr("config.app_config.EMAIL_PASSIVE_CONTEXT_ENABLED", True)

    class _FakeService:
        async def search(self, *a, **k):
            if service_search is not None:
                return service_search(*a, **k)
            return messages

    async def _fake_resolve(name, **k):
        return []

    class _FakeEmbedder:
        def encode(self, text, **k):
            return np.array([1.0, 0.0])

    monkeypatch.setattr(svc, "get_email_service", lambda: _FakeService())
    monkeypatch.setattr(gc, "resolve_contact", _fake_resolve)
    ef = embedder_factory or (lambda: _FakeEmbedder())
    monkeypatch.setattr(ModelManager, "_get_cached_embedder", staticmethod(ef))

    g = _G()
    return g.get_relevant_emails(query)


@pytest.mark.asyncio
async def test_raising_embedder_is_failed_relevance_unavailable(monkeypatch):
    """ANCHOR #83: fail-closed, but typed instead of a bare []."""
    msg = EmailMessage(provider="gmail", message_id="e1", sender="A <a@example.com>",
                        subject="Update", snippet="quasar project update",
                        date="2026-09-01T10:00:00")

    def _raise():
        raise RuntimeError(f"boom {MARKER}")

    result = await _run_email_gatherer(monkeypatch, [msg], "check my inbox",
                                        embedder_factory=_raise)
    status, reason = outcome_status(result)
    assert (status, reason) == ("failed", "relevance_unavailable")
    assert result == []
    assert MARKER not in reason


@pytest.mark.asyncio
async def test_raising_service_search_is_failed(monkeypatch):
    """ANCHOR #84: the outer except."""
    def _raise_search(*a, **k):
        raise RuntimeError(f"boom {MARKER}")

    result = await _run_email_gatherer(monkeypatch, [], "check my inbox",
                                        service_search=_raise_search)
    status, reason = outcome_status(result)
    assert (status, reason) == ("failed", "RuntimeError")
    assert result == []
    assert MARKER not in reason


@pytest.mark.asyncio
async def test_control_no_cue_is_no_results(monkeypatch):
    result = await _run_email_gatherer(monkeypatch, [], "what should I cook tonight")
    assert outcome_status(result) == ("no_results", "")
    assert result == []


@pytest.mark.asyncio
async def test_control_healthy_ranked_messages_succeeds(monkeypatch):
    msg = EmailMessage(provider="gmail", message_id="e2", sender="A <a@example.com>",
                        subject="Quasar", snippet="quasar project status",
                        date="2026-09-01T10:00:00")
    result = await _run_email_gatherer(monkeypatch, [msg], "check my inbox for quasar updates")
    assert len(result) == 1
    assert outcome_status(result) == ("succeeded", "")


# ---------------------------------------------------------------------------
# get_narrative_context — ANCHOR #81 (str return; the except re-raises
# instead of swallowing, so the builder's existing try/except — F5-
# integrated — records the failure).
# ---------------------------------------------------------------------------

class _NarrativeSource(KnowledgeRetrievalMixin):
    """Wraps only what get_narrative_context reads."""

    def __init__(self, corpus_manager):
        self.memory_coordinator = SimpleNamespace(corpus_manager=corpus_manager)


def test_narrative_producer_raise_reraises():
    corpus = SimpleNamespace(get_narrative_context=MagicMock(side_effect=RuntimeError(f"boom {MARKER}")))
    g = _NarrativeSource(corpus)
    with pytest.raises(RuntimeError):
        g.get_narrative_context()


def test_narrative_disabled_flag_returns_empty_string(monkeypatch):
    monkeypatch.setattr("config.app_config.NARRATIVE_CONTEXT_ENABLED", False)
    corpus = SimpleNamespace(get_narrative_context=MagicMock(side_effect=AssertionError("must not be called")))
    g = _NarrativeSource(corpus)
    assert g.get_narrative_context() == ""


def test_narrative_missing_corpus_manager_returns_empty_string(monkeypatch):
    monkeypatch.setattr("config.app_config.NARRATIVE_CONTEXT_ENABLED", True)
    g = _NarrativeSource(None)
    assert g.get_narrative_context() == ""


class _CharacterTokenizer:
    """Token-dense input: one token per character (test_independent_prompt_audit.py precedent)."""

    def count_tokens(self, text, model_name):
        return len(text)


def _retrieval_limits():
    return dict.fromkeys([
        "max_mems", "max_summaries", "max_reflections", "max_dreams", "max_semantic",
        "max_wiki", "max_skills", "max_proposals", "max_git_commits",
        "max_surfaced_threads", "max_reference_docs", "max_user_uploads",
        "max_proactive", "max_visual_memories", "max_personal_notes", "max_graph_sentences",
    ], 0) | {"max_recent": 2}


def _full_builder(monkeypatch, corpus_manager):
    """Deployed full builder, no optional stores/models — F5/F7a/F7b's
    `full_builder`/`retrieval_limits` fake-builder pattern, a local copy."""
    builder = UnifiedPromptBuilder.__new__(UnifiedPromptBuilder)
    builder.model_manager = SimpleNamespace(
        get_active_model_name=lambda: "audit",
        generate_once=AsyncMock(return_value="Compressed evidence with provenance."),
        active_model_name="audit",
    )
    builder.token_manager = TokenManager(
        SimpleNamespace(get_active_model_name=lambda: "audit"), _CharacterTokenizer(), 10000)
    builder._llm_compress_cache = {}
    builder.time_manager = None
    builder.memory_coordinator = SimpleNamespace(
        scorer=None, chroma_store=None, corpus_manager=corpus_manager,
        get_summaries=lambda count: [], get_reflections=AsyncMock(return_value=[]),
    )
    builder._skill_activation_policy = None
    builder._should_use_light_path = lambda *args: False
    builder._is_continuation_answer = lambda *args: False
    builder._hygiene_and_caps = AsyncMock(side_effect=lambda context, **kw: context)
    builder.context_gatherer = SimpleNamespace(
        memory_id_map={}, clear_memory_id_map=lambda: None,
        _get_recent_conversations=AsyncMock(return_value=[]),
        get_user_profile_context=AsyncMock(return_value=""),
        _get_web_search_results=AsyncMock(return_value=[]),
        get_narrative_context=_NarrativeSource(corpus_manager).get_narrative_context,
    )
    for name in ("GOOGLE_CALENDAR_ENABLED", "EMAIL_PASSIVE_CONTEXT_ENABLED", "DAEMON_NOTES_ENABLED"):
        monkeypatch.setattr("config.app_config." + name, False)
    monkeypatch.setattr("core.prompt.builder.LLM_COMPRESSION_ENABLED", False)
    return builder


@pytest.mark.asyncio
async def test_through_builder_narrative_raise_recorded_failed(monkeypatch):
    """Contract point 3's builder-level proof: the prompt still builds,
    narrative_state stays "", and F5's _section_outcomes records failed."""
    corpus = SimpleNamespace(get_narrative_context=MagicMock(side_effect=RuntimeError(f"boom {MARKER}")))
    builder = _full_builder(monkeypatch, corpus)
    result = await builder.build_prompt("Estimate the homework effort",
                                         retrieval_overrides=_retrieval_limits())
    assert "_build_time" in result, "builder must not silently return its error fallback"
    assert result["narrative_state"] == ""
    assert result["_section_outcomes"]["narrative"] == {"status": "failed", "reason": "RuntimeError"}
    assert MARKER not in result["_section_outcomes"]["narrative"]["reason"]


@pytest.mark.asyncio
async def test_through_builder_max_narrative_zero_skips_entry_control(monkeypatch):
    """Control (F5 precedent): the max_narrative=0 gate still skips the
    section entirely — no entry, and the producer is never even called."""
    corpus = SimpleNamespace(get_narrative_context=MagicMock(return_value="today's narrative"))
    builder = _full_builder(monkeypatch, corpus)
    overrides = _retrieval_limits() | {"max_narrative": 0}
    result = await builder.build_prompt("Estimate the homework effort", retrieval_overrides=overrides)
    assert result["narrative_state"] == ""
    assert "narrative" not in result["_section_outcomes"]
    corpus.get_narrative_context.assert_not_called()
