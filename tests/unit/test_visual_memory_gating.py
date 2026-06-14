"""Tests for visual-memory auto-injection gating in gatherer_knowledge.

Regression coverage for the 2026-06-09 incident: a technical project-update
message pulled a personal photo (user + cat) because the stored image-entity
``luke`` matched as a raw substring inside the path ``/home/lukeh/.claude/...``.

Two gates are exercised:
  1. Visual-intent gate (_query_wants_visual): a bare name mention does not
     surface a photo — the user must signal visual intent (a "show me" word) or
     the turn must be a recall intent.
  2. Word-boundary entity match (Step C): "luke" must NOT match "lukeh".
"""

import pytest

from core.prompt.gatherer_knowledge import (
    KnowledgeRetrievalMixin,
    _query_wants_visual,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class _FakeStore:
    def __init__(self, ids):
        self._ids = set(ids)

    def get_visual_entity_ids(self):
        return set(self._ids)


class _FakeRetriever:
    def __init__(self, ids):
        self._store = _FakeStore(ids)
        self.calls = []

    async def retrieve_visual_memories(self, query, max_images=3, target_entities=None):
        self.calls.append(set(target_entities) if target_entities else set())
        return {
            "text_results": [
                {"caption": "a cat", "image_path": "/x.jpg", "score": 0.9, "doc_id": "d1"}
            ],
            "images": [],
        }


class _FakeMC:
    graph_memory = None
    entity_resolver = None      # forces Step C (word-boundary) fallback
    chroma_store = None


class _Gatherer(KnowledgeRetrievalMixin):
    """Minimal harness exposing get_visual_memories with mocked collaborators."""

    def __init__(self, entity_ids):
        self.memory_id_map = {}
        self.memory_coordinator = _FakeMC()
        self._visual_retriever = _FakeRetriever(entity_ids)


@pytest.fixture(autouse=True)
def _enable_visual(monkeypatch):
    import config.app_config as ac
    monkeypatch.setattr(ac, "VISUAL_MEMORY_ENABLED", True, raising=False)


# ---------------------------------------------------------------------------
# _query_wants_visual — the visual-intent gate (pure function)
# ---------------------------------------------------------------------------

class TestQueryWantsVisual:
    @pytest.mark.parametrize("q", [
        "show me the cat",
        "can i see a picture",
        "what does my cat look like",
        "got any photos of biscuit",
        "pull up that pic",
    ])
    def test_visual_intent_words_pass(self, q):
        assert _query_wants_visual(q, None) is True

    def test_recall_intents_pass_without_visual_word(self):
        assert _query_wants_visual("do you remember biscuit", "factual_recall") is True
        assert _query_wants_visual("when did i get biscuit", "temporal_recall") is True

    @pytest.mark.parametrize("intent", [None, "project_work", "technical_help", "general"])
    def test_no_visual_word_and_non_recall_intent_fails(self, intent):
        # The exact failure shape from the incident: a technical message that
        # merely contains a name (or a path) and no visual intent.
        q = "the full revised plan is at /home/lukeh/.claude/plans/x.md"
        assert _query_wants_visual(q, intent) is False

    def test_punctuation_around_visual_word_still_passes(self):
        assert _query_wants_visual("show! me the cat.", None) is True


# ---------------------------------------------------------------------------
# get_visual_memories — end-to-end gating
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGetVisualMemoriesGating:
    async def test_path_substring_does_not_pull_photo(self):
        """The original bug: 'luke' must not match inside '/home/lukeh/...'.

        Visual intent is present ('show'), so only the word-boundary fix is
        under test here.
        """
        g = _Gatherer(["luke", "biscuit"])
        out = await g.get_visual_memories(
            "show me the file at /home/lukeh/.claude/plans/x.md", intent_type=None
        )
        assert out == {"text_results": [], "images": []}
        assert g._visual_retriever.calls == []  # retrieval never reached

    async def test_no_visual_intent_short_circuits_even_with_name_word(self):
        """The visual-intent gate fires before any entity work."""
        g = _Gatherer(["luke", "biscuit"])
        out = await g.get_visual_memories("luke approved the merge", intent_type="project_work")
        assert out == {"text_results": [], "images": []}
        assert g._visual_retriever.calls == []

    async def test_visual_word_plus_real_entity_pulls_photo(self):
        g = _Gatherer(["luke", "biscuit"])
        out = await g.get_visual_memories("show me luke", intent_type=None)
        assert out["text_results"]
        assert g._visual_retriever.calls == [{"luke"}]

    async def test_recall_intent_without_visual_word_pulls_photo(self):
        g = _Gatherer(["biscuit"])
        out = await g.get_visual_memories("do you remember biscuit", intent_type="factual_recall")
        assert out["text_results"]
        assert g._visual_retriever.calls == [{"biscuit"}]

    async def test_word_boundary_matches_standalone_name(self):
        """'luke' as a real word (not a substring) still matches under visual intent."""
        g = _Gatherer(["luke"])
        out = await g.get_visual_memories("show me luke, please", intent_type=None)
        assert g._visual_retriever.calls == [{"luke"}]
