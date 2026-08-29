"""Cross-section dedup must catch the same turn rendered with different role labels.

Regression for 2026-08-05: the session's own turns appeared in BOTH
[RECENT CONVERSATION] ("User: ...\nDaemon: ...") and [RELEVANT MEMORIES]
("User: ...\nAssistant: ...") because the first-500-char dedup key kept the
mid-string role label — "daemon:" vs "assistant:" made the keys differ, so
three duplicates of today's turns were injected into one prompt.

Regression for 2026-08-28: the duplicates came BACK because production
recent_conversations items are corpus entries with separate query/response
fields (no "content" key) — the old dedup keyed those on response-only while
retrieval docs keyed on content starting with the query, so the two shapes of
the SAME turn could never collide (4 of 8 prompts in one session carried
today's own turns twice, including a ~1,200-word email paste). All dedup
passes now key through _canonical_turn_key, which builds the same composite
for both shapes.
"""
import pytest

from core.prompt.hygiene import ContentHygiene, _canonical_turn_key


class _StubCoordinator:
    corpus_manager = None


def _hygiene():
    return ContentHygiene(memory_coordinator=_StubCoordinator(), context_gatherer=None)


def _ctx(recent, memories):
    return {
        "recent_conversations": recent,
        "memories": memories,
    }


QUERY = "I was taking about 900 mg a day for a week or so."
RESPONSE = "That's a completely reasonable line to hold, and honestly a responsible one."


@pytest.mark.asyncio
async def test_daemon_vs_assistant_label_deduped():
    recent = [{"content": f"User: {QUERY}\nDaemon: {RESPONSE}"}]
    memories = [{"content": f"User: {QUERY}\nAssistant: {RESPONSE}"}]
    out = await _hygiene()._hygiene_and_caps(_ctx(recent, memories))
    assert len(out["recent_conversations"]) == 1
    assert out["memories"] == [], "memory duplicating a recent turn must be dropped"


@pytest.mark.asyncio
async def test_whitespace_variance_deduped():
    recent = [{"content": f"User: {QUERY}\nDaemon: {RESPONSE}"}]
    memories = [{"content": f"User:  {QUERY}\n\nAssistant:   {RESPONSE}"}]
    out = await _hygiene()._hygiene_and_caps(_ctx(recent, memories))
    assert out["memories"] == []


@pytest.mark.asyncio
async def test_distinct_memory_survives():
    recent = [{"content": f"User: {QUERY}\nDaemon: {RESPONSE}"}]
    memories = [
        {"content": f"User: {QUERY}\nAssistant: {RESPONSE}"},
        {"content": "User: Tell me about FAISS.\nAssistant: FAISS is a vector index library."},
    ]
    out = await _hygiene()._hygiene_and_caps(_ctx(recent, memories))
    assert len(out["memories"]) == 1
    assert "FAISS" in out["memories"][0]["content"]


@pytest.mark.asyncio
async def test_plain_identical_content_still_deduped():
    # The pre-existing behavior (exact same string) must keep working.
    recent = [{"content": f"User: {QUERY}\nDaemon: {RESPONSE}"}]
    memories = [{"content": f"User: {QUERY}\nDaemon: {RESPONSE}"}]
    out = await _hygiene()._hygiene_and_caps(_ctx(recent, memories))
    assert out["memories"] == []


# --- 2026-08-28 regression: production shapes ---
# recent_conversations items in production are CORPUS entries: separate
# query/response fields, NO "content" key. Retrieval docs carry content.


@pytest.mark.asyncio
async def test_corpus_shape_vs_retrieval_content_deduped():
    recent = [{"query": QUERY, "response": RESPONSE, "timestamp": "2026-08-28T14:40:00"}]
    memories = [{"content": f"User: {QUERY}\nAssistant: {RESPONSE}"}]
    out = await _hygiene()._hygiene_and_caps(_ctx(recent, memories))
    assert len(out["recent_conversations"]) == 1
    assert out["memories"] == [], (
        "retrieval doc duplicating a corpus-shaped recent turn must be dropped"
    )


@pytest.mark.asyncio
async def test_corpus_shape_vs_daemon_labeled_content_deduped():
    recent = [{"query": QUERY, "response": RESPONSE}]
    memories = [{"content": f"User: {QUERY}\nDaemon: {RESPONSE}"}]
    out = await _hygiene()._hygiene_and_caps(_ctx(recent, memories))
    assert out["memories"] == []


@pytest.mark.asyncio
async def test_long_paste_query_deduped_across_shapes():
    # The live payloads were long email pastes — the query alone exceeds the
    # 500-char key window, so the collision must happen on query text alone.
    long_query = "I sent these. Hi Morgan and Robin, " + ("thank you for the help this summer. " * 30)
    recent = [{"query": long_query, "response": RESPONSE}]
    memories = [{"content": f"User: {long_query}\nAssistant: {RESPONSE}"}]
    out = await _hygiene()._hygiene_and_caps(_ctx(recent, memories))
    assert out["memories"] == []


@pytest.mark.asyncio
async def test_distinct_corpus_turns_not_collapsed():
    # Two different turns sharing neither query nor response must both survive.
    recent = [
        {"query": QUERY, "response": RESPONSE},
        {"query": "Totally different question about FAISS?", "response": "FAISS is a vector index library."},
    ]
    out = await _hygiene()._hygiene_and_caps(_ctx(recent, []))
    assert len(out["recent_conversations"]) == 2


def test_canonical_key_shape_independent():
    corpus_item = {"query": QUERY, "response": RESPONSE}
    content_item = {"content": f"User: {QUERY}\nAssistant: {RESPONSE}"}
    daemon_item = {"content": f"User:  {QUERY}\n\nDaemon:   {RESPONSE}"}
    assert _canonical_turn_key(corpus_item) == _canonical_turn_key(content_item)
    assert _canonical_turn_key(corpus_item) == _canonical_turn_key(daemon_item)


def test_canonical_key_empty_item():
    assert _canonical_turn_key({}) == ""
    assert _canonical_turn_key({"query": "", "response": ""}) == ""
