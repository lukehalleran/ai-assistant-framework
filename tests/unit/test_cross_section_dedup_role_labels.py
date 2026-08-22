"""Cross-section dedup must catch the same turn rendered with different role labels.

Regression for 2026-08-05: the session's own turns appeared in BOTH
[RECENT CONVERSATION] ("User: ...\nDaemon: ...") and [RELEVANT MEMORIES]
("User: ...\nAssistant: ...") because the first-500-char dedup key kept the
mid-string role label — "daemon:" vs "assistant:" made the keys differ, so
three duplicates of today's turns were injected into one prompt.
"""
import pytest

from core.prompt.hygiene import ContentHygiene


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
