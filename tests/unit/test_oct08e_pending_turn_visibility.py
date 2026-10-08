"""A delivered-but-not-yet-stored turn is still the newest turn
(2026-10-08, class: BC-38, BC-45). Drives THE deployed CorpusManager readers
(get_recent_memories / get_recent_within_hours) with the in-process pending
registry. Test text is synthetic."""
import json

import pytest

from memory.corpus_manager import CorpusManager


@pytest.fixture
def cm(tmp_path):
    import memory.pending_turns as pending_turns
    pending_turns._pending.clear()
    mgr = CorpusManager(str(tmp_path / "corpus.json"))
    mgr.add_entry("turn five question", "turn five answer", tags=["topic:general"])
    yield mgr
    pending_turns._pending.clear()


def _six(cm):
    import memory.pending_turns as pending_turns
    pending_turns.register(cm, "turn six question", "turn six answer", "turn six question")


def test_pending_turn_is_newest(cm):
    _six(cm)
    recent = cm.get_recent_memories(2)
    assert recent[0]["query"] == "turn six question"
    assert recent[0]["pending_storage"] is True
    assert recent[1]["query"] == "turn five question"
    within = cm.get_recent_within_hours(1)
    assert [e["query"] for e in within][:2] == ["turn six question", "turn five question"]


def test_no_duplicate_after_storage_lands(cm):
    import memory.pending_turns as pending_turns
    _six(cm)
    cm.add_entry("turn six question", "turn six answer", tags=["topic:general"])
    # Even before forget() runs, the corpus tail dedupes it.
    qs = [e["query"] for e in cm.get_recent_memories(5)]
    assert qs.count("turn six question") == 1
    pending_turns.forget(cm, "turn six question")
    qs = [e["query"] for e in cm.get_recent_memories(5)]
    assert qs.count("turn six question") == 1


def test_expired_entry_is_ignored(cm, monkeypatch):
    import memory.pending_turns as pending_turns
    _six(cm)
    monkeypatch.setattr(pending_turns, "PENDING_TURN_TTL_S", 0.0)
    assert [e["query"] for e in cm.get_recent_memories(5)] == ["turn five question"]


def test_failed_storage_leaves_no_ghost_after_forget_or_ttl(cm, monkeypatch):
    import memory.pending_turns as pending_turns
    _six(cm)
    pending_turns.forget(cm, "turn six question")   # what the task's finally does
    assert [e["query"] for e in cm.get_recent_memories(5)] == ["turn five question"]
    _six(cm)                                         # crashed task: only TTL removes it
    monkeypatch.setattr(pending_turns, "PENDING_TURN_TTL_S", 0.0)
    assert [e["query"] for e in cm.get_recent_memories(5)] == ["turn five question"]


def test_corpus_save_never_contains_pending(cm, tmp_path):
    _six(cm)
    cm.get_recent_memories(3)
    cm.get_recent_within_hours(1)
    cm.save_corpus()
    saved = json.loads((tmp_path / "corpus.json").read_text())
    assert all("turn six" not in json.dumps(e) for e in saved)
    assert all("pending_storage" not in e for e in saved)
    assert all("pending_storage" not in e for e in cm.corpus)


def test_pending_is_scoped_to_its_owner_and_capped(cm, tmp_path):
    import memory.pending_turns as pending_turns
    other = CorpusManager(str(tmp_path / "other.json"))
    _six(cm)
    assert other.get_recent_memories(3) == []
    for i in range(12):
        pending_turns.register(cm, f"q{i}", "a", None)
    assert len(pending_turns.pending_entries(cm)) == pending_turns.PENDING_TURN_CAP


def test_dispatch_storage_makes_the_turn_visible_until_storage_lands(cm):
    """Through the deployed handler: while store_interaction is still waiting
    (the personal-claim receipt window) the turn is already the newest."""
    import asyncio
    import types
    import gui.handlers as handlers

    async def scenario():
        release = asyncio.Event()

        class _Mem:
            corpus_manager = cm

            async def store_interaction(self, query, response, **kw):
                await release.wait()
                cm.add_entry(query, response, tags=["topic:general"])
                return "id-6"

        orch = types.SimpleNamespace(memory_system=_Mem(), current_topic="general")
        logger = types.SimpleNamespace(log_interaction=lambda **k: None)
        task = handlers._dispatch_storage(
            orch, "turn six question", "turn six answer", "turn six question",
            "turn six answer", "default", [], logger, None, None, "enhanced",
        )
        await asyncio.sleep(0)
        newest = cm.get_recent_memories(1)[0]
        assert newest["query"] == "turn six question"
        release.set()
        await task
        qs = [e["query"] for e in cm.get_recent_memories(5)]
        assert qs.count("turn six question") == 1
        assert "pending_storage" not in cm.get_recent_memories(1)[0]

    asyncio.run(scenario())
