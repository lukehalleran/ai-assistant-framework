"""Daily notes count REAL conversation turns, dated by content.

Incident 2026-10-20, classes BC-96 / BC-52: a 00:02 shutdown wrote a note for
the new day from a consolidated SUMMARY entry (created 00:01, covering the two
days before) and at 03:59 counted 6 "conversations" when 4 were real (the other
two: that summary and a reflection). main.py also asked for ``date.today()``
instead of the content day(s) of the session being closed.
"""

import asyncio
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory.corpus_manager import CorpusManager, is_consolidated_entry
from utils.daily_notes_generator import DailyNotesGenerator


def _turn(ts, q="what is the plan for the Tamsin project"):
    return {"query": q, "response": "a reply", "timestamp": ts}


def _entries():
    return [
        _turn(datetime(2026, 10, 19, 21, 0)),
        _turn(datetime(2026, 10, 19, 22, 41), "second real turn"),
        {"query": "", "response": "summary of the last two days", "type": "summary",
         "timestamp": datetime(2026, 10, 20, 0, 1)},
        {"query": "", "response": "reflection", "type": "reflection",
         "timestamp": datetime(2026, 10, 20, 0, 1)},
        {"query": "", "response": "tagged summary", "tags": ["@summary"],
         "timestamp": datetime(2026, 10, 20, 0, 2)},
        _turn(datetime(2026, 10, 20, 3, 59), "a real turn after midnight"),
    ]


@pytest.fixture
def corpus(tmp_path):
    cm = CorpusManager.__new__(CorpusManager)
    cm.corpus = _entries()
    cm._episodic_cache = None
    return cm


@pytest.fixture
def gen(corpus):
    with patch("utils.daily_notes_generator.DailyNotesGenerator.__init__", lambda self, **kw: None):
        g = DailyNotesGenerator.__new__(DailyNotesGenerator)
    g._corpus_manager = corpus
    return g


def test_consolidated_predicate_drops_summaries_and_reflections(corpus):
    assert len(corpus.derivable_entries()) == 6  # base iterator keeps them
    real = [e["query"] for e in corpus.derivable_entries() if not is_consolidated_entry(e)]
    assert real == ["what is the plan for the Tamsin project", "second real turn",
                    "a real turn after midnight"]


def test_note_for_the_new_day_counts_only_the_real_turn(gen):
    got = gen._get_conversations_for_date(date(2026, 10, 20))
    assert [c["query"] for c in got] == ["a real turn after midnight"]


def test_content_dates_follow_the_session_turns_not_write_times(gen):
    # Session started the evening of 10-19; the 00:01 summary must not add 10-20
    # on its own -- only the real 03:59 turn does here.
    assert gen.content_dates_since(datetime(2026, 10, 19, 20, 0)) == [
        date(2026, 10, 19), date(2026, 10, 20)]
    gen._corpus_manager.corpus = _entries()[:5]  # no real turn after midnight
    assert gen.content_dates_since(datetime(2026, 10, 19, 20, 0)) == [date(2026, 10, 19)]


def test_shutdown_writes_yesterdays_note_only(gen, monkeypatch):
    """Drive the deployed shutdown sequence at a 00:02 'now': the session's real
    turns are all dated 10-19, a 00:01 summary exists for 10-20."""
    import main

    gen._corpus_manager.corpus = _entries()[:5]
    gen.generate_for_date = AsyncMock(return_value=SimpleNamespace(
        success=True, skipped_reason=None, error=None, conversation_count=2))
    monkeypatch.setattr(main.daily_notes_generator, "DailyNotesGenerator", lambda **kw: gen)
    monkeypatch.setattr(main.app_config, "DAILY_NOTES_ENABLED", True)
    monkeypatch.setattr(main.app_config, "CURATION_ENABLED", False, raising=False)

    class _Date(date):
        @classmethod
        def today(cls):
            return cls(2026, 10, 20)

    monkeypatch.setattr(main, "date", _Date)
    ms = SimpleNamespace(
        corpus_manager=gen._corpus_manager,
        session_start=datetime(2026, 10, 19, 20, 0),
        run_shutdown_reflection=AsyncMock(), process_shutdown_memory=AsyncMock(),
        run_synthesis_dreaming=AsyncMock(),
    )
    orch = SimpleNamespace(memory_system=ms, model_manager=MagicMock())

    asyncio.run(main._do_shutdown_async(orch, [], []))
    called = [c.args[0] for c in gen.generate_for_date.call_args_list]
    assert called == [date(2026, 10, 19)]
