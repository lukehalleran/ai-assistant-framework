"""STM and tone windows drop owner [test] probe turns from PRIOR sessions.

Incident 2026-10-18, class BC-58 (test-state isolation): #91 stopped probe
turns reaching [RECENT CONVERSATION] and retrieval, but the STM analyzer's
window (``_analyze_stm``) and the tone detector's history (``_detect_tone``)
read ``corpus_manager`` directly and still saw every probe. The CURRENT
session's probes stay (a multi-turn probe needs its own previous exchange);
prior-session probes are dropped through THE session helper
``core.prompt.token_manager.drop_prior_session_test_turns``.

Each test drives the deployed method with a fake corpus holding a real and a
probe turn from the current session and from a prior session, over the
inline / whole-line envelope forms, clean and wrapped/indented. Invented
content only.

2026-10-23, class BC-94 (wall-clock fixture crossing a calendar boundary):
the entries were stamped with a wall-clock now minus N hours while the
session helper reads its own ``datetime.now()`` and treats a date change as a
session boundary, so the "current" probe (0.2 h ago) was YESTERDAY for the first
12 minutes of every day (and a fixture built before midnight was read after
it) and the prior-session probe survived. The clock the deployed code reads is
now pinned to the instant the fixtures are built from.
"""

import asyncio
from datetime import datetime, timedelta

import pytest

from core.context_pipeline import ContextPipeline

BODY = "Do you remember what we said about the budget sheet yesterday?"
PROBE_FORMS = {
    "inline": f"[test]{BODY}[/test]",
    "inline_wrapped": f"   [test]{BODY}[/test]   ",
    "whole_line": f"[test]\n{BODY}\n[/test]",
    "whole_line_indented": "  [test]\n  Do you remember what we said\n"
                           "  about the budget sheet yesterday?\n  [/test]",
}
FORMS = pytest.mark.parametrize("probe", list(PROBE_FORMS.values()), ids=list(PROBE_FORMS))


PINNED_NOW = datetime(2026, 10, 18, 15, 0, 0)


class _PinnedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return PINNED_NOW


@pytest.fixture(autouse=True)
def _pinned_clock(monkeypatch):
    """Every clock the deployed STM/tone/session path reads is PINNED_NOW."""
    import core.context_pipeline as cp
    import core.prompt.token_manager as tm

    monkeypatch.setattr(tm, "datetime", _PinnedDatetime)
    monkeypatch.setattr(cp, "datetime", _PinnedDatetime)


def _entry(hours_ago, query):
    ts = (PINNED_NOW - timedelta(hours=hours_ago)).isoformat()
    return {"query": query, "response": f"reply to {query[:20]}", "timestamp": ts}


def _window(probe):
    """Newest-first: current-session probe + real, prior-session probe + real."""
    return [
        _entry(0.2, probe + " current"),
        _entry(0.5, "Tamsin wants the budget sheet split into two tabs."),
        _entry(8.0, probe + " prior"),
        _entry(8.5, "Rowan sent the lease paperwork on Friday."),
    ]


class _WindowCorpus:
    def __init__(self, entries):
        self._entries = entries

    def get_recent_within_hours(self, hours=24, max_count=30):
        return list(self._entries)[:max_count]

    def get_recent_memories(self, count=3):
        return list(self._entries)[:count]


class _LegacyCorpus:
    def __init__(self, entries):
        self._entries = entries

    def get_recent_memories(self, count=3):
        return list(self._entries)[:count]


class _Analyzer:
    def __init__(self):
        self.recent = None

    async def analyze(self, recent_memories, user_query, last_assistant_response, graph_memory):
        self.recent = list(recent_memories)
        return {"topic": "x"}


def _pipeline(corpus):
    p = ContextPipeline.__new__(ContextPipeline)
    p.stm_analyzer = _Analyzer()
    p._stm_max_recent = 30
    p.memory_system = type("M", (), {"corpus_manager": corpus})()
    return p


def _queries(entries):
    return [e["query"] for e in entries]


@FORMS
@pytest.mark.parametrize("corpus_cls", [_WindowCorpus, _LegacyCorpus])
def test_stm_window_drops_prior_probe_keeps_current_and_real(probe, corpus_cls):
    p = _pipeline(corpus_cls(_window(probe)))
    asyncio.run(p._analyze_stm("[test]what did we decide?[/test]", None))
    got = _queries(p.stm_analyzer.recent)
    assert probe + " current" in got          # own session's probe stays
    assert probe + " prior" not in got        # prior session's probe dropped
    assert "Tamsin wants the budget sheet split into two tabs." in got
    assert "Rowan sent the lease paperwork on Friday." in got


@FORMS
def test_stm_history_fallback_drops_prior_probe(probe):
    p = _pipeline(None)
    p.memory_system = None
    asyncio.run(p._analyze_stm("[test]what did we decide?[/test]", _window(probe)))
    got = _queries(p.stm_analyzer.recent)
    assert probe + " prior" not in got
    assert probe + " current" in got


@FORMS
def test_tone_history_drops_prior_probe(probe, monkeypatch):
    import utils.emotional_context as ec

    seen = {}

    async def fake_analyze(**kwargs):
        seen["history"] = list(kwargs["conversation_history"])
        return None

    monkeypatch.setattr(ec, "analyze_emotional_context", fake_analyze)
    p = ContextPipeline.__new__(ContextPipeline)
    p.model_manager = None
    p._last_tone_level = None
    p._floor_chain = 0
    p.memory_system = type("M", (), {"corpus_manager": _LegacyCorpus(_window(probe))})()
    p._should_reset_tone_stickiness = lambda recent: False
    asyncio.run(p._detect_tone("[test]hello there[/test]", None))
    got = _queries(seen["history"])
    assert probe + " prior" not in got
    assert got[0] == probe + " current"
