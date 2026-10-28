"""A REAL turn never sees [test] probe turns; thread depth and topic follow.

Incident date: 2026-10-28 (live debug dump). Six ``[test]`` probes ran 11:47-11:51, the
owner restarted, and a real unwrapped "Hey" at 13:07 (1 h 16 min later, so the SAME session
by the 2-hour rule) carried all four probe exchanges in [RECENT CONVERSATION] / [LAST
EXCHANGE FOR CONTEXT]; [THREAD CONTEXT] said "message #4 ... about <probe topic>" and the
reply treated the probe as open business. Classes: BC-58 (one predicate, many readers), BC-91.

Every test drives the DEPLOYED function (``drop_prior_session_test_turns``,
``MemoryRetrievalMixin._get_recent_conversations``, ``ContextPipeline._analyze_stm``,
``ThreadManager``) over a real CorpusManager. A probe current turn keeps today's behaviour.
Fixture text is invented; every probe is asserted clean and wrapped/indented.
"""
import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.context_pipeline import ContextPipeline
from core.prompt.gatherer_memory import MemoryRetrievalMixin
from core.prompt.token_manager import drop_prior_session_test_turns
from memory.corpus_manager import CorpusManager
from memory.thread_manager import ThreadManager

BODY = "Rowan asks which resume line should say BS or BA?"
PROBES = {
    "inline": f"[test]{BODY}[/test]",
    "wrapped": f"   [test]{BODY}[/test]  ",
    "whole_line_indented": f"  [test]\n  {BODY}\n  [/test]",
}
FORMS = pytest.mark.parametrize("probe", list(PROBES.values()), ids=list(PROBES))
REAL_Q = "Tamsin sent the lab schedule."


def _entries(probe, now):
    """Newest-first: four probe exchanges 1-1.3 h ago (same session), one real turn before them."""
    probes = [
        {"query": probe, "response": f"probe reply {i}", "timestamp": now - timedelta(minutes=70 + i)}
        for i in range(4)
    ]
    real = {"query": REAL_Q, "response": "Got it.", "timestamp": now - timedelta(minutes=100)}
    return probes + [real]


@FORMS
def test_real_turn_drops_same_session_probes(probe):
    now = datetime.now()
    out = drop_prior_session_test_turns(_entries(probe, now), now, current_query="Hey")
    assert [e["query"] for e in out] == [REAL_Q]


@FORMS
def test_probe_turn_keeps_its_own_session_probes(probe):
    now = datetime.now()
    out = drop_prior_session_test_turns(_entries(probe, now), now, current_query=probe)
    assert len(out) == 5


@FORMS
def test_no_current_query_keeps_todays_behaviour(probe):
    now = datetime.now()
    assert len(drop_prior_session_test_turns(_entries(probe, now), now)) == 5


def _corpus(tmp_path, probe):
    cm = CorpusManager(str(tmp_path / "corpus.json"))
    now = datetime.now()
    cm.add_entry(REAL_Q, "Got it.", timestamp=now - timedelta(minutes=100),
                 thread_id="thread_real", thread_depth=2, thread_started=(now - timedelta(minutes=110)).isoformat(),
                 thread_topic="Lab Schedule")
    for i in range(4):
        cm.add_entry(probe, f"probe reply {i}", timestamp=now - timedelta(minutes=70 + i),
                     thread_id="thread_probe", thread_depth=i + 1, thread_topic="In The New")
    return cm


@FORMS
def test_gatherer_recent_conversations_real_turn_sees_no_probe(tmp_path, probe):
    cm = _corpus(tmp_path, probe)
    g = MemoryRetrievalMixin.__new__(MemoryRetrievalMixin)
    g.memory_coordinator = SimpleNamespace(corpus_manager=cm, get_memories=AsyncMock(return_value=[]))
    g.memory_id_map = {}
    out = asyncio.run(g._get_recent_conversations(limit=10, current_query="Hey"))
    responses = [m.get("response") for m in out]
    assert "Got it." in responses
    assert not any(str(r).startswith("probe reply") for r in responses)
    # a probe turn still sees its in-session probes
    out = asyncio.run(g._get_recent_conversations(limit=10, current_query=probe))
    assert any(str(m.get("response")).startswith("probe reply") for m in out)


class _Analyzer:
    def __init__(self):
        self.recent = None

    async def analyze(self, recent_memories, user_query, last_assistant_response, graph_memory):
        self.recent = list(recent_memories)
        return {"topic": "x"}


@FORMS
def test_stm_window_real_turn_sees_no_probe(tmp_path, probe):
    cm = _corpus(tmp_path, probe)
    p = ContextPipeline.__new__(ContextPipeline)
    p.stm_analyzer = _Analyzer()
    p._stm_max_recent = 30
    p.memory_system = SimpleNamespace(corpus_manager=cm)
    asyncio.run(p._analyze_stm("Hey", None))
    assert [e["query"] for e in p.stm_analyzer.recent] == [REAL_Q]


@FORMS
def test_thread_depth_and_topic_exclude_probes(tmp_path, probe):
    cm = _corpus(tmp_path, probe)
    tm = ThreadManager(corpus_manager=cm)
    ctx = tm.get_thread_context(current_query="Hey")
    assert ctx["thread_id"] == "thread_real" and ctx["thread_topic"] == "Lab Schedule"
    info = tm.detect_or_create_thread("Hey", False, current_topic="Greeting")
    assert info["depth"] == 1 and info["topic"] == "Greeting"  # new thread, not "#5 about In The New"
    # a probe current turn still reads its own session's probes (today's behaviour)
    assert tm.get_thread_context(current_query=probe)["thread_id"] == "thread_probe"
    assert tm.get_thread_context()["thread_id"] == "thread_probe"


def test_zone_aware_timestamps_are_compared_in_local_time():
    """BC-37 (2026-10-28): a timezone-aware timestamp (UTC, or an ISO string with an offset) of NOW is the current
    session. `_parse_entry_timestamp` returned aware values unconverted, so the session rule compared them with a
    naive local `now` and every such entry read as a prior session (all probes hidden from a probe turn)."""
    from datetime import datetime, timezone
    from core.prompt import token_manager as tm
    for ts in (datetime.now(timezone.utc), datetime.now().astimezone().isoformat()):
        items = [{"query": "[test]p[/test]", "response": "a", "timestamp": ts},
                 {"query": "real", "response": "b", "timestamp": ts}]
        assert tm.current_session_prefix_len(items, datetime.now()) == 2
        kept = tm.drop_prior_session_test_turns(items, None, current_query="[test]x[/test]")
        assert [e["response"] for e in kept] == ["a", "b"]
