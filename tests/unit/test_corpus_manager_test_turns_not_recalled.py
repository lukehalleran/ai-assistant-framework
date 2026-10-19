"""Owner ``[test]...[/test]`` probe turns are never recalled after their session.

Incident 2026-10-18, class BC-58 (test-state isolation): four identical
``[test]Do you remember what we talked about ...[/test]`` probes each recalled
the PREVIOUS probe's stored reply and copied its (wrong) wording, so one bad
reply propagated through every later test. The 10-16 rule kept probes in
retrieval "so an in-session probe can test recall"; that is superseded.

Every retrieval/recall path drops a probe turn at the same place it drops
junk/quarantined docs (THE predicate is ``memory.fact_source.is_test_turn``);
[RECENT CONVERSATION] keeps only the CURRENT session's probes so a multi-turn
probe reaches its own previous exchange. Each test drives the deployed
function with one real turn and one probe turn, in clean and wrapped/indented
forms, inline and whole-line envelopes. Invented content only.
"""

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.insight.sweep import run_sweep
from core.insight.types import FacetPlan, FacetQuery
from core.prompt.gatherer_memory import MemoryRetrievalMixin
from core.prompt.token_manager import drop_prior_session_test_turns
from memory.corpus_manager import CorpusManager
from memory.fact_source import is_test_turn
from memory.hybrid_retriever import HybridRetriever
from memory.memory_expander import MemoryExpander
from memory.memory_retriever import MemoryRetriever
from memory.utils import is_test_conversation_doc

REAL_Q = "Rowan asked whether the quarterly budget sheet needs a second tab."
REAL_A = "A second tab for the quarterly budget sheet would keep Rowan's notes apart."
PROBE_BODY = "Do you remember what we talked about with the budget sheet yesterday?"
PROBE_A = "Yesterday you said the budget sheet must be finished by end of day today."

PROBE_FORMS = {
    "inline": f"[test]{PROBE_BODY}[/test]",
    "inline_wrapped": f"   [test]{PROBE_BODY}[/test]   ",
    "whole_line": f"[test]\n{PROBE_BODY}\n[/test]",
    "whole_line_indented": "  [test]\n  Do you remember what we talked about\n"
                           "  with the budget sheet yesterday?\n  [/test]",
}
FORMS = pytest.mark.parametrize("probe", list(PROBE_FORMS.values()), ids=list(PROBE_FORMS))


def _doc(doc_id, user, assistant, **meta):
    return {
        "id": doc_id,
        "content": f"User: {user}\nAssistant: {assistant}",
        "metadata": {"timestamp": "2026-10-10T12:00:00", **meta},
        "relevance_score": 0.8,
        "collection": "conversations",
        "rank": 1,
    }


# --- the predicate shapes ------------------------------------------------------------------

class TestPredicateShapes:
    @FORMS
    def test_stored_doc_user_side_probe(self, probe):
        assert is_test_conversation_doc(content=f"User: {probe}\nAssistant: {PROBE_A}")
        assert is_test_turn({"content": f"User: {probe}\nAssistant: {PROBE_A}", "metadata": {}})

    def test_real_stored_doc_passes(self):
        assert not is_test_conversation_doc(content=f"User: {REAL_Q}\nAssistant: {REAL_A}")

    def test_marker_only_in_the_assistant_reply_is_not_a_probe(self):
        quoted = "You can wrap a message as [test]like this[/test] to mark it."
        assert not is_test_conversation_doc(content=f"User: how do I mark a probe?\nAssistant: {quoted}")

    def test_attachment_turn_reads_metadata_user_text(self):
        doc = {"content": "User: <merged attachment blob>\nAssistant: ok",
               "metadata": {"user_text": PROBE_FORMS["inline"]}}
        assert is_test_turn(doc)

    def test_chat_message_shape_is_not_reinterpreted(self):
        assert not is_test_turn({"role": "assistant", "content": PROBE_FORMS["inline"]})


# --- hybrid retrieval ----------------------------------------------------------------------

class TestHybridRetrieval:
    @FORMS
    def test_probe_never_reaches_ranking(self, probe):
        results = [
            _doc("real", REAL_Q, REAL_A),
            _doc("probe", probe, PROBE_A),
        ]
        r = HybridRetriever.__new__(HybridRetriever)
        r.chroma_store = MagicMock()
        r.chroma_store.query_multiple_collections = AsyncMock(return_value={"conversations": results})
        r.semantic_weight = 0.7
        r.keyword_weight = 0.3
        r._fast_mode = False
        out = asyncio.run(r.retrieve("budget sheet", limit=10))
        assert {m["id"] for m in out} == {"real"}


# --- fallback semantic search --------------------------------------------------------------

class TestFallbackSemanticSearch:
    @FORMS
    def test_probe_dropped_real_kept(self, probe):
        store = MagicMock()
        store.query_multiple_collections = AsyncMock(return_value={
            "conversations": [_doc("real", REAL_Q, REAL_A), _doc("probe", probe, PROBE_A)],
        })
        r = MemoryRetriever(corpus_manager=MagicMock(), chroma_store=store)
        out = asyncio.run(r._fallback_semantic_search("budget sheet", 10))
        assert [m["id"] for m in out] == ["real"]


# --- memory expander -----------------------------------------------------------------------

class _ExpanderStore:
    def __init__(self, docs):
        self._docs = docs

    def get_by_id(self, name, doc_id):
        return next((d for d in self._docs if d["id"] == doc_id), None)

    def list_all(self, name):
        return list(self._docs)

    def get_ids_by_timestamp_range(self, name, a, b):
        return [d["id"] for d in self._docs]


class TestMemoryExpander:
    @FORMS
    def test_probe_neighbour_excluded_and_probe_anchor_suppressed(self, probe):
        docs = [
            {**_doc("d0", REAL_Q, REAL_A), "metadata": {"timestamp": "2026-10-10T12:00:00"}},
            {**_doc("d1", probe, PROBE_A), "metadata": {"timestamp": "2026-10-10T12:10:00"}},
            {**_doc("d2", "Tamsin moved the meeting.", "Noted, the meeting moved."),
             "metadata": {"timestamp": "2026-10-10T12:20:00"}},
        ]
        ex = MemoryExpander(_ExpanderStore(docs))
        out = ex.expand("d0", window=2, collection="conversations")
        ids = [t["id"] for t in out["turns"]]
        assert "d0" in ids and "d2" in ids and "d1" not in ids
        anchor = ex.expand("d1", window=1, collection="conversations")
        assert anchor["turns"] == [] and "anchor suppressed" in anchor["error"].lower()


# --- insight sweep -------------------------------------------------------------------------

def _caps():
    return {"per_facet_cap": 10, "total_evidence_cap": 80, "evidence_snippet_chars": 280,
            "keyword_scan_max": 50, "expand_top_k": 3, "expand_window": 2, "sweep_timeout_s": 45.0}


class TestInsightSweep:
    @FORMS
    def test_probe_rows_and_expanded_turns_and_date_scan_excluded(self, probe, monkeypatch):
        probe_text = f"User: {probe}\nAssistant: {PROBE_A}"
        real_text = f"User: {REAL_Q} Here is a fuller note about the sheet.\nAssistant: {REAL_A}"
        store = MagicMock()
        store.query_collection.side_effect = lambda coll, q, n: (
            [_doc("c_real", REAL_Q + " a fuller note", REAL_A), _doc("c_probe", probe, PROBE_A)]
            if coll == "conversations" else [])
        corpus = MagicMock()
        corpus.search_keyword.return_value = []
        expander = MagicMock()
        expander.expand.return_value = {"turns": [
            {"id": "x_real", "content": real_text, "metadata": {"timestamp": "2026-10-11T10:00:00"}},
            {"id": "x_probe", "content": probe_text, "metadata": {"timestamp": "2026-10-11T11:00:00"}},
        ]}
        import core.insight.sweep as sweep
        monkeypatch.setattr(sweep, "window_scan_collection", lambda s, coll, *a, **k: (
            [_doc("w_real", REAL_Q + " window", REAL_A), _doc("w_probe", probe, PROBE_A)]
            if coll == "conversations" else []))
        plan = FacetPlan(facets=[FacetQuery(name="f", query_text="budget", keywords=["budget"])])
        items = asyncio.run(run_sweep(
            plan, chroma_store=store, corpus_manager=corpus, memory_expander=expander,
            caps=_caps(), date_window=("2026-10-01", "2026-10-31"),
        ))
        texts = " ".join(i.text for i in items)
        assert "Do you remember" not in texts and "[test]" not in texts.lower()
        assert "budget" in texts.lower()  # the real turns survived


# --- corpus keyword channel + keyword anchor + recent very_recent --------------------------

def _corpus(tmp_path, probe):
    cm = CorpusManager(corpus_file=str(tmp_path / "corpus.json"))
    now = datetime.now()
    cm.add_entry(f"Sent Morgan the budget sheet. {REAL_Q}", REAL_A, timestamp=now - timedelta(days=3))
    cm.add_entry(f"{probe} Morgan", PROBE_A + " Morgan agreed.", timestamp=now - timedelta(days=2))
    return cm


class TestCorpusRecallChannels:
    @FORMS
    def test_search_keyword_skips_probe(self, tmp_path, probe):
        cm = _corpus(tmp_path, probe)
        hits = cm.search_keyword("Morgan", include_entry=True)
        assert hits and all(not is_test_turn(h["entry"]) for h in hits)

    @FORMS
    def test_keyword_anchor_never_injects_probe(self, tmp_path, probe):
        cm = _corpus(tmp_path, probe)
        r = MemoryRetriever(corpus_manager=cm, chroma_store=None)
        out = r._keyword_anchor_memories("Please email Morgan about it", [])
        assert out and all("Do you remember" not in (m.get("content") or "") for m in out)

    @FORMS
    def test_retrieval_very_recent_skips_probe_and_refills(self, tmp_path, probe):
        cm = _corpus(tmp_path, probe)
        r = MemoryRetriever(corpus_manager=cm, chroma_store=None)
        out = r._get_recent_conversations(1)
        blob = " ".join(str(m) for m in out)
        assert out and "Do you remember" not in blob and "budget sheet" in blob  # the real turn refills


# --- [RECENT CONVERSATION]: current session kept, prior session dropped --------------------

def _entries(probe):
    now = datetime.now()
    # newest-first, like get_recent_memories
    return [
        {"query": probe, "response": PROBE_A, "timestamp": now - timedelta(minutes=5)},      # current probe
        {"query": REAL_Q, "response": REAL_A, "timestamp": now - timedelta(minutes=12)},     # current real
        {"query": probe, "response": PROBE_A, "timestamp": now - timedelta(days=3)},         # prior probe
        {"query": "Tamsin sent the lab schedule.", "response": "Got it.",
         "timestamp": now - timedelta(days=3, minutes=10)},                                   # prior real
    ]


class TestRecentConversationSessions:
    @FORMS
    def test_helper_drops_only_prior_session_probe(self, probe):
        out = drop_prior_session_test_turns(_entries(probe))
        queries = [e["query"] for e in out]
        assert len(out) == 3
        assert queries.count(probe) == 1 and out[0]["query"] == probe  # current-session probe kept
        assert REAL_Q in queries and "Tamsin sent the lab schedule." in queries

    def test_all_prior_session_keeps_real_drops_probe(self):
        probe = PROBE_FORMS["inline"]
        items = _entries(probe)[2:]
        out = drop_prior_session_test_turns(items)
        assert [e["query"] for e in out] == ["Tamsin sent the lab schedule."]

    @FORMS
    def test_gatherer_recent_conversations_uses_session_rule(self, tmp_path, probe):
        cm = CorpusManager(corpus_file=str(tmp_path / "corpus.json"))
        now = datetime.now()
        cm.add_entry("Tamsin sent the lab schedule.", "Got it.", timestamp=now - timedelta(days=3, minutes=10))
        cm.add_entry(probe, PROBE_A, timestamp=now - timedelta(days=3))
        cm.add_entry(REAL_Q, REAL_A, timestamp=now - timedelta(minutes=12))
        cm.add_entry(probe, "Current probe reply.", timestamp=now - timedelta(minutes=5))
        g = MemoryRetrievalMixin.__new__(MemoryRetrievalMixin)
        g.memory_coordinator = SimpleNamespace(
            corpus_manager=cm, get_memories=AsyncMock(return_value=[]))
        g.memory_id_map = {}
        out = asyncio.run(g._get_recent_conversations(limit=10))
        responses = [m.get("response") for m in out]
        assert PROBE_A not in responses                      # prior-session probe dropped
        assert "Current probe reply." in responses           # current-session probe kept
        assert REAL_A in responses and "Got it." in responses  # real turns of both sessions kept


class TestUserStatedEvidenceSession:
    """The [THE USER'S OWN STATEMENTS] channel follows the same session rule."""

    @FORMS
    def test_prior_session_probe_statement_is_not_evidence(self, probe):
        import numpy as np

        stmt = probe.replace(PROBE_BODY, "I work at Halvorsen Marine as an operations planner.") \
            if PROBE_BODY in probe else probe
        now = datetime.now()
        real = "I work at Brightwater Logistics as a data analyst."
        entries = [
            {"query": "ok", "response": "r", "timestamp": (now - timedelta(minutes=1)).isoformat()},
            {"query": stmt, "response": "r", "timestamp": (now - timedelta(days=3)).isoformat()},
            {"query": real, "response": "r", "timestamp": (now - timedelta(days=3, minutes=5)).isoformat()},
        ]

        class _Model:
            def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
                return np.array([[1.0, 0.0] if t == "where do I work?" else [0.9, 0.43589] for t in texts])

        class _Corpus:
            def get_recent_memories(self, count=3):
                return list(entries)

        g = MemoryRetrievalMixin.__new__(MemoryRetrievalMixin)
        g.memory_coordinator = SimpleNamespace(
            corpus_manager=_Corpus(), chroma_store=SimpleNamespace(get_st_model=lambda: _Model()))
        got = asyncio.run(g.get_user_stated_evidence("where do I work?", "factual_recall"))
        texts = [r["query"] for r in got]
        assert real in texts
        assert not any("Halvorsen" in t for t in texts)
