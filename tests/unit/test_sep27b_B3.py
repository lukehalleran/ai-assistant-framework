"""tests/unit/test_sep27b_B3.py

2026-09-27 lane B, batch B3 — small correctness items (class: BC-03, BC-70,
BC-58, BC-46). Four independent items, each against THE deployed function:

(1) `core/insight/detector.py` `_RECORD_ESTABLISHES_RE` / `_PERSONAL_PLUS_
    EXTERNAL_RE`: `.{0,N}` with DOTALL let the span cross a sentence
    boundary and pick up an unrelated verb/noun in the NEXT sentence — the
    same defect the 09-08 F4 fix closed on `_IMPLICIT_PERSONAL_COMPARISON_
    RE`. Both are now `[^.?!\\n]{0,N}` (no DOTALL), so the pairing must sit
    in one sentence; same-sentence positives are unaffected.
(2) `utils/query_checker.py:1113`: the LLM heavy-topic classification
    except-clause logged `f"...failed: {e}"`, which renders EMPTY for an
    exception whose str() is blank — now includes `type(e).__name__`.
(3) `gui/handlers._recent_completed_duplicate` (resend-dedup) matched on
    normalized query text + corpus timestamp only, with no notion of WHICH
    process wrote the entry: after a restart, an identical question inside
    the resend window was served the STORED PRE-RESTART reply. Corpus
    entries loaded from disk at process start now sit below a per-process
    baseline; only entries this process itself appended are dedup
    candidates.
(4) TRACE (no fix needed — closes with this test as evidence):
    `action_claim_guard.claims_calendar_state("It's already on the
    calendar — the career fair is the 28th.")` fires on a TRUE statement
    about a real event. Traced through `gui.handlers._apply_action_guard`
    end-to-end: when this turn's gathered `google_calendar` events include
    a matching event, the calendar STATE-claim backstop (already reworked
    in the 2026-09-22 lane, `_calendar_claim_matches_event` /
    `_calendar_claim_user_corroborated`) suppresses the notice; only an
    UNMATCHED claim (no such event gathered, and no user corroboration)
    gets one.
"""
import asyncio
import json
import logging
from datetime import timedelta
from types import SimpleNamespace

import core.action_claim_guard as action_claim_guard
import core.insight.detector as detector
import gui.handlers as handlers
import utils.query_checker as query_checker
from memory.corpus_manager import CorpusManager


# ─── (1) DOTALL sentence-crossing fix ───────────────────────────────────

class TestRecordEstablishesSentenceBoundary:
    def test_same_sentence_still_matches(self):
        text = "What my record establishes here is pretty clear."
        assert detector._RECORD_ESTABLISHES_RE.search(text) is not None

    def test_does_not_cross_into_the_next_sentence(self):
        # "record" anchor in sentence 1, no verb until "says" in sentence 2 —
        # the pre-fix DOTALL pattern crossed the "." and matched anyway.
        text = (
            "What my record contains is nothing you'd expect. "
            "My therapist says I'm doing great lately."
        )
        assert detector._RECORD_ESTABLISHES_RE.search(text) is None


class TestPersonalPlusExternalSentenceBoundary:
    def test_same_sentence_still_matches(self):
        text = "Compare my history against the wikipedia consensus on this."
        assert detector._PERSONAL_PLUS_EXTERNAL_RE.search(text) is not None

    def test_does_not_cross_into_the_next_sentence(self):
        text = "Check my history for context. Also see this wikipedia article on the topic."
        assert detector._PERSONAL_PLUS_EXTERNAL_RE.search(text) is None

    def test_reverse_order_same_sentence_still_matches(self):
        text = "The wikipedia summary and my own record roughly agree on the dates."
        assert detector._PERSONAL_PLUS_EXTERNAL_RE.search(text) is not None


# ─── (2) empty exception message ────────────────────────────────────────

class TestHeavyTopicLLMFailureLogsExceptionType:
    async def test_logs_type_name_when_str_is_empty(self, caplog, monkeypatch):
        async def _boom(q, model_manager):
            raise ValueError()  # str(ValueError()) == "" — the live defect

        monkeypatch.setattr(query_checker, "_classify_heavy_topic_llm", _boom)
        caplog.set_level(logging.DEBUG, logger="query_checker")
        analysis = await query_checker.analyze_query_async(
            "what is the weather like today", model_manager=object(),
        )
        assert analysis is not None
        matching = [
            r for r in caplog.records
            if "LLM heavy topic classification failed" in r.message
        ]
        assert matching, "expected the failure to be logged"
        assert "ValueError" in matching[0].message


# ─── (3) process-scoped resend-dedup ────────────────────────────────────

class TestResendDedupIsProcessScoped:
    """A reply stored by a PREVIOUS process (restart inside the resend window)
    is never served; a reply this process wrote is. The deployed
    CorpusManager records its own writes (`written_this_process`); a store
    without that capability keeps the old text+time match (the pinned
    contract in test_insight_completion_fixes.py::TestCompletedResendWindow)."""

    def _real_corpus_manager(self, tmp_path, entries):
        path = tmp_path / "corpus.json"
        path.write_text(json.dumps(entries))
        return CorpusManager(corpus_file=str(path))

    def _orch(self, cm):
        return SimpleNamespace(memory_system=SimpleNamespace(corpus_manager=cm))

    def test_pre_restart_entry_is_never_served(self, tmp_path):
        now = handlers._dt.now()
        cm = self._real_corpus_manager(tmp_path, [{
            "query": "what time is my meeting",
            "response": "3pm (pre-restart reply)",
            "timestamp": now.isoformat(),
        }])
        assert handlers._recent_completed_duplicate(self._orch(cm), "what time is my meeting") is None

    def test_entry_written_by_this_process_is_served(self, tmp_path):
        now = handlers._dt.now()
        cm = self._real_corpus_manager(tmp_path, [{
            "query": "what time is my meeting",
            "response": "3pm (pre-restart reply)",
            "timestamp": now.isoformat(),
        }])
        cm.add_entry("what time is my meeting", "3pm (this process's own reply)", timestamp=now)
        assert (handlers._recent_completed_duplicate(self._orch(cm), "what time is my meeting")
                == "3pm (this process's own reply)")

    def test_outside_the_resend_window_is_never_served_either(self, tmp_path):
        old = handlers._dt.now() - timedelta(seconds=600)
        cm = self._real_corpus_manager(tmp_path, [])
        cm.add_entry("what time is my meeting", "old reply", timestamp=old)
        assert handlers._recent_completed_duplicate(self._orch(cm), "what time is my meeting") is None

    def test_store_without_the_capability_keeps_the_old_match(self):
        now = handlers._dt.now()
        cm = SimpleNamespace(corpus=[{
            "query": "what time is my meeting", "response": "servable",
            "timestamp": now.isoformat(),
        }])
        assert handlers._recent_completed_duplicate(self._orch(cm), "what time is my meeting") == "servable"


# ─── (4) TRACE: calendar state-claim backstop already verifies ─────────

CAREER_FAIR_REPLY = "It's already on the calendar — the career fair is the 28th."


def _guard_ctx(user_text="", raw_context=None):
    return SimpleNamespace(
        user_text=user_text,
        user_text_ws=user_text,
        orchestrator=SimpleNamespace(),
        raw_context=raw_context or {},
    )


class TestCalendarStateClaimTrace:
    def test_detector_fires_on_the_true_statement(self):
        # Confirms the FOLLOWUPS-quoted shape is still detected — the fix
        # (if any were needed) belongs at the backstop, not the detector.
        claims = action_claim_guard.claims_calendar_state(CAREER_FAIR_REPLY)
        assert claims and "career fair" in claims[0].lower()

    def test_notice_is_suppressed_when_the_event_is_actually_gathered(self):
        events = [{"summary": "Career Fair", "start": "2026-09-28T09:00:00"}]
        out = asyncio.run(handlers._apply_action_guard(
            _guard_ctx("", {"google_calendar": events}), CAREER_FAIR_REPLY,
            executed_kinds=set(), proposed_kinds=set(), self_repair=False,
        ))
        assert "I don't see that on your calendar" not in out

    def test_notice_still_fires_when_no_matching_event_was_gathered(self):
        events = [{"summary": "Project Review", "start": "2026-09-24T14:00:00"}]
        out = asyncio.run(handlers._apply_action_guard(
            _guard_ctx("", {"google_calendar": events}), CAREER_FAIR_REPLY,
            executed_kinds=set(), proposed_kinds=set(), self_repair=False,
        ))
        assert "I don't see that on your calendar" in out

    def test_fails_open_when_no_calendar_context_was_gathered_at_all(self):
        out = asyncio.run(handlers._apply_action_guard(
            _guard_ctx("", {"google_calendar": []}), CAREER_FAIR_REPLY,
            executed_kinds=set(), proposed_kinds=set(), self_repair=False,
        ))
        assert out == ""
