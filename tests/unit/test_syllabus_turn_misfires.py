"""2026-08-29 syllabus-session misfires (live-turn reproductions).

Turn 1 (12:44): "Ok. I am in bathroom with shower running. I will login paste
in syllabus here shortly." ran a 13s agentic-search loop. Tier 2 matched graph
entities {'running','shower'} and the recall-signal test used BARE SUBSTRING —
'how' ⊂ "sHOWer" (4th occurrence of the substring class after
'solve'⊂"resolution", 'document'⊂"documented", 'cat'⊂"catalog").
Fix: RECALL_SIGNAL_WORDS compiled through _compile_keyword_matcher.

Turn 4 (12:52): the grounding verifier appended "⚠️ Correction: Please verify
the due date with the official syllabus…" at confidence 0.9 to a summary whose
every date came straight from the syllabus the user had just PASTED. Root
causes: (a) the verifier's query view was truncated at 500 chars so the pasted
source containing the dates was invisible; (b) nothing rejected advice-shaped
corrections that assert no falsehood. Fixes: head+tail query slices for long
pastes, prompt abstain/user-source rules, and a deterministic demotion of
advice-shaped verdicts in _parse_verdict.
"""

import json

import pytest

from core.agentic.gate import _RECALL_SIGNAL_HIT
from core.grounding_check import (
    GroundingVerdict,
    _build_verifier_prompt,
    _is_advice_shaped,
    _parse_verdict,
    _truncate_query,
)

LIVE_TURN1 = (
    "ok. i am in bathroom with shower running. i will login paste in "
    "syllabus here shortly. before i get in the shower."
)


class TestRecallSignalWordBounding:
    def test_live_turn_shower_does_not_hit_how(self):
        assert _RECALL_SIGNAL_HIT(LIVE_TURN1) is False

    @pytest.mark.parametrize("q", [
        "how does the scoring work",
        "do you remember what i said about Morgan",
        "remind me what the deadline was",
        "what did we decide on the project",
        "tell me anything about my sleep patterns",
    ])
    def test_real_recall_signals_still_hit(self, q):
        assert _RECALL_SIGNAL_HIT(q) is True

    @pytest.mark.parametrize("q", [
        # substring traps: shower/somewhat/nowhere must not hit how/what/where
        "the showerhead is broken",
        "i was somewhat tired after the gym",
        "that got me nowhere yesterday",
    ])
    def test_substring_traps_do_not_hit(self, q):
        assert _RECALL_SIGNAL_HIT(q) is False


# The exact live correction text that shipped on the MGT-6203 turn.
_LIVE_ADVICE_CORRECTION = (
    "Please verify the due date with the official syllabus or course "
    "calendar, as it may differ from what was stated."
)


class TestAdviceShapedVerdictDemotion:
    def _verdict(self, correction, why_false="", present=True, conf=0.9):
        return GroundingVerdict(
            false_claim_present=present, claim="HW 1 due on Sep 13",
            why_false=why_false, confidence=conf, correction=correction)

    def test_live_correction_is_advice_shaped(self):
        assert _is_advice_shaped(self._verdict(_LIVE_ADVICE_CORRECTION)) is True

    def test_parse_demotes_live_verdict_to_no_flag(self):
        raw = json.dumps({
            "false_claim_present": True,
            "claim": "HW 1 due on Sep 13",
            "why_false": "The due date could not be verified.",
            "confidence": 0.9,
            "correction": _LIVE_ADVICE_CORRECTION,
        })
        v = _parse_verdict(raw)
        assert v is not None
        assert v.false_claim_present is False
        assert v.correction == ""

    def test_real_correction_survives(self):
        raw = json.dumps({
            "false_claim_present": True,
            "claim": "the refrigerator mother theory lands closer to truth",
            "why_false": "The refrigerator mother theory is discredited.",
            "confidence": 0.95,
            "correction": "The refrigerator mother theory was discredited "
                          "decades ago; autism is not caused by parenting style.",
        })
        v = _parse_verdict(raw)
        assert v is not None
        assert v.false_claim_present is True
        assert "discredited" in v.correction

    def test_cannot_verify_shapes_demoted(self):
        for c in (
            "Double-check the deadline in the course calendar.",
            "I cannot verify this date from the available context.",
            "It is recommended to confirm the policy with the instructor.",
        ):
            assert _is_advice_shaped(self._verdict(c)) is True

    def test_flag_with_empty_correction_and_reason_demoted(self):
        assert _is_advice_shaped(self._verdict("", why_false="")) is True

    def test_advice_opener_with_real_falsehood_reason_survives(self):
        # why_false genuinely asserts an error → keep the flag even though the
        # correction opens with "check": under-demote, never over-demote a
        # real catch
        v = self._verdict(
            "Check the syllabus again — the correct due date is Sep 20.",
            why_false="The stated Sep 13 date is incorrect; the syllabus says Sep 20.",
        )
        assert _is_advice_shaped(v) is False

    def test_negative_verdicts_untouched(self):
        v = self._verdict(_LIVE_ADVICE_CORRECTION, present=False)
        assert _is_advice_shaped(v) is False


class TestVerifierSeesPastedSource:
    def test_long_paste_keeps_head_and_tail(self):
        head = "FUCK. Okay I have the docs. " + "x" * 3000
        tail = "y" * 2000 + " | Homework 1 | 9/13/2026 (11:59pm ET) | 15% |"
        q = head + tail
        out = _truncate_query(q)
        assert out.startswith("FUCK. Okay I have the docs.")
        assert "9/13/2026" in out          # the table at the END survives
        assert "snipped" in out

    def test_short_query_unchanged(self):
        assert _truncate_query("hello there") == "hello there"

    def test_prompt_carries_abstain_and_user_source_rules(self):
        p = _build_verifier_prompt("q", "r")
        assert "abstain" in p.lower()
        assert "pasted" in p.lower()
        assert "please verify" in p.lower()  # the never-output-advice rule


class TestFactPasteGuard:
    """Turn 4 also STORED syllabus boilerplate as facts (live 12:51:
    'assignments | is_a | the most important avenue of learning…',
    'questions | is | multiple choices', 'students | is | responsible…').
    On paste-sized messages only user-anchored triples survive."""

    def _fact(self, subj, rel, obj):
        return {"content": f"{subj} | {rel} | {obj}",
                "metadata": {"subject": subj, "relation": rel, "object": obj}}

    def _live_facts(self):
        return [
            self._fact("assignments", "is_a",
                       "the most important avenue of learning in this course"),
            self._fact("questions", "is", "multiple choices"),
            self._fact("students", "is",
                       "responsible for making sure that individual assignments "
                       "are submitted in a timely manner before"),
            self._fact("user", "enrolled_in", "MGT 6203"),
        ]

    def test_paste_sized_message_keeps_only_user_facts(self):
        from memory.memory_storage import _paste_guard_filter
        query = "FUCK. Okay I have the docs. " + "syllabus text " * 500
        kept = _paste_guard_filter(query, self._live_facts())
        assert [f["metadata"]["subject"] for f in kept] == ["user"]

    def test_normal_message_untouched(self):
        from memory.memory_storage import _paste_guard_filter
        facts = self._live_facts()
        assert _paste_guard_filter("I adopted a cat named Daisy", facts) is facts

    def test_role_subjects_survive(self):
        from memory.memory_storage import _paste_guard_filter
        query = "x" * 2000
        facts = [self._fact("user's advisor", "is", "Morgan")]
        assert len(_paste_guard_filter(query, facts)) == 1

    def test_object_shaped_items_dropped_not_crashed(self):
        from memory.memory_storage import _paste_guard_filter
        class F:  # non-dict fact object with metadata attr
            metadata = {"subject": "assignments", "relation": "is", "object": "x"}
        assert _paste_guard_filter("y" * 2000, [F()]) == []


class TestHybridKeywordCap:
    def test_paste_query_keywords_capped_and_ordered(self):
        from memory.hybrid_retriever import KEYWORD_MATCH_MAX_KEYWORDS
        from utils.query_rewriter import _ordered_keywords
        import itertools, string
        vocab = ["".join(p) for p in itertools.product(string.ascii_lowercase, repeat=3)]
        query = "FUCK. Okay I have the docs. " + " ".join(vocab[:1000])
        kws = _ordered_keywords(query)[:KEYWORD_MATCH_MAX_KEYWORDS]
        assert len(kws) == KEYWORD_MATCH_MAX_KEYWORDS
        # user's own words (message head) come first
        assert "docs" in kws[:5]

    def test_short_query_unaffected(self):
        from memory.hybrid_retriever import KEYWORD_MATCH_MAX_KEYWORDS
        from utils.query_rewriter import _ordered_keywords
        kws = _ordered_keywords("did I email Morgan about the syllabus")
        assert len(kws) <= KEYWORD_MATCH_MAX_KEYWORDS
