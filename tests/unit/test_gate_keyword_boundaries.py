"""CGR-20260913-002 (dm01_raw_substring; BC-01, BC-02) — deployed-function
outcome fixtures for the seven `evaluate_agentic_gate` sites that matched
CONTINUATION_PHRASES / an email-intent verb tuple / SEARCH_SIGNAL_WORDS /
EXPLICIT_SEARCH_KEYWORDS with raw `in` against the lowered user text.

Each site is exercised through the DEPLOYED `evaluate_agentic_gate` (never a
bare helper) via the smallest fixture/context that reaches it, reusing the
corpus/offer-slot construction from tests/unit/test_agentic_gate.py and
tests/unit/test_deferred_request_clarify.py. Counterexamples prove a phrase
contained inside a longer, unrelated word no longer counts; paired positive
controls prove the legitimate phrasing still fires. Per DEVELOPMENT_WORKFLOW
§3, every fixture is asserted in both a clean and a wrapped/indented form.
"""

import pytest
from unittest.mock import MagicMock

import core.agentic.gate as gate
from core.agentic.gate import evaluate_agentic_gate


def _clean(text: str) -> str:
    return text


def _wrap_edges(text: str) -> str:
    """Leading indentation + trailing newline — the realistic 'wrapped' shape
    for the short (<=6/<12 word) messages these anchors match; `.strip()`
    already normalizes pure edge whitespace, so this proves the edge-wrap
    case explicitly rather than leaving it untested."""
    return f"\n   {text}   \n"


def _wrap_mid(before: str, after: str) -> str:
    """A client line-wrap INSIDE the message but away from the matched
    keyword — the actual live shape from DEVELOPMENT_WORKFLOW §3 ("...a new
    doc I\\n  think will be helpful")."""
    return f"{before}\n  {after}"


_SHAPES = [("clean", _clean)]


@pytest.fixture(autouse=True)
def _clear_gate_slots():
    gate._DEFERRED_REQUEST_SLOT.clear()
    gate._reset_insight_offer_state()
    yield
    gate._DEFERRED_REQUEST_SLOT.clear()
    gate._reset_insight_offer_state()


# ===========================================================================
# Anchor #3 (line 724) — tone-deferral affirmation, CONTINUATION_PHRASES
# ===========================================================================

class TestAnchor3ToneDeferralAffirmation:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("wrap", [_clean, _wrap_edges], ids=["clean", "wrapped"])
    async def test_yesterday_does_not_affirm(self, wrap):
        """'yes' contained inside "yesterday" must not re-run the deferred
        request — the 'yes' ⊂ "yesterday" containment class."""
        gate._arm_deferred_request("please run the cleanup script")
        d = await evaluate_agentic_gate(user_text=wrap("yesterday"))
        assert "deferred-request affirmation" not in (d.reason or "")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("wrap", [_clean, _wrap_edges], ids=["clean", "wrapped"])
    async def test_yes_please_affirms(self, wrap):
        """Paired positive control: a genuine affirmation still re-runs the
        deferred request veto-exempt."""
        gate._arm_deferred_request("please run the cleanup script")
        d = await evaluate_agentic_gate(user_text=wrap("yes please"))
        assert d.should_trigger is True
        assert "deferred-request affirmation" in d.reason
        assert d.veto_exempt is True


# ===========================================================================
# Anchor #4 (line 770) — insight-offer affirmation, CONTINUATION_PHRASES
# ===========================================================================

class TestAnchor4InsightOfferAffirmation:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("wrap", [_clean, _wrap_edges], ids=["clean", "wrapped"])
    async def test_measure_does_not_affirm(self, wrap):
        """'sure' contained inside "measure" must not consume the one-shot
        insight offer as an affirmation — the 'sure' ⊂ "measure" containment
        class."""
        gate._INSIGHT_OFFER_SLOT["statement"] = "I keep repeating the same pattern with my sister"
        d = await evaluate_agentic_gate(user_text=wrap("measure"))
        assert "insight-offer affirmation" not in (d.reason or "")
        assert "insight" not in d.modes

    @pytest.mark.asyncio
    @pytest.mark.parametrize("wrap", [_clean, _wrap_edges], ids=["clean", "wrapped"])
    async def test_go_ahead_affirms(self, wrap):
        """Paired positive control."""
        gate._INSIGHT_OFFER_SLOT["statement"] = "I keep repeating the same pattern with my sister"
        d = await evaluate_agentic_gate(user_text=wrap("go ahead"))
        assert d.should_trigger is True
        assert d.modes == ["insight"]
        assert "insight-offer affirmation" in d.reason


# ===========================================================================
# Anchor #5 (line 973) — email-intent verbs
# ===========================================================================

class TestAnchor5EmailIntentVerbs:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("wrap", [_clean, _wrap_edges], ids=["clean", "wrapped"])
    async def test_rewrite_does_not_satisfy_write(self, wrap):
        """'write' contained inside "rewrite" must not grant the email-verb
        arm."""
        d = await evaluate_agentic_gate(
            user_text=wrap("Please rewrite my email so it sounds more formal")
        )
        assert d.should_trigger is False
        assert "tools" not in d.modes

    @pytest.mark.asyncio
    @pytest.mark.parametrize("wrap", [_clean, _wrap_edges], ids=["clean", "wrapped"])
    async def test_sender_does_not_satisfy_send(self, wrap):
        """'send' contained inside "sender" must not grant the email-verb
        arm."""
        d = await evaluate_agentic_gate(
            user_text=wrap("Reply to the sender of that email please")
        )
        assert d.should_trigger is False
        assert "tools" not in d.modes

    @pytest.mark.asyncio
    @pytest.mark.parametrize("wrap", [_clean, _wrap_edges], ids=["clean", "wrapped"])
    async def test_negated_send_does_not_grant_arm(self, wrap):
        """BC-02: a negated email-send verb must not count."""
        d = await evaluate_agentic_gate(
            user_text=wrap("Don't send that email, I want to review it first")
        )
        assert d.should_trigger is False
        assert "tools" not in d.modes

    @pytest.mark.asyncio
    @pytest.mark.parametrize("wrap", [_clean, _wrap_edges], ids=["clean", "wrapped"])
    async def test_write_an_email_still_fires(self, wrap):
        """Paired positive control for the anchor itself (isolated from the
        earlier send-address/send-imperative elifs)."""
        d = await evaluate_agentic_gate(
            user_text=wrap("Hey, write an email for the professor explaining the delay")
        )
        assert d.should_trigger is True
        assert "tools" in d.modes

    @pytest.mark.asyncio
    async def test_send_an_email_to_still_fires(self):
        """Paired positive control (task's exact phrasing) — satisfied via
        the sibling send-imperative elif, exercised end-to-end."""
        d = await evaluate_agentic_gate(user_text="send an email to Morgan about the meeting")
        assert d.should_trigger is True
        assert "tools" in d.modes


class TestAnchor5NewlyMatchedInflections:
    """The chokepoint's e-drop inflection rule makes 'write'/'compose' ALSO
    match their present-participle forms ("writing"/"composing"), which the
    old raw-substring check MISSED entirely (no literal "write"/"compose"
    substring survives dropping the 'e'). Sense-preserving — accepted and
    reported in the response packet as newly-matched forms."""

    @pytest.mark.asyncio
    async def test_writing_an_email_now_recognized(self):
        d = await evaluate_agentic_gate(
            user_text="Hey, I'm writing an email for the professor"
        )
        assert d.should_trigger is True
        assert "tools" in d.modes

    @pytest.mark.asyncio
    async def test_composing_an_email_now_recognized(self):
        d = await evaluate_agentic_gate(
            user_text="Hey, I'm composing an email for the professor"
        )
        assert d.should_trigger is True
        assert "tools" in d.modes


# ===========================================================================
# Anchor #6 (line 1080) — casual-skip filter, SEARCH_SIGNAL_WORDS
# ===========================================================================

class TestAnchor6CasualSkipSearchSignal:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("wrap", [_clean, _wrap_edges], ids=["clean", "wrapped"])
    async def test_research_is_not_a_search_signal(self, wrap):
        """'search' contained inside "research" must not defeat the casual-
        skip filter — the 'how' ⊂ "shower" family named in the request."""
        d = await evaluate_agentic_gate(user_text=wrap("This is pure research"))
        assert d.should_trigger is False
        assert d.reason == "casual/short message"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("wrap", [_clean, _wrap_edges], ids=["clean", "wrapped"])
    async def test_outlook_is_not_a_search_signal(self, wrap):
        """'look' contained inside "outlook" must not defeat the casual-skip
        filter."""
        d = await evaluate_agentic_gate(user_text=wrap("I use outlook daily"))
        assert d.should_trigger is False
        assert d.reason == "casual/short message"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("wrap", [_clean, _wrap_edges], ids=["clean", "wrapped"])
    async def test_look_up_the_latest_is_a_search_signal(self, wrap):
        """Paired positive control (task's exact phrasing)."""
        d = await evaluate_agentic_gate(
            user_text=wrap("look up the latest developments on this")
        )
        assert d.reason != "casual/short message"


# ===========================================================================
# Anchor #7 (line 1107) — continuation-override check, CONTINUATION_PHRASES
# ===========================================================================

class TestAnchor7ContinuationOverride:

    @staticmethod
    def _agentic_corpus():
        corpus = MagicMock()
        corpus.get_recent_memories = MagicMock(return_value=[
            {"query": "search for python tutorials?", "response": "Let me search for that..."},
        ])
        return corpus

    @pytest.mark.asyncio
    @pytest.mark.parametrize("wrap", [_clean, _wrap_edges], ids=["clean", "wrapped"])
    async def test_yesterday_does_not_read_as_continuation(self, wrap):
        """'yes' ⊂ "yesterday" must not mark the prior turn as inherited —
        the deployed decision's `reason` flips from "casual/short message"
        (correct) to "no trigger" (bug: the false continuation match makes
        `_prev_was_agentic` true, which suppresses the casual-skip reason)."""
        d = await evaluate_agentic_gate(user_text=wrap("yesterday"), corpus_manager=self._agentic_corpus())
        assert d.reason == "casual/short message"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("wrap", [_clean, _wrap_edges], ids=["clean", "wrapped"])
    async def test_yes_please_reads_as_continuation(self, wrap):
        """Paired positive control (matches test_agentic_gate.py's
        TestContinuationOverride shape)."""
        d = await evaluate_agentic_gate(user_text=wrap("yes please"), corpus_manager=self._agentic_corpus())
        assert d.reason != "casual/short message"


# ===========================================================================
# Anchor #8 (line 1185) — file-retrieval affirmation, CONTINUATION_PHRASES
# ===========================================================================

class TestAnchor8FileRetrievalAffirmation:

    @staticmethod
    def _file_offer_corpus():
        corpus = MagicMock()
        corpus.get_recent_memories.return_value = [
            {"query": "can you check the report",
             "response": "I can't read files this turn. Want me to pull that up next turn?"},
        ]
        return corpus

    @pytest.mark.asyncio
    @pytest.mark.parametrize("wrap", [_clean, _wrap_edges], ids=["clean", "wrapped"])
    async def test_pressure_is_not_an_affirmation(self, wrap):
        """'sure' ⊂ "pressure" must not consume the file-offer as an
        affirmation."""
        d = await evaluate_agentic_gate(user_text=wrap("pressure"), corpus_manager=self._file_offer_corpus())
        assert "tools" not in d.modes
        assert d.should_trigger is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("wrap", [_clean, _wrap_edges], ids=["clean", "wrapped"])
    async def test_go_ahead_is_an_affirmation(self, wrap):
        """Paired positive control (task's exact phrasing)."""
        d = await evaluate_agentic_gate(user_text=wrap("go ahead"), corpus_manager=self._file_offer_corpus())
        assert d.should_trigger is True
        assert "tools" in d.modes


# ===========================================================================
# Anchor #9 (line 1514) — intent-veto exemption, EXPLICIT_SEARCH_KEYWORDS
# ===========================================================================

class TestAnchor9ExplicitSearchVetoExemption:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("wrap", [_clean, _wrap_mid], ids=["clean", "wrapped"])
    async def test_research_does_not_grant_exemption(self, wrap):
        """'search' ⊂ "research" must not grant the intent-veto exemption."""
        text = ("I have been doing a lot of research", "lately on this topic")
        user_text = wrap(*text) if wrap is _wrap_mid else wrap(" ".join(text))
        d = await evaluate_agentic_gate(user_text=user_text)
        assert d.veto_exempt is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("wrap", [_clean, _wrap_mid], ids=["clean", "wrapped"])
    async def test_negated_search_does_not_grant_exemption(self, wrap):
        """BC-02: a negated explicit search keyword must not grant the
        veto exemption ("don't search for it")."""
        text = ("don't search for that,", "just tell me what you already found")
        user_text = wrap(*text) if wrap is _wrap_mid else wrap(" ".join(text))
        d = await evaluate_agentic_gate(user_text=user_text)
        assert d.veto_exempt is False

    @pytest.mark.asyncio
    async def test_search_for_grants_exemption(self):
        """Paired positive control (task's exact phrasing)."""
        d = await evaluate_agentic_gate(user_text="search for the latest developments")
        assert d.veto_exempt is True

    @pytest.mark.asyncio
    async def test_pull_up_the_article_grants_exemption(self):
        """Paired positive control (task's exact phrasing)."""
        d = await evaluate_agentic_gate(user_text="pull up the article")
        assert d.veto_exempt is True


# ===========================================================================
# Follow-up sibling: CONTINUATION_PHRASES negation characterization
# ===========================================================================

class TestContinuationNegationCharacterization:
    """Contract v2 deliberately does NOT add negation semantics to
    CONTINUATION_PHRASES (#3/#4/#7/#8) this batch — only the request-cue
    vocabularies (#5, #9) are negation-aware (see the response packet's
    "siblings"). This RECORDS today's actual outcome for a negated
    continuation phrase so a future request has a baseline to compare
    against; it is not asserting correct behavior."""

    @pytest.mark.asyncio
    async def test_please_dont_do_it_still_reads_as_an_affirmation(self):
        gate._arm_deferred_request("please run the cleanup script")
        d = await evaluate_agentic_gate(user_text="please don't do it")
        # CHARACTERIZATION (unchanged by this batch): 'do it' still matches
        # despite the "don't" immediately before it in the same message.
        assert "deferred-request affirmation" in (d.reason or "")
        assert d.should_trigger is True
