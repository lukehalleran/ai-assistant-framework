"""2026-08-29 calendar-task turn, round 2 (15:59 retest after restart).

Round 1 fixes verified live (gate → tools mode, zero Tavily, action intent
detected and forced) — but the turn still failed downstream:

1. XML-PATH FORCING WAS NATIVE-ONLY — "forcing propose_action" set
   tool_choice/tools_override (ignored on the XML protocol) and told kimi-3
   to "call propose_action" (native-tools vocabulary); the model emitted
   nothing and the controller read silence as "ready to answer (implicit)".
   Fixes: ACTION_PATTERN parses attributes GENERICALLY (calendar's
   summary/start_time/end_time are now expressible), the forced round gets
   protocol-appropriate syntax with a concrete example, and a forced round
   that returns no action marker retries exactly once.

2. SYNTHESIS NARRATION SHIPPED — 55s of synthesis produced 188 chars of
   "let me grab the full text back out of memory…" with no guard (the 08-28
   promissory guards covered only decision-answer REUSE). Fixes:
   narration_shaped_final() + regenerate_final_answer() (one bounded
   no-reasoning retry), wired in handlers before the final replacement yield.

3. MEMORY TOP-UP REFILLED 1 → 30 — the gate's quality floor honestly
   returned 1 memory; the builder top-up then added 30 UNGATED recent
   conversations (10K tokens of crisis history on a logistics turn).
   Fix: MEMORY_TOPUP_FLOOR=3 survival minimum.

4. content_type=lyrics STORED FOR THE QUERY — a terminal-wrapped single
   sentence (4 short indented lines, no '?') passed the structural lyrics
   heuristic. Fix: indented continuation lines = wrapped prose, never verse.
"""
import inspect

import pytest

from core.agentic.controller import AgenticSearchController
from core.agentic.protocols import XMLMarkerHandler
from core.actions.registry import ACTION_SPECS
from core.actions.types import ActionType
from core.content_type_detector import detect_content_type, _looks_like_lyrics


LIVE_NARRATION = (
    "I already pulled the syllabus details earlier today, so let me grab the "
    "full text back out of memory to make sure I have every date exact before "
    "we touch the calendar."
)

WRAPPED_QUERY = (
    "Please search for documents related to the MGT class\n"
    "  I am enrolled in, catalog all the dates and\n"
    "  deadlines, and place each in the appropriate time\n"
    "  slot on my Google calendar"
)


def _bare_controller() -> AgenticSearchController:
    """Controller instance without running __init__ (the methods under test
    only touch regex class attrs, model_manager, and stashed prompt attrs)."""
    return object.__new__(AgenticSearchController)


# ===========================================================================
# 1a. Generic <action> attribute parsing
# ===========================================================================

class TestGenericActionAttrs:

    def _parse(self, text):
        handler = XMLMarkerHandler()
        return [d for d in handler.parse_response(text) if d.wants_action]

    def test_calendar_event_marker_carries_all_fields(self):
        text = (
            '<action type="calendar_create_event" summary="MGT 6203 HW 1 due" '
            'start_time="2026-09-13T23:59:00" end_time="2026-09-13T23:59:59" '
            'reason="user asked">HW 1 deadline</action>'
        )
        ds = self._parse(text)
        assert len(ds) == 1
        d = ds[0]
        assert d.action_type == "calendar_create_event"
        assert d.action_params["summary"] == "MGT 6203 HW 1 due"
        assert d.action_params["start_time"] == "2026-09-13T23:59:00"
        assert d.action_params["end_time"] == "2026-09-13T23:59:59"
        assert d.action_params["message"] == "HW 1 deadline"
        assert d.action_reason == "user asked"

    def test_marker_with_attrs_but_empty_body_is_valid(self):
        text = (
            '<action type="calendar_create_event" summary="Office hours" '
            'start_time="2026-09-04T21:00:00" end_time="2026-09-04T22:00:00" '
            'reason="user asked"></action>'
        )
        ds = self._parse(text)
        assert len(ds) == 1
        assert ds[0].action_params["summary"] == "Office hours"
        assert "message" not in ds[0].action_params

    def test_multiple_markers_one_decision_each(self):
        text = "\n".join(
            f'<action type="calendar_create_event" summary="HW {i} due" '
            f'start_time="2026-10-0{i}T23:59:00" end_time="2026-10-0{i}T23:59:59" '
            f'reason="user asked"></action>'
            for i in range(1, 4)
        )
        ds = self._parse(text)
        assert len(ds) == 3
        assert [d.action_params["summary"] for d in ds] == [
            "HW 1 due", "HW 2 due", "HW 3 due"]

    def test_legacy_message_shape_unchanged(self):
        text = '<action type="send_telegram" recipient="@luke" reason="asked">ping me at 5</action>'
        ds = self._parse(text)
        assert len(ds) == 1
        d = ds[0]
        assert d.action_type == "send_telegram"
        assert d.action_params["recipient"] == "@luke"
        assert d.action_params["message"] == "ping me at 5"
        assert "@luke" in d.action_summary

    def test_typeless_or_paramless_markers_ignored(self):
        assert self._parse('<action reason="x">body</action>') == []
        assert self._parse('<action type="send_email"></action>') == []


# ===========================================================================
# 1b. XML-aware forced-round prompt + forced retry
# ===========================================================================

class TestXmlForcedRound:

    def test_force_prompt_uses_marker_syntax_and_spec_fields(self):
        spec = ACTION_SPECS[ActionType.CALENDAR_CREATE_EVENT]
        p = AgenticSearchController._build_xml_action_force_prompt(
            "place my deadlines on my calendar", ActionType.CALENDAR_CREATE_EVENT, spec)
        assert '<action type="calendar_create_event"' in p
        for field in ("summary", "start_time", "end_time"):
            assert field in p
        assert "propose_action" not in p          # native vocabulary stays out
        assert "one <action> marker per item" in p.lower() or "per item" in p

    def test_forced_retry_wired_in_loop(self):
        src = inspect.getsource(AgenticSearchController)
        assert "_action_force_retry_sent" in src
        # the retry must RE-ARM the force so the second round gets the directive
        assert "_force_propose_pending = True" in src

    def test_xml_branch_wired_in_force_block(self):
        src = inspect.getsource(AgenticSearchController)
        assert "_build_xml_action_force_prompt" in src


# ===========================================================================
# 2. Final-synthesis narration guard + recovery
# ===========================================================================

class TestNarrationShapedFinal:

    def test_live_narration_detected(self):
        assert _bare_controller().narration_shaped_final(LIVE_NARRATION) is True

    def test_loop_meta_opener_detected(self):
        assert _bare_controller().narration_shaped_final(
            "The first round of results missed the mark — here is what I found "
            "anyway. " + "Real content. " * 60) is True

    def test_long_substantive_answer_with_incidental_promissory_ok(self):
        text = ("Here is the complete catalog of deadlines. " * 20
                + "If anything looks off, let me check with you next time.")
        assert len(text) > 600
        assert _bare_controller().narration_shaped_final(text) is False

    def test_plain_answer_ok(self):
        assert _bare_controller().narration_shaped_final(
            "HW 1 is due Sep 13 at 11:59 PM Eastern; all seven deadlines are "
            "on your calendar proposal list now.") is False

    def test_empty_ok(self):
        assert _bare_controller().narration_shaped_final("") is False


class _MM:
    def __init__(self, out):
        self.out = out
        self.kwargs = None

    async def generate_once(self, prompt, **kwargs):
        self.kwargs = dict(kwargs, prompt=prompt)
        return self.out


GOOD_RECOVERY = (
    "Here is the full catalog for MGT 6203: HW 1 due Sep 13, HW 2 due Sep 27, "
    "HW 3 due Oct 11, HW 4 due Oct 25, HW 5 due Nov 8, HW 6 due Nov 22, and "
    "HW 7 due Dec 8, all at 11:59 PM Eastern. I could not create the calendar "
    "events this round, so tell me and I will queue the proposals."
)


class TestRegenerateFinalAnswer:

    @pytest.mark.asyncio
    async def test_recovery_returns_vetted_text(self):
        c = _bare_controller()
        c._last_final_prompt = "FINAL PROMPT"
        c._last_final_system_prompt = "SYS"
        c._last_final_model = "kimi-3"
        c.model_manager = _MM(GOOD_RECOVERY)
        out = await c.regenerate_final_answer()
        assert out == GOOD_RECOVERY
        assert c.model_manager.kwargs["disable_reasoning"] is True
        assert c.model_manager.kwargs["prompt"].startswith("FINAL PROMPT")
        assert "CLOSED" in c.model_manager.kwargs["prompt"]

    @pytest.mark.asyncio
    async def test_recovery_rejects_narration_again(self):
        c = _bare_controller()
        c._last_final_prompt = "FINAL PROMPT"
        c._last_final_system_prompt = ""
        c._last_final_model = "kimi-3"
        c.model_manager = _MM(LIVE_NARRATION)
        assert await c.regenerate_final_answer() is None

    @pytest.mark.asyncio
    async def test_recovery_rejects_stub_output(self):
        c = _bare_controller()
        c._last_final_prompt = "FINAL PROMPT"
        c._last_final_system_prompt = ""
        c._last_final_model = "kimi-3"
        c.model_manager = _MM("Ok.")
        assert await c.regenerate_final_answer() is None

    @pytest.mark.asyncio
    async def test_no_stashed_prompt_no_call(self):
        c = _bare_controller()
        assert await c.regenerate_final_answer() is None

    def test_final_prompt_stashed_in_generate(self):
        src = inspect.getsource(AgenticSearchController._generate_final_response)
        assert "_last_final_prompt" in src

    def test_handlers_wiring(self):
        import gui.handlers as handlers
        src = inspect.getsource(handlers._run_agentic_search)
        assert "narration_shaped_final" in src
        assert "regenerate_final_answer" in src
        assert "agentic_narration_recovered" in src


# ===========================================================================
# 3. Memory top-up survival floor
# ===========================================================================

class TestMemoryTopupFloor:

    def test_floor_constant(self):
        from core.prompt.builder import MEMORY_TOPUP_FLOOR
        assert 1 <= MEMORY_TOPUP_FLOOR <= 5

    def test_topup_targets_floor_not_intent_cap(self):
        import core.prompt.builder as b
        src = inspect.getsource(b.UnifiedPromptBuilder)
        assert "min(max(0, int(eff_max_mems or 0)), MEMORY_TOPUP_FLOOR)" in src


# ===========================================================================
# 4. Wrapped-prose lyrics guard (the stored content_type='lyrics' misfire)
# ===========================================================================

class TestWrappedProseNotLyrics:

    def test_live_query_not_lyrics(self):
        r = detect_content_type(WRAPPED_QUERY)
        assert r.content_type != "lyrics"

    def test_indented_continuations_rejected(self):
        assert _looks_like_lyrics(WRAPPED_QUERY) is False

    def test_column_zero_verse_still_lyrics(self):
        text = "Wake up older\nNot older or wiser but probably safe\nTake your time here\nThings that you'll want"
        assert _looks_like_lyrics(text) is True
