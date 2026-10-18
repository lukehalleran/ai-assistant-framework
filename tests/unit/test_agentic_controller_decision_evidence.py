"""Decision round sees the turn's evidence; nudge only on narration.

Incident 2026-10-18 (live [test] probes, kimi-3, native_tools): 115 of 126
agentic turns since 09-30 called no tool, and every one paid TWO full calls --
the decision round (answer text discarded) and a full synthesis -- because the
A4/B1 reuse gate refused reuse whenever a retrieval key was non-empty and the
decision prompt rendered only a digest + counts. Separately the round-1
"narrated tool intent" nudge matched a bare substring list ('search',
'commits', ...), so a FINISHED answer saying "search results" was thrown away
and re-asked (+7 s, +15K tokens), with XML examples shown to a native-tools
model.

Classes: BC-69 (latency: duplicate full-context call), and the bare-substring
match class for the nudge (BC-01 keyword-boundary family; closed here with a
structural test over the deployed `_usable_decision_answer` /
`_PROMISSORY_RE` / `narration_shaped_final`).
"""
import types
from datetime import datetime as _real_datetime
from unittest.mock import MagicMock

import pytest

from core.agentic.controller import AgenticSearchController
from core.agentic.tools import ToolExecutor
from core.agentic.types import AgenticSearchSession, SearchDecision, SearchProtocol

from tests.unit.test_agentic_decision_answer_reuse import LONG_ANSWER


# ---------------------------------------------------------------------------
# Fixtures (invented values only)
# ---------------------------------------------------------------------------

def rich_context():
    return {
        "recent_conversations": [
            {"timestamp": "2026-10-17 20:00", "query": "How is Rowan's rehearsal going?",
             "response": "Rehearsal went long but the Tamsin duet came together."},
        ],
        "memories": [
            {"timestamp": "2026-10-10", "content": "Planning the Tamsin birthday dinner for Friday.",
             "response": "Noted: dinner Friday at the lakeside cafe."},
        ],
        "user_profile": "Hobbies: cello, climbing.\nPets: one cat named Biscuit.",
        "recent_summaries": [{"timestamp": "2026-10-16", "content": "Discussed cello practice routines."}],
        "semantic_summaries": [{"timestamp": "2026-09-30", "content": "Talked about climbing gym schedules."}],
        "personal_notes": [
            {"content": "Cello recital repertoire: Bach suite no. 1, Faure Elegie.",
             "metadata": {"title": "Recital plan", "tags": "music"}},
        ],
        "dreams": [{"timestamp": "2026-10-01", "content": "A dream about an endless staircase."}],
        "reference_docs": [
            {"content": "Daemon scores memories by relevance and recency.",
             "metadata": {"title": "Scoring", "section": "overview"}},
        ],
        "user_uploads": [
            {"content": "Syllabus: week 3 covers regression diagnostics.",
             "metadata": {"title": "upload:Syllabus.pdf"}},
        ],
        "recent_reflections": [{"timestamp": "2026-10-15", "content": "The user values steady routines."}],
    }


def _mock_tool_executor():
    class _Exec:
        pass
    ex = _Exec()
    ex._current_web_source_map = {}
    ex._current_wiki_source_map = {}
    ex.get_tool_health = lambda: "All tools nominal."
    ex._merge_web_ids = types.MethodType(ToolExecutor._merge_web_ids, ex)
    return ex


class _FrozenDatetime(_real_datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 10, 18, 9, 30, 0)


class _NoPending:
    def get_pending(self):
        return []


@pytest.fixture
def controller(monkeypatch):
    manager = MagicMock()
    manager.api_models = {}
    c = AgenticSearchController(model_manager=manager, web_search_manager=MagicMock())
    c._tool_executor = _mock_tool_executor()
    monkeypatch.setattr("core.agentic.controller.datetime", _FrozenDatetime)
    monkeypatch.setattr(ToolExecutor, "_get_pending_actions_store",
                        staticmethod(lambda: _NoPending()))
    return c


def _wrap(text):
    """Wrapped/indented variant of a live text (workflow section 3)."""
    return "\n".join("    " + line for line in text.replace(". ", ".\n").splitlines())


async def _run(controller, initial_context, decision_fn, final_called=None):
    async def fake_final(query, system_prompt, model_name, session, initial_context=None):
        if final_called is not None:
            final_called.append(True)
        yield "SYNTHESIZED ANSWER."

    controller._generate_final_response = fake_final
    controller._get_model_decision = decision_fn
    events = []
    async for ev in controller.run_agentic_search(
        query="What did I plan for Friday?", system_prompt="sys", model_name="test-model",
        initial_search_terms=[], skip_initial_search=True,
        initial_context=initial_context,
    ):
        events.append(ev)
    return "".join(e for e in events if isinstance(e, str)), controller


def _answering_model(prompts, answer=LONG_ANSWER):
    async def fake_decision(*a, **k):
        prompts.append(k.get("prompt") if "prompt" in k else a[0])
        return [SearchDecision(wants_answer=True, partial_response=answer)]
    return fake_decision


# ---------------------------------------------------------------------------
# Item 1: one evidence renderer for both calls
# ---------------------------------------------------------------------------

# _build_final_prompt output on the BASE tree (pristine 2cddc8d / base5_1018, run once
# with the frozen clock and no pending action), embedded verbatim.
BASE_FINAL_PROMPT = "[RECENT CONVERSATION — THIS SESSION'S HISTORY]\nContext only — do not reply to these turns as if they were the current message. But they are established ground truth for this session: if a search result contradicts what was already settled here, surface the conflict and trust the session unless the new evidence is clearly stronger — do NOT silently override it.\nEntries ending in [...truncated] are PREVIEWS, not complete messages — never quote one as if it were the full text; use search_memory (conversations) to retrieve the full stored message first.\n[MEM_RECENT_1] 2026-10-17 20:00: User: How is Rowan's rehearsal going?\n   Daemon: Rehearsal went long but the Tamsin duet came together.\n\n[RELEVANT MEMORIES]\n[MEM_SEMANTIC_1] 2026-10-10: Planning the Tamsin birthday dinner for Friday.\n   Response: Noted: dinner Friday at the lakeside cafe.\n\n[USER PROFILE]\nStored facts — reference naturally but do not add names, apps, or details not written here.\nHobbies: cello, climbing.\nPets: one cat named Biscuit.\n\n[RECENT SUMMARIES]\n1) [2026-10-16] Discussed cello practice routines.\n\n[SEMANTIC SUMMARIES]\n1) [2026-09-30] Talked about climbing gym schedules.\n\n[USER'S PERSONAL NOTES]\n1) Recital plan [tags: music]: Cello recital repertoire: Bach suite no. 1, Faure Elegie.\n\n[RECENT DREAMS]\n1) [2026-10-01] A dream about an endless staircase.\n\n[DAEMON DOCUMENTATION]\n1) **Scoring** (overview)\nDaemon scores memories by relevance and recency.\n\n[USER UPLOADED ITEMS]\n1) **Syllabus.pdf**\nSyllabus: week 3 covers regression diagnostics.\n\n[RECENT REFLECTIONS]\n1) [2026-10-15] The user values steady routines.\n\n[TIME CONTEXT]\nCurrent time: Sunday, 2026-10-18 09:30:00\n\n[TOOL RESULTS - 0 rounds] (each block is labeled by its source kind: web search, memory, file, computation, fetched page, pre-gathered web)\n[Search Round 1] Query: dinner\nMEMORY: dinner Friday.\n\n[CURRENT USER QUERY — RESPOND TO THIS]\nWhat did I plan for Friday?\n\n[TOOL STATUS — report these accurately, never claim a tool works if it says UNAVAILABLE]\nAll tools nominal.\n\nPlease provide a comprehensive answer based on ALL context above:\n- Use your memories, facts, and personal notes to personalize the response\n- Cite web sources when stating facts from search results\n- Use [TIME CONTEXT] above as the authoritative source for today's date and day of the week.\n- Note any uncertainties or conflicting information\n- Focus on answering the user's specific question\n- If asked about tool status, ONLY report what [TOOL STATUS] says — do NOT rely on prior conversation"


def _final_prompt(controller):
    session = AgenticSearchSession(query="What did I plan for Friday?")
    session.accumulated_context = "[Search Round 1] Query: dinner\nMEMORY: dinner Friday."
    return controller._build_final_prompt(
        query="What did I plan for Friday?", session=session,
        initial_context=rich_context(),
    )


class TestFinalPromptUnchanged:
    def test_rich_fixture_byte_identical_to_base(self, controller):
        assert _final_prompt(controller) == BASE_FINAL_PROMPT

    def test_rules_are_the_closing_block(self, controller):
        out = _final_prompt(controller)
        assert out.endswith(controller._final_answer_rules())

    def test_renderer_reports_keys_that_produced_a_part(self, controller):
        parts, keys = controller._render_evidence_sections(rich_context())
        assert keys == frozenset(rich_context().keys())
        assert len(parts) == len(keys)
        parts, keys = controller._render_evidence_sections({"memories": [], "google_calendar": ["x"]})
        assert parts == [] and keys == frozenset()


class TestDecisionRoundSeesEvidence:
    @pytest.mark.asyncio
    async def test_rich_context_answer_is_reused_and_prompt_has_evidence(self, controller):
        prompts, final_called = [], []
        text, c = await _run(controller, rich_context(), _answering_model(prompts), final_called)
        assert final_called == []
        assert LONG_ANSWER in text
        p = prompts[0]
        for needle in ("Tamsin birthday dinner", "Bach suite no. 1",
                       "week 3 covers regression diagnostics", "Discussed cello practice routines"):
            assert needle in p
        # evidence block FIRST, ahead of the per-round clock/question
        assert p.index("[EVIDENCE ALREADY RETRIEVED") < p.index("Today is")
        # the closing rules ride in instruction 1
        assert "Focus on answering the user's specific question" in p

    @pytest.mark.asyncio
    async def test_receipts_on_reuse(self, controller):
        prompts = []
        _, c = await _run(controller, rich_context(), _answering_model(prompts))
        sess = c._last_session
        assert sess.answer_call == "decision_reuse"
        assert {"memories", "personal_notes", "user_uploads"} <= set(sess.decision_rendered_keys)
        assert sess.omitted_sections == []
        assert "[RELEVANT MEMORIES]" in sess.visible_sources["sections"]

    @pytest.mark.asyncio
    async def test_unrendered_keys_do_not_block_reuse(self, controller):
        ctx = {
            "google_calendar": [{"summary": "Dentist"}],
            "graph_context": ["Rowan -> friend_of -> user"],
            "relevant_emails": [{"subject": "Hi"}],
        }
        prompts, final_called = [], []
        text, _ = await _run(controller, ctx, _answering_model(prompts), final_called)
        assert final_called == []
        assert LONG_ANSWER in text

    @pytest.mark.asyncio
    async def test_budget_emptied_key_still_blocks_reuse(self, controller):
        ctx = {"memories": [], "_budget_emptied": ["memories"]}
        prompts, final_called = [], []
        text, _ = await _run(controller, ctx, _answering_model(prompts), final_called)
        assert final_called == [True]
        assert "SYNTHESIZED ANSWER." in text

    @pytest.mark.asyncio
    async def test_budget_emptied_unrendered_key_does_not_block(self, controller):
        ctx = {"google_calendar": [], "_budget_emptied": ["google_calendar"]}
        prompts, final_called = [], []
        await _run(controller, ctx, _answering_model(prompts), final_called)
        assert final_called == []

    @pytest.mark.asyncio
    async def test_evidence_block_is_identical_prefix_every_round(self, controller):
        prompts = []
        calls = {"n": 0}

        async def fake_decision(*a, **k):
            prompts.append(k["prompt"])
            calls["n"] += 1
            if calls["n"] == 1:
                return [SearchDecision(wants_memory_search=True, memory_query="x",
                                       memory_collection="conversations")]
            return [SearchDecision(is_done=True, partial_response=LONG_ANSWER)]

        class _Res:
            start_events = []
            end_events = []
            round_data = MagicMock(duration_ms=1.0)
            formatted_context = "MEMORY: relevant context."
            memory_collection = "conversations"
            is_expand = False
            decision = SearchDecision()

        async def fake_dispatch(*a, **k):
            return _Res()

        controller._dispatch_single = fake_dispatch
        await _run(controller, rich_context(), fake_decision)
        assert len(prompts) == 2
        block, _, _ = controller._build_evidence_block(rich_context())
        header = prompts[0].split("\n", 1)[0]
        for p in prompts:
            assert p.startswith(header)
            assert block in p
            assert p.index(block) == prompts[0].index(block)
        assert prompts[0][: prompts[0].index("Today is")] == prompts[1][: prompts[1].index("Today is")]

    def test_oversized_notes_dropped_from_block_and_blocks_reuse(self, controller):
        ctx = rich_context()
        ctx["personal_notes"] = [{"content": "x " * 5000, "metadata": {"title": "Big"}}]
        ctx["memories"] = [{"timestamp": "t", "content": "short memory"}]
        # item content is clipped per formatter, so shrink the ceiling instead
        # of relying on the oversized text alone
        controller.context_budget_tokens = 20  # ceiling = 100 tokens
        block, keys, _ = controller._build_evidence_block(ctx)
        assert "personal_notes" not in keys
        assert "recent_conversations" in keys  # never on the ladder
        session = AgenticSearchSession(query="q")
        session.decision_rendered_keys = keys
        # personal_notes is a direct-rendered key the decision block lost, so
        # the gate stays conservative and blocks reuse (synthesis will run its
        # own ladder; it may keep the notes).
        assert controller._first_unmet_retrieval_key(ctx, session) == "personal_notes"


# ---------------------------------------------------------------------------
# Item 2: nudge only for narration
# ---------------------------------------------------------------------------

FINISHED_ANSWER = (
    "Your search results from last week show three commits touching the scheduler, "
    "and the commits all landed on Tuesday. The search results also confirm that the "
    "regression tests passed on every one of those commits, so nothing needs a revert. "
    "That is the full picture as of this morning."
)


class TestNudgeOnlyOnNarration:
    @pytest.mark.parametrize("wrap", [False, True])
    @pytest.mark.asyncio
    async def test_finished_answer_mentioning_search_and_commits_is_not_nudged(self, controller, wrap):
        answer = _wrap(FINISHED_ANSWER) if wrap else FINISHED_ANSWER
        prompts, final_called = [], []
        text, _ = await _run(controller, {}, _answering_model(prompts, answer), final_called)
        assert len(prompts) == 1, "finished answer was re-asked"
        assert final_called == []
        assert "three commits" in text

    @pytest.mark.parametrize("wrap", [False, True])
    @pytest.mark.asyncio
    async def test_promissory_text_triggers_nudge_once(self, controller, wrap):
        plan = "Let me search for that."
        plan = _wrap(plan) if wrap else plan
        prompts = []
        calls = {"n": 0}

        async def fake_decision(*a, **k):
            prompts.append(k["prompt"])
            calls["n"] += 1
            if calls["n"] == 1:
                return [SearchDecision(wants_answer=True, partial_response=plan)]
            return [SearchDecision(wants_answer=True, partial_response=LONG_ANSWER)]

        text, _ = await _run(controller, {}, fake_decision)
        assert len(prompts) == 2
        assert "NOT executed" in prompts[1]

    def test_native_nudge_has_no_xml_examples(self):
        txt = AgenticSearchController._tool_nudge_text(SearchProtocol.NATIVE_TOOLS)
        assert "<github>" not in txt and "<git_stats>" not in txt and "<action" not in txt
        assert "function-calling" in txt

    def test_xml_nudge_keeps_marker_examples(self):
        txt = AgenticSearchController._tool_nudge_text(SearchProtocol.XML_MARKERS)
        assert "<github>open issues</github>" in txt

    @pytest.mark.asyncio
    async def test_native_session_nudge_text_reaches_prompt(self, controller, monkeypatch):
        monkeypatch.setattr(controller, "detect_protocol", lambda m: SearchProtocol.NATIVE_TOOLS)
        prompts = []
        calls = {"n": 0}

        async def fake_decision(*a, **k):
            prompts.append(k["prompt"])
            calls["n"] += 1
            if calls["n"] == 1:
                return [SearchDecision(wants_answer=True, partial_response="Let me search for that.")]
            return [SearchDecision(wants_answer=True, partial_response=LONG_ANSWER)]

        await _run(controller, {}, fake_decision)
        assert "<github>" not in prompts[1]
        assert "function-calling" in prompts[1]
