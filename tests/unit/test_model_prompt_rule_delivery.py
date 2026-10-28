"""Behavioural siblings: prompt rules must be DELIVERED to the model (source-text rows).

Incident dates: 2026-09-10 (STM abbreviation expansion "doc" -> "doctor"; planner
embellished "turned 2" into a celebration), 2026-09-03 (STM invented a user_question
for a short fragment), 2026-08-24 (LLM web trigger could not set document_source),
2026-08-29 (trigger searched the web for the user's own "Monday"), 2026-08-03 (fact
extractor taught role from an activity), 2026-09-01 (XML tool vocabulary gave calendar
times with a UTC offset). Class: BC-63 (a grep of the module text stands in for the
prompt the model is actually sent).

Each test calls THE deployed entry point with a recording fake model and asserts what
the model RECEIVED (and, where a deterministic backstop exists, what happens when the
fake misbehaves). Prompt text is the contract in these rows, so the assertion is on
the prompt a real call would send — not on the source file.
"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# ------------------------------------------------------------------ STM analyzer
STM_JSON = {
    "topic": "Joke riff", "user_question": "", "intent": "continue the bit",
    "tone": "playful", "reference_type": "recall",
    "temporal_facts": [], "open_threads": [], "constraints": [],
}


class _RecordingModel:
    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    async def generate_once(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.payload if isinstance(self.payload, str) else json.dumps(self.payload)


async def _analyze(query, payload=STM_JSON, memories=()):
    from core.stm_analyzer import STMAnalyzer
    mm = _RecordingModel(payload)
    result = await STMAnalyzer(mm).analyze(
        recent_memories=list(memories), user_query=query,
        last_assistant_response="Happy to help with the joke.",
    )
    return mm.prompts[0], result


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "classic Morgan move",
    "    classic Morgan\n    move\n   ",
])
async def test_stm_prompt_tells_the_model_not_to_invent_a_question_for_a_fragment(query):
    prompt, _ = await _analyze(query)
    assert "SHORT FRAGMENT" in prompt
    assert 'do NOT invent a "user_question"' in prompt
    assert query in prompt  # the fragment itself reaches the model with the rule


ABBREV_QUERY = "Cool. Managed to push today and there is a new doc I think will be helpful"
ABBREV_PAYLOAD = {**STM_JSON, "topic": "New doctor", "user_question": "User mentions a new doctor is helpful",
                  "reference_type": "new_event",
                  "temporal_facts": ["user has a new doctor who is helpful"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [ABBREV_QUERY, "    " + ABBREV_QUERY.replace("push today", "push\n    today") + "\n  "])
async def test_stm_prompt_carries_the_abbreviation_rule_and_analysis_flags_an_expansion(query):
    memories = [{"query": "pushed a commit today", "response": "nice work"}]
    prompt, result = await _analyze(query, payload=ABBREV_PAYLOAD, memories=memories)
    assert "Never expand an abbreviation" in prompt
    # misbehaving model expanded "doc" -> "doctor": the deployed backstop records it
    assert ["doc", "doctor"] in result["abbreviation_conflicts"]


@pytest.mark.asyncio
async def test_stm_does_not_flag_an_expansion_the_user_established():
    memories = [{"query": "my new doctor is nice", "response": "good to hear"}]
    _, result = await _analyze(ABBREV_QUERY, payload=ABBREV_PAYLOAD, memories=memories)
    assert "abbreviation_conflicts" not in result


# ------------------------------------------------------------- web-search LLM trigger
async def _trigger_prompt(monkeypatch, query, reply=None):
    import utils.web_search_trigger as wst
    monkeypatch.setattr(wst.location_resolver, "get_user_location", lambda: None)
    monkeypatch.setattr(wst.institution_resolver, "get_user_institution", lambda: None)
    mm = MagicMock()
    mm.generate_once = AsyncMock(return_value=reply or json.dumps({
        "should_search": False, "confidence": 0.9, "reason": "personal", "search_terms": [],
        "search_depth": "quick", "num_searches": 1}))
    parsed = await wst._classify_with_llm_unified(query, mm)
    return mm.generate_once.await_args.args[0], parsed


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "write up what we just discussed so I can send it to my sister",
    "    write up what we just discussed\n    so I can send it to my sister\n  ",
])
async def test_trigger_prompt_asks_for_document_source_and_the_answer_threads_through(monkeypatch, query):
    reply = json.dumps({
        "should_search": False, "confidence": 0.9, "reason": "doc of this conversation",
        "search_terms": [], "search_depth": "quick", "num_searches": 1,
        "needs_document_generation": True, "document_topic": "trip plan",
        "document_type": "summary", "document_source": " Conversation ",
    })
    prompt, parsed = await _trigger_prompt(monkeypatch, query, reply)
    assert '"document_source"' in prompt and "THIS conversation" in prompt
    assert parsed.document_source == "conversation"  # normalised, case/whitespace tolerant


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "is this useful info for Monday?",
    "    is this useful\n    info for Monday?\n  ",
])
async def test_trigger_prompt_tells_the_model_monday_is_the_users_own_schedule(monkeypatch, query):
    prompt, parsed = await _trigger_prompt(monkeypatch, query)
    assert "user's Monday" in prompt
    assert "nothing on the web about" in prompt
    assert parsed is not None and parsed.should_search is False


# ------------------------------------------------------------------- response planner
class _PlanModel:
    def __init__(self, plan):
        self.plan = plan
        self.prompts = []

    async def generate_once(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return json.dumps(self.plan)


async def _plan(query, plan):
    from core.response_planner import ResponsePlanner
    mm = _PlanModel(plan)
    ctx = SimpleNamespace(original_query=query, tone_level=None, intent=None,
                          topics=[], thread_context=None, last_exchange=None)
    result = await ResponsePlanner(model_manager=mm).create_plan(query=query, context=ctx)
    return mm.prompts[0], result


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "Biscuit turned 2 last week, can't believe it.",
    "    Biscuit turned 2 last week,\n    can't believe it.\n  ",
])
async def test_planner_prompt_forbids_embellishment_and_the_guard_drops_one(query):
    prompt, result = await _plan(query, {
        "key_points": ["Biscuit turned 2 last week", "Plan a birthday party celebration"],
        "tone": "warm", "avoid": [], "strategy": "Share the milestone"})
    assert "detail the user did not state" in prompt
    assert '"turned 2" is an age' in prompt
    assert result.key_points == ["Biscuit turned 2 last week"]
    assert result.dropped_points == ["Plan a birthday party celebration"]


# ------------------------------------------------------------- LLM fact extractor prompt
@pytest.mark.asyncio
@pytest.mark.parametrize("message", [
    "I'm testing the app and monitoring myself this week",
    "    I'm testing the app and\n    monitoring myself this week\n  ",
])
async def test_fact_extractor_prompt_does_not_teach_role_from_an_activity(message):
    from memory.llm_fact_extractor import LLMFactExtractor
    mm = _RecordingModel("[]")
    ex = LLMFactExtractor(mm)
    await ex.extract_triples([{"query": message, "response": "ok"}])
    prompt = mm.prompts[0]
    assert "Never derive a role or title from an activity" in prompt
    assert '"relation": "role"' not in prompt
    assert message.strip() in prompt  # the extractor strips the envelope, keeps the words


# ------------------------------------------------------------- XML tool vocabulary
def test_xml_protocol_system_prompt_teaches_offsetless_local_calendar_times():
    from core.agentic.protocols import get_protocol_handler
    from core.agentic.types import SearchProtocol
    handler = get_protocol_handler(SearchProtocol.XML_MARKERS, actions_available=True)
    prompt = handler.augment_system_prompt("BASE SYSTEM PROMPT", max_rounds=3)
    assert prompt.startswith("BASE SYSTEM PROMPT")
    assert "WITHOUT a UTC offset" in prompt
    assert "calendar_update_event, calendar_delete_event" in prompt
