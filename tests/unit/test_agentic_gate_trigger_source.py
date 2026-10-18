"""Which gate arm triggered the agentic loop (``AgenticDecision.trigger_source``).

Incident date: 2026-10-17 (lane AGT). Bug class: BC-72 (a telemetry/debug field
that is never populated). Turn telemetry's ``gate_reason`` ("triggered:
memory") names the MODE, not the tier/arm, so 115 of 126 agentic turns since
2026-09-30 (no tool called) could not be attributed to the arm that started
the loop. These tests drive THE deployed ``evaluate_agentic_gate`` and the
deployed handler helpers.
"""
import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agentic.gate import TRIGGER_SOURCES, AgenticDecision, apply_intent_veto, evaluate_agentic_gate


def _llm(**kw):
    base = dict(
        should_search=False, search_terms=[], needs_memory_search=False,
        needs_knowledge_search=False, needs_document_generation=False,
    )
    base.update(kw)
    return MagicMock(**base)


async def _tier4(text, decision):
    with patch(
        "utils.web_search_trigger.analyze_for_web_search_llm",
        new_callable=AsyncMock, return_value=decision,
    ):
        return await evaluate_agentic_gate(text, model_manager=MagicMock())


class _Corpus:
    def __init__(self, response, query="x"):
        self._e = {"query": query, "response": response,
                   "timestamp": datetime.now(timezone.utc)}

    def get_recent_memories(self, n=1):
        return [self._e]


# (message, expected source) for the deterministic Tier-1 arms
TIER1_CASES = [
    ("calculate fibonacci 10", "tier1_computation"),
    ("do you remember my brother's name?", "tier1_memory"),
    ("explain in depth how photosynthesis works", "tier1_knowledge"),
    ("check out https://example.com for details", "tier1_url"),
    ("read the file config/config.yaml please", "tier1_file"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("text,expected", TIER1_CASES)
@pytest.mark.parametrize("wrap", ["clean", "wrapped"])
async def test_tier1_arms_name_their_source(text, expected, wrap):
    # wrapped form: indented, hard-wrapped before the last word, trailing blank line
    msg = text if wrap == "clean" else "    " + "\n    ".join(text.rsplit(" ", 1)) + "\n   "
    d = await evaluate_agentic_gate(msg)
    assert d.should_trigger
    assert d.trigger_source == expected
    assert d.trigger_source in TRIGGER_SOURCES


@pytest.mark.asyncio
async def test_tier1_web_keyword_source():
    with patch("utils.web_search_trigger.paid_search_block_reason", return_value=None):
        d = await evaluate_agentic_gate("search the web for python tutorials please")
    assert d.should_trigger and "web_search" in d.modes
    assert d.trigger_source in {"tier1_web_keyword", "tier1_fresh_public"}


@pytest.mark.asyncio
async def test_tier2_entity_recall_source():
    with patch("memory.graph_utils.extract_graph_entities", return_value={"biscuit"}):
        d = await evaluate_agentic_gate(
            "what do you know about Biscuit?", entity_resolver=MagicMock(),
        )
    assert d.trigger_source == "tier2_entity_recall"


@pytest.mark.asyncio
async def test_email_by_name_source():
    d = await evaluate_agentic_gate("email Rowan about dinner on Friday")
    assert d.should_trigger
    assert d.trigger_source in {"tier1_tool_keyword", "tier1_email", "tier1_action"}
    assert d.trigger_source in TRIGGER_SOURCES


@pytest.mark.asyncio
async def test_tier3_doc_gen_source():
    with patch(
        "knowledge.document_generator.detect_document_intent",
        return_value={"topic": "AI overview", "doc_type": "report", "focus": None},
    ):
        d = await evaluate_agentic_gate("write a report about recent AI developments")
    assert d.trigger_source == "tier3_doc_gen"


@pytest.mark.asyncio
async def test_tier3_self_note_source():
    with patch(
        "knowledge.daemon_notes_manager.detect_self_note_intent",
        return_value={"topic": "architecture decisions", "category": "architecture"},
    ):
        d = await evaluate_agentic_gate("save a note: the cache uses a ring buffer")
    assert d.trigger_source == "tier3_self_note"


@pytest.mark.asyncio
async def test_tier4_llm_memory_source():
    d = await _tier4(
        "Can you describe the color of my cat's fur exactly?",
        _llm(should_search=False, search_terms=[], needs_memory_search=True),
    )
    assert d.should_trigger and d.trigger_source == "tier4_llm_memory"


@pytest.mark.asyncio
async def test_tier4_llm_knowledge_source():
    d = await _tier4(
        "Which dynasty built the great wall of China originally?",
        _llm(should_search=False, search_terms=[], needs_knowledge_search=True),
    )
    assert d.should_trigger and d.trigger_source == "tier4_llm_knowledge"


@pytest.mark.asyncio
async def test_tier4_llm_web_source():
    with patch("utils.web_search_trigger.paid_search_block_reason", return_value=None):
        d = await _tier4(
            "Could you look into the current ferry timetable for the island route",
            _llm(should_search=True, search_terms=["ferry timetable"]),
        )
    assert d.should_trigger and d.trigger_source == "tier4_llm_web"


@pytest.mark.asyncio
async def test_continuation_file_source():
    corpus = MagicMock()
    corpus.get_recent_memories.return_value = [
        {"query": "verify the save",
         "response": "I can't read files this turn. Want me to pull that up next turn?"},
    ]
    d = await evaluate_agentic_gate("yes please", corpus_manager=corpus)
    assert d.should_trigger
    assert d.trigger_source == "continuation_file"


def test_offer_affirmation_and_explicit_action_sources():
    from core.actions.types import PendingActionsStore
    from core.agentic.tools import ToolExecutor

    store = PendingActionsStore(ttl_seconds=300, max_pending=5, persist=False)
    corpus = _Corpus("Want me to go ahead and create the recurring calendar event for the study group?")
    with patch.object(ToolExecutor, "_get_pending_actions_store", return_value=store):
        d = asyncio.run(evaluate_agentic_gate("yes", corpus_manager=corpus))
    assert d.forced_action
    assert d.trigger_source == "offer_affirmation"


@pytest.mark.asyncio
async def test_explicit_action_source():
    d = await evaluate_agentic_gate(
        "add a calendar event for the study group on Tuesday at 3pm"
    )
    assert d.forced_action
    assert d.trigger_source in {"tier1_action", "tier1_tool_keyword"}


@pytest.mark.asyncio
async def test_untriggered_message_has_empty_source():
    d = await evaluate_agentic_gate("ok thanks")
    assert not d.should_trigger
    assert d.trigger_source == ""


@pytest.mark.asyncio
async def test_vetoed_decision_clears_source():
    d = await evaluate_agentic_gate("do you remember my brother's name?")
    assert d.trigger_source == "tier1_memory"
    # Same-instance before/after: the veto flips should_trigger and the source.
    d.veto_exempt = False
    intent = SimpleNamespace(intent_type="casual_social", confidence=0.95)
    vetoed = apply_intent_veto(d, intent)
    if not vetoed.should_trigger:
        assert vetoed.trigger_source == ""


def test_default_decision_has_empty_source():
    assert AgenticDecision(should_trigger=False).trigger_source == ""


# --------------------------------------------------------------------------
# Debug record + telemetry carry the key
# --------------------------------------------------------------------------
def test_gate_trigger_source_helper_is_mock_safe():
    from gui.handlers import _gate_trigger_source

    assert _gate_trigger_source(None) == ""
    assert _gate_trigger_source(MagicMock()) == ""  # auto-vivified attr is not a str
    assert _gate_trigger_source(SimpleNamespace(trigger_source="tier1_url")) == "tier1_url"


def test_debug_record_carries_gate_trigger_source_next_to_gate_reason():
    from gui.handlers import _build_debug_record, _gate_debug_summary, _gate_trigger_source

    d = AgenticDecision(should_trigger=True, modes=["memory"], reason="triggered: memory",
                        trigger_source="tier1_memory")
    rec = _build_debug_record(
        "agentic-search", "q", "p", "s", "r", "m", 1, 1, 2, [], MagicMock(),
        gate_reason=_gate_debug_summary(d),
        gate_trigger_source=_gate_trigger_source(d),
    )
    assert rec["gate_reason"].startswith("triggered: memory")
    assert rec["gate_trigger_source"] == "tier1_memory"
    # Default (caller without a gate) stays an empty string, key present.
    rec2 = _build_debug_record("enhanced", "q", "p", "s", "r", "m", 1, 1, 2, [], MagicMock())
    assert rec2["gate_trigger_source"] == ""


def test_turn_telemetry_row_keeps_gate_trigger_source(tmp_path, monkeypatch):
    import json

    from utils import turn_telemetry

    path = tmp_path / "turns.jsonl"
    monkeypatch.setattr("config.app_config.TURN_TELEMETRY_ENABLED", True, raising=False)
    monkeypatch.setattr("config.app_config.TURN_TELEMETRY_PATH", str(path), raising=False)
    assert turn_telemetry.record_turn({"gate_reason": "triggered: memory",
                                       "gate_trigger_source": "tier1_memory"})
    row = json.loads(path.read_text().splitlines()[-1])
    assert row["gate_trigger_source"] == "tier1_memory"


def test_handler_gate_telemetry_wiring_sets_the_key():
    """The telemetry.update block in the deployed submit path must carry the
    key (source-level pin: the block is inside a 600-line coroutine)."""
    import inspect

    import gui.handlers as h

    src = inspect.getsource(h)
    assert '"gate_trigger_source": _gate_trigger_source(_gate_decision)' in src
