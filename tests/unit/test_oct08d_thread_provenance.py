"""Threads come from the USER's words (2026-10-08, class: BC-75, BC-51).

Live incident: an idle-shutdown run stored a `deadline` thread from a
DAEMON reply ("You've got the Thursday check-in with him at 2:30 anyway")
while the user's own text only offered to meet. These tests drive THE
deployed ThreadExtractor.extract_new_threads with a fake generate_once.
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from memory.thread_extractor import ThreadExtractor
from memory.thread_models import ThreadType


def _extract(convos, items):
    mm = MagicMock()
    mm.generate_once = AsyncMock(return_value=json.dumps(items))
    threads = asyncio.run(ThreadExtractor(model_manager=mm).extract_new_threads(convos))
    return threads, mm


def _deadline_item(**over):
    base = {
        "topic": "Thursday check-in with professor",
        "summary": "Check-in with Prof. Rivera on Thursday at 2:30",
        "thread_type": "deadline",
        "urgency": 0.8,
        "resolution_hint": "check-in happens",
        "deadline_date": "2026-10-08",
    }
    base.update(over)
    return base


def test_assistant_only_deadline_is_not_stored():
    convos = [
        {"query": "happy to meet tomorrow", "response": "You've got the Thursday check-in at 2:30 anyway."},
        {"query": "ok thanks", "response": "Anytime."},
    ]
    threads, _ = _extract(convos, [_deadline_item()])
    assert threads == []


def test_user_stated_deadline_is_stored_with_excerpt_and_zone_kept():
    line = "my check-in with Prof. Rivera is Thursday at 2:30 EST"
    convos = [{"query": line, "response": "Noted."}, {"query": "thanks", "response": "ok"}]
    threads, _ = _extract(convos, [_deadline_item()])
    assert len(threads) == 1
    t = threads[0]
    assert t.source_summary == line
    assert "EST" in f"{t.topic} {t.summary}"
    assert "(EST as written)" in t.summary  # LLM dropped it; appended verbatim


def test_zone_already_present_is_not_duplicated():
    convos = [{"query": "check-in Thursday at 2:30 EST", "response": "ok"}]
    threads, _ = _extract(convos, [_deadline_item(summary="Check-in Thursday at 2:30 EST")])
    assert threads[0].summary.count("EST") == 1


def test_non_deadline_topic_thread_with_user_support_unchanged():
    convos = [{"query": "still thinking about the budget spreadsheet layout", "response": "Sure."}]
    item = {"topic": "Budget spreadsheet layout", "summary": "Unfinished layout discussion",
            "thread_type": "unfinished", "urgency": 0.3, "resolution_hint": "", "deadline_date": None}
    threads, _ = _extract(convos, [item])
    assert len(threads) == 1
    assert threads[0].thread_type == ThreadType.UNFINISHED
    assert threads[0].summary == "Unfinished layout discussion"
    assert threads[0].source_summary == "still thinking about the budget spreadsheet layout"


def test_unfinished_thread_is_not_gated_on_user_support():
    convos = [{"query": "hi", "response": "We could revisit the migration plan later."}]
    item = {"topic": "Migration plan", "summary": "assistant raised", "thread_type": "unfinished",
            "urgency": 0.2, "resolution_hint": "", "deadline_date": None}
    threads, _ = _extract(convos, [item])
    assert len(threads) == 1


def test_commitment_supported_by_key_object_words():
    convos = [{"query": "I should call the dentist about the crown", "response": "ok"}]
    item = {"topic": "Call dentist about crown", "summary": "s", "thread_type": "commitment",
            "urgency": 0.5, "resolution_hint": "", "deadline_date": None}
    threads, _ = _extract(convos, [item])
    assert len(threads) == 1
    assert threads[0].source_summary == "I should call the dentist about the crown"


def test_attachment_text_is_not_user_evidence():
    convos = [{"query": "see attached\nSam's meeting is Friday at 4:00", "user_text": "see attached",
               "response": "ok"}]
    item = _deadline_item(topic="Sam meeting Friday", summary="Sam meeting Friday at 4:00")
    threads, _ = _extract(convos, [item])
    assert threads == []


def test_prompt_labels_assistant_lines_as_context_only():
    convos = [{"query": "hello there", "response": "Thursday at 2:30 works."}]
    _, mm = _extract(convos, [])
    prompt = mm.generate_once.call_args.args[0]
    assert 'lines starting "Assistant:" are context only' in prompt
    assert "never a source for a commitment, deadline, date or time" in prompt
