"""2026-09-30 (BC-72, BC-58): a tool's wrapped empty result must classify "empty".

Live 2026-09-30 19:39: Gmail search returned nothing and the turn record said
status 'ok', because ``_classify_tool_result`` tested ``startswith("[No ")`` on
text that begins with the handler's "---/**Round N: …**" wrapper.
"""
from types import SimpleNamespace

from core.agentic.controller import _classify_tool_result


def _email_wrapped(result_text, round_number=1):
    # Copied from core/agentic/tools.py `_dispatch_email_search`'s
    # formatted_context f-string shape (not a paraphrase).
    return (
        f"\n---\n**Round {round_number}: Email Search**\n"
        f"{result_text}\n---\n"
    )


def _res(text):
    return SimpleNamespace(formatted_context=text)


def test_wrapped_empty_email_result_is_empty():
    body = "[No emails found in the last 14 days matching midterm professor]"
    assert _classify_tool_result(_res(_email_wrapped(body))) == ("empty", "")


def test_wrapped_results_are_ok():
    body = "[EMAIL RESULTS] 2 message(s)\n1. From: a\n2. From: b"
    assert _classify_tool_result(_res(_email_wrapped(body)))[0] == "ok"


def test_wrapped_failure_is_failed():
    body = "[EMAIL SEARCH FAILED — no mailbox was searched. Gmail: token revoked.]"
    assert _classify_tool_result(_res(_email_wrapped(body)))[0] == "failed"


def test_unwrapped_empty_and_blank():
    assert _classify_tool_result(_res("[No results]")) == ("empty", "")
    assert _classify_tool_result(_res("")) == ("empty", "")
