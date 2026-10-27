"""The per-turn session-place extraction runs in the post-response hook, not in the next prompt build.

Incident date: 2026-10-27 (wave-2 follow-up: ~12 ms/turn, ~130 ms with a stated move, spent on the
event loop inside ``get_user_location()`` via ``WebSearchManager._get_user_location``).
Class: BC-41 (blocking work on the loop).

Drives THE deployed ``core.orchestrator.run_post_response_hooks`` with the real resolver, a provider
callable and a spy around the deployed extractor (``_stated_place_from_text``): the hook extracts the
new turn once; the following ``get_user_location()`` only reads.  Statements asserted clean and wrapped.
"""
import time
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

import utils.location_resolver as lr
from core.orchestrator import POST_RESPONSE_HOOKS, PostResponseHookContext, run_post_response_hooks


def _turn(query):
    ts = (datetime.now() - timedelta(seconds=5)).isoformat()  # clock-safe: compared only against session_start (also now-relative)
    return {"query": query, "response": "ok", "timestamp": ts}


@pytest.fixture
def resolver(monkeypatch, tmp_path):
    monkeypatch.setattr(lr, "LOCATION_ENABLED", True)
    monkeypatch.setattr(lr, "LOCATION_IP_LOOKUP_ENABLED", False)
    monkeypatch.setattr(lr, "LOCATION_OVERRIDE", "")
    r = lr.LocationResolver(profile_path=str(tmp_path / "none.json"), session_start=time.time() - 3600)  # clock-safe: now-relative only
    monkeypatch.setattr(lr, "_resolver", r)
    return r


def _only_session_hook(monkeypatch):
    rows = [(n, f) for n, f in POST_RESPONSE_HOOKS if n == "session_location"]
    assert rows, "session_location hook is not registered"
    monkeypatch.setattr("core.orchestrator.POST_RESPONSE_HOOKS", rows)


@pytest.mark.parametrize("text", [
    "I just moved to Portland, Oregon.",
    "  I just moved\n  to Portland,\n  Oregon.",
])
def test_hook_extracts_once_and_next_lookup_only_reads(resolver, monkeypatch, text):
    entries = [_turn(text)]
    resolver.turn_provider = lambda: entries
    calls = []
    real = lr._stated_place_from_text
    monkeypatch.setattr(lr, "_stated_place_from_text", lambda t: (calls.append(t), real(t))[1])
    _only_session_hook(monkeypatch)

    run_post_response_hooks(PostResponseHookContext(orchestrator=SimpleNamespace(), user_input=text))
    assert len(calls) == 1, "the hook must run the extraction"
    assert lr.get_user_location() == "Portland, Oregon"
    assert len(calls) == 1, "get_user_location must only read after the hook"


def test_hook_is_inert_when_location_disabled(resolver, monkeypatch):
    monkeypatch.setattr(lr, "LOCATION_ENABLED", False)
    resolver.turn_provider = lambda: [_turn("I moved to Salem, Oregon")]
    calls = []
    monkeypatch.setattr(lr, "_stated_place_from_text", lambda t: calls.append(t))
    _only_session_hook(monkeypatch)
    run_post_response_hooks(PostResponseHookContext(orchestrator=SimpleNamespace(), user_input="x"))
    assert calls == []
