"""The day-count anchor writer is a POST_RESPONSE_HOOKS entry (class: BC-103, BC-75).

Drives the deployed registry (run_post_response_hooks) -- the chokepoint both the GUI path and
process_user_query run. The prompt build is read-only; only this hook writes the store.
"""
import json
from types import SimpleNamespace

import core.orchestrator as orch
import memory.day_count_anchors as dca


def ctx(user_input, response="You're 23 days into the cold-shower habit."):
    return orch.PostResponseHookContext(orchestrator=SimpleNamespace(), user_input=user_input,
                                        response_text=response)


def stored():
    try:
        return json.load(open(dca.store_path()))["claims"]
    except FileNotFoundError:
        return []


def only_hook(monkeypatch):
    monkeypatch.setattr(orch, "POST_RESPONSE_HOOKS",
                        [(n, f) for n, f in orch.POST_RESPONSE_HOOKS if n == "day_count_anchors"])


def test_registered_once():
    assert [n for n, _ in orch.POST_RESPONSE_HOOKS].count("day_count_anchors") == 1


def test_hook_records_user_text_only(monkeypatch):
    only_hook(monkeypatch)
    orch.run_post_response_hooks(ctx("I'm 16 days into the cold-shower habit"))
    assert [c["count"] for c in stored()] == [16]             # the response's "23 days" never lands


def test_hook_skips_test_probes_and_assistant_only_counts(monkeypatch):
    only_hook(monkeypatch)
    orch.run_post_response_hooks(ctx("[test]I'm 16 days into the cold-shower habit[/test]"))
    orch.run_post_response_hooks(ctx("thanks, that helps"))
    assert stored() == []


def test_hook_never_raises_on_a_corrupt_store(monkeypatch):
    only_hook(monkeypatch)
    open(dca.store_path(), "w").write("{ broken")
    orch.run_post_response_hooks(ctx("I'm 16 days into the cold-shower habit"))     # isolated: no exception
    assert open(dca.store_path()).read() == "{ broken"
