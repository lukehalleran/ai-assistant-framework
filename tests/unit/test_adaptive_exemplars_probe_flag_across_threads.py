"""The probe flag must survive every thread hop between ingress and a teacher.

Incident 2026-10-20 (follow-up to #93): ``_PROBE_TURN`` is a ContextVar set in
``handle_submit``; ``loop.run_in_executor`` does NOT copy contextvars
(``asyncio.to_thread`` does). Trace result: no executor hop on the path from
``handle_submit`` to a teach call (record sites: tone, need, intent, web-search
trigger, agentic gate, orchestrator citation hook, action-claim guard) drops the
flag today -- the only ``run_in_executor`` sites are wiki/web/HTTP/encode work
that never teaches. This file (a) drives the deployed teachers through both
hop styles, (b) documents that a bare executor hop DOES drop the flag, and
(c) pins the inventory of ``run_in_executor`` sites so a new hop must be
reviewed against this contract. Class: BC-63 (adjacent: context lost across
an executor boundary).
"""
import ast
import asyncio
import contextvars
from pathlib import Path

import pytest

import utils.adaptive_exemplars as ae

REPO = Path(__file__).resolve().parents[2]
MSG = "I feel completely hopeless and cannot go on like this anymore tonight"


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = ae.AdaptiveExemplarStore(str(tmp_path / "ex.json"))
    monkeypatch.setattr(ae, "_store", s)
    return s


def _teach_all():
    from utils.tone_detector import _learn_tone_exemplar
    from core.intent_classifier import _learn_intent_exemplar
    from core.action_claim_guard import record_claim_exemplar
    _learn_tone_exemplar(MSG, "high", "keyword")
    _learn_intent_exemplar(MSG, "temporal_recall", "regex")
    record_claim_exemplar("calendar", "I have added that to your calendar for you", "test")
    ae.get_store().record("web_search", "no_search", MSG, "gate_veto")


def _run(hop):
    async def main():
        token = ae.mark_probe_turn(True)
        try:
            await hop()
        finally:
            ae.end_probe_turn(token)
    asyncio.run(main())


def test_to_thread_hop_teaches_nothing(store):
    async def hop():
        await asyncio.to_thread(_teach_all)
    _run(hop)
    assert store._data == {}


def test_copy_context_executor_hop_teaches_nothing(store):
    async def hop():
        ctx = contextvars.copy_context()
        await asyncio.get_running_loop().run_in_executor(None, ctx.run, _teach_all)
    _run(hop)
    assert store._data == {}


def test_bare_executor_hop_loses_the_flag(store):
    """Documents WHY new executor hops must copy the context."""
    seen = {}

    async def hop():
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: seen.setdefault("flag", ae._PROBE_TURN.get()))
    _run(hop)
    assert seen["flag"] is False


# Every run_in_executor call site in shipped code, verified 2026-10-21 not to
# reach a teach call. A new site fails this test: confirm it cannot teach (or
# route it through asyncio.to_thread / copy_context().run) and add it here.
KNOWN_EXECUTOR_SITES = {
    "core/actions/email.py": 1,
    "core/actions/google_auth.py": 1,
    "core/prompt/gatherer_knowledge.py": 3,
    "integrations/wikipedia_api.py": 1,
    "knowledge/web_search_manager.py": 4,
    "main.py": 1,
    "memory/storage/multi_collection_chroma_store.py": 1,
    "processing/gate_system.py": 1,
}
SCAN_DIRS = ("core", "gui", "utils", "api", "models", "memory", "knowledge",
             "processing", "integrations")


def test_executor_hop_inventory_is_pinned():
    found = {}
    files = [REPO / "main.py"]
    for d in SCAN_DIRS:
        files += (REPO / d).rglob("*.py")
    for f in files:
        try:
            tree = ast.parse(f.read_text())
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        n = sum(1 for node in ast.walk(tree)
                if isinstance(node, ast.Attribute) and node.attr == "run_in_executor")
        if n:
            found[str(f.relative_to(REPO))] = n
    assert found == KNOWN_EXECUTOR_SITES
