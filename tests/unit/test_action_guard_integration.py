"""Integration tests for the action-guard wiring in gui.handlers.

Exercises the proposal → affirmation → save loop and the confab self-repair at
the handler-helper level, with a lightweight fake orchestrator and a
DaemonNotesManager redirected to a tmp directory (so the real daemon_notes/ dir
is never touched).
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import gui.handlers as h
import knowledge.daemon_notes_manager as dnm_mod
from core.action_claim_guard import ActionKind, DetectedAction
from core.pending_proposal import PendingProposalStore


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeMM:
    async def generate_once(self, *a, **k):
        return "A concise working note generated for the test."

    def get_active_model_name(self):
        return "test-model"


class _FakeCorpus:
    def __init__(self, recent):
        self._recent = list(recent)

    def get_recent_memories(self, n):
        return self._recent[-n:]


class _FakeMemSys:
    def __init__(self, corpus=None):
        self.chroma_store = None  # no embedding in tests
        self.corpus_manager = corpus
        self.session_id = "test-session"

    async def store_interaction(self, **k):
        return None


class _FakeOrch:
    def __init__(self, corpus=None):
        self.model_manager = _FakeMM()
        self.memory_system = _FakeMemSys(corpus)


@pytest.fixture
def notes_in_tmp(tmp_path, monkeypatch):
    """Redirect DaemonNotesManager writes into tmp_path/daemon_notes."""
    real = dnm_mod.DaemonNotesManager

    def factory(**kwargs):
        kwargs.setdefault("output_dir", tmp_path / "daemon_notes")
        kwargs.setdefault("repo_root", tmp_path)
        return real(**kwargs)

    monkeypatch.setattr(dnm_mod, "DaemonNotesManager", factory)
    return tmp_path / "daemon_notes"


async def _drain(agen):
    chunks = []
    async for c in agen:
        chunks.append(c)
    return chunks


# ---------------------------------------------------------------------------
# Happy path: proposal captured in turn 1, affirmed + saved in turn 2
# ---------------------------------------------------------------------------

TURN1_RESPONSE = (
    "Here's a rough shape for it:\n"
    "Week 1 — catch up on the missed videos, then review the bad quiz.\n"
    "Week 2 — start the homework early and run the MLE project in parallel.\n"
    "Want me to drop this 2-week plan into a daemon note so it's there tomorrow?"
)


@pytest.mark.asyncio
async def test_proposal_then_affirmation_saves_note(notes_in_tmp):
    orch = _FakeOrch()
    store = h._get_pending_proposal_store(orch)
    assert store is not None

    # Turn 1: assistant proposes a note → capture it.
    store.bump_turn()
    h._capture_proposal(orch, TURN1_RESPONSE)
    assert store.peek() is not None
    assert "Week 1" in store.peek().body
    assert "Want me to drop this" not in store.peek().body  # offer tail stripped

    # Turn 2: user affirms.
    store.bump_turn()
    affirmed = store.consume_if_affirmed("sure that makes sense")
    assert affirmed is not None

    ctx = SimpleNamespace(orchestrator=orch, user_text="sure that makes sense", handled=False)
    chunks = await _drain(h._run_pending_proposal(ctx, affirmed))

    assert ctx.handled is True
    # A progress chunk + a result chunk.
    assert any(c.get("is_progress") for c in chunks)
    result = chunks[-1]["content"]
    assert "Self-note saved" in result

    # The note file actually exists, and contains the real plan.
    md_files = list(notes_in_tmp.glob("*.md"))
    assert len(md_files) == 1
    body = md_files[0].read_text()
    assert "Week 1" in body
    # And it's indexed.
    index = json.loads((notes_in_tmp / "index.json").read_text())
    assert len(index) == 1


# ---------------------------------------------------------------------------
# Confab safety net: a completion claim with nothing executed → self-repair
# ---------------------------------------------------------------------------

CONFAB = "Done — saving the 2-week plan as a note so it's waiting for you tomorrow."


@pytest.mark.asyncio
async def test_claim_guard_self_repairs_unbacked_note(notes_in_tmp):
    # Recent conversation holds the plan the model "saved".
    corpus = _FakeCorpus([
        {"query": "regression plan?", "response": "Week 1 catch up on videos. Week 2 homework + project."},
    ])
    orch = _FakeOrch(corpus=corpus)
    h._get_pending_proposal_store(orch).bump_turn()
    ctx = SimpleNamespace(orchestrator=orch, user_text="sure", handled=False)

    suffix = await h._apply_action_guard(
        ctx, CONFAB, executed_kinds=set(), proposed_kinds=set(), self_repair=True,
    )
    assert "saved that note" in suffix.lower()
    md_files = list(notes_in_tmp.glob("*.md"))
    assert len(md_files) == 1
    assert "Week" in md_files[0].read_text()


@pytest.mark.asyncio
async def test_claim_guard_no_repair_when_note_executed(notes_in_tmp):
    orch = _FakeOrch()
    h._get_pending_proposal_store(orch).bump_turn()
    ctx = SimpleNamespace(orchestrator=orch, user_text="x", handled=False)
    # NOTE already executed this turn → claim is backed → no suffix, no file.
    suffix = await h._apply_action_guard(
        ctx, CONFAB, executed_kinds={ActionKind.NOTE}, proposed_kinds=set(), self_repair=True,
    )
    assert suffix == ""
    assert list(notes_in_tmp.glob("*.md")) == []


@pytest.mark.asyncio
async def test_claim_guard_external_correction_no_autosend(notes_in_tmp):
    orch = _FakeOrch()
    h._get_pending_proposal_store(orch).bump_turn()
    ctx = SimpleNamespace(orchestrator=orch, user_text="x", handled=False)
    suffix = await h._apply_action_guard(
        ctx, "I've emailed your advisor the update.",
        executed_kinds=set(), proposed_kinds=set(), self_repair=True,
    )
    assert "didn't actually" in suffix.lower()
    assert "email" in suffix.lower()
    # Nothing was auto-sent and no note was written for an external claim.
    assert list(notes_in_tmp.glob("*.md")) == []


@pytest.mark.asyncio
async def test_external_claim_suppressed_when_proposed(notes_in_tmp):
    orch = _FakeOrch()
    h._get_pending_proposal_store(orch).bump_turn()
    ctx = SimpleNamespace(orchestrator=orch, user_text="x", handled=False)
    # A proposal card was created this turn → don't nag about the email claim.
    suffix = await h._apply_action_guard(
        ctx, "I've emailed your advisor the update.",
        executed_kinds=set(), proposed_kinds={ActionKind.EMAIL}, self_repair=True,
    )
    assert suffix == ""
