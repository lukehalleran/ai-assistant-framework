"""A retrieved turn must reach the prompt in one of the two sections.

Cross-section dedup (ContentHygiene._hygiene_and_caps) removes a memory when the same turn
sits in recent_conversations. The token budget runs afterwards and may end the recent section
early (prior-session cap, section cap, total budget). Before 2026-11-22 the turn then reached
neither [RECENT CONVERSATION] nor [RELEVANT MEMORIES]: on a replay of LongMemEval half A the
hand-off restores 33 turns on 21 questions, six of them marked evidence. Both deployed functions
run here.
"""
import asyncio
from datetime import datetime, timedelta

from core.prompt.hygiene import MEMORIES_SHADOWED_BY_RECENT, ContentHygiene, _canonical_turn_key
from core.prompt.token_manager import TokenManager
from models.tokenizer_manager import TokenizerManager

NOW = datetime(2031, 3, 20, 9, 0, 0)


class _ApiModel:
    def get_active_model_name(self):
        return "gpt-4o"

    def is_api_model(self, name):
        return True

    def get_context_limit(self):
        return 128000


def _turn(n: int):
    filler = " ".join(f"detail{n}x{i}" for i in range(120))
    return f"Question number {n} about harbour schedule {n}", f"Answer number {n}: {filler}"


def _recent(n: int, newest_first_index: int) -> dict:
    q, r = _turn(n)
    ts = NOW - timedelta(days=30, minutes=newest_first_index)
    return {"query": q, "response": r, "timestamp": ts.isoformat()}


def _memory(n: int) -> dict:
    q, r = _turn(n)
    return {"content": f"User: {q}\nAssistant: {r}", "metadata": {"timestamp": (NOW - timedelta(days=30)).isoformat()}}


def _context() -> dict:
    # newest first, ten turns of one earlier session; memories hold copies of turn 1 and turn 8
    recent = [_recent(n, n) for n in range(10)]
    return {"recent_conversations": recent, "memories": [_memory(8), _memory(77), _memory(1)]}


def _keys(items) -> set:
    return {_canonical_turn_key(i) for i in items}


def _run(budget: int, drop_record: bool = False) -> dict:
    ctx = asyncio.run(ContentHygiene(None, None)._hygiene_and_caps(_context()))
    assert [_canonical_turn_key(m) for m in ctx["memories"]] == [_canonical_turn_key(_memory(77))]
    assert len(ctx[MEMORIES_SHADOWED_BY_RECENT]) == 2
    if drop_record:
        ctx.pop(MEMORIES_SHADOWED_BY_RECENT)
    mm = _ApiModel()
    tm = TokenManager(model_manager=mm, tokenizer_manager=TokenizerManager(mm), token_budget=budget)
    return tm._manage_token_budget(ctx, now=NOW)


def test_memory_returns_when_the_budget_cuts_its_recent_copy():
    out = _run(budget=3000)
    shown_recent = _keys(out["recent_conversations"])
    assert 1 <= len(out["recent_conversations"]) < 10, "the prior-session cap must end the section early"
    assert _canonical_turn_key(_memory(1)) in shown_recent
    assert _canonical_turn_key(_memory(8)) not in shown_recent
    mems = [_canonical_turn_key(m) for m in out["memories"]]
    # back at the position it held (first), ahead of the unrelated memory
    assert mems[:2] == [_canonical_turn_key(_memory(8)), _canonical_turn_key(_memory(77))]
    # the copy whose recent item is still shown stays removed
    assert _canonical_turn_key(_memory(1)) not in mems
    assert not (shown_recent & set(mems)), "no turn appears in both sections"


def test_nothing_returns_when_every_recent_item_is_shown():
    out = _run(budget=10 ** 6)
    assert len(out["recent_conversations"]) == 10
    assert [_canonical_turn_key(m) for m in out["memories"]] == [_canonical_turn_key(_memory(77))]


def test_without_the_dedup_record_the_turn_is_lost():
    """Sensitivity: the hand-off from dedup to the budget is what keeps the turn."""
    out = _run(budget=3000, drop_record=True)
    assert _canonical_turn_key(_memory(8)) not in _keys(out["recent_conversations"]) | _keys(out["memories"])
