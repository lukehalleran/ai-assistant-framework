"""[YOUR STATED FIGURES] prompt block (builder -> formatter -> budget -> instruction).

Incident 2026-10-20: Daemon restated the user's CONDITIONAL ("if I can get 6 done
today then I am only 6 days behind") as plain fact and quoted a superseded ESTIMATE.
Prompt instructions alone do not hold (BC-46), so the user's own figures, each with
its date and modality, are a STRUCTURAL input rendered with no LLM. Classes: BC-75,
BC-52, BC-46, BC-99 (budget-metered section), BC-91 (header vocabulary), BC-82 (live flags).

Drives THE deployed builder (build_prompt), formatter (_assemble_prompt), token
manager and section_instructions with a sandboxed ledger; every value is invented.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

import memory.claim_ledger_store as store
from config import app_config
from core.prompt import section_instructions
from core.prompt.builder import UnifiedPromptBuilder
from core.prompt.formatter import PromptFormatter, _sanitize_embedded_headers
from memory import claim_ledger as cl
from tests.unit.test_independent_prompt_audit import full_builder, retrieval_limits

QUERY = "How far behind am I on the reading?"


def when(days_ago: float) -> str:
    return (datetime.now() - timedelta(days=days_ago)).isoformat(timespec="seconds")


def seed(*specs):
    """specs: (key, value, modality, days_ago, quote, desc)"""
    ledger = store.get_ledger()
    for key, value, modality, days_ago, quote, desc in specs:
        at = when(days_ago)
        ledger.add(cl.LedgerClaim(
            id=cl.make_claim_id(key, "duration", at, value, quote + modality), subject_key=key,
            description=desc, kind="duration", value=value, unit="days", modality=modality,
            stated_at=at, quote=quote))
    return ledger


SPECS = (
    ("days_behind", "6", "conditional", 3, "if I can get 6 done today then I am only 6 days behind", "days behind the class"),
    ("days_spent", "12", "estimate", 2, "12 days i think?", "days spent on the material"),
)


@pytest.fixture()
def on(monkeypatch):
    monkeypatch.setattr(app_config, "CLAIM_LEDGER_ENABLED", True)
    monkeypatch.setattr(app_config, "CLAIM_LEDGER_RENDER_BLOCK", True)


async def build(monkeypatch, query=QUERY):
    builder = full_builder(monkeypatch, [{"query": "hi", "response": "hello"}], budget=10000)
    return builder, await builder.build_prompt(query, retrieval_overrides=retrieval_limits())


# --- gating -------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_block_present_with_conditional_and_estimate_wording(monkeypatch, on):
    seed(*SPECS)
    _builder, ctx = await build(monkeypatch)
    text = "\n".join(ctx["stated_figures"])
    assert "CONDITIONAL" in text and "not a fact" in text and "your ESTIMATE" in text and "~12 days" in text
    assert len(ctx["stated_figures"]) == 2


@pytest.mark.asyncio
async def test_block_absent_when_the_ledger_is_off(monkeypatch):
    seed(*SPECS)
    monkeypatch.setattr(app_config, "CLAIM_LEDGER_RENDER_BLOCK", True)        # render alone is not enough
    _b, ctx = await build(monkeypatch)
    assert ctx["stated_figures"] == []


@pytest.mark.asyncio
async def test_block_absent_when_render_block_is_off_same_instance(monkeypatch):
    seed(*SPECS)
    monkeypatch.setattr(app_config, "CLAIM_LEDGER_ENABLED", True)
    builder = full_builder(monkeypatch, [{"query": "hi", "response": "hello"}], budget=10000)
    off = await builder.build_prompt(QUERY, retrieval_overrides=retrieval_limits())
    assert off["stated_figures"] == []
    monkeypatch.setattr(app_config, "CLAIM_LEDGER_RENDER_BLOCK", True)        # flip on the SAME builder
    on_ = await builder.build_prompt(QUERY, retrieval_overrides=retrieval_limits())
    assert len(on_["stated_figures"]) == 2


@pytest.mark.asyncio
async def test_block_absent_on_a_local_repo_audit(monkeypatch, on):
    seed(*SPECS)
    _b, ctx = await build(monkeypatch, "Do a read-only audit of the repo at /home/lukeh/project and run pytest")
    assert ctx["stated_figures"] == []


@pytest.mark.asyncio
async def test_block_never_on_the_light_path(monkeypatch, on):
    seed(*SPECS)
    builder = full_builder(monkeypatch, [{"query": "hi", "response": "hello"}], budget=10000)
    ctx = await builder._build_lightweight_context("ok cool")
    assert not ctx.get("stated_figures")


@pytest.mark.asyncio
async def test_an_unreadable_store_never_fails_the_prompt_build(monkeypatch, on):
    open(store.store_path(), "w").write("{ broken")
    _b, ctx = await build(monkeypatch)
    assert ctx["stated_figures"] == [] and "_build_time" in ctx


def test_line_and_token_caps(monkeypatch, on):
    specs = [(f"k{i}", str(i + 2), "asserted", 1 + i * 0.01, f"my figure number {i} is {i + 2} days right now",
              f"a figure number {i}") for i in range(20)]
    seed(*specs)
    monkeypatch.setattr(app_config, "CLAIM_LEDGER_MAX_BLOCK_TOKENS", 10_000)
    assert len(store.stated_figure_lines()) == 12                              # max_block_lines
    monkeypatch.setattr(app_config, "CLAIM_LEDGER_MAX_BLOCK_LINES", 5)
    assert len(store.stated_figure_lines()) == 5
    monkeypatch.setattr(app_config, "CLAIM_LEDGER_MAX_BLOCK_LINES", 12)
    monkeypatch.setattr(app_config, "CLAIM_LEDGER_MAX_BLOCK_TOKENS", 60)
    lines = store.stated_figure_lines()
    assert 1 <= len(lines) < 12 and sum((len(l) + 2) // 3 for l in lines) <= 60
    assert all(l.endswith(")") or "earlier" in l for l in lines)               # whole lines, never cut mid-line


# --- formatter + budget -----------------------------------------------------------------------------------

def test_formatter_renders_the_header_with_count_and_lines():
    lines = ["- days behind: 6 days — CONDITIONAL, you said \"x\" (Sat 10-17); not a fact",
             "- days spent: ~12 days — your ESTIMATE (\"12 days i think?\", Sun 10-18)"]
    fmt = PromptFormatter(token_manager=None, time_manager=None)
    out = fmt._assemble_prompt({"stated_figures": lines}, QUERY, "")
    assert "[YOUR STATED FIGURES] n=2\n" + "\n".join(lines) in out
    assert "[YOUR STATED FIGURES]" not in fmt._assemble_prompt({"stated_figures": []}, QUERY, "")


def test_a_recalled_text_quoting_the_header_is_neutralised_and_an_echo_is_stripped():
    from core.response_parser import ResponseParser
    assert "[YOUR STATED FIGURES" not in _sanitize_embedded_headers("see [YOUR STATED FIGURES] n=2 above")
    echoed = "[YOUR STATED FIGURES] n=1\n- days: 6 days\n\nReal answer."
    assert ResponseParser.strip_prompt_artifacts(echoed) == "Real answer."


def test_budget_meters_and_trims_the_section_by_priority():
    from core.prompt.token_manager import PRIORITY_ORDER
    assert dict(PRIORITY_ORDER)["stated_figures"] == 8
    from tests.unit.test_independent_prompt_audit import manager
    mgr = manager(40)
    big = ["- figure %d: %d days — you said \"%s\" (Mon 10-19)" % (i, i, "word " * 15) for i in range(12)]
    out = mgr._manage_token_budget({"stated_figures": list(big), "wiki": ["W " * 200]})
    assert len(out["stated_figures"]) < len(big) or out["stated_figures"] == []   # metered: the budget acts on it


# --- instruction ------------------------------------------------------------------------------------------

def test_instruction_appears_only_when_the_section_exists():
    with_block = section_instructions.conditional_instruction_tail({"stated_figures": ["- x"]})
    assert "[YOUR STATED FIGURES]" in with_block and "the block wins" in with_block
    assert "CONDITIONAL lines are not facts" in with_block and "keep their hedge" in with_block
    assert "IN THIS TURN" in with_block and "name the source" in with_block
    assert "STATED FIGURES" not in section_instructions.conditional_instruction_tail({"stated_figures": []})
    assert "STATED FIGURES" not in section_instructions.conditional_instruction_tail({"personal_notes": "n"})
