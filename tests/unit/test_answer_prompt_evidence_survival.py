"""Sentinel transport test: admitted evidence must reach every answer-generation request.

Plan: docs/PLAN_LONGMEMEVAL_ANSWER_PATH.md, plan v4 (Step 1 item 4, "Fix B"). class: BC-58.

Invariant: every admitted record's text -- a sentinel placed beyond character
400, 500 and 2,000 of a memory's content/response or a recent turn's
query/response -- appears at least once in each answer-generation request, in a
block that also carries that record's timestamp label and its role label.

What is driven (nothing is re-implemented here):

* agentic decision, synthesis and error-fallback: the DEPLOYED
  ``AgenticSearchController.run_agentic_search`` with a fake model manager that
  records every request it is sent. The assertion is therefore at the
  MODEL-CALL boundary (``generate_once_with_tools`` for the decision round,
  ``generate_async`` for synthesis / error fallback). Only the tool-dispatch
  and (for the fallback) the decision seam are faked, as in
  ``helpers_agentic_loop``.
* decision-answer reuse reuses the DECISION prompt, so the decision test also
  covers it.
* main prompt: ``PromptFormatter._assemble_prompt`` -- the deployed function
  that turns the context into the prompt string. This is the builder level, not
  the model-call boundary: the string it returns is what
  ``build_prompt_from_context`` hands to the response generator
  (builder.py:2568 -> returned prompt), but the generator itself needs live
  services, so the boundary is asserted one hop earlier.

Role/date check: the nearest timestamp label BEFORE a sentinel must be the one
belonging to that sentinel's record, and the nearest role marker before it must
be the record's role (User vs Assistant/Daemon/Response).
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

from core.agentic.controller import AgenticSearchController
from core.agentic.types import SearchDecision, SearchProtocol, _ToolResult

OFFSETS = (450, 800, 2500)
USER_ROLES = ("User:",)
ASSISTANT_ROLES = ("Assistant:", "Daemon:", "Response:")
ALL_ROLES = USER_ROLES + ASSISTANT_ROLES


def _pad(tag: str, offset: int) -> str:
    """Filler (no sentinel text) of exactly ``offset`` chars, then the sentinel."""
    filler = (f"filler-{tag} " * 400)[:offset]
    return filler + f" SENT_{tag}_{offset} " + "tail words. "


def _ts_label(ts: datetime) -> str:
    # deployed renderers print "YYYY-MM-DD HH:MM (relative)"; match the stable prefix
    return ts.strftime("%Y-%m-%d %H:%M")


def _build_context() -> Tuple[Dict[str, Any], List[Tuple[str, str, str]]]:
    """Return (initial_context, expectations[(sentinel, ts_label, role_kind)]).

    Shapes follow the deployed readers: memories carry ``content`` (hybrid
    retriever's "User: q\\nAssistant: r" document) or the corpus manager's
    ``query``/``response``;
    recent_conversations carry ``query``/``response`` (newest first).
    """
    now = datetime.now()
    expect: List[Tuple[str, str, str]] = []

    # memories: three records, one per role/field
    mem_ts = [now - timedelta(days=40 + i) for i in range(3)]
    memories = []
    # M1: combined content, user sentinel past 450 in the user half
    m1_user = _pad("M1U", 450)
    m1_asst = _pad("M1A", 800)
    memories.append({
        "content": f"User: {m1_user}\nAssistant: {m1_asst}",
        "timestamp": mem_ts[0].isoformat(),
        "metadata": {"timestamp": mem_ts[0].isoformat()},
    })
    expect += [("SENT_M1U_450", _ts_label(mem_ts[0]), "user"),
               ("SENT_M1A_800", _ts_label(mem_ts[0]), "assistant")]
    # M2: corpus-manager shape (query/response, no content), long separate response
    m2_resp = _pad("M2R", 2500)
    memories.append({
        "query": "what did I say about the garden",
        "response": m2_resp,
        "timestamp": mem_ts[1].isoformat(),
        "metadata": {"timestamp": mem_ts[1].isoformat()},
    })
    expect.append(("SENT_M2R_2500", _ts_label(mem_ts[1]), "assistant"))
    # M3: content only, sentinel far out
    m3 = _pad("M3U", 2500)
    memories.append({
        "content": f"User: {m3}",
        "timestamp": mem_ts[2].isoformat(),
        "metadata": {"timestamp": mem_ts[2].isoformat()},
    })
    expect.append(("SENT_M3U_2500", _ts_label(mem_ts[2]), "user"))

    # recent turns, newest first; r[0] is in the current session (last-exchange copy applies)
    rec_ts = [now - timedelta(minutes=10), now - timedelta(minutes=20), now - timedelta(minutes=30)]
    recent = []
    r0_q = _pad("R0U", 800)
    r0_a = _pad("R0A", 800)
    recent.append({"query": r0_q, "response": r0_a, "timestamp": rec_ts[0].isoformat()})
    expect += [("SENT_R0U_800", _ts_label(rec_ts[0]), "user"),
               ("SENT_R0A_800", _ts_label(rec_ts[0]), "assistant")]
    r1_a = _pad("R1A", 450)
    recent.append({"query": "short question one", "response": r1_a, "timestamp": rec_ts[1].isoformat()})
    expect.append(("SENT_R1A_450", _ts_label(rec_ts[1]), "assistant"))
    r2_a = _pad("R2A", 2500)
    recent.append({"query": "short question two", "response": r2_a, "timestamp": rec_ts[2].isoformat()})
    expect.append(("SENT_R2A_2500", _ts_label(rec_ts[2]), "assistant"))

    return {"memories": memories, "recent_conversations": recent}, expect


def _find_violations(text: str, expect: List[Tuple[str, str, str]]) -> List[str]:
    """Every sentinel must be present, after its own record's label and role marker."""
    labels = sorted({lab for _, lab, _ in expect})
    bad: List[str] = []
    for sentinel, label, role in expect:
        positions = [m.start() for m in re.finditer(re.escape(sentinel), text)]
        if not positions:
            bad.append(f"{sentinel}: ABSENT")
            continue
        ok = False
        for pos in positions:
            before = text[:pos]
            last_label = max(((before.rfind(l), l) for l in labels), default=(-1, ""))
            if last_label[1] != label or last_label[0] < 0:
                continue
            seg = before[last_label[0]:]
            last_role = max(((seg.rfind(r), r) for r in ALL_ROLES), default=(-1, ""))
            want = USER_ROLES if role == "user" else ASSISTANT_ROLES
            if last_role[0] >= 0 and last_role[1] in want:
                ok = True
                break
        if not ok:
            bad.append(f"{sentinel}: present but not under its own date+role label")
    return bad


# --------------------------------------------------------------------------
# agentic harness
# --------------------------------------------------------------------------

class _RecordingManager:
    """Fake model manager: records every request; no tool calls, canned answer."""

    api_models: Dict[str, Any] = {}
    active_model_name = "test-model"

    def __init__(self) -> None:
        self.decision_prompts: List[str] = []
        self.final_prompts: List[str] = []

    async def generate_once_with_tools(self, prompt, **kw):
        self.decision_prompts.append(prompt)
        # no tool_calls, empty text -> wants_answer with nothing to reuse
        return {"content": ""}

    async def generate_once(self, prompt, **kw):
        self.decision_prompts.append(prompt)
        return ""

    async def generate_async(self, prompt, **kw):
        self.final_prompts.append(prompt)
        return "CANNED FINAL ANSWER."

    def _log_cache_usage(self, *a, **k):
        pass


def _controller(mgr) -> AgenticSearchController:
    ctrl = AgenticSearchController(
        model_manager=mgr, web_search_manager=MagicMock(), max_rounds=3,
    )
    ctrl.detect_protocol = lambda m: SearchProtocol.NATIVE_TOOLS
    return ctrl


async def _drive(ctrl, ctx) -> List[Any]:
    out = []
    async for ev in ctrl.run_agentic_search(
        query="What did we decide about the garden?", system_prompt="sys",
        model_name="test-model", initial_search_terms=[], skip_initial_search=True,
        initial_context=ctx,
    ):
        out.append(ev)
    return out


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def test_main_prompt_keeps_admitted_evidence():
    from core.prompt.formatter import PromptFormatter

    tm = MagicMock()
    tm.count_tokens = MagicMock(return_value=10)
    fmt = PromptFormatter(token_manager=tm, time_manager=None)
    ctx, expect = _build_context()
    prompt = fmt._assemble_prompt(context=ctx, user_input="What did we decide about the garden?")
    assert _find_violations(prompt, expect) == []


@pytest.mark.asyncio
async def test_agentic_decision_request_keeps_admitted_evidence():
    mgr = _RecordingManager()
    ctx, expect = _build_context()
    await _drive(_controller(mgr), ctx)
    assert mgr.decision_prompts, "no decision request was recorded"
    assert _find_violations(mgr.decision_prompts[0], expect) == []


@pytest.mark.asyncio
async def test_agentic_synthesis_request_keeps_admitted_evidence():
    mgr = _RecordingManager()
    ctx, expect = _build_context()
    await _drive(_controller(mgr), ctx)
    assert mgr.final_prompts, "no synthesis request was recorded"
    assert _find_violations(mgr.final_prompts[-1], expect) == []


@pytest.mark.asyncio
async def test_agentic_error_fallback_request_keeps_admitted_evidence():
    mgr = _RecordingManager()
    ctrl = _controller(mgr)
    ctx, expect = _build_context()
    calls = {"n": 0}

    async def fake_decision(prompt, system_prompt, model_name, handler, session,
                            tool_choice="auto", tools_override=None, forced_action_type=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return [SearchDecision(wants_search=True, search_query="garden")]
        raise RuntimeError("boom: decision round failed")

    async def fake_dispatch(decision, round_number, session, crisis_level, sandbox_session):
        return _ToolResult(decision=decision, round_data=None,
                           formatted_context="TOOL OUTPUT: something found",
                           start_events=[], end_events=[])

    ctrl._get_model_decision = fake_decision
    ctrl._dispatch_single = fake_dispatch
    await _drive(ctrl, ctx)
    assert calls["n"] >= 2
    assert mgr.final_prompts, "error fallback made no answer request"
    assert _find_violations(mgr.final_prompts[-1], expect) == []


# --------------------------------------------------------------------------
# Fix B (plan v4): the remaining admitted-record sections, tool results, boundary
# --------------------------------------------------------------------------

_SECTION_OFFSETS = (700, 2500)  # past every historical clip (400/500/600)


def _section_ctx(key: str):
    """One section of ``key`` records, sentinels past every old clip limit.

    Returns (initial_context, expectations[(sentinel, label)]). ``label`` is the
    record's timestamp label, or its title for personal notes (no date label).
    """
    now = datetime.now()
    records, expect = [], []
    for i, off in enumerate(_SECTION_OFFSETS):
        ts = now - timedelta(days=30 + 3 * i)
        body = _pad(f"{key[:4].upper()}{i}", off)
        sentinel = f"SENT_{key[:4].upper()}{i}_{off}"
        if key == "personal_notes":
            title = f"NoteTitle{i}"
            records.append({"content": body, "metadata": {"title": title}})
            expect.append((sentinel, title))
        else:
            records.append({"content": body, "timestamp": ts.isoformat()})
            expect.append((sentinel, _ts_label(ts)))
    return {key: records}, expect


def _find_label_violations(text: str, expect: List[Tuple[str, str]]) -> List[str]:
    labels = sorted({lab for _, lab in expect})
    bad: List[str] = []
    for sentinel, label in expect:
        positions = [m.start() for m in re.finditer(re.escape(sentinel), text)]
        if not positions:
            bad.append(f"{sentinel}: ABSENT")
            continue
        ok = False
        for pos in positions:
            before = text[:pos]
            last = max(((before.rfind(l), l) for l in labels), default=(-1, ""))
            if last[0] >= 0 and last[1] == label:
                ok = True
                break
        if not ok:
            bad.append(f"{sentinel}: present but not under its own label")
    return bad


async def _decision_and_final(ctx):
    mgr = _RecordingManager()
    await _drive(_controller(mgr), ctx)
    assert mgr.decision_prompts, "no decision request was recorded"
    assert mgr.final_prompts, "no synthesis request was recorded"
    return mgr


@pytest.mark.asyncio
async def test_agentic_requests_keep_admitted_summaries():
    ctx, expect = _section_ctx("recent_summaries")
    mgr = await _decision_and_final(ctx)
    assert _find_label_violations(mgr.decision_prompts[0], expect) == []
    assert _find_label_violations(mgr.final_prompts[-1], expect) == []


@pytest.mark.asyncio
async def test_agentic_requests_keep_admitted_personal_notes():
    ctx, expect = _section_ctx("personal_notes")
    mgr = await _decision_and_final(ctx)
    assert _find_label_violations(mgr.decision_prompts[0], expect) == []
    assert _find_label_violations(mgr.final_prompts[-1], expect) == []


@pytest.mark.asyncio
async def test_agentic_requests_keep_admitted_dreams():
    ctx, expect = _section_ctx("dreams")
    mgr = await _decision_and_final(ctx)
    assert _find_label_violations(mgr.decision_prompts[0], expect) == []
    assert _find_label_violations(mgr.final_prompts[-1], expect) == []


@pytest.mark.asyncio
async def test_agentic_requests_keep_admitted_reflections():
    ctx, expect = _section_ctx("recent_reflections")
    mgr = await _decision_and_final(ctx)
    assert _find_label_violations(mgr.decision_prompts[0], expect) == []
    assert _find_label_violations(mgr.final_prompts[-1], expect) == []


async def _synthesis_after_tool(formatted: str) -> str:
    """Drive the deployed controller; one tool round returns ``formatted``."""
    mgr = _RecordingManager()
    ctrl = _controller(mgr)
    calls = {"n": 0}

    async def fake_decision(prompt, system_prompt, model_name, handler, session,
                            tool_choice="auto", tools_override=None, forced_action_type=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return [SearchDecision(wants_search=True, search_query="garden")]
        return [SearchDecision(is_done=True)]

    async def fake_dispatch(decision, round_number, session, crisis_level, sandbox_session):
        return _ToolResult(decision=decision, round_data=None, formatted_context=formatted,
                           start_events=[], end_events=[])

    ctrl._get_model_decision = fake_decision
    ctrl._dispatch_single = fake_dispatch
    await _drive(ctrl, {})
    assert mgr.final_prompts, "no synthesis request was recorded"
    return mgr.final_prompts[-1]


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="not in Fix B scope: reported, not fixed (plan v4) -- "
           "core/agentic/formatters.py:296 clips search_memory results (500 chars; 2000 for conversations)")
async def test_search_memory_tool_result_reaches_synthesis_whole():
    from core.agentic.formatters import AgenticFormatter
    fmt = AgenticFormatter()
    rows = [{"content": _pad("TOOLC", 2500), "id": "abcdef123456", "relevance_score": 0.9,
             "metadata": {"timestamp": datetime.now().isoformat()}}]
    text = fmt.format_memory_results(rows, "conversations")
    rows2 = [{"content": _pad("TOOLF", 700), "id": "fedcba654321", "relevance_score": 0.8,
              "metadata": {"subject": "a", "relation": "b"}}]
    text += "\n\n" + fmt.format_memory_results(rows2, "facts")
    final = await _synthesis_after_tool(text)
    assert "SENT_TOOLC_2500" in final and "SENT_TOOLF_700" in final


@pytest.mark.asyncio
async def test_expanded_memory_tool_result_reaches_synthesis_whole():
    """format_expanded_results does not clip: passes on base (not in Fix B scope)."""
    from core.agentic.formatters import AgenticFormatter
    result = {"collection": "conversations", "total_in_collection": 5, "turns": [
        {"id": "abcdef123456", "timestamp": datetime.now().isoformat(),
         "content": _pad("EXPN", 2500), "is_anchor": True}]}
    final = await _synthesis_after_tool(AgenticFormatter().format_expanded_results(result))
    assert "SENT_EXPN_2500" in final


@pytest.mark.asyncio
async def test_large_evidence_below_final_ceiling_keeps_admitted_records(capsys):
    """20 memories x ~2,000 chars and 10 recent turns x ~2,000 chars plus a ~40,000-char
    tool result: every admitted sentinel survives with its date and role. This request
    (~93K chars, ~23K estimated tokens) stays BELOW the final-request ceiling; the
    over-ceiling cases are the ``test_over_ceiling_*`` tests below."""
    now = datetime.now()
    expect: List[Tuple[str, str, str]] = []
    memories = []
    for i in range(20):
        ts = now - timedelta(days=30 + i)
        body = _pad(f"BM{i}", 1850)
        memories.append({"content": f"User: short question {i}\nAssistant: {body}",
                         "timestamp": ts.isoformat(), "metadata": {"timestamp": ts.isoformat()}})
        expect.append((f"SENT_BM{i}_1850", _ts_label(ts), "assistant"))
    recent = []
    for i in range(10):
        ts = now - timedelta(hours=2 + 3 * i)
        body = _pad(f"BR{i}", 1850)
        recent.append({"query": f"short recent question {i}", "response": body,
                       "timestamp": ts.isoformat()})
        expect.append((f"SENT_BR{i}_1850", _ts_label(ts), "assistant"))
    mgr = _RecordingManager()
    ctrl = _controller(mgr)
    calls = {"n": 0}

    async def fake_decision(prompt, system_prompt, model_name, handler, session,
                            tool_choice="auto", tools_override=None, forced_action_type=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return [SearchDecision(wants_search=True, search_query="garden")]
        return [SearchDecision(is_done=True)]

    async def fake_dispatch(decision, round_number, session, crisis_level, sandbox_session):
        return _ToolResult(decision=decision, round_data=None,
                           formatted_context="TOOL OUTPUT " + ("x" * 40000),
                           start_events=[], end_events=[])

    ctrl._get_model_decision = fake_decision
    ctrl._dispatch_single = fake_dispatch
    await _drive(ctrl, {"memories": memories, "recent_conversations": recent})
    req = mgr.final_prompts[-1]
    print(f"BOUNDARY_REQUEST_CHARS={len(req)}")
    assert _find_violations(req, expect) == []


@pytest.mark.asyncio
async def test_measure_request_sizes_on_default_fixture(capsys):
    """Prints decision and synthesis request sizes for the default sentinel fixture."""
    mgr = _RecordingManager()
    ctx, _ = _build_context()
    await _drive(_controller(mgr), ctx)
    with capsys.disabled():
        print(f"\nMEASURE decision_chars={len(mgr.decision_prompts[0])} "
              f"synthesis_chars={len(mgr.final_prompts[-1])}")


# --------------------------------------------------------------------------
# Over the final-request ceiling (review docs/REVIEW_LONGMEMEVAL_FIX_COMMITS.md, P2)
#
# Permitted exclusion (plan v4 Fix B): when a request exceeds
# context_budget_tokens * 5, the controller's ladder (_TRIMMABLE_PREFIXES) drops
# WHOLE low-priority sections in its fixed order. It never drops recent turns or
# memories and never cuts inside a record. Each drop is visible in the receipts:
# visible_sources["sections"] (post-trim headers) and, for admitted-evidence keys,
# omitted_sections.
# --------------------------------------------------------------------------

_C = AgenticSearchController
_LADDER_KEYS = [key for prefix in _C._TRIMMABLE_PREFIXES
                for header, key in _C._EVIDENCE_HEADER_KEYS if header == prefix]
_HEADER_OF = {key: header for header, key in _C._EVIDENCE_HEADER_KEYS}
# reference_docs is left out of the fixture: it has its own origin gate.
_OVER_SECTIONS = [k for k in _LADDER_KEYS if k != "reference_docs"]


def _over_ceiling_ctx():
    """~200K chars of admitted evidence (~50K estimated tokens, ceiling 40K).

    Returns (ctx, core_expect[(sentinel, label, role)], section_expect{key: [(sentinel, label)]}).
    """
    now = datetime.now()
    core: List[Tuple[str, str, str]] = []
    memories, recent = [], []
    for i in range(20):
        ts = now - timedelta(days=30 + i)
        memories.append({"content": f"User: short question {i}\nAssistant: {_pad(f'OM{i}', 1850)}",
                         "timestamp": ts.isoformat(), "metadata": {"timestamp": ts.isoformat()}})
        core.append((f"SENT_OM{i}_1850", _ts_label(ts), "assistant"))
    for i in range(10):
        ts = now - timedelta(hours=2 + 3 * i)
        recent.append({"query": f"short recent question {i}", "response": _pad(f"OR{i}", 1850),
                       "timestamp": ts.isoformat()})
        core.append((f"SENT_OR{i}_1850", _ts_label(ts), "assistant"))
    ctx: Dict[str, Any] = {"memories": memories, "recent_conversations": recent}
    sections: Dict[str, List[Tuple[str, str]]] = {}
    for s, key in enumerate(_OVER_SECTIONS):
        tag = "OS" + "".join(w[0] for w in key.split("_")).upper()
        records, expect = [], []
        for i in range(10):
            ts = now - timedelta(days=100 + 20 * s + i)
            body = _pad(f"{tag}{i}", 2800)
            if key == "personal_notes":
                records.append({"content": body, "metadata": {"title": f"OverNote{i}"}})
                expect.append((f"SENT_{tag}{i}_2800", f"OverNote{i}"))
            else:
                records.append({"content": body, "timestamp": ts.isoformat()})
                expect.append((f"SENT_{tag}{i}_2800", _ts_label(ts)))
        ctx[key] = records
        sections[key] = expect
    return ctx, core, sections


def _whole_or_absent(req: str, sections) -> Tuple[List[str], List[str], List[str]]:
    """Each section is either kept with every record whole, or absent entirely."""
    kept, dropped, bad = [], [], []
    for key in _OVER_SECTIONS:
        expect = sections[key]
        present = [s for s, _ in expect if s in req]
        if len(present) == len(expect):
            kept.append(key)
            bad += _find_label_violations(req, expect)
        elif not present:
            dropped.append(key)
        else:
            bad.append(f"{key}: PARTIAL {len(present)}/{len(expect)}")
    return kept, dropped, bad


def _assert_ladder_drop(ctrl, req, core, sections):
    ceiling = ctrl.context_budget_tokens * 5
    assert ctrl._estimate_tokens(req) <= ceiling
    assert _find_violations(req, core) == []          # recent turns + memories whole
    kept, dropped, bad = _whole_or_absent(req, sections)
    assert bad == []
    assert dropped, "the ladder never fired: the request did not cross the ceiling"
    assert dropped == _OVER_SECTIONS[:len(dropped)]   # fixed ladder order, whole sections
    return kept, dropped


@pytest.mark.asyncio
async def test_over_ceiling_decision_and_synthesis_drop_whole_sections_only():
    ctx, core, sections = _over_ceiling_ctx()
    mgr = _RecordingManager()
    ctrl = _controller(mgr)
    parts, _ = ctrl._render_evidence_sections(ctx)
    pre_trim = ctrl._estimate_tokens("\n\n".join(parts))
    assert pre_trim > ctrl.context_budget_tokens * 5   # demonstrably over the ceiling
    await _drive(ctrl, ctx)
    assert mgr.decision_prompts and mgr.final_prompts

    kept_d, dropped_d = _assert_ladder_drop(ctrl, mgr.decision_prompts[0], core, sections)
    session = ctrl._last_session
    assert not set(dropped_d) & set(session.decision_rendered_keys)
    assert set(k for k in kept_d if k in _C._ADMITTED_EVIDENCE_KEYS) <= set(session.decision_rendered_keys)

    kept_f, dropped_f = _assert_ladder_drop(ctrl, mgr.final_prompts[-1], core, sections)
    shown = set(session.visible_sources["sections"])
    assert not {_HEADER_OF[k] for k in dropped_f} & shown
    assert {_HEADER_OF[k] for k in kept_f} <= shown
    admitted_dropped = {k for k in dropped_f if k in _C._ADMITTED_EVIDENCE_KEYS}
    assert admitted_dropped <= set(session.omitted_sections)
    assert not set(kept_f) & set(session.omitted_sections)
    print(f"\nOVER_CEILING pre_trim_tokens={pre_trim} "
          f"decision_chars={len(mgr.decision_prompts[0])} dropped={dropped_d} "
          f"final_chars={len(mgr.final_prompts[-1])} dropped={dropped_f}")


@pytest.mark.asyncio
async def test_over_ceiling_with_tool_result_drops_whole_sections_only():
    ctx, core, sections = _over_ceiling_ctx()
    mgr = _RecordingManager()
    ctrl = _controller(mgr)
    calls = {"n": 0}

    async def fake_decision(prompt, system_prompt, model_name, handler, session,
                            tool_choice="auto", tools_override=None, forced_action_type=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return [SearchDecision(wants_search=True, search_query="garden")]
        return [SearchDecision(is_done=True)]

    async def fake_dispatch(decision, round_number, session, crisis_level, sandbox_session):
        return _ToolResult(decision=decision, round_data=None,
                           formatted_context="TOOL OUTPUT " + ("x" * 40000),
                           start_events=[], end_events=[])

    ctrl._get_model_decision = fake_decision
    ctrl._dispatch_single = fake_dispatch
    await _drive(ctrl, ctx)
    req = mgr.final_prompts[-1]
    assert "TOOL OUTPUT" in req
    kept, dropped = _assert_ladder_drop(ctrl, req, core, sections)
    session = ctrl._last_session
    shown = set(session.visible_sources["sections"])
    assert not {_HEADER_OF[k] for k in dropped} & shown
    admitted_dropped = {k for k in dropped if k in _C._ADMITTED_EVIDENCE_KEYS}
    assert admitted_dropped <= set(session.omitted_sections)
    assert not set(kept) & set(session.omitted_sections)
    print(f"\nOVER_CEILING_TOOL final_chars={len(req)} "
          f"final_tokens={ctrl._estimate_tokens(req)} dropped={dropped}")
