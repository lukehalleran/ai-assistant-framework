"""Main prompt under real budget pressure: what happens to each admitted record.

Plan: docs/PLAN_LONGMEMEVAL_ANSWER_PATH.md, plan v4, Step 3 Fix B "Main prompt"
(a free unit test of survival under real budget pressure). class: BC-58.

The sentinel test (tests/unit/test_answer_prompt_evidence_survival.py, added on the
fix/agentic-admitted-evidence-unclipped branch) checks the
main prompt only at the formatter, with token counting mocked to 10 per call, so
the budget never acts. This test drives the deployed chain from the budget to the
prompt string with the real tokenizer:

  TokenManager._manage_token_budget  (core/prompt/token_manager.py; "the last
      content mutation before assembly", core/prompt/builder.py:2441)
  -> PromptFormatter._assemble_prompt (what build_prompt_from_context hands the
      response generator, builder.py:2568)

with the budget from the deployed _compute_token_budget for an API model with a
128K context (10,000 tokens with the shipped config) and tiktoken o200k_base, as
for openai/gpt-4o. The record caps are the shipped ones: memories 512 tokens,
other list items 800, a conversation entry's query 600.

Measured on 2026-11-02 against the 500 recaptured LongMemEval base prompts (arm D):
453 carry at least one middle-out marker, 1,921 in all; 1,880 of them in [RELEVANT
MEMORIES] (median 835 chars removed). The flagged-message probes saw 9 partial
answer-bearing messages, so the trims mostly land on records without the answer.

It is a MEASUREMENT. Every record is classified as whole, middle-trimmed (head and
tail kept, the "[middle-out snipped N chars]" marker in the gap), or dropped, and a
table is printed. Trimming and dropping are designed behaviour. The test fails
only on:
  * a cut without the marker (a record losing its tail, or text missing in more
    than one place);
  * the input not being over budget, or the real tokenizer not being in use;
  * the memories floor (prompt_memories_reserve_floor) not holding;
  * a whole memory or recent turn not under its own date label and role.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import pytest

from core.prompt.builder import _compute_token_budget
from core.prompt.formatter import PromptFormatter
from core.prompt.token_manager import TokenManager
from models.tokenizer_manager import TokenizerManager, _TiktokenWrapper

MODEL = "gpt-4o"
MARKER = "[middle-out snipped"
QUESTION = "What dance did I say I wanted to learn?"
_WORDS = ("garden river morning project notes weather music dinner travel market "
          "letter window bicycle library coffee station mountain kitchen").split()


class _FakeModelManager:
    """Only what the budget and tokenizer read: an API model with a 128K context."""

    def get_active_model_name(self):
        return MODEL

    def is_api_model(self, name):
        return True

    def get_context_limit(self):
        return 128000


def _ts_label(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:%M")


def _body(rid: str, n_chars: int, every: int = 250) -> Tuple[str, List[str]]:
    """``n_chars`` of word filler with START/END tags and a sentinel every ``every``
    chars. Returns (text, ordered sentinels incl. START and END)."""
    words: List[str] = []
    while sum(len(w) + 1 for w in words) < n_chars:
        words.append(_WORDS[len(words) % len(_WORDS)])
    filler = " ".join(words)
    out, sentinels = f"START_{rid} ", [f"START_{rid}"]
    for k, start in enumerate(range(0, len(filler), every)):
        tag = f"SENT_{rid}_{k}"
        out += filler[start:start + every] + f" {tag} "
        sentinels.append(tag)
    out += f"END_{rid}"
    sentinels.append(f"END_{rid}")
    return out, sentinels


def _build_context(now: datetime):
    """~40K tokens of admitted context across seven sections.

    Returns (context, records) where records maps rid -> dict(section, sentinels,
    label, role, rank).
    """
    ctx: Dict[str, Any] = {}
    records: Dict[str, Dict[str, Any]] = {}

    def add(section, rid, sentinels, label, role, rank):
        records[rid] = {"section": section, "sentinels": sentinels, "label": label,
                        "role": role, "rank": rank}

    # recent turns, newest first: 3 in the current session, 9 from earlier days
    recent = []
    sizes = [1200, 3000, 6000, 1400, 2500, 6000, 1400, 2500, 6000, 1400, 2500, 6000]
    for i, n in enumerate(sizes):
        ts = now - (timedelta(minutes=5 + 10 * i) if i < 3 else timedelta(days=i, hours=3))
        rid = f"RC{i}"
        text, sents = _body(rid, n)
        recent.append({"query": f"short question {i}", "response": text,
                       "timestamp": ts.isoformat()})
        add("recent_conversations", rid, sents, _ts_label(ts), "assistant", i)
    ctx["recent_conversations"] = recent

    # memories, ranked: the first is LongMemEval-shaped (a 1,420-char assistant
    # message with the answer near char 1,281: dc439ea3)
    memories = []
    sizes = [1420, 1400, 2500, 6000, 1400, 2500, 6000, 1400, 2500, 6000,
             1400, 2500, 6000, 1400, 2500, 6000, 1400, 2500, 6000, 1400]
    for i, n in enumerate(sizes):
        ts = now - timedelta(days=40 + i, hours=1)
        rid = f"ME{i}"
        text, sents = _body(rid, n)
        if i % 2 == 0:  # hybrid-retriever shape
            memories.append({"content": f"User: question about topic {i}\nAssistant: {text}",
                             "timestamp": ts.isoformat(),
                             "metadata": {"timestamp": ts.isoformat()}})
        else:           # corpus-manager shape
            memories.append({"query": f"question about topic {i}", "response": text,
                             "timestamp": ts.isoformat(),
                             "metadata": {"timestamp": ts.isoformat()}})
        add("memories", rid, sents, _ts_label(ts), "assistant", i)
    ctx["memories"] = memories

    def plain(section, prefix, sizes, day0, titled=False):
        rows = []
        for i, n in enumerate(sizes):
            ts = now - timedelta(days=day0 + i, hours=2)
            rid = f"{prefix}{i}"
            text, sents = _body(rid, n)
            if titled:
                rows.append({"content": text, "metadata": {"title": f"Note {prefix}{i}"}})
                add(section, rid, sents, f"Note {prefix}{i}", None, i)
            else:
                rows.append({"content": text, "timestamp": ts.isoformat()})
                add(section, rid, sents, _ts_label(ts), None, i)
        ctx[section] = rows

    plain("recent_summaries", "RS", [2000] * 6, 100)
    plain("semantic_summaries", "SS", [2000] * 6, 120)
    plain("recent_reflections", "RR", [1500] * 4, 140)
    plain("dreams", "DR", [1500] * 3, 160)
    plain("personal_notes", "PN", [3500] * 4, 180, titled=True)
    return ctx, records


def _classify(prompt: str, sentinels: List[str]) -> Tuple[str, str]:
    """(kind, detail). kind: whole | middle_trimmed | dropped | CUT_NO_MARKER | SCATTERED."""
    present = [s in prompt for s in sentinels]
    if all(present):
        return "whole", ""
    if not any(present):
        return "dropped", ""
    pattern = "".join("1" if p else "0" for p in present)
    m = re.fullmatch(r"(1*)(0+)(1+)", pattern)
    if not m:
        kind = "CUT_NO_MARKER" if re.fullmatch(r"1+0+", pattern) else "SCATTERED"
        return kind, pattern
    head_n = len(m.group(1))
    start = prompt.find(sentinels[head_n - 1]) if head_n else -1
    end = prompt.find(sentinels[head_n + len(m.group(2))])
    gap = prompt[start + 1 if start >= 0 else max(0, end - 400):end]
    if MARKER not in gap:
        return "CUT_NO_MARKER", pattern
    return "middle_trimmed", pattern


def _under_label_and_role(prompt: str, rec: Dict[str, Any]) -> bool:
    first = prompt.find(rec["sentinels"][0])
    before = prompt[:first]
    lab = before.rfind(rec["label"])
    if lab < 0:
        return False
    if rec["role"] is None:
        return True
    seg = before[lab:]
    roles = ("Assistant:", "Daemon:", "Response:") if rec["role"] == "assistant" else ("User:",)
    others = ("User:",) if rec["role"] == "assistant" else ("Assistant:", "Daemon:", "Response:")
    last_mine = max(seg.rfind(r) for r in roles)
    last_other = max(seg.rfind(r) for r in others)
    return last_mine > last_other


def _run():
    mm = _FakeModelManager()
    tok_mgr = TokenizerManager(mm)
    budget = _compute_token_budget(mm)
    tm = TokenManager(model_manager=mm, tokenizer_manager=tok_mgr, token_budget=budget)
    now = datetime.now()
    ctx, records = _build_context(now)
    pre = sum(tm.get_token_count(tm._extract_text(it), MODEL)
              for v in ctx.values() for it in v)
    trimmed = tm._manage_token_budget(dict(ctx), now=now)
    prompt = PromptFormatter(token_manager=tm, time_manager=None)._assemble_prompt(
        context=trimmed, user_input=QUESTION)
    return tm, tok_mgr, budget, pre, trimmed, prompt, records


@pytest.fixture(scope="module")
def run():
    return _run()


def test_real_tokenizer_and_input_over_budget(run):
    tm, tok_mgr, budget, pre, _trimmed, _prompt, _records = run
    tok = tok_mgr.get_tokenizer(MODEL)
    # count_tokens falls back to len // 4 silently when the tokenizer fails; the
    # o200k encoder proves it did not
    assert isinstance(tok, _TiktokenWrapper) and tok._enc.name == "o200k_base"
    assert pre > 2 * budget, f"input only {pre} tokens against a {budget}-token budget"
    assert tm._prompt_token_usage <= budget


def test_no_record_is_cut_without_the_marker(run, capsys):
    tm, _tok, budget, pre, _trimmed, prompt, records = run
    kinds: Dict[str, Dict[str, int]] = {}
    bad: List[str] = []
    for rid, rec in records.items():
        kind, detail = _classify(prompt, rec["sentinels"])
        sec = kinds.setdefault(rec["section"], {"in": 0, "whole": 0, "middle_trimmed": 0,
                                                "dropped": 0})
        sec["in"] += 1
        if kind in sec:
            sec[kind] += 1
        else:
            bad.append(f"{rid} ({rec['section']}): {kind} {detail}")
    with capsys.disabled():
        print(f"\nMAIN_PROMPT budget={budget} input_tokens={pre} "
              f"metered_after={tm._prompt_token_usage} "
              f"prompt_tokens={tm.get_token_count(prompt, MODEL)} prompt_chars={len(prompt)}")
        print(f"{'section':22} {'in':>3} {'whole':>5} {'middle':>6} {'dropped':>7}")
        for sec, k in kinds.items():
            print(f"{sec:22} {k['in']:>3} {k['whole']:>5} {k['middle_trimmed']:>6} {k['dropped']:>7}")
    assert bad == []


def test_memories_floor_holds_and_answer_shaped_record_is_whole(run):
    from core.prompt.token_manager import FLOOR_PROTECTED_SECTIONS, _live_memory_cfg
    _key, _default = next((k, d) for n, k, d in FLOOR_PROTECTED_SECTIONS if n == "memories")
    floor = int(_live_memory_cfg(_key, _default))
    assert floor >= 1
    _tm, _tok, _budget, _pre, _trimmed, prompt, records = run
    for i in range(floor):
        kind, _ = _classify(prompt, records[f"ME{i}"]["sentinels"])
        assert kind != "dropped", f"memory rank {i} dropped despite floor {floor}"
    # the 1,420-char LongMemEval-shaped message (answer near char 1,281) is under the
    # 512-token memory cap, so it arrives whole, not middle-trimmed
    assert _classify(prompt, records["ME0"]["sentinels"])[0] == "whole"


def test_whole_memories_and_recent_turns_keep_date_and_role(run):
    _tm, _tok, _budget, _pre, _trimmed, prompt, records = run
    bad = [rid for rid, rec in records.items()
           if rec["section"] in ("memories", "recent_conversations")
           and _classify(prompt, rec["sentinels"])[0] == "whole"
           and not _under_label_and_role(prompt, rec)]
    assert bad == []
