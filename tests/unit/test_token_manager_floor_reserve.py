"""Summary/reflection survival floors are RESERVED inside the token budget.

Incident 2026-10-15 22:05 (live retest after PR #77): budget 10000; profile, capped
recent conversations, notes and ~11 memories filled it, and recent_summaries,
semantic_summaries, recent_reflections, semantic_reflections, git and wiki were
all emptied -- although the builder's survival floors (2 summaries / 1 reflection)
had topped the sections up just before the budget ran. Bug class: BC-99
(priority admission starves lower tiers), BC-24 (floors re-admitted/defeated
around the budget).

All tests drive the deployed ``_manage_token_budget`` (explicit ``now``, BC-94)
with a word-count tokenizer.
"""
import logging
from datetime import datetime, timedelta

import pytest

from config.app_config import config as app_cfg
from core.prompt.token_manager import TokenManager

NOW = datetime(2026, 10, 15, 22, 5, 0)


class _MM:
    def get_active_model_name(self):
        return "test-model"


class _Tok:
    def count_tokens(self, text, model_name=None):
        return len((text or "").split())


def _tm(budget=10000):
    return TokenManager(_MM(), _Tok(), budget)


def _words(n, tag):
    return {"content": " ".join([tag] * n)}


def _conv(i, ts, words):
    return {"query": f"q{i}", "response": " ".join([f"r{i}"] * words), "timestamp": ts.isoformat()}


def live_2205_ctx():
    newest = NOW - timedelta(minutes=5)
    convs = [_conv(i, newest - timedelta(hours=5 * i), 230) for i in range(16)]
    return {
        "user_profile": " ".join(["prof"] * 2900),
        "recent_conversations": convs,
        "personal_notes": [_words(150, f"note{i}") for i in range(5)],
        "memories": [_words(250, f"m{i}") for i in range(19)],
        "recent_summaries": [_words(300, f"s{i}") for i in range(5)],
        "semantic_summaries": [_words(300, f"ss{i}") for i in range(5)],
        "recent_reflections": [_words(100, f"f{i}") for i in range(5)],
        "semantic_reflections": [_words(100, f"sf{i}") for i in range(3)],
        "git_commits": [_words(120, f"g{i}") for i in range(4)],
    }


def _metered(tm, out):
    n = 0
    for k, v in out.items():
        if k.startswith("_") or k in ("stm_summary",) or not isinstance(v, list):
            continue
        n += sum(tm.get_token_count(tm._extract_text(i), "m") for i in v)
    return n


@pytest.fixture
def mem_cfg(monkeypatch):
    mem = app_cfg.setdefault("memory", {})
    def _set(**kw):
        for k, v in kw.items():
            monkeypatch.setitem(mem, k, v)
    return _set


class TestLive2205Shape:
    def test_floors_survive_and_memories_yield(self, mem_cfg):
        mem_cfg(prompt_min_summaries_floor=2, prompt_min_reflections_floor=1,
                prompt_floor_reserve_max_share=0.15)
        tm = _tm()
        out = tm._manage_token_budget(live_2205_ctx(), now=NOW)
        assert len(out["recent_summaries"]) >= 2
        assert len(out["recent_reflections"]) >= 1
        assert _metered(tm, out) <= 10000
        # memories were reduced instead
        mem_cfg(prompt_floor_reserve_max_share=0.0)
        base = _tm()._manage_token_budget(live_2205_ctx(), now=NOW)
        assert len(base["recent_summaries"]) == 0   # the incident, reproduced by turning the reserve off
        assert len(out["memories"]) < len(base["memories"])

    def test_floor_zero_is_identical_to_no_reserve(self, mem_cfg):
        mem_cfg(prompt_min_summaries_floor=0, prompt_min_reflections_floor=0,
                prompt_floor_reserve_max_share=0.15)
        a = _tm()._manage_token_budget(live_2205_ctx(), now=NOW)
        mem_cfg(prompt_min_summaries_floor=2, prompt_min_reflections_floor=1,
                prompt_floor_reserve_max_share=0.0)
        b = _tm()._manage_token_budget(live_2205_ctx(), now=NOW)
        assert a == b

    def test_live_read_same_instance(self, mem_cfg):
        tm = _tm()
        mem_cfg(prompt_floor_reserve_max_share=0.0)
        off = tm._manage_token_budget(live_2205_ctx(), now=NOW)
        mem_cfg(prompt_floor_reserve_max_share=0.15)
        on = tm._manage_token_budget(live_2205_ctx(), now=NOW)
        assert len(off["recent_summaries"]) == 0 and len(on["recent_summaries"]) >= 2


class TestBound:
    def test_only_first_floor_reserved_and_logged(self, mem_cfg, caplog):
        mem_cfg(prompt_min_summaries_floor=2, prompt_min_reflections_floor=1,
                prompt_floor_reserve_max_share=0.15)
        budget = 5000   # bound 750: summaries (600) fit, reflections (+300) do not
        ctx = {
            "user_profile": " ".join(["prof"] * 2900),
            "memories": [_words(250, f"m{i}") for i in range(10)],
            "recent_summaries": [_words(300, f"s{i}") for i in range(3)],
            "recent_reflections": [_words(300, f"f{i}") for i in range(3)],
        }
        tm = _tm(budget)
        with caplog.at_level(logging.INFO):
            out = tm._manage_token_budget(ctx, now=NOW)
        assert len(out["recent_summaries"]) >= 2
        assert any("Floor reserve for 'recent_reflections' skipped" in r.getMessage()
                   for r in caplog.records)
        assert not any("Floor reserve for 'recent_summaries' skipped" in r.getMessage()
                       for r in caplog.records)
        assert _metered(tm, out) <= budget


class TestFewerItemsThanFloor:
    def test_reserves_only_what_exists(self, mem_cfg):
        mem_cfg(prompt_min_summaries_floor=2, prompt_min_reflections_floor=1,
                prompt_floor_reserve_max_share=0.15)
        ctx = live_2205_ctx()
        ctx["recent_summaries"] = [_words(300, "s0")]
        ctx["recent_reflections"] = []
        tm = _tm()
        out = tm._manage_token_budget(ctx, now=NOW)
        assert len(out["recent_summaries"]) == 1
        assert _metered(tm, out) <= 10000


class TestSecondPassAndStringPaths:
    """Referee gaps: the second (trim) pass must not take floor items, and the
    pass-1 string path must honour other sections' reserves."""

    def _ctx(self):
        ctx = live_2205_ctx()
        # Big plain-string sections that do not fit in pass 1 stay in `trimmed`
        # and force the iterative second pass.
        ctx["wiki"] = " ".join(["wiki"] * 4000)
        ctx["codebase_changes"] = " ".join(["diff"] * 4000)
        return ctx

    def test_second_pass_keeps_floors(self, mem_cfg):
        mem_cfg(prompt_min_summaries_floor=2, prompt_min_reflections_floor=1,
                prompt_floor_reserve_max_share=0.15)
        tm = _tm()
        out = tm._manage_token_budget(self._ctx(), now=NOW)
        assert len(out["recent_summaries"]) >= 2
        assert len(out["recent_reflections"]) >= 1

    def test_second_pass_below_floor_untouched_when_floor_zero(self, mem_cfg):
        # floor 0 => the old trim semantics (a 1-item list can be emptied)
        mem_cfg(prompt_min_summaries_floor=0, prompt_min_reflections_floor=0,
                prompt_floor_reserve_max_share=0.15)
        out = _tm()._manage_token_budget(self._ctx(), now=NOW)
        assert len(out["recent_reflections"]) == 0
