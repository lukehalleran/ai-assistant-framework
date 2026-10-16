"""Prior-session [RECENT CONVERSATION] share cap in TokenManager._manage_token_budget.

Incident 2026-10-15 14:48 (live probe, temporal_recall): 26 prior-session
turns (~6.5K of a 10K budget) were admitted first (priority 7), emptying
memories / summaries / reflections / wiki. Bug class: new mechanism
(priority-ordered admission starving lower tiers; draft in bc_draft.md), BC-24.

All tests drive the deployed ``_manage_token_budget`` with an explicit ``now``
(BC-94) and a word-count tokenizer.
"""
from datetime import datetime, timedelta

import pytest

import core.prompt.token_manager as tmod
from core.prompt.token_manager import TokenManager, current_session_prefix_len

NOW = datetime(2026, 10, 15, 14, 47, 0)


class _MM:
    def get_active_model_name(self):
        return "test-model"


class _Tok:
    def count_tokens(self, text, model_name=None):
        return len((text or "").split())


def _conv(i, ts, words):
    return {"query": f"q{i}", "response": " ".join([f"r{i}"] * words), "timestamp": ts.isoformat()}


def _tm(budget=10000):
    return TokenManager(_MM(), _Tok(), budget)


def _tokens(tm, items):
    return sum(tm.get_token_count(tm._extract_text(i), "m") for i in items)


def _words(n, tag):
    return {"content": " ".join([tag] * n)}


def probe_ctx(prior=26, words=250):
    newest = datetime(2026, 10, 15, 11, 50)
    convs = [_conv(i, newest - timedelta(hours=4 * i), words) for i in range(prior)]
    return {
        "user_profile": " ".join(["prof"] * 2900),
        "recent_conversations": convs,
        "personal_notes": [_words(480, "note")],
        "memories": [_words(180, f"m{i}") for i in range(10)],
        "recent_summaries": [_words(150, f"s{i}") for i in range(8)],
        "recent_reflections": [_words(100, f"f{i}") for i in range(5)],
    }


def run(ctx, share, budget=10000, sect=1.0):
    # These tests isolate the recent_conversations share caps; the summary/
    # reflection floor reserve (2026-10-15, test_token_manager_floor_reserve.py)
    # is switched off here so "share 1.0 == old behaviour" stays literally true.
    from config.app_config import config as _cfg
    _mem = _cfg.setdefault("memory", {})
    _had = "prompt_floor_reserve_max_share" in _mem
    _prev = _mem.get("prompt_floor_reserve_max_share")
    _mem["prompt_floor_reserve_max_share"] = 0.0
    old = (tmod.RECENT_PRIOR_SESSION_MAX_SHARE, tmod.RECENT_MAX_SHARE)
    tmod.RECENT_PRIOR_SESSION_MAX_SHARE, tmod.RECENT_MAX_SHARE = share, sect
    try:
        tm = _tm(budget)
        out = tm._manage_token_budget(ctx, now=NOW)
    finally:
        tmod.RECENT_PRIOR_SESSION_MAX_SHARE, tmod.RECENT_MAX_SHARE = old
        if _had:
            _mem["prompt_floor_reserve_max_share"] = _prev
        else:
            _mem.pop("prompt_floor_reserve_max_share", None)
    return tm, out


def kept(out):
    return {k: len(out.get(k) or []) for k in
            ("recent_conversations", "memories", "recent_summaries", "recent_reflections")}


class TestProbeShape:
    def test_share_one_reproduces_base_starvation(self):
        _, out = run(probe_ctx(), 1.0)
        assert kept(out)["memories"] == 0

    def test_default_share_keeps_long_term_memory(self):
        tm, out = run(probe_ctx(), 0.40)
        k = kept(out)
        assert _tokens(tm, out["recent_conversations"]) <= 4000
        assert k["memories"] >= 3
        assert k["recent_summaries"] >= 1

    def test_share_one_equals_no_cap(self):
        # share=1.0 is exactly the pre-change first pass: every conversation
        # the total budget admits stays, nothing is removed by the cap.
        tm, out = run(probe_ctx(), 1.0)
        assert kept(out) == {"recent_conversations": 26, "memories": 0,
                             "recent_summaries": 0, "recent_reflections": 0}


class TestCurrentSession:
    def test_long_current_session_not_cut_by_share(self):
        base = datetime(2026, 10, 15, 14, 45)
        convs = [_conv(i, base - timedelta(minutes=5 * i), 100) for i in range(30)]
        ctx = {"recent_conversations": convs}
        _, capped = run(dict(ctx), 0.40)
        _, off = run(dict(ctx), 1.0)
        assert current_session_prefix_len(convs, NOW) == 30
        assert len(capped["recent_conversations"]) == len(off["recent_conversations"]) == 30

    def test_mixed_current_kept_prior_stops_at_cap(self):
        cur = [_conv(i, datetime(2026, 10, 15, 14, 40) - timedelta(minutes=5 * i), 100) for i in range(4)]
        prior = [_conv(10 + i, datetime(2026, 10, 14, 10, 0) - timedelta(hours=i), 300) for i in range(20)]
        tm, out = run({"recent_conversations": cur + prior}, 0.40)
        got = out["recent_conversations"]
        assert got[:4] == cur
        assert _tokens(tm, got[4:]) <= 4000
        assert 0 < len(got) - 4 < 20
        assert got[4:] == prior[: len(got) - 4]  # newest-first prefix, no gaps


class TestPrefixLen:
    def test_gap_boundary(self):
        t = datetime(2026, 10, 15, 14, 0)
        items = [_conv(0, t, 1), _conv(1, t - timedelta(minutes=30), 1),
                 _conv(2, t - timedelta(hours=2, minutes=30), 1)]
        assert current_session_prefix_len(items, NOW) == 2

    def test_calendar_day_boundary(self):
        now = datetime(2026, 10, 15, 0, 20)
        items = [_conv(0, datetime(2026, 10, 15, 0, 10), 1),
                 _conv(1, datetime(2026, 10, 14, 23, 55), 1)]
        assert current_session_prefix_len(items, now) == 1

    def test_newest_item_already_old_is_zero(self):
        items = [_conv(0, NOW - timedelta(hours=2), 1), _conv(1, NOW - timedelta(hours=2, minutes=5), 1)]
        assert current_session_prefix_len(items, NOW) == 0

    def test_empty(self):
        assert current_session_prefix_len([], NOW) == 0

    def test_unreadable_timestamps_are_never_capped(self):
        items = [{"query": "q", "response": "r r r"}, {"query": "q", "response": "r r r"}]
        assert current_session_prefix_len(items, NOW) == 2


def long_current_ctx():
    base = NOW - timedelta(minutes=2)
    c = {"recent_conversations": [_conv(i, base - timedelta(minutes=12 * i), 250) for i in range(30)]}
    c.update({k: v for k, v in probe_ctx().items() if k != "recent_conversations"})
    return c


class TestSectionCap:
    def test_long_current_session_capped_and_memory_survives(self):
        ctx = long_current_ctx()
        assert current_session_prefix_len(ctx["recent_conversations"], NOW) == 30
        tm, out = run(ctx, 0.40, sect=0.60)
        got = out["recent_conversations"]
        assert got == ctx["recent_conversations"][: len(got)]
        assert len(got) >= tmod.RECENT_ALWAYS_ADMIT
        assert _tokens(tm, got) <= 6000
        assert kept(out)["memories"] >= 3

    def test_both_shares_one_reproduces_base(self):
        _, out = run(long_current_ctx(), 1.0, sect=1.0)
        assert kept(out)["recent_conversations"] == 28 and kept(out)["memories"] == 0

    def test_newest_five_exempt_from_section_cap(self):
        ctx = long_current_ctx()
        tm, out = run(ctx, 0.40, sect=0.05)  # cap far below 5 items
        assert len(out["recent_conversations"]) == tmod.RECENT_ALWAYS_ADMIT

    def test_short_session_unaffected(self):
        base = NOW - timedelta(minutes=2)
        convs = [_conv(i, base - timedelta(minutes=5 * i), 250) for i in range(4)]
        _, a = run({"recent_conversations": convs}, 0.40, sect=0.60)
        _, b = run({"recent_conversations": convs}, 1.0, sect=1.0)
        assert a["recent_conversations"] == b["recent_conversations"] == convs
