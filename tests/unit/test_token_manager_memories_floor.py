"""[RELEVANT MEMORIES] keeps a reserved survival floor inside the token budget.

Incidents 2026-10-18..20: the budget emptied ``memories`` (priority 5) on 37 of 54
prompt builds -- distress turns and a recall question whose answer was in memory
included -- because ~8.7K of the 10K budget was spent before it was reached and the
BC-99 floor reserve covered only summaries/reflections. Log shape:
``Stopped adding memories at item 0/14: budget=10000, current=9199, item_tokens=435``.
Bug class: BC-99 (priority admission starves lower tiers; closure extended to memories).

Tests drive the deployed ``TokenManager._manage_token_budget`` (explicit ``now``).
"""
from datetime import datetime, timedelta

import pytest

import config.app_config as app_config
from core.prompt.token_manager import TokenManager

NOW = datetime(2026, 10, 19, 21, 0, 0)


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


def incident_ctx():
    newest = NOW - timedelta(minutes=5)
    return {
        "user_profile": " ".join(["prof"] * 2900),
        "narrative_state": " ".join(["narr"] * 500),
        # section share cap (50%) is what bounds this in prod: ~5K of history
        "recent_conversations": [_conv(i, newest - timedelta(minutes=3 * i), 300) for i in range(14)],
        "personal_notes": [_words(200, f"note{i}") for i in range(6)],
        "user_uploads": [_words(120, "up")],
        "relevant_emails": [_words(100, f"mail{i}") for i in range(3)],
        "memories": [_words(435, f"mem{i}") for i in range(14)],
        "recent_summaries": [_words(300, f"s{i}") for i in range(3)],
        "recent_reflections": [_words(100, f"f{i}") for i in range(3)],
    }


def _metered(tm, out):
    n = 0
    for k, v in out.items():
        if k.startswith("_") or k == "stm_summary":
            continue
        if isinstance(v, list):
            n += sum(tm.get_token_count(tm._extract_text(i), "m") for i in v)
        elif isinstance(v, str) and v:
            n += tm.get_token_count(v, "m")
    return n


def _section_tokens(tm, ctx):
    return {k: (sum(tm.get_token_count(tm._extract_text(i), "m") for i in v)
                if isinstance(v, list) else tm.get_token_count(v, "m"))
            for k, v in ctx.items() if v}


@pytest.fixture
def mem_cfg(monkeypatch):
    mem = app_config.config.setdefault("memory", {})

    def _set(**kw):
        for k, v in kw.items():
            monkeypatch.setitem(mem, k, v)
    return _set


def test_incident_shape_keeps_memories_and_other_floors(mem_cfg):
    mem_cfg(prompt_memories_reserve_floor=2, prompt_min_summaries_floor=2,
            prompt_min_reflections_floor=1, prompt_floor_reserve_max_share=0.25)
    tm = _tm()
    out = tm._manage_token_budget(incident_ctx(), now=NOW)
    assert len(out["memories"]) >= 2
    assert len(out["recent_summaries"]) >= 2
    assert len(out["recent_reflections"]) >= 1
    assert _metered(tm, out) <= 10000


def test_memories_floor_zero_is_the_old_behaviour(mem_cfg):
    mem_cfg(prompt_memories_reserve_floor=0, prompt_min_summaries_floor=2,
            prompt_min_reflections_floor=1, prompt_floor_reserve_max_share=0.25)
    tm = _tm()
    out = tm._manage_token_budget(incident_ctx(), now=NOW)
    assert len(out.get("memories") or []) == 0   # the incident, reproduced with the floor off
    assert len(out["recent_summaries"]) >= 2     # sibling floors unaffected


def test_live_read_same_instance(mem_cfg):
    tm = _tm()
    mem_cfg(prompt_memories_reserve_floor=0, prompt_floor_reserve_max_share=0.25)
    off = tm._manage_token_budget(incident_ctx(), now=NOW)
    mem_cfg(prompt_memories_reserve_floor=2)
    on = tm._manage_token_budget(incident_ctx(), now=NOW)
    assert len(off.get("memories") or []) == 0 and len(on["memories"]) >= 2


def test_default_share_is_one_value():
    from config.schema import MemorySection  # noqa: F401  (schema default must match)
    from core.prompt.token_manager import FLOOR_RESERVE_MAX_SHARE_DEFAULT
    assert MemorySection().prompt_floor_reserve_max_share == FLOOR_RESERVE_MAX_SHARE_DEFAULT
    assert float((app_config.config.get("memory") or {}).get(
        "prompt_floor_reserve_max_share", FLOOR_RESERVE_MAX_SHARE_DEFAULT)) == FLOOR_RESERVE_MAX_SHARE_DEFAULT


def test_section_token_measurements(mem_cfg, capsys):
    mem_cfg(prompt_memories_reserve_floor=2, prompt_min_summaries_floor=2,
            prompt_min_reflections_floor=1, prompt_floor_reserve_max_share=0.25)
    tm = _tm()
    before = _section_tokens(tm, incident_ctx())
    after = _section_tokens(tm, tm._manage_token_budget(incident_ctx(), now=NOW))
    mem_cfg(prompt_memories_reserve_floor=0)
    old = _section_tokens(tm, tm._manage_token_budget(incident_ctx(), now=NOW))
    with capsys.disabled():
        print("\nMEASURE before:", before)
        print("MEASURE floor3:", after)
        print("MEASURE floor0:", old)
    assert after.get("memories", 0) > old.get("memories", 0)


class TestMemoriesRenderedTelemetry:
    def test_extra_and_capture_delivery_reach_the_turn_record(self):
        from types import SimpleNamespace
        from gui import handlers

        orch = SimpleNamespace(_last_memories_rendered=2)
        extra = handlers._memories_rendered_extra(orch)
        assert extra == {"memories_rendered": 2}
        ctx = SimpleNamespace(telemetry={}, t_ingress=0.0)
        handlers._capture_delivery(ctx, dict(extra))
        assert ctx.telemetry["memories_rendered"] == 2

    def test_zero_is_recorded_and_unknown_is_omitted(self):
        from types import SimpleNamespace
        from gui import handlers

        assert handlers._memories_rendered_extra(SimpleNamespace(_last_memories_rendered=0)) == {"memories_rendered": 0}
        assert handlers._memories_rendered_extra(SimpleNamespace()) == {}
        assert handlers._memories_rendered_extra(SimpleNamespace(_last_memories_rendered=None)) == {}
