"""A routed live commit history survives the budget, and cut context is named.

Incident 2026-10-24 13:57 ([test] probe "what have I been building lately, and
how much code went into it?"): git_commits (10 items) was emptied by the budget
along with 6 other sections; the model saw nothing about commits and answered
"your building lately has actually been light" -- absence of context read as
absence of activity. Bug classes: BC-99 (priority admission starves a lower
tier; routed turns now reserve a floor), BC-24 (floor defeated around the
budget), BC-46 (the omission was invisible, so absence read as evidence).

All tests drive the deployed ``_manage_token_budget`` (explicit ``now``, BC-94)
and the deployed formatter ``_assemble_prompt``; the tokenizer is a word counter.
"""
from datetime import datetime, timedelta

import pytest

import config.app_config as app_config
from core.prompt.formatter import PromptFormatter
from core.prompt.token_manager import TokenManager, build_omitted_note

NOW = datetime(2026, 10, 24, 13, 57, 0)


class _MM:
    def get_active_model_name(self):
        return "test-model"


class _Tok:
    def count_tokens(self, text, model_name=None):
        return len((text or "").split())


def _fmt():
    return PromptFormatter(token_manager=None, time_manager=None)


def _tm(budget=10000):
    return TokenManager(_MM(), _Tok(), budget)


def _conv(i, words):
    ts = NOW - timedelta(minutes=3 * i)
    return {"query": f"q{i}", "response": " ".join([f"r{i}"] * words), "timestamp": ts.isoformat()}


def _commit(i, routed):
    meta = {"commit_hash": f"{i:07x}", "timestamp": "2026-10-23T10:00:00-05:00"}
    if routed:
        meta["retrieval_source"] = "local_git"
    return {"content": f"feat: invented change number {i} " + "word " * 200, "metadata": meta}


def incident_ctx(routed):
    """24 recent turns, 12 stated figures, 10 memories, 10 commits (+ filler that
    fills a 10K budget the way the live turn did)."""
    return {
        "user_profile": " ".join(["prof"] * 2900),
        "recent_conversations": [_conv(i, 200) for i in range(24)],
        "stated_figures": [f"figure {i}: invented value {i} on 2026-10-0{i % 9 + 1}" for i in range(12)],
        "memories": [{"content": " ".join([f"m{i}"] * 250)} for i in range(14)],
        "personal_notes": [{"content": " ".join([f"n{i}"] * 150)} for i in range(14)],
        "reference_docs": [{"content": " ".join([f"d{i}"] * 150)} for i in range(14)],
        "semantic_summaries": [{"content": " ".join(["ss"] * 200)} for _ in range(5)],
        "git_commits": [_commit(i, routed) for i in range(10)],
    }


@pytest.fixture
def mem_cfg(monkeypatch):
    mem = app_config.config.setdefault("memory", {})

    def _set(**kw):
        for k, v in kw.items():
            monkeypatch.setitem(mem, k, v)
    return _set


class TestRoutedGitReserve:
    def test_unrouted_ambient_commits_are_still_emptied(self, mem_cfg):
        mem_cfg(prompt_git_commits_reserve_floor=5)
        out = _tm()._manage_token_budget(incident_ctx(routed=False), now=NOW)
        assert not out.get("git_commits")                      # the incident, unchanged
        assert "git_commits" in out["_budget_emptied"]

    def test_routed_keeps_at_least_the_floor(self, mem_cfg):
        mem_cfg(prompt_git_commits_reserve_floor=5, prompt_floor_reserve_max_share=0.25)
        tm = _tm()
        out = tm._manage_token_budget(incident_ctx(routed=True), now=NOW)
        assert len(out["git_commits"]) >= 5
        assert "git_commits" not in out.get("_budget_emptied", [])

    def test_reserve_holds_against_equal_priority_crowding(self, mem_cfg):
        """Priority 8 alone is not a guarantee: a user-evidence section admitted
        earlier in list order can still spend the whole budget."""
        def crowded():
            ctx = incident_ctx(routed=True)
            ctx["user_stated_evidence"] = [
                {"content": " ".join([f"e{i}"] * 100)} for i in range(80)]
            return ctx
        tm = _tm()
        mem_cfg(prompt_git_commits_reserve_floor=0)
        off = tm._manage_token_budget(crowded(), now=NOW)
        mem_cfg(prompt_git_commits_reserve_floor=5)
        on = tm._manage_token_budget(crowded(), now=NOW)
        assert len(off.get("git_commits") or []) < 5      # the failure the reserve closes
        assert len(on["git_commits"]) >= 5

    def test_floor_zero_disables_reserve_same_instance(self, mem_cfg):
        tm = _tm()
        mem_cfg(prompt_git_commits_reserve_floor=0)
        off = tm._manage_token_budget(incident_ctx(routed=True), now=NOW)
        mem_cfg(prompt_git_commits_reserve_floor=5)
        on = tm._manage_token_budget(incident_ctx(routed=True), now=NOW)
        assert len(on["git_commits"]) >= 5
        assert len(on["git_commits"]) >= len(off.get("git_commits") or [])

    def test_reserve_respects_max_share(self, mem_cfg):
        mem_cfg(prompt_git_commits_reserve_floor=5, prompt_floor_reserve_max_share=0.0)
        out = _tm()._manage_token_budget(incident_ctx(routed=True), now=NOW)
        # no reserve may be granted at share 0; commits still win on priority 8
        # but are not GUARANTEED -- the call must simply stay within budget
        total = sum(
            _tm().get_token_count(_tm()._extract_text(i), "m")
            for k, v in out.items() if isinstance(v, list) and not k.startswith("_") for i in v
        )
        assert total <= 10000


class TestOmittedNote:
    def test_incident_renders_note_naming_commit_history(self, mem_cfg):
        mem_cfg(prompt_git_commits_reserve_floor=5)
        out = _tm()._manage_token_budget(incident_ctx(routed=False), now=NOW)
        note = out["budget_omitted_note"]
        assert "commit history (10)" in note
        assert "absence here is not evidence" in note
        assert len(note.split()) <= 50
        assert "do not mention it unless the user asks what you can see" in note
        prompt = _fmt()._assemble_prompt(dict(out), "q")
        assert "[NOT SHOWN FOR SPACE]" in prompt and "commit history (10)" in prompt

    def test_nothing_emptied_renders_no_line(self):
        ctx = {"memories": [{"content": "one small memory"}], "git_commits": [_commit(1, True)]}
        out = _tm()._manage_token_budget(ctx, now=NOW)
        assert "budget_omitted_note" not in out
        prompt = _fmt()._assemble_prompt(dict(out), "q")
        assert "NOT SHOWN FOR SPACE" not in prompt

    def test_empty_sections_never_listed_and_stale_note_dropped(self):
        ctx = {"memories": [{"content": "m"}], "recent_summaries": [], "budget_omitted_note": "stale"}
        out = _tm()._manage_token_budget(ctx, now=NOW)
        assert "budget_omitted_note" not in out

    def test_cap_and_labels(self):
        ctx = {k: [{"content": "x"}] for k in (
            "git_commits", "semantic_summaries", "daemon_self_notes",
            "proactive_insights", "user_uploads", "proposed_features")}
        note = build_omitted_note(list(ctx), ctx)
        assert "+2 more" in note and "commit history (1)" in note
        assert "_" not in note
