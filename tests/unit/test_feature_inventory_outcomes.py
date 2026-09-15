"""[ACTIVE FEATURES] renders "could not check" for failed/unavailable
sections; obsidian= is driven by OBSIDIAN_ENABLED instead of by whether
personal_notes happened to come back non-empty this turn.

Design source: docs/execution/generalization/failure_outcome_design.md,
"Real defects" (formatter.py:793: "A failed vault search prints
obsidian=OFF") and CGR-007. Batch: docs/execution/generalization/batches/
F6a.md (attempt 2).

Drives the deployed PromptFormatter directly (core/prompt/formatter.py),
the pattern tests/unit/test_sep10_web_search_gap.py already uses.
`_section_outcomes` is F5's dict: {section: {"status": ... , "reason": ...}}.

FAILING FIRST: run against the UNEDITED formatter.py (digest
2e83856b7f14957c95ae5d330d3e59aeb91f478d7d900ba840b034904bd9e3db); failures
and the pinned pre-edit output are recorded in batches/F6a.md.
"""
from unittest.mock import MagicMock

import pytest

from core.prompt.formatter import PromptFormatter

# Every config flag _build_feature_inventory reads, defaulted False so every
# test is deterministic regardless of this checkout's config/env.
_FLAG_DEFAULTS = {
    "KNOWLEDGE_GRAPH_ENABLED": False, "FACT_VERIFICATION_ENABLED": False,
    "TRUTH_SCORER_ENABLED": False, "CROSS_DEDUP_ENABLED": False,
    "GIT_MEMORY_ENABLED": False, "OBSIDIAN_ENABLED": False,
    "REFERENCE_DOCS_AUTO_SEED": False, "WEB_SEARCH_ENABLED": False,
    "THREAD_SURFACING_ENABLED": False, "PROACTIVE_SURFACING_ENABLED": False,
    "NARRATIVE_CONTEXT_ENABLED": False, "INTENT_ENABLED": False,
    "ESCALATION_ENABLED": False, "PROCEDURAL_SKILLS_ENABLED": False,
}


def _patch_flags(monkeypatch, **overrides):
    flags = dict(_FLAG_DEFAULTS)
    flags.update(overrides)
    for name, value in flags.items():
        monkeypatch.setattr(f"config.app_config.{name}", value)


def _formatter():
    return PromptFormatter(token_manager=MagicMock(), time_manager=None)


def _outcome(status, reason=""):
    return {"status": status, "reason": reason}


def _base_context(**overrides):
    ctx = {
        "graph_context": [], "git_commits": [], "personal_notes": [],
        "reference_docs": [], "unresolved_threads": [],
        "proactive_insights": [], "procedural_skills": [],
        "narrative_state": "",
    }
    ctx.update(overrides)
    return ctx


class TestObsidianFlagSemantics:
    """Contract point 2: obsidian= reads OBSIDIAN_ENABLED; suffix priority
    is could-not-check, else a note count, else nothing."""

    def test_failed_personal_notes_with_flag_enabled(self, monkeypatch):
        _patch_flags(monkeypatch, OBSIDIAN_ENABLED=True)
        context = _base_context(
            _section_outcomes={"personal_notes": _outcome("failed", "ConnectionError")})

        result = _formatter()._build_feature_inventory(context)

        assert "obsidian=ON(could not check)" in result

    def test_flag_disabled_and_no_outcomes_shows_bare_off(self, monkeypatch):
        _patch_flags(monkeypatch, OBSIDIAN_ENABLED=False)
        context = _base_context()  # no _section_outcomes key at all

        result = _formatter()._build_feature_inventory(context)

        assert "obsidian=OFF" in result
        assert "obsidian=OFF(" not in result

    def test_enabled_with_three_notes_shows_count(self, monkeypatch):
        _patch_flags(monkeypatch, OBSIDIAN_ENABLED=True)
        context = _base_context(personal_notes=[{"content": "a"}, {"content": "b"}, {"content": "c"}])

        result = _formatter()._build_feature_inventory(context)

        assert "obsidian=ON(3 notes)" in result

    def test_enabled_no_notes_no_failure_shows_bare_on(self, monkeypatch):
        """The changed case: today this reads OFF ("feature off"); the
        vault is enabled and simply had nothing to return this turn."""
        _patch_flags(monkeypatch, OBSIDIAN_ENABLED=True)
        context = _base_context(personal_notes=[])

        result = _formatter()._build_feature_inventory(context)

        assert "obsidian=ON" in result
        assert "obsidian=ON(" not in result


class TestPerItemCouldNotCheckSuffix:
    """Contract point 3: (could not check) REPLACES the count suffix for
    the six list-backed items; narrative (no count) gets it appended."""

    def test_graph_context_unavailable_replaces_count_not_appends(self, monkeypatch):
        _patch_flags(monkeypatch, KNOWLEDGE_GRAPH_ENABLED=True)
        # Leftover items alongside a failed status must not be counted.
        context = _base_context(
            graph_context=["stale edge 1", "stale edge 2"],
            _section_outcomes={"graph_context": _outcome("unavailable", "timeout")})

        result = _formatter()._build_feature_inventory(context)

        assert "knowledge_graph=ON(could not check)" in result
        assert "edges)" not in result

    @pytest.mark.parametrize(
        ("context_key", "cfg_flag", "expected"),
        [
            ("git_commits", "GIT_MEMORY_ENABLED", "git_commits=ON(could not check)"),
            ("reference_docs", "REFERENCE_DOCS_AUTO_SEED", "reference_docs=ON(could not check)"),
            ("unresolved_threads", "THREAD_SURFACING_ENABLED", "threads=ON(could not check)"),
            ("proactive_insights", "PROACTIVE_SURFACING_ENABLED", "insights=ON(could not check)"),
            ("procedural_skills", "PROCEDURAL_SKILLS_ENABLED", "skills=ON(could not check)"),
        ],
    )
    def test_replace_suffix_for_each_itemized_section(self, monkeypatch, context_key, cfg_flag, expected):
        _patch_flags(monkeypatch, **{cfg_flag: True})
        context = _base_context(
            **{context_key: []},
            _section_outcomes={context_key: _outcome("failed", "ValueError")})

        result = _formatter()._build_feature_inventory(context)

        assert expected in result, f"{context_key}: expected {expected!r} in {result!r}"

    @pytest.mark.parametrize(
        ("flag", "expected"),
        [(True, "narrative=ON(could not check)"), (False, "narrative=OFF(could not check)")],
    )
    def test_narrative_failed_appends_could_not_check(self, monkeypatch, flag, expected):
        _patch_flags(monkeypatch, NARRATIVE_CONTEXT_ENABLED=flag)
        context = _base_context(_section_outcomes={"narrative": _outcome("failed", "RuntimeError")})

        result = _formatter()._build_feature_inventory(context)

        assert expected in result


class TestCouldNotCheckLine:
    """Contract point 4: one extra line after the four category lines for
    NOT CHECKED sections not already shown by an inventory item."""

    def test_two_non_itemized_sections_produce_exact_sorted_line(self, monkeypatch):
        _patch_flags(monkeypatch)
        context = _base_context(_section_outcomes={
            "upcoming_schedule": _outcome("failed", "ValueError"),
            "relevant_emails": _outcome("unavailable", "timeout"),
        })

        result = _formatter()._build_feature_inventory(context)

        assert result.split("\n")[-1] == "Could not check this turn: relevant_emails, upcoming_schedule"

    def test_itemized_sections_excluded_from_the_line_even_when_failed(self, monkeypatch):
        _patch_flags(monkeypatch, KNOWLEDGE_GRAPH_ENABLED=True)
        context = _base_context(_section_outcomes={
            "graph_context": _outcome("failed", "X"),
            "upcoming_schedule": _outcome("failed", "Y"),
        })

        result = _formatter()._build_feature_inventory(context)
        last_line = result.split("\n")[-1]

        assert last_line == "Could not check this turn: upcoming_schedule"
        assert "graph_context" not in last_line

    def test_web_search_failed_produces_no_extra_line_label_unchanged(self, monkeypatch):
        _patch_flags(monkeypatch, WEB_SEARCH_ENABLED=True)
        context = _base_context(_section_outcomes={"web_search": _outcome("failed", "TavilyError")})

        result = _formatter()._build_feature_inventory(context)

        assert "web_search=ON(no search this turn)" in result
        assert "Could not check this turn" not in result

    def test_line_omitted_when_no_not_checked_sections(self, monkeypatch):
        _patch_flags(monkeypatch)
        context = _base_context(_section_outcomes={
            "recent": _outcome("succeeded"),
            "personal_notes": _outcome("no_results"),
        })

        result = _formatter()._build_feature_inventory(context)

        assert "Could not check this turn" not in result
        assert result.count("\n") == 3  # exactly 4 lines


def test_reason_labels_never_appear_in_output(monkeypatch):
    """Privacy: no reason label, query text or exception text in the prompt."""
    _patch_flags(monkeypatch, OBSIDIAN_ENABLED=True, KNOWLEDGE_GRAPH_ENABLED=True)
    marker = "F6aMARKQ77_sensitive_detail_must_not_leak"
    context = _base_context(_section_outcomes={
        "personal_notes": _outcome("failed", marker),
        "graph_context": _outcome("unavailable", marker),
        "narrative": _outcome("failed", marker),
        "upcoming_schedule": _outcome("failed", marker),
    })

    result = _formatter()._build_feature_inventory(context)

    assert marker not in result
    assert "ConnectionError" not in result


class TestControls:
    """Point 5: no outcomes key, or all succeeded/no_results, means
    byte-identical output apart from the obsidian flag semantics."""

    def test_memory_proactive_analysis_unaffected_by_missing_outcomes_key(self, monkeypatch):
        """Unaffected-by-design control: passes before AND after the edit."""
        _patch_flags(
            monkeypatch, KNOWLEDGE_GRAPH_ENABLED=True, FACT_VERIFICATION_ENABLED=True,
            TRUTH_SCORER_ENABLED=True, GIT_MEMORY_ENABLED=True, REFERENCE_DOCS_AUTO_SEED=True,
            THREAD_SURFACING_ENABLED=True, PROACTIVE_SURFACING_ENABLED=True,
            NARRATIVE_CONTEXT_ENABLED=True, INTENT_ENABLED=True, ESCALATION_ENABLED=True,
            PROCEDURAL_SKILLS_ENABLED=True,
        )
        context = _base_context(
            graph_context=["e1", "e2"], git_commits=[{"content": "c1"}],
            reference_docs=[{"content": "d1"}], unresolved_threads=[{"topic": "t1"}],
            proactive_insights=["i1"], procedural_skills=[{"metadata": {}}],
        )

        lines = _formatter()._build_feature_inventory(context).split("\n")

        assert lines[0] == "Memory: knowledge_graph=ON(2 edges) | fact_verification=ON | truth_scorer=ON | dedup=OFF"
        assert lines[2] == "Proactive: threads=ON(1 open) | insights=ON(1) | narrative=ON"
        assert lines[3] == "Analysis: intent=ON | escalation=ON | skills=ON(1)"

    def test_no_outcomes_key_obsidian_reflects_new_semantics(self, monkeypatch):
        """The case that CHANGES vs. the unedited source (pre-edit actual
        output pinned in batches/F6a.md's failing-first section — there
        this line reads obsidian=OFF)."""
        _patch_flags(
            monkeypatch, GIT_MEMORY_ENABLED=True, OBSIDIAN_ENABLED=True,
            REFERENCE_DOCS_AUTO_SEED=True,
        )
        context = _base_context(
            git_commits=[{"content": "c1"}], personal_notes=[],
            reference_docs=[{"content": "d1"}],
        )

        lines = _formatter()._build_feature_inventory(context).split("\n")

        assert lines[1] == "Knowledge: git_commits=ON(1) | obsidian=ON | reference_docs=ON(1) | web_search=OFF"

    def test_all_succeeded_outcomes_no_could_not_check_anywhere(self, monkeypatch):
        _patch_flags(
            monkeypatch, OBSIDIAN_ENABLED=True, KNOWLEDGE_GRAPH_ENABLED=True,
            GIT_MEMORY_ENABLED=True, REFERENCE_DOCS_AUTO_SEED=True,
            THREAD_SURFACING_ENABLED=True, PROACTIVE_SURFACING_ENABLED=True,
            NARRATIVE_CONTEXT_ENABLED=True, PROCEDURAL_SKILLS_ENABLED=True,
        )
        names = ("graph_context", "git_commits", "personal_notes", "reference_docs",
                  "unresolved_threads", "proactive_insights", "procedural_skills", "narrative")
        context = _base_context(_section_outcomes={n: _outcome("succeeded") for n in names})
        context["_section_outcomes"]["upcoming_schedule"] = _outcome("no_results")

        result = _formatter()._build_feature_inventory(context)

        assert "could not check" not in result
        assert "Could not check this turn" not in result


def test_could_not_check_line_appears_inside_active_features_section(monkeypatch):
    """The deployed path used by existing formatter tests
    (test_feature_inventory.py's test_section_in_assembled_prompt)."""
    _patch_flags(monkeypatch)
    context = _base_context(
        recent_conversations=[], memories=[], user_profile="", summaries=[],
        recent_summaries=[], semantic_summaries=[], reflections=[],
        recent_reflections=[], semantic_reflections=[], dreams=[],
        semantic_chunks=[], wiki=[], user_uploads=[], proposed_features=[],
        web_search_results=None, codebase_changes={},
        _section_outcomes={
            "upcoming_schedule": _outcome("failed", "ValueError"),
            "relevant_emails": _outcome("unavailable", "timeout"),
        },
    )

    prompt = _formatter()._assemble_prompt(context=context, user_input="hello")

    assert "[ACTIVE FEATURES]" in prompt
    start = prompt.index("[ACTIVE FEATURES]")
    next_header = prompt.find("\n[", start + 1)
    section = prompt[start:next_header if next_header != -1 else len(prompt)]

    assert "Could not check this turn: relevant_emails, upcoming_schedule" in section
