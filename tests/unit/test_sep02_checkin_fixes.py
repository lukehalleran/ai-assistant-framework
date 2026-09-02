"""2026-09-02 daily-check-in session fixes:

1. Grounding verifier spliced a heat-safety PSA into a warm reply, "correcting"
   the user's OWN "100 out" — figure-parity demotion + safety-PSA demotion.
2. A read-only git audit query was mined into profile facts (head_sha=<40hex>,
   git_status="working tree has 13 modified files", role=auditor).
3. The debug dump showed mode=agentic-search but not WHY — gate_reason surfaced.
"""
import core.grounding_check as gc
from core.grounding_check import (
    GroundingVerdict,
    claim_figure_user_stated,
    _is_safety_psa,
)
from memory.fact_extractor import _is_repo_audit_junk, _clean_triple


# ── Issue 1: grounding figure-parity + safety-PSA demotions ────────────────
class TestFigureUserStated:
    def test_echoed_user_figure_demotes(self):
        # The live Turn-5 shape: reply said "100-degree day"; user said "100 out".
        assert claim_figure_user_stated(
            "working on a 100-degree day", "it is like 100 out"
        )

    def test_unrelated_figure_not_demoted(self):
        assert not claim_figure_user_stated(
            "the GDP grew 15 percent", "it is like 100 out"
        )

    def test_single_digit_ignored(self):
        # 1-digit figures collide spuriously — under-fire by design.
        assert not claim_figure_user_stated("5 things", "I have 5 tabs open")

    def test_empty_inputs(self):
        assert not claim_figure_user_stated("", "100 out")
        assert not claim_figure_user_stated("100 degrees", "")


class TestSafetyPsaDemotion:
    def _v(self, correction, why_false="the response omits heat safety"):
        return GroundingVerdict(
            false_claim_present=True, claim="a 100-degree day",
            why_false=why_false, confidence=0.9, correction=correction,
        )

    def test_heat_psa_demoted(self):
        v = self._v(
            "Working in extreme heat, such as 100 degrees Fahrenheit, can pose "
            "serious health risks, including heat exhaustion and heat stroke. "
            "It's important to maintain a safe and comfortable working environment."
        )
        assert _is_safety_psa(v)

    def test_real_dose_correction_survives(self):
        # A genuine catch carries a contradiction/contrast — must NOT demote.
        v = self._v(
            "The safe dose is 400mg, not 4000mg — exceeding it can pose serious "
            "health risks.",
            why_false="the response stated 4000mg which is incorrect",
        )
        assert not _is_safety_psa(v)

    def test_non_safety_correction_unaffected(self):
        v = self._v("The capital of France is Paris.", why_false="it said Lyon")
        assert not _is_safety_psa(v)

    def test_parse_verdict_demotes_heat_psa(self):
        import json
        raw = json.dumps({
            "false_claim_present": True,
            "claim": "a 100-degree day",
            "why_false": "the response does not mention heat safety",
            "confidence": 0.9,
            "correction": "Working in extreme heat can pose serious health "
                          "risks including heat stroke. Stay hydrated.",
        })
        verdict = gc._parse_verdict(raw)
        assert verdict is not None
        assert verdict.false_claim_present is False  # demoted


# ── Issue 2: repo-audit fact junk guard ────────────────────────────────────
class TestRepoAuditJunk:
    def test_head_sha_relation(self):
        assert _is_repo_audit_junk(
            "user", "head_sha", "d18d58617b3d9b6f945b292694ef978a599c18b4"
        )

    def test_git_status_object(self):
        assert _is_repo_audit_junk(
            "user", "git_status", "working tree has 13 modified files"
        )

    def test_role_auditor(self):
        assert _is_repo_audit_junk("user", "role", "auditor")

    def test_bare_sha_object(self):
        assert _is_repo_audit_junk(
            "user", "commit", "a52b5e26112b4f0faa6d21a9774ae8d8c0fb630f"
        )

    def test_real_career_fact_survives(self):
        assert not _is_repo_audit_junk("user", "role", "data analyst")
        assert not _is_repo_audit_junk("user", "works_at", "Georgia Tech")

    def test_clean_triple_drops_git_status(self):
        assert _clean_triple(
            "user", "git_status", "working tree has 13 modified files"
        ) is None

    def test_clean_triple_keeps_real_fact(self):
        out = _clean_triple("user", "hobby", "running")
        assert out is not None and out[1] == "hobby"


# ── Profile timeline-fact bloat cap ────────────────────────────────────────
class TestProfileTimelineCap:
    """A churny relation (works_on/plan) accumulated dozens of history values;
    the temporal-query mini-timeline joined ALL of them into one giant line
    that ate the profile budget and truncated real facts out of the prompt."""

    def _profile(self):
        import os as _os
        import tempfile
        from memory.user_profile import UserProfile
        td = tempfile.mkdtemp()
        return UserProfile(profile_path=_os.path.join(td, "profile.json"))

    def test_timeline_capped_to_recent_entries(self):
        from datetime import datetime, timedelta
        p = self._profile()
        base = datetime(2026, 1, 1, 12, 0, 0)
        for i in range(15):  # 15 distinct values -> long accumulated history
            p.add_fact(relation="works_on", value=f"task alpha {i}",
                       confidence=0.9, timestamp=base + timedelta(days=i))
        ctx = p.get_context_injection(
            max_tokens=2000, query="my works timeline over the last month")
        lines = [l for l in ctx.splitlines() if l.startswith("works_on timeline:")]
        assert lines, f"timeline line missing:\n{ctx}"
        line = lines[0]
        assert line.count(" → ") <= 5           # capped at 6 entries
        assert "…→" in line                     # older entries dropped-marker
        assert "task alpha 14" in line          # newest kept
        assert "task alpha 8 " not in line      # entry beyond the cap dropped

    def test_short_history_uncapped_no_marker(self):
        from datetime import datetime, timedelta
        p = self._profile()
        base = datetime(2026, 1, 1, 12, 0, 0)
        for i in range(3):
            p.add_fact(relation="works_on", value=f"task beta {i}",
                       confidence=0.9, timestamp=base + timedelta(days=i))
        ctx = p.get_context_injection(
            max_tokens=2000, query="my works timeline over the last month")
        lines = [l for l in ctx.splitlines() if l.startswith("works_on timeline:")]
        assert lines
        assert "…→" not in lines[0]             # nothing dropped


# ── Issue 3: gate_reason debug surfacing ───────────────────────────────────
class TestGateReasonSummary:
    def test_summary_formats_reason_and_modes(self):
        from gui.handlers import _gate_debug_summary

        class _D:
            reason = "triggered: web_search, memory"
            modes = ["web_search", "memory"]
            veto_exempt = False
            deferred_request = None

        s = _gate_debug_summary(_D())
        assert "triggered: web_search, memory" in s
        assert "modes=" in s

    def test_summary_none_decision(self):
        from gui.handlers import _gate_debug_summary
        assert _gate_debug_summary(None) == ""

    def test_summary_veto_reason(self):
        from gui.handlers import _gate_debug_summary

        class _D:
            reason = "tone-veto: acute tone (CrisisLevel.MEDIUM)"
            modes = []
            veto_exempt = False
            deferred_request = None

        assert "tone-veto" in _gate_debug_summary(_D())
