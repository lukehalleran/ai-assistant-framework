"""Fixes from the 2026-08-28 evening thread audit (post-restart session).

1. Intent: bare \bissue\b classified "disclosure timing … is often a huge
   issue" (health-research) as technical_help@0.75 — everyday-English sense,
   same class as "commit to" (08-27). Removed; tracker-sense forms kept.
2. [DAEMON DOCUMENTATION] origin gate: 54 OMSA course-transcript titles (206
   chunks, ingested without type='user_upload') rendered as Daemon
   self-knowledge; only repo-docs/-origin chunks may render there now.
3. Accuracy-clause dedup: LIGHT SUPPORT + GROUNDING PRESENCE both carried
   GROUNDING_ACCURACY_CLAUSE in one prompt; orchestrator strips the later copy.
4. Recap-accuracy bullet: turn 1 asserted "made the fin aid call" — an event
   the user never reported; clause now forbids inferring task completion.

(The turn-8 decision-answer narration fix is tested in
test_agentic_decision_answer_reuse.py.)
"""
import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.intent_classifier import IntentClassifier, IntentType


LIVE_TURN8_QUERY = (
    "Let's make sure we're accurate here. My prior was that especially around "
    "that specific sort of health stuff, disclosure timing if at all to "
    "doctors is often a huge issue"
)


class TestBareIssueIntentFix:
    @pytest.fixture(scope="class")
    def clf(self):
        return IntentClassifier()

    def test_live_health_research_query_not_technical(self, clf):
        r = clf.classify(LIVE_TURN8_QUERY)
        assert r.intent != IntentType.TECHNICAL_HELP

    @pytest.mark.parametrize("q", [
        "my mom's health issues are getting worse",
        "sleep has been a huge issue lately",
        "the issue with my mom came up again in therapy",
    ])
    def test_everyday_issue_not_technical(self, clf, q):
        assert clf.classify(q).intent != IntentType.TECHNICAL_HELP

    @pytest.mark.parametrize("q", [
        "I opened an issue on the repo for the crash",
        "there's a github issue tracking that bug",
        "how do I fix this docker build error",
        "the app crashes with a traceback on startup",
    ])
    def test_technical_forms_still_classify(self, clf, q):
        r = clf.classify(q)
        assert r.intent == IntentType.TECHNICAL_HELP
        assert r.confidence >= 0.65


class TestSelfDocsOriginGate:
    @pytest.mark.asyncio
    async def test_downloads_origin_chunks_filtered(self):
        from core.prompt.context_gatherer import ContextGatherer
        import core.prompt.gatherer_knowledge as gk

        manager = MagicMock()
        manager.get_documents = AsyncMock(return_value=[
            {"content": "# Daemon Architecture Guide",
             "metadata": {"title": "ARCHITECTURE_GUIDE",
                          "file_path": f"{gk._SELF_DOCS_DIR}/ARCHITECTURE_GUIDE.md"}},
            {"content": ">> In some previous lessons we've seen survival models",
             "metadata": {"title": "OMSA_ISyE6501_M16L7",
                          "file_path": "/home/lukeh/Downloads/OMSA_ISyE6501_M16L7.txt"}},
            {"content": "legacy self-doc chunk with no path",
             "metadata": {"title": "QUICK_REFERENCE"}},
            {"content": "an upload with the proper tag",
             "metadata": {"title": "upload:syllabus.docx", "type": "user_upload",
                          "file_path": "/home/lukeh/Downloads/syllabus.docx"}},
        ])
        g = ContextGatherer.__new__(ContextGatherer)
        # reference_docs_manager is a lazily-created property backed by
        # _reference_docs_manager — inject the stub behind it.
        g._reference_docs_manager = manager
        g.memory_id_map = {}

        docs = await g.get_reference_docs("how does your memory work")
        titles = [d["metadata"]["title"] for d in docs]
        assert "ARCHITECTURE_GUIDE" in titles
        assert "QUICK_REFERENCE" in titles          # no-path legacy kept (fail-open)
        assert "OMSA_ISyE6501_M16L7" not in titles  # foreign-origin dropped
        assert "upload:syllabus.docx" not in titles # user_upload filter still works

    def test_self_docs_dir_points_at_repo_docs(self):
        # Repo-root-relative, NOT the literal checkout folder name — CI checks
        # out under a different directory name (me-shaped assertion, fixed 09-01).
        from pathlib import Path
        import core.prompt.gatherer_knowledge as gk
        repo_docs = Path(gk.__file__).resolve().parents[2] / "docs"
        assert gk._SELF_DOCS_DIR == str(repo_docs)
        assert gk._SELF_DOCS_DIR.endswith("/docs")


class TestAccuracyClauseDedup:
    def test_orchestrator_strips_duplicate_clause(self):
        # Wiring pin: the escalation append site dedups the clause.
        import core.orchestrator as orch
        src = inspect.getsource(orch)
        assert "GROUNDING_ACCURACY_CLAUSE in escalation_instructions" in src

    def test_replace_semantics_leave_single_copy(self):
        # Deployed constants through the same replace the orchestrator applies.
        from core.grounding_check import GROUNDING_ACCURACY_CLAUSE
        from core.escalation_tracker import EscalationTracker, ResponseStrategy
        from core.tone_instructions import get_tone_instructions
        from utils.tone_detector import CrisisLevel

        system_prompt = "base" + get_tone_instructions(CrisisLevel.CONCERN)
        t = EscalationTracker()
        t.current_strategy = ResponseStrategy.GROUNDING_PRESENCE
        esc = t.get_strategy_instructions()
        assert GROUNDING_ACCURACY_CLAUSE in system_prompt
        assert GROUNDING_ACCURACY_CLAUSE in esc
        esc = esc.replace("\n" + GROUNDING_ACCURACY_CLAUSE, "").replace(
            GROUNDING_ACCURACY_CLAUSE, "")
        combined = system_prompt + esc
        assert combined.count("ACCURACY FLOOR") == 1


class TestRecapAccuracyBullet:
    def test_clause_forbids_inferred_completion(self):
        from core.grounding_check import GROUNDING_ACCURACY_CLAUSE
        assert "ONLY events" in GROUNDING_ACCURACY_CLAUSE
        assert "was completed unless they said so" in GROUNDING_ACCURACY_CLAUSE

    def test_bullet_reaches_light_support_block(self):
        from core.tone_instructions import get_tone_instructions
        from utils.tone_detector import CrisisLevel
        assert "ONLY events" in get_tone_instructions(CrisisLevel.CONCERN)
