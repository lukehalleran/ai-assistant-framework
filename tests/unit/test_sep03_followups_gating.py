"""2026-09-03 follow-ups — retrieval gating on casual chat, `role` facts, timer.

* Passive email fired on a capitalized PET name as a "contact" with a hardcoded
  30-day window (a shelter newsletter reached a cat-chat turn); casual and
  emotional intents now zero it and pet names are not contact seeds.
* A fresh upload surfaced regardless of relevance for a week (now 3 days).
* Personal notes had no casual-intent gate, a forced below-threshold backfill,
  and a 0.45 blended bar; casual_social zeros them, backfill is off, bar 0.60.
* `role=` facts ("bug fixer", "PR approver") were minted from activities by two
  few-shot examples; `role` now normalizes to `occupation`, needs a work cue,
  and can never be promoted from the learned-relation store.
* The daily-notes systemd job ran main.py's heavy imports under the system
  interpreter; a thin entrypoint + unit templates replace it.
"""
import ast
import inspect
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import memory.llm_fact_extractor as llm_fx
from core.intent_classifier import _PROFILES, IntentClassifier, IntentType
from core.prompt import gatherer_knowledge as gk
from memory.entity_resolver import normalize_relation
from memory.fact_source import _relation_cue_supported
from memory.graph_memory import GraphMemory
from memory.graph_models import GraphNode
from memory.learned_relations import LearnedRelationStore
from memory.user_profile_schema import canonicalize_profile_relation

REPO = Path(__file__).resolve().parents[2]


# ── passive email ─────────────────────────────────────────────────────────
class TestEmailPassiveTweaks:
    def _gatherer(self, monkeypatch, graph=None):
        from core.email import service as svc
        from config import app_config
        calls = {}

        class _Svc:
            async def search(self, terms, **kw):
                calls["terms"] = terms; calls.update(kw); return []
        monkeypatch.setattr(svc, "get_email_service", lambda: _Svc())
        monkeypatch.setattr(app_config, "EMAIL_PASSIVE_CONTEXT_ENABLED", True, raising=False)
        g = gk.KnowledgeRetrievalMixin.__new__(gk.KnowledgeRetrievalMixin)
        g._distress_active = False
        g.memory_coordinator = SimpleNamespace(graph_memory=graph)
        return g, calls

    @pytest.mark.asyncio
    async def test_window_uses_config(self, monkeypatch):
        from config.app_config import EMAIL_DEFAULT_WINDOW_DAYS
        g, calls = self._gatherer(monkeypatch)
        await gk.KnowledgeRetrievalMixin.get_relevant_emails(g, "did Morgan email me back")
        assert calls.get("window_days") == EMAIL_DEFAULT_WINDOW_DAYS != 30

    @pytest.mark.asyncio
    async def test_pet_name_not_a_contact_seed(self, monkeypatch, tmp_path):
        gm = GraphMemory(persist_path=str(tmp_path / "g.json"))
        gm.add_entity(GraphNode(entity_id="biscuit", display_name="Biscuit", entity_type="animal"))
        g, calls = self._gatherer(monkeypatch, graph=gm)
        out = await gk.KnowledgeRetrievalMixin.get_relevant_emails(g, "I saw Biscuit chase a moth")
        assert out == [] and calls == {}

    @pytest.mark.asyncio
    async def test_unknown_name_still_seeds(self, monkeypatch, tmp_path):
        gm = GraphMemory(persist_path=str(tmp_path / "g.json"))
        g, calls = self._gatherer(monkeypatch, graph=gm)
        await gk.KnowledgeRetrievalMixin.get_relevant_emails(g, "I saw Morgan at the store")
        assert calls.get("terms") == "Morgan"

    def test_non_contact_helper_underfires(self, tmp_path):
        gm = GraphMemory(persist_path=str(tmp_path / "g.json"))
        gm.add_entity(GraphNode(entity_id="morgan", display_name="Morgan", entity_type="person"))
        assert gk._is_non_contact_entity(None, "Biscuit") is False
        assert gk._is_non_contact_entity(gm, "Morgan") is False
        assert gk._is_non_contact_entity(gm, "Nobody") is False


# ── intent profiles ───────────────────────────────────────────────────────
class TestCasualProfileZeros:
    def test_profile_dicts(self):
        casual = _PROFILES[IntentType.CASUAL_SOCIAL]["retrieval"]
        assert casual["max_personal_notes"] == 0
        assert casual["max_relevant_emails"] == 0
        assert casual["max_graph_sentences"] == 0
        assert _PROFILES[IntentType.EMOTIONAL_SUPPORT]["retrieval"]["max_relevant_emails"] == 0

    def test_deployed_classifier_carries_the_keys(self):
        clf = IntentClassifier()
        res = clf.classify("Hey")
        assert res.intent_type == IntentType.CASUAL_SOCIAL
        for key in ("max_personal_notes", "max_relevant_emails", "max_graph_sentences"):
            assert res.retrieval_overrides.get(key) == 0

    def test_builder_reads_the_new_keys(self):
        from core.prompt import builder
        src = inspect.getsource(builder)
        assert '_ro.get("max_relevant_emails"' in src
        assert '_ro.get("max_graph_sentences"' in src


# ── uploads window ────────────────────────────────────────────────────────
class TestUploadsWindow:
    def test_window_is_three_days(self):
        assert gk.USER_UPLOADS_MAX_AGE_DAYS == 3

    def test_five_day_old_irrelevant_doc_dropped(self):
        now = datetime.now()
        def doc(days):
            return {"metadata": {"timestamp": (now - timedelta(days=days)).isoformat(),
                                 "filename": "homework.docx"}, "relevance_score": 0.1,
                    "content": "exponential smoothing"}
        assert gk._upload_is_live(doc(1)) is True
        assert gk._upload_is_live(doc(5)) is False


# ── personal notes gate ───────────────────────────────────────────────────
class TestNotesGate:
    def test_section_instruction_has_no_hardcoded_score(self):
        from core.prompt import section_instructions as si
        assert "0.30" not in si._PERSONAL_NOTES

    def test_builder_disables_forced_backfill_for_notes(self):
        from core.prompt import builder
        assert "personal_notes, min_results=0" in inspect.getsource(builder)

    def test_threshold_triplet(self):
        import yaml
        from config.app_config import PERSONAL_NOTES_GATE_THRESHOLD
        assert PERSONAL_NOTES_GATE_THRESHOLD == 0.60
        cfg = yaml.safe_load((REPO / "config" / "config.yaml").read_text())
        assert cfg["obsidian"]["gate_threshold"] == 0.60


# ── role facts ────────────────────────────────────────────────────────────
class TestRoleRelationHygiene:
    def test_role_normalizes_to_occupation(self):
        assert normalize_relation("role") == "occupation"
        assert normalize_relation("job_title") == "occupation"
        assert canonicalize_profile_relation("role", "app builder") == "occupation"

    def test_activity_span_has_no_occupation_cue(self):
        assert _relation_cue_supported("I built this app and I'm testing it now", "role") is False
        assert _relation_cue_supported("Fixed a small bug", "job_title") is False
        assert _relation_cue_supported("I work as a data analyst", "role") is True
        assert _relation_cue_supported("I'm a nurse", "occupation") is True

    def test_learned_store_never_tracks_or_promotes_role(self, tmp_path):
        store = LearnedRelationStore(path=str(tmp_path / "lr.json"))
        assert store.record("role", "bug fixer") is False
        store._data["relations"]["role"] = {"count": 3, "days": ["2026-08-01", "2026-08-02", "2026-08-03"],
                                            "last_seen": "2026-08-03", "examples": ["x"]}
        assert "role" not in store.promoted(min_days=3)

    def test_prompt_no_longer_teaches_role(self):
        src = inspect.getsource(llm_fx)
        assert '"relation": "role"' not in src
        assert "Never derive a role or title from an activity" in src


# ── daily-note entrypoint + systemd templates ─────────────────────────────
HEAVY = ("core.orchestrator", "gui", "chromadb", "memory.storage", "knowledge.WikiManager",
         "models.model_manager", "processing.gate_system")


class TestDailyNoteCatchupEntrypoint:
    def test_top_level_imports_are_light(self):
        tree = ast.parse((REPO / "scripts" / "daily_note_catchup.py").read_text())
        names = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
        assert not any(n.startswith(h) for n in names for h in HEAVY), names

    def _run(self, monkeypatch, result):
        import scripts.daily_note_catchup as dn
        import utils.daily_notes_generator as gen
        import models.model_manager as mm
        monkeypatch.setattr(mm, "ModelManager", lambda: object())
        fake = MagicMock(); fake.generate_yesterday_if_missing = AsyncMock(return_value=result)
        monkeypatch.setattr(gen, "DailyNotesGenerator", lambda model_manager=None: fake)
        return dn.run_catchup()

    def test_exit_codes(self, monkeypatch):
        assert self._run(monkeypatch, None) == 0
        assert self._run(monkeypatch, SimpleNamespace(success=True, output_path="x", conversation_count=1,
                                                      skipped_reason="", error="")) == 0
        assert self._run(monkeypatch, SimpleNamespace(success=False, skipped_reason="no conversations",
                                                      error="")) == 0
        assert self._run(monkeypatch, SimpleNamespace(success=False, skipped_reason="", error="boom")) == 1

    def test_main_delegates(self):
        src = (REPO / "main.py").read_text()
        assert "from scripts.daily_note_catchup import run_catchup" in src

    def test_unit_templates(self):
        d = REPO / "scripts" / "systemd"
        svc = (d / "daemon-daily-notes.service").read_text()
        assert "%h/Daemon_v1/scripts/daily_note_catchup.py" in svc and "%h/.pyenv" in svc
        assert "/home/" not in svc
        assert "OnCalendar=*-*-* 02:00:00" in (d / "daemon-daily-notes.timer").read_text()
        assert "OnFailure=daemon-daily-notes-failed.service" in (d / "daemon-daily-notes.service.d" / "onfailure.conf").read_text()
        assert (d / "daemon-daily-notes-failed.service").exists() and (d / "README.md").exists()
