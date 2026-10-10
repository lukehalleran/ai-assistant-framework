"""2026-10-10 (FOLLOWUPS L91): a `self_harm` profile fact (an old crisis
disclosure) surfaced in [USER PROFILE] on a student-loans turn. Nothing gated
sensitive facts on relevance. Fix: relation_classifier.is_sensitive_relation
(one categorized set) + a selection gate in UserProfile.get_relevant_facts,
driven per turn by the prompt gatherer (elevated tone / explicit history ask /
topical relation).

Everything runs through the DEPLOYED selection: the real
MemoryRetrievalMixin.get_user_profile_context over a real UserProfile.
Fixture data only — never the owner's profile.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

import memory.relation_classifier as rc
import memory.user_profile as up
from core.prompt.gatherer_memory import MemoryRetrievalMixin
from memory.user_profile import UserProfile

LOANS = "can you help me figure out my student loans repayment plan"
SENSITIVE_VALUE = "SYNTH_CRISIS_DISCLOSURE_2019"

_CONCEPTS = [
    {"loan", "loans", "student", "repayment", "plan", "payment", "debt", "balance"},
    {"self", "harm", "hurt", "myself", "suicide", "crisis", "disclosure"},
    {"gym", "squat", "bench"},
]


class FakeEmbedder:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, **kw):
        out = []
        for t in texts:
            words = {w.strip(".,:;'\"?!").lower() for w in t.replace("_", " ").split()}
            v = np.array([float(len(words & c)) for c in _CONCEPTS] + [0.01])
            out.append(v / np.linalg.norm(v))
        return np.array(out)


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    up._fact_emb_cache.clear()
    up._query_emb_cache.clear()
    monkeypatch.setattr(up, "_profile_embedder", lambda: FakeEmbedder())
    yield
    up._fact_emb_cache.clear()
    up._query_emb_cache.clear()


def _profile(tmp_path):
    p = UserProfile(str(tmp_path / "profile.json"))
    base = datetime.now() - timedelta(hours=2)
    # The sensitive fact is the NEWEST, so recency alone would surface it.
    rows = [
        ("loan_servicer", "synthetic servicer", 0),
        ("gym_schedule", "monday squat day", 1),
        ("self_harm", SENSITIVE_VALUE, 2),
    ]
    for rel, val, mins in rows:
        assert p.add_fact(rel, val, 0.9, "fixture", timestamp=base + timedelta(minutes=mins))
    return p


def _host(profile, distress=False):
    g = MemoryRetrievalMixin.__new__(MemoryRetrievalMixin)
    g.memory_id_map = {}
    g.user_profile = profile
    g._distress_active = distress
    return g


async def _inject(profile, query, distress=False):
    return await _host(profile, distress).get_user_profile_context(query, max_tokens=1000)


class TestRelationClassifier:
    @pytest.mark.parametrize("rel", [
        "self_harm", "suicidal_ideation", "trauma_history", "previous_addiction",
        "substance_abuse", "sexual_health", "psychiatric_diagnosis", "Self Harm",
        "childhood_trauma_event", "alcohol_use_disorder", "hiv_status",
        # live-profile relation names (referee check): substance + psychiatric-symptom families
        "drug_experience", "experience_with_psychedelics", "hallucination",
        "hallucination_experience", "experience_with_hallucination",
        "addictive_personality", "previous_substance_use", "psychological_addiction",
    ])
    def test_sensitive(self, rel):
        assert rc.is_sensitive_relation(rel)

    @pytest.mark.parametrize("rel", [
        "", None, "likes", "condition", "diagnosis", "symptom", "medication_name",
        "drank_alcohol", "gym_schedule", "sleep_quality", "student_status", "loan_servicer",
        # routine scheduling / ordinary health state: deliberately NOT sensitive
        "therapy_appointment", "therapy_schedule", "therapist_name", "has_therapist",
        "psychiatrist_appointment", "alcohol_consumed", "opinion_on_alcohol",
        "anxiety_level", "anxiety_trigger", "sleep_disorders",
    ])
    def test_not_sensitive(self, rel):
        assert not rc.is_sensitive_relation(rel)

    def test_history_ask_phrases_derive_from_gate_list(self):
        from core.agentic.gate import MEMORY_KEYWORDS
        from core.prompt.gatherer_memory import PROFILE_HISTORY_ASK_PHRASES
        assert set(PROFILE_HISTORY_ASK_PHRASES) <= set(MEMORY_KEYWORDS)


@pytest.mark.asyncio
class TestGatedSelection:
    async def test_loans_turn_omits_sensitive_fact(self, tmp_path):
        out = await _inject(_profile(tmp_path), LOANS)
        assert SENSITIVE_VALUE not in out and "self_harm" not in out
        assert "loan_servicer" in out  # the relevant non-sensitive fact is intact

    async def test_non_sensitive_facts_unaffected(self, tmp_path):
        out = await _inject(_profile(tmp_path), "what is my gym schedule squat")
        assert "gym_schedule" in out and SENSITIVE_VALUE not in out

    async def test_elevated_tone_turn_includes_it(self, tmp_path):
        out = await _inject(_profile(tmp_path), LOANS, distress=True)
        assert SENSITIVE_VALUE in out

    async def test_topical_query_includes_it_by_cosine(self, tmp_path):
        out = await _inject(_profile(tmp_path), "I keep thinking about how I hurt myself back then")
        assert SENSITIVE_VALUE in out

    async def test_topical_query_includes_it_by_relation_word_without_embedder(self, tmp_path, monkeypatch):
        monkeypatch.setattr(up, "_profile_embedder", lambda: None)
        out = await _inject(_profile(tmp_path), "tell me about my self harm history")
        assert SENSITIVE_VALUE in out
        out = await _inject(_profile(tmp_path), LOANS)
        assert SENSITIVE_VALUE not in out

    async def test_explicit_history_ask_includes_it(self, tmp_path):
        out = await _inject(_profile(tmp_path), "what do you know about me?")
        assert SENSITIVE_VALUE in out

    async def test_loose_what_are_my_does_not_open_the_gate(self, tmp_path):
        out = await _inject(_profile(tmp_path), "what are my student loans repayment options")
        assert SENSITIVE_VALUE not in out

    async def test_nothing_is_deleted_and_flag_does_not_leak(self, tmp_path):
        p = _profile(tmp_path)
        before = {f["relation"] for cat in p.get_all_facts().values() for f in cat}
        await _inject(p, LOANS)
        after = {f["relation"] for cat in p.get_all_facts().values() for f in cat}
        assert before == after and "self_harm" in after
        assert up.SENSITIVE_OPEN.get() is None
        # a direct, ungated call (shutdown / CLI path) is unchanged
        assert SENSITIVE_VALUE in p.get_context_injection(query=LOANS)

    async def test_held_back_log_is_count_only(self, tmp_path, monkeypatch):
        p = _profile(tmp_path)  # built first: add_fact logs its own lines
        lines = []
        monkeypatch.setattr(up.logger, "info", lambda msg, *a, **k: lines.append(str(msg)))
        await _inject(p, LOANS)
        held = [l for l in lines if "held back" in l]
        assert held and "1 sensitive" in held[0]
        assert not any(SENSITIVE_VALUE in l or "self_harm" in l for l in lines)
