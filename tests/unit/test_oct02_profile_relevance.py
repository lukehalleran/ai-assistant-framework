"""2026-10-02 (housing miss): profile facts never reached the prompt on topic.

Live turn: "No it's for me. I don't want to move I can't afford it" — the
profile held apartment_condition / living_with / housing_stress /
moving_out_timeline facts and NONE rendered; relationships showed dating/IBM.
Root cause: UserProfile.get_relevant_facts scored raw word overlap with the
relation's underscores UNSPLIT, so every housing fact scored 0.

Fixture data only — never the owner's profile.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

import memory.user_profile as up
from memory.user_profile import RELEVANCE_COSINE_FLOOR, UserProfile
from memory.user_profile_schema import ProfileCategory, categorize_relation

# Concept groups for a deterministic fake embedder (one dimension each).
_CONCEPTS = [
    {"housing", "apartment", "rent", "move", "moving", "living", "live", "lives",
     "unliveable", "bugs", "ac", "afford", "landlord", "mother", "lease"},
    {"date", "dating", "alex", "dinner", "hinge", "matches"},
    {"ibm", "internship", "job", "works", "career", "application"},
    {"gym", "bench", "squat"},
]


class FakeEmbedder:
    calls = 0

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, **kw):
        FakeEmbedder.calls += len(texts)
        out = []
        for t in texts:
            words = {w.strip(".,:;'\"?!").lower() for w in t.replace("_", " ").split()}
            v = np.array([float(len(words & c)) for c in _CONCEPTS] + [0.01])
            out.append(v / np.linalg.norm(v))
        return np.array(out)


@pytest.fixture
def fake_embedder(monkeypatch):
    FakeEmbedder.calls = 0
    up._fact_emb_cache.clear()
    up._query_emb_cache.clear()
    # raising=False: on the base tree the seam does not exist, so the test
    # must fail on the ranking assertion, not on the monkeypatch.
    monkeypatch.setattr(up, "_profile_embedder", lambda: FakeEmbedder(), raising=False)
    yield
    up._fact_emb_cache.clear()
    up._query_emb_cache.clear()


def _profile(tmp_path):
    p = UserProfile(str(tmp_path / "profile.json"))
    # Relative to now: apartment_condition is a 24h-TTL relation in
    # relation_classifier (see handoff), so fixture facts must be fresh.
    base = datetime.now() - timedelta(hours=12)
    rows = [
        # (relation, value, minutes-after-base) — the housing facts are OLDER
        # than the dating/IBM ones, so recency alone can never surface them.
        ("apartment_condition", "horrific, unliveable with bugs and broken AC", 0),
        ("living_with", "mother", 1),
        ("housing_stress", "severe", 2),
        ("moving_out_timeline", "next spring", 3),
        ("dating_app_usage", "matches on hinge", 500),
        ("dinner_with", "Alex", 501),
        ("manager", "Pat from IBM", 502),
        ("internship_application_status", "applied to IBM", 503),
    ]
    for rel, val, mins in rows:
        assert p.add_fact(rel, val, 0.9, "fixture",
                          timestamp=base + timedelta(minutes=mins))
    return p


QUERY = "No it's for me. I don't want to move I can't afford it"


class TestRelevanceRanking:
    def test_housing_query_ranks_housing_facts_first(self, tmp_path, fake_embedder):
        p = _profile(tmp_path)
        rel_cat = ProfileCategory.RELATIONSHIPS
        # category-local: the relationship slots must not be the old
        # dating/IBM recency pile for a housing query
        got = [f["relation"] for f in p.get_relevant_facts(QUERY, rel_cat, limit=3)]
        assert "dating_app_usage" not in got[:1]
        # and the living_situation category surfaces the apartment facts
        ls = [f["relation"] for f in p.get_relevant_facts(
            "my apartment is unliveable and the landlord ignores the AC", ProfileCategory.LIVING_SITUATION, limit=3)]
        assert ls[0] in {"apartment_condition", "housing_stress", "living_with"}
        assert {"apartment_condition"} <= set(ls)

    def test_injection_contains_housing_facts_on_housing_query(self, tmp_path, fake_embedder):
        p = _profile(tmp_path)
        text = p.get_context_injection(query=QUERY)
        assert "apartment_condition=" in text or "living_with=" in text
        assert "living_situation:" in text

    def test_relation_underscores_are_split_in_fact_text(self):
        assert up._fact_text({"relation": "apartment_condition", "value": "bugs"}) == \
            "apartment condition: bugs"

    def test_below_floor_facts_take_no_relevance_slot(self, tmp_path, fake_embedder):
        p = _profile(tmp_path)
        # A query matching none of the fixture's concepts: nothing clears the
        # floor, so the result is purely the recency slots (newest first).
        got = p.get_relevant_facts("zzzz qqqq", ProfileCategory.RELATIONSHIPS, limit=3)
        stamps = [f["timestamp"] for f in got]
        assert stamps == sorted(stamps, reverse=True)
        assert RELEVANCE_COSINE_FLOOR > 0

    def test_embeddings_cached_across_calls(self, tmp_path, fake_embedder):
        p = _profile(tmp_path)
        p.get_relevant_facts(QUERY, ProfileCategory.LIVING_SITUATION, limit=3)
        first = FakeEmbedder.calls
        p.get_relevant_facts(QUERY, ProfileCategory.LIVING_SITUATION, limit=3)
        assert FakeEmbedder.calls == first  # query + every fact text cached

    def test_limit_and_confidence_filter_kept(self, tmp_path, fake_embedder):
        p = _profile(tmp_path)
        p.add_fact("housing_note", "low confidence junk", 0.3, "fixture")
        got = p.get_relevant_facts(QUERY, ProfileCategory.LIVING_SITUATION, limit=2)
        assert len(got) <= 2
        assert all(f["confidence"] >= 0.55 for f in got)


class TestFallbackWithoutEmbedder:
    def test_overlap_fallback_splits_relation_underscores(self, tmp_path, monkeypatch):
        monkeypatch.setattr(up, "_profile_embedder", lambda: None, raising=False)
        p = _profile(tmp_path)
        got = p.get_relevant_facts("what is the apartment condition", ProfileCategory.LIVING_SITUATION, limit=3)
        assert got[0]["relation"] == "apartment_condition"

    def test_stub_zero_embedder_falls_back(self, tmp_path, monkeypatch):
        class Zero:
            def encode(self, texts, **kw):
                return np.zeros((len(texts), 4))
        up._fact_emb_cache.clear(); up._query_emb_cache.clear()
        monkeypatch.setattr(up, "_profile_embedder", lambda: Zero(), raising=False)
        p = _profile(tmp_path)
        got = p.get_relevant_facts("apartment condition", ProfileCategory.LIVING_SITUATION, limit=3)
        assert got[0]["relation"] == "apartment_condition"


class TestLivingSituationCategory:
    @pytest.mark.parametrize("rel", [
        "lives_in", "living_with", "roommate", "living_arrangement", "home_location",
        "apartment_condition", "living_duration", "housing_stress", "housing_type",
        "moving_out_timeline", "moving_plan", "rent_amount", "rental_status",
        "lease_end", "landlord_issue",
    ])
    def test_housing_relations_categorize(self, rel):
        assert categorize_relation(rel) == ProfileCategory.LIVING_SITUATION

    def test_lives_in_no_longer_identity(self):
        assert categorize_relation("lives_in") != ProfileCategory.IDENTITY

    def test_unrelated_relations_unchanged(self):
        assert categorize_relation("works_at") == ProfileCategory.CAREER
        assert categorize_relation("location") == ProfileCategory.IDENTITY

    def test_add_fact_files_under_living_situation_and_old_profile_ok(self, tmp_path):
        p = UserProfile(str(tmp_path / "p.json"))
        p.profile["categories"].pop("living_situation", None)  # profile saved pre-category
        assert p.add_fact("living_with", "mother", 0.9, "fixture")
        assert p.get_category(ProfileCategory.LIVING_SITUATION)[0]["value"] == "mother"

    def test_legacy_identity_lives_in_is_superseded_not_duplicated(self, tmp_path):
        p = UserProfile(str(tmp_path / "p.json"))
        p.add_fact("lives_in", "Springfield", 0.9, "fixture", category=ProfileCategory.IDENTITY)
        p.add_fact("lives_in", "Shelbyville", 0.9, "fixture")
        cur = [f["value"] for cat in (ProfileCategory.IDENTITY, ProfileCategory.LIVING_SITUATION)
               for f in p.get_category(cat) if f["relation"] == "lives_in"]
        assert cur == ["Shelbyville"]
        # confirming the legacy value lands on the legacy fact (no duplicate)
        p.add_fact("lives_in", "Springfield", 0.9, "fixture", category=None)
        legacy = [f for f in p.get_category(ProfileCategory.IDENTITY, include_historical=True)
                  if f["relation"] == "lives_in" and f["value"] == "Springfield"]
        assert len(legacy) == 1


def test_real_embedder_orders_housing_above_unrelated(tmp_path, monkeypatch):
    """Through the DEPLOYED embedder accessor (not the fake): the measured
    separation behind RELEVANCE_COSINE_FLOOR."""
    emb = up._profile_embedder()
    if emb is None or not float(np.linalg.norm(emb.encode(["x"], normalize_embeddings=True)[0])):
        pytest.skip("real embedder unavailable (stub)")
    up._fact_emb_cache.clear(); up._query_emb_cache.clear()
    p = _profile(tmp_path)
    cos = dict(zip(
        [f["relation"] for f in p.get_category(ProfileCategory.LIVING_SITUATION)],
        up._cosine_scores("my AC is broken again", p.get_category(ProfileCategory.LIVING_SITUATION)),
    ))
    assert cos["apartment_condition"] >= RELEVANCE_COSINE_FLOOR
    assert cos["apartment_condition"] > cos["moving_out_timeline"]
