"""Learned-relation promotion loop (memory/learned_relations.py).

2026-08-05: the fixed core vocabulary had no slot for care-team facts and the
owner had to hand-edit vocab after the miss was observed. Recurring invented
relations that survive the extraction gates are now tracked and auto-promoted
into the extractor prompt — with guards so the 08-02 junk-relation explosion
(630/696 single-use inventions) can't come back through this door.
"""
import json
from datetime import datetime, timedelta

import pytest

import memory.learned_relations as lr
from memory.learned_relations import LearnedRelationStore
from memory.llm_fact_extractor import CORE_RELATIONS, LLMFactExtractor


def _store(tmp_path):
    return LearnedRelationStore(path=str(tmp_path / "learned.json"))


def _days_ago(n):
    return datetime.now() - timedelta(days=n)


class TestRecordGuards:
    def test_core_relation_never_tracked(self, tmp_path):
        s = _store(tmp_path)
        assert not s.record("likes", "diet coke")
        assert s.promoted(min_days=0) == []

    def test_bad_shapes_rejected(self, tmp_path):
        s = _store(tmp_path)
        assert not s.record("Asked About The Thing", "x")
        assert not s.record("a_b_c_d_e", "x")  # > 3 tokens
        assert not s.record("", "x")
        assert not s.record("123_rel", "x")

    def test_ephemeral_relation_never_tracked(self, tmp_path):
        s = _store(tmp_path)
        # current_mood is transient state, not vocabulary
        assert not s.record("current_mood", "tired")

    def test_valid_invented_relation_tracked(self, tmp_path):
        s = _store(tmp_path)
        assert s.record("therapy_schedule", "biweekly with Harper")


class TestPromotion:
    def test_one_day_is_not_enough(self, tmp_path):
        s = _store(tmp_path)
        for _ in range(10):  # many uses, single day
            s.record("resume_strategy", "two versions")
        assert s.promoted(min_days=3) == []

    def test_recurrence_across_days_promotes(self, tmp_path):
        s = _store(tmp_path)
        for n in (3, 2, 1):
            s.record("resume_strategy", "two versions", when=_days_ago(n))
        assert s.promoted(min_days=3) == ["resume_strategy"]

    def test_promoted_cap(self, tmp_path):
        s = _store(tmp_path)
        for i in range(20):
            rel = f"rel_number_{chr(ord('a') + i)}"
            for n in (3, 2, 1):
                s.record(rel, "x", when=_days_ago(n))
        assert len(s.promoted(min_days=3)) == lr.MAX_PROMOTED


class TestPersistence:
    def test_roundtrip(self, tmp_path):
        path = str(tmp_path / "learned.json")
        s = LearnedRelationStore(path=path)
        for n in (3, 2, 1):
            s.record("therapy_schedule", "biweekly", when=_days_ago(n))
        s2 = LearnedRelationStore(path=path)
        assert s2.promoted(min_days=3) == ["therapy_schedule"]

    def test_corrupt_store_is_cold_start(self, tmp_path):
        path = tmp_path / "learned.json"
        path.write_text("{not json")
        s = LearnedRelationStore(path=str(path))
        assert s.promoted(min_days=0) == []
        assert s.record("therapy_schedule", "x")  # still usable

    def test_disabled_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LEARNED_RELATIONS_ENABLED", "0")
        s = _store(tmp_path)
        assert not s.record("therapy_schedule", "x")
        assert s.promoted(min_days=0) == []


class TestPromptWiring:
    def _extractor(self):
        return LLMFactExtractor(model_manager=None)

    def test_core_relations_rendered(self):
        prompt = self._extractor()._build_prompt(["I love hiking"])
        for rel in ("doctor_communication", "medication_name", "lives_in"):
            assert rel in prompt

    def test_promoted_relations_rendered(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lr, "_STORE_PATH", str(tmp_path / "learned.json"))
        monkeypatch.setattr(lr, "_store", None)
        s = lr.get_learned_relation_store()
        for n in (3, 2, 1):
            s.record("therapy_schedule", "biweekly", when=_days_ago(n))
        prompt = self._extractor()._build_prompt(["I love hiking"])
        assert "LEARNED relations" in prompt
        assert "therapy_schedule" in prompt

    def test_no_learned_line_when_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lr, "_STORE_PATH", str(tmp_path / "learned.json"))
        monkeypatch.setattr(lr, "_store", None)
        prompt = self._extractor()._build_prompt(["I love hiking"])
        assert "LEARNED relations" not in prompt


class TestCoveragePromptBuild:
    """2026-08-05 coverage fix: newest-first budget selection. 2026-09-02:
    Daemon responses are excluded from the prompt entirely."""

    def _extractor(self, max_chars=1200):
        return LLMFactExtractor(model_manager=None, max_input_chars=max_chars)

    def test_responses_excluded_from_prompt(self):
        pairs = [{"query": "short question", "response": "R" * 2000}]
        prompt = self._extractor(4000)._build_prompt(pairs)
        assert "short question" in prompt
        assert "R" * 50 not in prompt  # generated text is not extraction input

    def test_budget_drops_oldest_not_newest(self):
        pairs = [
            {"query": f"message number {i} " + "x" * 300, "response": ""}
            for i in range(10)
        ]
        prompt = self._extractor(1200)._build_prompt(pairs)
        assert "message number 9" in prompt, "newest turn must survive budget truncation"
        assert "message number 0" not in prompt, "oldest turns are the ones dropped"

    def test_chronological_order_preserved(self):
        pairs = [{"query": f"marker_{i}", "response": ""} for i in range(3)]
        prompt = self._extractor(4000)._build_prompt(pairs)
        assert prompt.index("marker_0") < prompt.index("marker_1") < prompt.index("marker_2")
