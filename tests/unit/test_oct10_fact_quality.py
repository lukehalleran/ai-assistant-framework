"""2026-10-10 fact quality (FOLLOWUPS L123 b+c; class: BC-75, BC-46).

(b) a PAST employer stored as current (`works_at=Mercer` from resume talk);
(c) junk graph-feeding triples: self-reference, chat-interjection objects,
    future-tense planned-act relations.
Everything is driven through THE deployed functions.
"""
from unittest.mock import MagicMock

import pytest

from memory.fact_extractor import (
    FactExtractor,
    _clean_triple,
    _is_junk_object,
    is_self_reference,
)
from memory.llm_fact_extractor import _normalize_triple
from memory.memory_storage import MemoryStorage
from memory.fact_source import (
    current_status_claim_supported,
    find_supporting_user_span,
    relation_is_future_planned,
)


def _t(rel, obj, subj="user"):
    return {"subject": subj, "relation": rel, "object": obj}


class TestCurrentStatusNeedsPresentTenseCue:
    def test_resume_talk_mints_no_current_employer(self):
        msgs = ["ok so we have project, mercer job and skill g2g"]
        assert find_supporting_user_span(_t("works_at", "Mercer"), msgs) is None

    def test_present_tense_statement_supported(self):
        ev = find_supporting_user_span(_t("works_at", "Northwind"), ["I work at Northwind now"])
        assert ev is not None and ev.anchor == "first_person"

    def test_present_perfect_is_still_current(self):
        msgs = ["I have worked at Northwind for three years"]
        assert find_supporting_user_span(_t("works_at", "Northwind"), msgs) is not None

    @pytest.mark.parametrize("text", [
        "I used to work at Northwind",
        "I worked at Northwind last summer",
        "I quit my job at Northwind",
        "Northwind was my first job",
        "my former employer was Northwind",
    ])
    def test_past_framing_is_not_current(self, text):
        assert find_supporting_user_span(_t("works_at", "Northwind"), [text]) is None

    def test_enrollment_family_shares_the_rule(self):
        assert find_supporting_user_span(_t("enrolled_in", "CS 6200"), ["I'm taking CS 6200 this fall"]) is not None
        assert find_supporting_user_span(_t("enrolled_in", "CS 6200"), ["CS 6200 class notes are done"]) is None

    def test_other_relations_unaffected(self):
        assert current_status_claim_supported("likes", "mercer job") is True
        assert current_status_claim_supported("dropped", "CS 6200") is True

    def test_regex_extractor_skips_past_employer(self):
        ex = FactExtractor(use_rebel=False, use_regex=True)
        past = ex._extract_with_regex("I worked at Deloitte.")
        assert not [f for f in past if f[1] == "works_at"]
        now = ex._extract_with_regex("I work at Deloitte.")
        assert [f for f in now if f[1] == "works_at" and f[2] == "deloitte"]


class TestGraphFeedingJunk:
    def test_self_reference(self):
        assert is_self_reference("user", "User")
        assert is_self_reference("User", "the user")
        assert not is_self_reference("user", "andrew")
        assert _clean_triple("user", "name", "User") is None

    def test_real_middle_name_still_passes(self):
        assert _clean_triple("user", "name", "andrew") == ("user", "name", "andrew")

    def test_future_planned_relation(self):
        assert relation_is_future_planned("will_do")
        assert relation_is_future_planned("will do")
        assert not relation_is_future_planned("would_like")
        assert not relation_is_future_planned("plans")
        assert _clean_triple("user", "will do", "shower") is None
        assert _is_junk_object("shower", "will_do")

    def test_ordinary_triples_unaffected(self):
        assert _clean_triple("user", "likes", "pizza") == ("user", "likes", "pizza")
        assert not _is_junk_object("Daemon", "works_on")


class TestLLMPathSelfReference:
    def test_self_reference_dropped_on_llm_path(self):
        assert _normalize_triple({"subject": "I", "relation": "name", "object": "User"}) is None

    def test_real_name_kept_on_llm_path(self):
        t = _normalize_triple({"subject": "I", "relation": "middle_name", "object": "andrew"})
        assert t is not None and t["object"].lower() == "andrew"


def _storage(resolve=None, ids=("user", "obj")):
    """Deployed _ingest_fact_to_graph against a fake graph/resolver."""
    st = MemoryStorage.__new__(MemoryStorage)
    st.graph_memory = MagicMock()
    st.graph_memory.get_entity.return_value = None
    st.entity_resolver = MagicMock()
    st.entity_resolver.resolve.return_value = resolve
    st.entity_resolver.resolve_or_create.side_effect = list(ids)
    return st


class TestIngestEdgeRules:
    def test_self_loop_never_created(self):
        # "User" resolves to the existing user node: object already known, so
        # the worthiness check is bypassed - the self-loop rule must not be.
        st = _storage(resolve="user", ids=("user", "user"))
        st._ingest_fact_to_graph("user", "name", "User", confidence=0.9)
        st.graph_memory.add_relation.assert_not_called()

    def test_ephemeral_relation_never_an_edge(self):
        st = _storage(resolve="shower", ids=("user", "shower"))
        st._ingest_fact_to_graph("user", "upcoming_appointment", "Dr Lee", confidence=0.9)
        st.graph_memory.add_relation.assert_not_called()

    def test_ordinary_edge_still_created(self):
        st = _storage(resolve="pizza", ids=("user", "pizza"))
        st._ingest_fact_to_graph("user", "likes", "pizza", confidence=0.9)
        st.graph_memory.add_relation.assert_called_once()
