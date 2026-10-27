"""Employment statements reach the profile through the shutdown LLM path.

Incident 2026-10-12 (class: BC-04): "I just started a new job at Northwind as a
data analyst" produced no per-turn fact (the per-turn extractor is regex-only by
design; the owner rule forbids adding a regex for this).  The employer arrives via
the shutdown LLM pass, so this pins the pieces that make that path work: the
relations are in the prompt's CORE_RELATIONS, ``job_title`` canonicalizes to
``occupation``, and THE deployed provenance join supports both triples.
Clean AND wrapped forms.
"""
import pytest

from memory.entity_resolver import normalize_relation
from memory.fact_source import find_supporting_user_span
from memory.llm_fact_extractor import CORE_RELATIONS

TEXT = "I just started a new job at Northwind as a data analyst"


def test_core_relations_carry_employment():
    assert "works_at" in CORE_RELATIONS and "occupation" in CORE_RELATIONS


def test_job_title_canonicalizes_to_a_core_relation():
    assert normalize_relation("job_title") in CORE_RELATIONS


@pytest.mark.parametrize("relation,obj", [("works_at", "Northwind"), ("occupation", "data analyst")])
def test_join_supports_the_employment_triples(relation, obj):
    for text in (TEXT, "   " + TEXT + "\n"):
        triple = {"subject": "user", "relation": relation, "object": obj}
        assert find_supporting_user_span(triple, [{"query": text}]) is not None, text
