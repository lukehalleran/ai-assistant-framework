"""Query-rewriter synonym expansion is sense-gated (R1 item 1).

Incident 2026-10-25: "what's been going on with the daemon codebase last couple
days work" was expanded with "job shift employment" and pulled employment
memories into a code question; a [test]-wrapped multi-line probe was keyworded
with its envelope tag. class: BC-05
Drives the deployed utils.query_rewriter.rewrite_query.
"""
import pytest

from utils.query_rewriter import rewrite_query

EMPLOYMENT = {"job", "shift", "employment"}
CODE_Q = "what's been going on with the quillfern codebase last couple days work"


def _wrapped(q):
    return f"[test]\n  {q}\n[/test]"


def _indented(q):
    return "\n".join("    " + part for part in q.replace(" last", "\nlast").split("\n"))


@pytest.mark.parametrize("form", [lambda q: q, _wrapped, _indented])
def test_code_question_gets_no_employment_terms(form):
    out = rewrite_query(form(CODE_Q), use_topic_extraction=False).split()
    assert not EMPLOYMENT & set(out)
    assert "codebase" in out and "work" in out  # user's own words stay


def test_topic_supplied_code_anchor_gates_expansion():
    out = rewrite_query("how is work going", use_topic_extraction=True,
                        topics=["quillfern codebase"]).split()
    assert not EMPLOYMENT & set(out)


@pytest.mark.parametrize("form", [lambda q: q, _wrapped, _indented])
def test_employment_question_still_expands(form):
    out = rewrite_query(form("my shift at work ran late"), use_topic_extraction=False).split()
    assert EMPLOYMENT <= set(out)


def test_code_group_still_expands_in_software_context():
    out = rewrite_query("debug the quillfern codebase", use_topic_extraction=False).split()
    assert {"coding", "programming", "refactor"} <= set(out)


def test_wrapped_three_line_query_rewritten_whole():
    raw = "[test]\nTamsin asked about the garden\nand the pruning schedule\nfor spring\n[/test]"
    out = rewrite_query(raw, use_topic_extraction=False).split()
    assert "test" not in out
    for w in ("tamsin", "garden", "pruning", "schedule", "spring"):
        assert w in out
