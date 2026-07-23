"""
Unit tests for knowledge/literature_oracle.py — all offline (LLM + network mocked).

The load-bearing invariants under test:
  * adjudication parse failures are LOUD (PARSE_FAILURE after exactly 1 retry, never a default)
  * the evidence rule is code-enforced (downgrade to UNCERTAIN on missing/non-verbatim/
    over-length/hallucinated-URL citations)
  * escalation fires at most once, only under its trigger, and not when disabled
  * llm_prior_verdict is logged but never drives the verdict
  * blinding: gate-derived fields can never reach a prompt
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from knowledge.literature_oracle import (
    Citation,
    EvidenceItem,
    LiteratureOracle,
    OracleRecord,
    OracleVerdict,
    QueryPlan,
    _extract_json,
    _norm_text,
    append_record,
    check_budget,
    load_done_ids,
)

QUERY_JSON = json.dumps({
    "queries": ["shannon entropy boltzmann gibbs entropy equivalence",
                "information theory statistical mechanics duality theorem"],
    "canonical_idx": 0,
    "relationship_idx": 1,
})

ARXIV_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678v1</id>
    <title>Entropy in Information Theory and Statistical Mechanics</title>
    <summary>We review the formal equivalence between Shannon entropy and Gibbs entropy.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2345.6789v2</id>
    <title>A Second Paper</title>
    <summary>Another abstract body.</summary>
  </entry>
</feed>"""


def _mk_oracle(mm=None, wsm=None, **kw):
    kw.setdefault("model_name", "test-model")
    return LiteratureOracle(mm or MagicMock(), web_search_manager=wsm, **kw)


def _ev(url="http://arxiv.org/abs/1234.5678v1", text="the maximum entropy principle "
        "recovers statistical mechanics from information theory", source="arxiv", **kw):
    return EvidenceItem(source=source, url=url, title=kw.pop("title", "Paper"), text=text, **kw)


def _adj_json(verdict="DOCUMENTED_EXACT", citations=None, prior="", name="", **extra):
    d = {"verdict": verdict, "name": name, "citations": citations or [],
         "prior_verdict": prior, "prior_citation_claim": "", "reasoning": "because"}
    d.update(extra)
    return json.dumps(d)


# ---------------------------------------------------------------- query plan
class TestQueryPlan:
    async def test_code_enforcement_clamps_and_truncates(self):
        mm = MagicMock()
        six = [f"query number {i} " + ("x" * 900 if i == 2 else "") for i in range(6)]
        mm.generate_once = AsyncMock(return_value=json.dumps(
            {"queries": six, "canonical_idx": 0, "relationship_idx": 1}))
        plan = await _mk_oracle(mm).generate_queries("a", "b", "claim")
        assert len(plan.queries) == 4
        assert all(len(q) <= 400 for q in plan.queries)
        assert not plan.fallback_used

    async def test_fallback_templates_on_garbage(self):
        mm = MagicMock()
        mm.generate_once = AsyncMock(return_value="I think you should search for stuff!")
        plan = await _mk_oracle(mm).generate_queries("percolation", "renormalization", "claim")
        assert plan.fallback_used
        assert len(plan.queries) == 2
        assert "percolation" in plan.queries[0] and "renormalization" in plan.queries[0]
        assert "duality" in plan.queries[1] or "isomorphism" in plan.queries[1]

    async def test_fallback_on_llm_exception(self):
        mm = MagicMock()
        mm.generate_once = AsyncMock(side_effect=RuntimeError("api down"))
        plan = await _mk_oracle(mm).generate_queries("a", "b", "c")
        assert plan.fallback_used

    async def test_out_of_range_indices_default(self):
        mm = MagicMock()
        mm.generate_once = AsyncMock(return_value=json.dumps(
            {"queries": ["q1", "q2"], "canonical_idx": 9, "relationship_idx": -3}))
        plan = await _mk_oracle(mm).generate_queries("a", "b", "c")
        assert plan.canonical_idx == 0
        assert plan.relationship_idx == 1


# ---------------------------------------------------------------- json utils
class TestJsonUtils:
    def test_extract_json_strips_fences_and_prose(self):
        raw = 'Sure! Here you go:\n```json\n{"verdict": "NOT_FOUND", "citations": []}\n```\nHope that helps.'
        assert _extract_json(raw)["verdict"] == "NOT_FOUND"

    def test_extract_json_none_on_garbage(self):
        assert _extract_json("no braces here") is None
        assert _extract_json("") is None
        assert _extract_json('{"unclosed": ') is None

    def test_norm_text_unicode_and_whitespace(self):
        assert _norm_text("The  “Fluctuation–Dissipation”\n Theorem") == \
            'the "fluctuation-dissipation" theorem'


# ------------------------------------------------------------- evidence rule
class TestEvidenceRule:
    def test_downgrade_no_citation(self):
        parsed = json.loads(_adj_json("DOCUMENTED_EXACT", citations=[]))
        verdict, cits, downgraded = LiteratureOracle.enforce_evidence_rule(parsed, [_ev()])
        assert verdict == OracleVerdict.UNCERTAIN
        assert downgraded and not cits

    def test_downgrade_passage_not_verbatim(self):
        parsed = json.loads(_adj_json("DOCUMENTED_EXACT", citations=[
            {"url": "http://arxiv.org/abs/1234.5678v1",
             "supporting_passage": "this passage was never in the evidence",
             "rationale": "r"}]))
        verdict, _, downgraded = LiteratureOracle.enforce_evidence_rule(parsed, [_ev()])
        assert verdict == OracleVerdict.UNCERTAIN and downgraded

    def test_downgrade_hallucinated_url(self):
        ev = _ev()
        parsed = json.loads(_adj_json("DOCUMENTED_EXACT", citations=[
            {"url": "http://example.com/other", "supporting_passage": ev.text[:50], "rationale": "r"}]))
        verdict, _, downgraded = LiteratureOracle.enforce_evidence_rule(parsed, [ev])
        assert verdict == OracleVerdict.UNCERTAIN and downgraded

    def test_valid_verbatim_citation_survives_with_case_and_whitespace_noise(self):
        ev = _ev(text="The Maximum Entropy   principle recovers statistical mechanics")
        parsed = json.loads(_adj_json("DOCUMENTED_ADJACENT", citations=[
            {"url": ev.url, "supporting_passage": "the maximum entropy principle recovers",
             "rationale": "r"}]))
        verdict, cits, downgraded = LiteratureOracle.enforce_evidence_rule(parsed, [ev])
        assert verdict == OracleVerdict.DOCUMENTED_ADJACENT
        assert len(cits) == 1 and not downgraded

    def test_passage_word_limit_boundary(self):
        words_40 = " ".join(f"w{i}" for i in range(40))
        words_41 = " ".join(f"w{i}" for i in range(41))
        ev = _ev(text=f"prefix {words_41} suffix")
        ok = json.loads(_adj_json("DOCUMENTED_EXACT", citations=[
            {"url": ev.url, "supporting_passage": words_40, "rationale": "r"}]))
        verdict, cits, _ = LiteratureOracle.enforce_evidence_rule(ok, [ev])
        assert verdict == OracleVerdict.DOCUMENTED_EXACT and len(cits) == 1
        too_long = json.loads(_adj_json("DOCUMENTED_EXACT", citations=[
            {"url": ev.url, "supporting_passage": words_41, "rationale": "r"}]))
        verdict, _, downgraded = LiteratureOracle.enforce_evidence_rule(too_long, [ev])
        assert verdict == OracleVerdict.UNCERTAIN and downgraded

    def test_non_documented_verdicts_pass_through(self):
        for v in ("NOT_FOUND", "UNCERTAIN"):
            parsed = json.loads(_adj_json(v))
            verdict, cits, downgraded = LiteratureOracle.enforce_evidence_rule(parsed, [_ev()])
            assert verdict == OracleVerdict(v) and not downgraded and not cits

    def test_passage_may_come_from_title(self):
        ev = _ev(title="The Data Rate Theorem for Control Under Communication Constraints", text="body")
        parsed = json.loads(_adj_json("DOCUMENTED_EXACT", citations=[
            {"url": ev.url, "supporting_passage": "data rate theorem for control", "rationale": "r"}]))
        verdict, cits, _ = LiteratureOracle.enforce_evidence_rule(parsed, [ev])
        assert verdict == OracleVerdict.DOCUMENTED_EXACT and len(cits) == 1


# -------------------------------------------------------------- adjudication
class TestAdjudication:
    async def test_parse_failure_after_exactly_one_retry(self):
        mm = MagicMock()
        mm.generate_once = AsyncMock(return_value="")  # empty visible content, both attempts
        parsed, raw, failures = await _mk_oracle(mm).adjudicate("a", "b", "c", [_ev()])
        assert parsed is None and failures == 2
        assert mm.generate_once.await_count == 2  # exactly one retry, no third attempt

    async def test_retry_recovers(self):
        mm = MagicMock()
        mm.generate_once = AsyncMock(side_effect=["garbage", _adj_json("NOT_FOUND")])
        parsed, _, failures = await _mk_oracle(mm).adjudicate("a", "b", "c", [_ev()])
        assert parsed["verdict"] == "NOT_FOUND" and failures == 1

    async def test_unrecognized_verdict_is_parse_failure(self):
        mm = MagicMock()
        mm.generate_once = AsyncMock(return_value=json.dumps({"verdict": "MAYBE_KINDA"}))
        parsed, _, failures = await _mk_oracle(mm).adjudicate("a", "b", "c", [_ev()])
        assert parsed is None and failures == 2

    def test_verdict_first_knob_flips_schema_order(self):
        ev = [_ev()]
        p_first = _mk_oracle(verdict_first=True)._adj_prompt("a", "b", "c", ev)
        p_last = _mk_oracle(verdict_first=False)._adj_prompt("a", "b", "c", ev)
        assert p_first.rindex('"verdict"') < p_first.rindex('"reasoning"')
        assert p_last.rindex('"reasoning"') < p_last.rindex('"verdict"')

    def test_blinding_no_gate_fields_in_prompts(self):
        ev = [_ev()]
        oracle = _mk_oracle()
        adj = oracle._adj_prompt("neural network", "control theory", "a structural claim", ev)
        from knowledge.literature_oracle import _QUERY_PROMPT
        for banned in ("composite", "coherence_level", "ACCEPT", "status", "novelty"):
            assert banned not in adj
            assert banned not in _QUERY_PROMPT


# -------------------------------------------------------------------- arXiv
class TestArxiv:
    def test_xml_parse_fixture(self):
        items = LiteratureOracle._parse_arxiv_atom(ARXIV_FIXTURE, "q")
        assert len(items) == 2
        assert items[0].source == "arxiv"
        assert items[0].url == "http://arxiv.org/abs/1234.5678v1"
        assert "Shannon entropy" in items[0].text

    def test_xml_parse_malformed_returns_empty(self):
        assert LiteratureOracle._parse_arxiv_atom("<not <valid xml", "q") == []
        assert LiteratureOracle._parse_arxiv_atom(
            '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>', "q") == []

    async def test_politeness_spacing(self):
        oracle = _mk_oracle()
        oracle._arxiv_last_call = time.monotonic()  # a request "just happened"
        resp = MagicMock(status_code=200, text=ARXIV_FIXTURE)
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=MagicMock(get=AsyncMock(return_value=resp)))
        client.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=client), \
             patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            items = await oracle.retrieve_arxiv("some query")
        assert len(items) == 2
        waits = [c.args[0] for c in mock_sleep.await_args_list]
        assert any(2.5 < w <= 3.11 for w in waits), f"expected ~3.1s politeness wait, got {waits}"

    async def test_retrieval_failure_returns_empty_not_parse_failure(self):
        oracle = _mk_oracle()
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=MagicMock(
            get=AsyncMock(side_effect=RuntimeError("net down"))))
        client.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=client), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            items = await oracle.retrieve_arxiv("q")
        assert items == []


# -------------------------------------------------------------------- merge
class TestMerge:
    def test_dedupe_cap_order_and_extract_preference(self):
        arxiv = [_ev(url=f"http://arxiv.org/abs/{i}", source="arxiv") for i in range(3)]
        quick = [EvidenceItem(source="tavily_quick", url="http://x.com/page", title="t",
                              text="snippet", score=0.9)]
        extract = [EvidenceItem(source="tavily_extract", url="http://x.com/page", title="t",
                                text="full extracted content", score=0.9)]
        filler = [EvidenceItem(source="tavily_quick", url=f"http://y.com/{i}", title="t",
                               text="s", score=0.1 * i) for i in range(8)]
        merged = LiteratureOracle.merge_evidence(arxiv, quick, extract, filler)
        assert len(merged) == 8  # capped
        assert all(e.source == "arxiv" for e in merged[:3])  # arXiv first
        xcom = [e for e in merged if e.url == "http://x.com/page"]
        assert len(xcom) == 1 and xcom[0].source == "tavily_extract"  # extract beat snippet

    def test_arxiv_slot_reservation_web_never_crowded_out(self):
        arxiv = [_ev(url=f"http://arxiv.org/abs/{i}", source="arxiv") for i in range(8)]
        web = [EvidenceItem(source="tavily_quick", url=f"http://w.com/{i}", title="t",
                            text="s", score=1.0 - 0.1 * i) for i in range(6)]
        merged = LiteratureOracle.merge_evidence(arxiv, web)
        assert len(merged) == 8
        assert sum(1 for e in merged if e.source == "arxiv") == 4  # reserved slots hold
        assert sum(1 for e in merged if e.source == "tavily_quick") == 4

    def test_arxiv_backfills_when_web_sparse(self):
        arxiv = [_ev(url=f"http://arxiv.org/abs/{i}", source="arxiv") for i in range(8)]
        web = [EvidenceItem(source="tavily_quick", url="http://w.com/0", title="t",
                            text="s", score=0.9)]
        merged = LiteratureOracle.merge_evidence(arxiv, web)
        assert len(merged) == 8
        assert sum(1 for e in merged if e.source == "arxiv") == 7  # backfill


# ---------------------------------------------------------------- score path
def _score_mocks(adj_responses, tavily_side_effect=None):
    """mm with [query_gen] + adj_responses; oracle with arXiv stubbed empty."""
    mm = MagicMock()
    mm.generate_once = AsyncMock(side_effect=[QUERY_JSON, *adj_responses])
    wsm = MagicMock()
    oracle = _mk_oracle(mm, wsm=wsm)
    oracle.retrieve_arxiv = AsyncMock(return_value=[])
    oracle.retrieve_tavily = AsyncMock(
        side_effect=tavily_side_effect or [([], 0.0)] * 10)
    return mm, oracle


class TestScoreCandidate:
    async def test_parse_failure_verdict_recorded_loudly(self):
        mm, oracle = _score_mocks(["", ""])
        rec = await oracle.score_candidate("c1", "a", "b", "claim", run_id="r1")
        assert rec.verdict == OracleVerdict.PARSE_FAILURE
        assert rec.parse_failures == 2

    async def test_prior_logged_but_never_drives_verdict(self):
        mm, oracle = _score_mocks([_adj_json("NOT_FOUND", prior="DOCUMENTED_EXACT")])
        oracle.escalate = False
        rec = await oracle.score_candidate("c1", "a", "b", "claim", run_id="r1")
        assert rec.verdict == OracleVerdict.NOT_FOUND
        assert rec.llm_prior_verdict == "DOCUMENTED_EXACT"

    async def test_escalation_fires_once_and_recovers(self):
        ev_url = "http://x.com/extract"
        passage = "backpropagation is the adjoint method"
        quick = ([EvidenceItem(source="tavily_quick", url="http://x.com/q", title="t",
                               text="thin snippet", score=0.5)], 1.0)
        extract = ([EvidenceItem(source="tavily_extract", url=ev_url, title="t",
                                 text=f"long page where {passage} appears verbatim", score=0.9)], 2.0)
        adj1 = _adj_json("DOCUMENTED_EXACT", citations=[
            {"url": "http://nowhere.com", "supporting_passage": "x", "rationale": "r"}])
        adj2 = _adj_json("DOCUMENTED_EXACT", citations=[
            {"url": ev_url, "supporting_passage": passage, "rationale": "r"}])
        # QUERY_JSON has 2 queries -> 2 QUICK calls, then the STANDARD escalation call
        mm, oracle = _score_mocks([adj1, adj2],
                                  tavily_side_effect=[quick, quick, extract])
        rec = await oracle.score_candidate("c1", "a", "b", "claim", run_id="r1")
        assert rec.escalated
        assert rec.verdict == OracleVerdict.DOCUMENTED_EXACT
        assert not rec.downgraded_from_prior
        from knowledge.web_search_manager import WebSearchDepth
        standard_calls = [c for c in oracle.retrieve_tavily.await_args_list
                          if c.kwargs.get("depth") == WebSearchDepth.STANDARD]
        assert len(standard_calls) == 1
        assert mm.generate_once.await_count == 3  # query + adj1 + adj2, no more
        assert rec.credits_used == pytest.approx(4.0)

    async def test_escalation_at_most_once_final_uncertain(self):
        adj_bad = _adj_json("DOCUMENTED_EXACT", citations=[])
        mm, oracle = _score_mocks([adj_bad, adj_bad])
        rec = await oracle.score_candidate("c1", "a", "b", "claim", run_id="r1")
        assert rec.escalated
        assert rec.verdict == OracleVerdict.UNCERTAIN
        assert rec.downgraded_from_prior
        assert mm.generate_once.await_count == 3  # never a third adjudication

    async def test_no_escalation_when_disabled(self):
        adj_bad = _adj_json("DOCUMENTED_EXACT", citations=[])
        mm, oracle = _score_mocks([adj_bad])
        oracle.escalate = False
        rec = await oracle.score_candidate("c1", "a", "b", "claim", run_id="r1")
        assert not rec.escalated
        assert rec.verdict == OracleVerdict.UNCERTAIN and rec.downgraded_from_prior
        assert mm.generate_once.await_count == 2  # query + single adjudication

    async def test_honest_uncertain_with_documented_prior_escalates(self):
        adj1 = _adj_json("UNCERTAIN", prior="DOCUMENTED_EXACT")
        adj2 = _adj_json("NOT_FOUND")
        mm, oracle = _score_mocks([adj1, adj2])
        rec = await oracle.score_candidate("c1", "a", "b", "claim", run_id="r1")
        assert rec.escalated
        assert rec.verdict == OracleVerdict.NOT_FOUND


# ------------------------------------------------------------- JSONL + budget
class TestPersistence:
    def test_jsonl_roundtrip_and_resume_with_truncated_tail(self, tmp_path):
        path = str(tmp_path / "runs" / "results.jsonl")
        for cid in ("c1", "c2"):
            append_record(path, OracleRecord(
                run_id="r", candidate_id=cid, concept_a="a", concept_b="b", claim_text="t"))
        with open(path, "a") as f:
            f.write('{"candidate_id": "c3", "trunc')  # interrupted write
        assert load_done_ids(path) == {"c1", "c2"}
        with open(path) as f:
            row = json.loads(f.readline())
        assert row["verdict"] == "UNCERTAIN" and row["candidate_id"] == "c1"

    def test_load_done_ids_missing_file(self, tmp_path):
        assert load_done_ids(str(tmp_path / "nope.jsonl")) == set()

    def test_budget_guard_values(self):
        limiter = MagicMock()
        limiter.get_remaining_credits.return_value = 4.0
        assert check_budget(limiter) == 4.0
        from knowledge.literature_oracle import ORACLE_WORST_CASE_CREDITS_PER_CANDIDATE
        assert check_budget(limiter) < ORACLE_WORST_CASE_CREDITS_PER_CANDIDATE  # driver must stop
        broken = MagicMock()
        broken.get_remaining_credits.side_effect = RuntimeError("no state")
        assert check_budget(broken) == 0.0


class TestCitationModel:
    def test_citation_word_cap_validator(self):
        Citation(url="u", supporting_passage=" ".join(["w"] * 40))
        with pytest.raises(ValueError):
            Citation(url="u", supporting_passage=" ".join(["w"] * 41))


def test_append_record_bare_filename(tmp_path, monkeypatch):
    """append_record on a bare filename (no directory component) must not raise
    FileNotFoundError from os.makedirs('')."""
    from knowledge.literature_oracle import append_record, OracleRecord
    monkeypatch.chdir(tmp_path)
    rec = OracleRecord.model_construct(candidate_id="x")
    try:
        append_record("results.jsonl", rec)
    except FileNotFoundError:
        raise AssertionError("makedirs('') guard missing")
    assert (tmp_path / "results.jsonl").exists()
