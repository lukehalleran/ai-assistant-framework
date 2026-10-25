"""Sampling sheet for the log-only checkers (lane MEAS, oct25, 2026-10-24).

class: BC-58 (unmeasured guard). Synthetic fixture logs only (invented text).
"""
import importlib.util
import json
import os

_SPEC = importlib.util.spec_from_file_location(
    "sample_checker_findings",
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "sample_checker_findings.py"))
scf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(scf)


def _rec(day, hh, **kw):
    base = {"ts": f"2026-10-{day:02d}T{hh:02d}:00:00-05:00", "query": "Rowan asks about Tamsin", "response_len": 500}
    base.update(kw)
    return base


def _records():
    recs = []
    for day in (20, 21, 22):
        for hh in range(10, 16):
            recs.append(_rec(day, hh, review_fired=True, review_passed=hh % 2 == 0, review_confidence=0.8,
                             grounding_flagged=hh % 3 == 0, grounding_confidence=0.9,
                             grounding_verdict="Tamsin is not Rowan" if hh % 3 == 0 else None,
                             personal_claim_status="checked", personal_claim_reason="ok",
                             personal_claim_supported_count=int(hh % 3 == 0),
                             personal_claim_insufficient_count=int(hh % 3 != 0),
                             personal_claim_contradicted_count=0))
    return recs


def test_stratified_across_days_and_kinds():
    pools = scf.build_sheets(_records(), {}, n=6, seed=1)
    pool, sample = pools["review"]
    assert pool == 18 and len(sample) == 6
    assert {it["day"] for it in sample} == {"2026-10-20", "2026-10-21", "2026-10-22"}
    assert {it["kind"] for it in sample} == {"pass", "fail"}
    assert {it["kind"] for it in pools["personal_claim"][1]} == {"has_supported", "has_insufficient"}


def test_seed_is_deterministic():
    a = scf.build_sheets(_records(), {}, n=5, seed=7)
    b = scf.build_sheets(_records(), {}, n=5, seed=7)
    c = scf.build_sheets(_records(), {}, n=5, seed=8)
    ids = lambda s: [it["id"] for it in s["grounding"][1]]
    assert ids(a) == ids(b)
    assert ids(a) != ids(c)


def test_test_probes_excluded():
    recs = _records()
    recs.append(_rec(22, 20, review_fired=True, review_passed=False, test_probe=True))
    recs.append(_rec(22, 21, review_fired=True, review_passed=False, test_env=True))
    recs.append(_rec(22, 22, review_fired=True, review_passed=False, query="[test] hello [/test]"))
    pool, sample = scf.build_sheets(recs, {}, n=100, seed=1)["review"]
    assert pool == 18 and len(sample) == 18


def test_excerpts_bounded_and_gaps_flagged():
    recs = [_rec(20, 10, query="q" * 5000, review_fired=True, review_passed=False,
                 response_plan={"p": "x" * 5000})]
    it = scf.build_sheets(recs, {}, n=1, seed=1)["review"][1][0]
    assert len(it["query_excerpt"]) <= scf.QUERY_MAX
    assert len(it["evidence_excerpt"]) <= scf.EVIDENCE_MAX
    assert "reply_text_not_persisted" in it["context_gaps"]
    assert it["judgeable"] is False and it["label"] == "" and it["note"] == ""


def test_log_join_recovers_review_issues(tmp_path):
    log = tmp_path / "daemon_debug_x.log"
    log.write_text("2026-10-20 10:00:05 [WARNING] [gradio_gui] [REVIEW GATE] Response failed review "
                   "(confidence=0.90, issues=['skipped a plan point']). Log-only\n")
    events = scf.load_log_events([str(log)])
    it = scf.build_sheets([_rec(20, 10, review_fired=True, review_passed=False)], events, n=1, seed=1)["review"][1][0]
    assert "skipped a plan point" in it["claim_span"] and it["judgeable"] is True


def test_cli_prints_counts_only(tmp_path, capsys):
    tr = tmp_path / "t.jsonl"
    tr.write_text("\n".join(json.dumps(r) for r in _records()) + "\n")
    out = tmp_path / "sheets"
    assert scf.main(["--turn-records", str(tr), "--debug-logs", "--out", str(out), "-n", "4"]) == 0
    printed = capsys.readouterr().out
    assert "Rowan" not in printed and "Tamsin" not in printed
    assert sorted(os.listdir(out)) == ["grounding.jsonl", "personal_claim.jsonl", "review.jsonl"]


def test_prefers_finding_lines_over_turn_records(tmp_path):
    f = tmp_path / "checker_findings.jsonl"
    rows = [{"ts": f"2026-10-2{i}T10:00:00-05:00", "checker": "review", "verdict": "fail",
             "claim_span": "missed a point", "evidence_excerpt": "plan", "reply_excerpt": "reply",
             "query_excerpt": "q"} for i in range(3)]
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    findings = scf.load_findings(str(f))
    out = scf.build_sheets(_records(), {}, n=10, seed=1, findings=findings)
    pool, sample = out["review"]
    assert pool == 3 and all(it["source"] == "checker_findings" and it["judgeable"] for it in sample)
    assert out["grounding"][0] == 18  # no finding lines for grounding: falls back to turn records
