"""Tone evaluation harness (plan 2026-10-08, B6b; D4, D6, section 9; BC-47, BC-64, BC-82).

Every hop is the deployed code: the harness calls utils.tone_detector._decision_model_crisis_arbiter
through a REAL ModelManager whose System One transport is an httpx.MockTransport; the LLM arbiter is
the only fake (it would need a chat API). No test opens a socket: the dry-run test fails any socket
use outright. Fixtures are synthetic nonsense text so the "no case text in any output" checks are
meaningful (6-character-run scan of every output against every case).
"""
import asyncio
import importlib.util
import json
import math
import socket
import sys
from pathlib import Path

import httpx
import pytest
import yaml

from config import app_config
from models.model_manager import ModelManager
from utils import tone_detector as tone

REPO = Path(app_config.__file__).resolve().parents[1]  # the code under test (clone or base)
_spec = importlib.util.spec_from_file_location("probe_decision_model_eval", REPO / "scripts" / "probe_decision_model_eval.py")
h = importlib.util.module_from_spec(_spec)
sys.modules["probe_decision_model_eval"] = h
_spec.loader.exec_module(h)

SERVED = "typesafe/jev-1.13-20260917"
FIXTURE = json.loads((REPO / "tests" / "fixtures" / "decision_model" / "tone_score_ok.json").read_text())
CASES = [
    {"id": "t1", "text": "Qorvex plimbs the vantrel wickets at dusk.", "expected": "HIGH", "group": "high", "split": "select"},
    {"id": "t2", "text": "Zynthe marbled quillfeathers on Ostrava Tuesday.", "expected": "MEDIUM", "group": "adversarial_severe", "split": "select"},
    {"id": "t3", "text": "Brumely fobbed twelve glintwork spindles aloft.", "expected": "CONCERN", "group": "concern", "split": "select"},
    {"id": "t4", "text": "Yarrowick snaffled the dulcimer moxie quietly.", "expected": "CONVERSATIONAL", "group": "borderline_conv", "split": "select"},
]
PROBS = {  # text -> four-level distribution the fake System One returns
    CASES[0]["text"]: [0.3, 0.1, 0.1, 0.5],
    CASES[1]["text"]: [0.0, 0.0, 1.0, 0.0],
    CASES[2]["text"]: [0.0, 1.0, 0.0, 0.0],
    CASES[3]["text"]: [1.0, 0.0, 0.0, 0.0],
}
LLM_LEVELS = {CASES[0]["text"]: "HIGH", CASES[1]["text"]: "CONVERSATIONAL", CASES[2]["text"]: "CONCERN", CASES[3]["text"]: "CONVERSATIONAL"}


def _body(probs, cost=1.9e-05):
    body = json.loads(json.dumps(FIXTURE))
    body["answers"]["tone"]["probabilities"] = {str(i): p for i, p in enumerate(probs)}
    body["answers"]["tone"]["score"] = sum(i * p for i, p in enumerate(probs))
    body["usage"]["cost"] = cost
    return body


class FakeSystemOne:
    def __init__(self, status_by_call=None, cost=1.9e-05):
        self.calls, self.status_by_call, self.cost = 0, status_by_call or {}, cost

    def __call__(self, request):
        self.calls += 1
        status = self.status_by_call.get(self.calls, 200)
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "x", "code": status}})
        text = json.loads(request.content)["state"]
        return httpx.Response(200, json=_body(PROBS[text], self.cost))


@pytest.fixture
def rig(monkeypatch):
    """A real ModelManager on a mock System One transport + the fake LLM arbiter + synthetic cases."""
    def build(system_one):
        mm = ModelManager(api_key="k-test")
        mm.system_one_transport = httpx.MockTransport(system_one)

        async def fake_llm(message, model_manager=None):
            level = LLM_LEVELS.get(message)
            return (tone.CrisisLevel[level], 0.7) if level else None

        monkeypatch.setattr(tone, "_llm_crisis_fallback", fake_llm)
        monkeypatch.setattr(h, "make_model_manager", lambda: mm)
        monkeypatch.setattr(h.cases_mod, "tone_cases", lambda split="select": [dict(c) for c in CASES])
        return mm
    return build


def _runs(text, n=6):
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _assert_no_case_text(blob, cases):
    for c in cases:
        assert c["text"] not in blob
        leaked = [r for r in _runs(c["text"]) if r in blob]
        assert not leaked, (c["id"], leaked[:3])


# ------------------------------------------------------------------ modes and refusals
def test_dry_run_makes_no_network_call(monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("network used in a dry run")
    for name in ("socket", "create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, boom)
    monkeypatch.setattr(h, "make_model_manager", boom)
    assert h.main([]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "calls per backend" in out and "price estimate" in out
    assert "cases: 102" in out and "state_too_large" in out


def test_run_without_budget_refuses(monkeypatch, capsys):
    monkeypatch.setattr(h, "make_model_manager", lambda: pytest.fail("built a ModelManager"))
    assert h.main(["--run"]) == 2
    assert "--budget-usd" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["--run", "--budget-usd", "2.5"], ["--budget-usd", "2.01"], ["--run", "--budget-usd", "0"]])
def test_budget_above_cap_refuses(monkeypatch, capsys, argv):
    monkeypatch.setattr(h, "make_model_manager", lambda: pytest.fail("built a ModelManager"))
    assert h.main(argv) == 2
    assert "refusing" in capsys.readouterr().err


def test_e2e_is_out_of_scope(capsys):
    assert h.main(["--e2e"]) == 2
    assert "not implemented in R1 scope" in capsys.readouterr().out


def test_output_under_data_dir_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        h.write_outputs(REPO / "data" / "eval_out", {"mode": "x"}, [], {})


# ------------------------------------------------------------------ statistics against hand values
def test_wilson_matches_hand_values():
    z2 = 1.959964 ** 2
    lo, hi = h.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(z2 / (10 + z2), abs=1e-9)
    lo, hi = h.wilson(10, 10)
    assert hi == pytest.approx(1.0, abs=1e-12) and lo == pytest.approx(10 / (10 + z2), abs=1e-9)
    lo, hi = h.wilson(5, 10)  # symmetric about 0.5
    assert lo == pytest.approx(1 - hi, abs=1e-9) and 0.23 < lo < 0.24
    assert h.wilson(0, 0) == (None, None)


def test_clopper_pearson_zero_misses_in_59_is_under_five_percent():
    assert h.clopper_pearson_upper(0, 59) == pytest.approx(1 - 0.05 ** (1 / 59), abs=1e-9)
    assert h.clopper_pearson_upper(0, 59) < 0.05 < h.clopper_pearson_upper(0, 58)
    assert h.clopper_pearson_upper(0, 1) == pytest.approx(0.95, abs=1e-9)
    assert h.clopper_pearson_upper(3, 3) == 1.0 and h.clopper_pearson_upper(0, 0) is None
    ub = h.clopper_pearson_upper(1, 93)  # one miss needs about 93 cases (section 9)
    assert ub < 0.05 and sum(math.comb(93, i) * 0.05 ** i * 0.95 ** (93 - i) for i in (0, 1)) < 0.05
    assert h.clopper_pearson_upper(1, 80) > 0.05


def test_brier_and_reliability_match_hand_values():
    rows = [([0.7, 0.1, 0.1, 0.1], 0), ([0.0, 0.0, 0.0, 1.0], 3)]
    assert h.brier(rows) == pytest.approx(((0.09 + 3 * 0.01) + 0.0) / 2)
    assert h.brier([]) is None
    rel = h.reliability(rows)
    assert [r["n"] for r in rel] == [0, 0, 0, 1, 1] and rel[3]["conf"] == pytest.approx(0.7) and rel[4]["acc"] == 1.0
    assert h.percentile([1, 2, 3, 4], 0.5) == 2 and h.percentile([], 0.5) is None


def test_case_cost_weights():
    assert h.case_cost(3, 0) == h.case_cost(2, 0) == 20.0   # severe miss
    assert h.case_cost(3, 2) == h.case_cost(1, 0) == 3.0    # other missed distress level
    assert h.case_cost(0, 2) == h.case_cost(1, 3) == 1.0    # false alarm / over-call
    assert h.case_cost(2, 2) == 0.0 and h.case_cost(2, None) == 0.0


# ------------------------------------------------------------------ run, budget, 402
def _run(rig, system_one, budget=1.0, per_mtok=0.0):
    mm = rig(system_one)
    with h.jev_enabled_in_process():
        return asyncio.run(asyncio.wait_for(h.run_cases([dict(c) for c in CASES], mm, budget, per_mtok), 20))


def test_run_calls_deployed_arbiters_and_records_probabilities_not_text(rig):
    fake = FakeSystemOne()
    rows, spent, stopped = _run(rig, fake)
    assert stopped is None and fake.calls == 4 and len(rows) == 4
    assert rows[0]["jev"]["probs"] == [0.3, 0.1, 0.1, 0.5] and rows[0]["jev"]["served_model"] == SERVED
    assert rows[1]["llm"]["level"] == "CONVERSATIONAL" and rows[0]["llm"]["level"] == "HIGH"
    assert spent == pytest.approx(4 * 1.9e-05)
    _assert_no_case_text(json.dumps(rows), CASES)


def test_jev_is_enabled_for_the_process_only_and_restored():
    before = (app_config.DECISION_MODEL_CFG, app_config.decision_model_mode("tone_arbiter"))
    with h.jev_enabled_in_process():
        assert app_config.decision_model_mode("tone_arbiter") == "shadow"
        assert app_config.DECISION_MODEL_TONE_POLICY == "unset"
    assert (app_config.DECISION_MODEL_CFG, app_config.decision_model_mode("tone_arbiter")) == before
    assert before[1] == "off"


def test_402_from_the_transport_stops_the_run(rig):
    fake = FakeSystemOne(status_by_call={2: 402})
    rows, _, stopped = _run(rig, fake)
    assert stopped == "http_402" and fake.calls == 2 and len(rows) == 2
    assert rows[1]["jev"]["status"] == "unavailable" and rows[1]["jev"]["reason"] == "http_402"


def test_budget_stops_the_run_when_summed_cost_reaches_it(rig):
    fake = FakeSystemOne(cost=0.6)
    rows, spent, stopped = _run(rig, fake, budget=1.0)
    assert stopped == "budget" and len(rows) == 2 and spent == pytest.approx(1.2)


def test_missing_cost_is_estimated_and_flagged(rig):
    class NoCost(FakeSystemOne):
        def __call__(self, request):
            resp = super().__call__(request)
            body = resp.json()
            body["usage"].pop("cost")
            return httpx.Response(200, json=body)

    rows, spent, _ = _run(rig, NoCost())
    assert all(r["jev"]["cost_estimated"] and r["jev"]["cost_usd"] > 0 for r in rows) and spent > 0


def test_state_too_large_makes_no_call_and_is_unavailable_not_wrong(rig):
    mm = rig(FakeSystemOne())
    long_case = dict(CASES[0], id="t_long", text="Qorvex " * 20000)
    LLM_LEVELS[long_case["text"]] = "HIGH"
    try:
        with h.jev_enabled_in_process():
            rows, _, _ = asyncio.run(asyncio.wait_for(h.run_cases([long_case], mm, 1.0, 0.0), 10))
    finally:
        LLM_LEVELS.pop(long_case["text"], None)
    assert rows[0]["jev"]["reason"] == "state_too_large" and rows[0]["jev"]["probs"] is None
    s = h.summarize(rows, "argmax", {})
    assert s["jev"]["unavailable"] == 1 and s["jev"]["answered"] == 0 and s["llm"]["answered"] == 1


# ------------------------------------------------------------------ main(): outputs, replay, select
def _main_run(rig, tmp_path, extra=()):
    rig(FakeSystemOne())
    out = tmp_path / "run"
    assert h.main(["--run", "--budget-usd", "1", "--llm-usd-per-mtok", "0", "--out", str(out), *extra]) == 0
    return out


def test_run_writes_report_and_metrics_without_case_text(rig, tmp_path, capsys):
    out = _main_run(rig, tmp_path)
    metrics = (out / "metrics.json").read_text()
    report = (out / "report.md").read_text()
    _assert_no_case_text(metrics + report + capsys.readouterr().out, CASES)
    doc = json.loads(metrics)
    assert doc["meta"]["jev_slug"] == "typesafe/jev-1.13" and "llm_model" in doc["meta"]
    assert [c["id"] for c in doc["cases"]] == ["t1", "t2", "t3", "t4"]
    for needle in ("LLM arbiter model", "Brier", "Wilson95", "one-sided 95% CP upper bound", "unavailable", "Paired discordant"):
        assert needle in report
    jev = doc["summary"]["jev"]
    assert jev["severe"]["n"] == 2 and jev["adversarial"]["n"] == 1 and doc["summary"]["llm"]["severe"]["misses"] == 1


def test_replay_reproduces_a_saved_runs_verdicts_exactly(rig, tmp_path, monkeypatch):
    out = _main_run(rig, tmp_path, ["--policy", "weighted"])
    saved = json.loads((out / "metrics.json").read_text())
    out2 = tmp_path / "replay"
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("replay opened a socket"))
    monkeypatch.setattr(h, "make_model_manager", lambda: pytest.fail("replay built a ModelManager"))
    assert h.main(["--replay", str(out / "metrics.json"), "--policy", "weighted", "--out", str(out2)]) == 0
    again = json.loads((out2 / "metrics.json").read_text())

    def verdicts(doc):
        return [(c["id"], c["llm"]["level"], c["jev"]["verdict"], c["jev"]["probs"]) for c in doc["cases"]]

    assert verdicts(again) == verdicts(saved)
    assert again["summary"]["jev"] == saved["summary"]["jev"] and again["summary"]["paired"] == saved["summary"]["paired"]
    assert again["meta"]["llm_model"] == saved["meta"]["llm_model"]
    # t1 is [0.3, 0.1, 0.1, 0.5]: weighted score 1.8 -> MEDIUM (2); argmax -> HIGH (3). Same probabilities, new policy.
    assert [c["jev"]["verdict"] for c in saved["cases"]][0] == 2
    out3 = tmp_path / "replay_argmax"
    assert h.main(["--replay", str(out / "metrics.json"), "--policy", "argmax", "--out", str(out3)]) == 0
    assert json.loads((out3 / "metrics.json").read_text())["cases"][0]["jev"]["verdict"] == 3


def test_select_freezes_a_zero_severe_miss_policy_and_prints_yaml(rig, tmp_path, capsys):
    out = _main_run(rig, tmp_path)
    capsys.readouterr()
    rows = json.loads((out / "metrics.json").read_text())["cases"]
    choice = h.select_policy(rows)
    assert choice is not None and choice["policy"] == "argmax" and choice["cost"] == 0.0 and choice["cases"] == 4
    assert h.main(["--select", "--replay", str(out / "metrics.json"), "--out", str(tmp_path / "sel")]) == 0
    printed = capsys.readouterr().out
    lines = printed[printed.index("  tone_policy:"):]
    parsed = yaml.safe_load("decision_model:\n" + lines)["decision_model"]
    assert parsed == {"tone_policy": "argmax", "tone_policy_params": {}}


def test_select_picks_cumulative_taus_when_argmax_misses_a_severe_case():
    sev = {"id": "s", "group": "high", "expected": "HIGH", "llm": {"level": "HIGH"},
           "jev": {"status": "ok", "probs": [0.6, 0.0, 0.0, 0.4]}}
    calm = {"id": "c", "group": "borderline_conv", "expected": "CONVERSATIONAL", "llm": {"level": "CONVERSATIONAL"},
            "jev": {"status": "ok", "probs": [0.9, 0.05, 0.0, 0.05]}}
    assert h.jev_verdict(sev["jev"]["probs"], "argmax", {}) == 0  # argmax would be a severe miss
    choice = h.select_policy([sev, calm])
    assert choice["policy"] in ("weighted", "cumulative")
    assert h.jev_verdict(sev["jev"]["probs"], choice["policy"], choice["params"]) == 3
    assert h.jev_verdict(calm["jev"]["probs"], choice["policy"], choice["params"]) == 0
    parsed = yaml.safe_load("x:\n" + "\n".join(h.yaml_lines(choice)))["x"]
    key = "cuts" if choice["policy"] == "weighted" else "taus"
    assert parsed["tone_policy"] == choice["policy"] and len(parsed["tone_policy_params"][key]) == 3
    # the owner's YAML round-trips through THE deployed schema-level verdict function
    assert tone.tone_verdict(sev["jev"]["probs"], parsed["tone_policy"], parsed["tone_policy_params"]).name == "HIGH"


def test_select_reports_nothing_frozen_when_no_policy_is_safe():
    impossible = {"id": "s", "group": "high", "expected": "HIGH", "llm": {"level": "HIGH"},
                  "jev": {"status": "ok", "probs": [1.0, 0.0, 0.0, 0.0]}}
    assert h.select_policy([impossible]) is None


def test_paired_discordant_counts():
    def item(i, e, v):
        return {"id": i, "group": "g", "expected": e, "verdict": v, "latency_ms": 1}
    llm = [item("a", 3, 3), item("b", 3, 0), item("c", 0, 0), item("d", 0, 2)]
    jev = [item("a", 3, 0), item("b", 3, 3), item("c", 0, 3), item("d", 0, 0)]
    p = h.paired(llm, jev)
    assert p["exact"] == {"b_llm_right_jev_wrong": 2, "c_jev_right_llm_wrong": 2}
    assert p["distress_caught"] == {"b_llm_right_jev_wrong": 1, "c_jev_right_llm_wrong": 1}
    assert p["calm_kept"] == {"b_llm_right_jev_wrong": 1, "c_jev_right_llm_wrong": 1}


# ------------------------------------------------------------------ owner-private and audit: no text out
def test_private_mode_writes_aggregates_only_and_prints_a_banner(rig, tmp_path, capsys, monkeypatch):
    private = [
        {"text": "Plinthor wangled eleven marzipan hatchlings.", "label": "HIGH"},
        {"text": "Grexlow tumbled amid ferrous tamarind ledgers.", "label": "CONVERSATIONAL"},
    ]
    path = tmp_path / "owner.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in private))
    PROBS.update({private[0]["text"]: [0.0, 0.0, 0.0, 1.0], private[1]["text"]: [1.0, 0.0, 0.0, 0.0]})
    LLM_LEVELS.update({private[0]["text"]: "HIGH", private[1]["text"]: "CONVERSATIONAL"})
    try:
        rig(FakeSystemOne())
        out = tmp_path / "priv"
        assert h.main(["--run", "--budget-usd", "1", "--llm-usd-per-mtok", "0", "--private", str(path), "--out", str(out)]) == 0
    finally:
        for r in private:
            PROBS.pop(r["text"], None)
            LLM_LEVELS.pop(r["text"], None)
    stdout = capsys.readouterr().out
    assert "PRIVATE MODE" in stdout
    blob = stdout + (out / "metrics_private.json").read_text() + (out / "report_private.md").read_text()
    _assert_no_case_text(blob, private)
    doc = json.loads((out / "metrics_private.json").read_text())
    assert "cases" not in doc and doc["summary"]["cases"] == 2 and doc["meta"]["split"] == "private"
    assert not (out / "metrics.json").exists()


def test_private_file_with_an_unknown_label_is_refused(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"text": "x y z", "label": "FINE"}))
    with pytest.raises(SystemExit):
        h.load_private(path)


def test_audit_writes_counts_only(tmp_path, capsys):
    recs = [
        {"query": f"Vexlington crumpled {i} thistledown parchments.", "tone_dm_mode": "shadow", "tone_dm_agrees": True,
         "tone_dm_level": "CONVERSATIONAL" if i % 2 else "CONCERN"} for i in range(10)
    ] + [{"query": "Mordane skiffed the ignored quibbling.", "tone_dm_mode": "shadow", "tone_dm_agrees": False, "tone_dm_level": "HIGH"},
         {"query": "Zoblin dithered amid hushed quartz.", "tone_dm_mode": "off", "tone_dm_agrees": True, "tone_dm_level": "HIGH"}]
    records = tmp_path / "turn_records.jsonl"
    records.write_text("\n".join(json.dumps(r) for r in recs) + "\nnot json\n")
    answers = iter(["h", "c", "n", "", "c", "c", "n", "h", "c", "n"])
    res = h.audit(records, 8, 7, tmp_path / "aud", ask=lambda prompt: next(answers))
    assert res["eligible_agreements"] == 10 and res["sampled"] == 8
    assert res["labelled"] + res["skipped"] == 8 and res["both_conversational_but_owner_severe"] >= 0
    saved = (tmp_path / "aud" / "audit_counts.json").read_text()
    _assert_no_case_text(saved, [{"id": "q", "text": r["query"]} for r in recs])
    assert set(json.loads(saved)) >= {"eligible_agreements", "sampled", "boundary_filter"}
