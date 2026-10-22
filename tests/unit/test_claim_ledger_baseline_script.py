"""Offline claim-ledger baseline script (scripts/claim_ledger_baseline.py).

Incident 2026-10-20 (conditional restated as fact; superseded estimate quoted).
Classes: BC-75, BC-52. The script is dry-run by default and never spends in a
test: every model is a local fake.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import claim_ledger_baseline as base  # noqa: E402

TURNS = [
    ("2026-10-17T10:00:00", "Quick update. If I can get 6 done today then I am only 6 days behind.",
     "Sounds like a plan, keep going."),
    ("2026-10-18T09:00:00", "Covered 7 weeks of material over the last like. 12 days i think?",
     "Nice progress on the material."),
    ("2026-10-19T09:00:00", "Anything else I should line up for the sprint?",
     "Plan the review block first."),
    ("2026-10-20T02:47:00", "Worked it out properly: so about fifteen days of work in total.",
     "Great, that is roughly twelve days then."),
    ("2026-10-20T20:26:00", "Thanks, signing off for tonight.",
     "Rest well, after being six days behind you did well."),
]


def write_corpus(path: Path):
    path.write_text(json.dumps([{"query": q, "response": r, "timestamp": ts} for ts, q, r in TURNS]),
                    encoding="utf-8")
    return path


def user_message(prompt: str) -> str:
    return re.search(r"<<<\n(.*?)\n>>>", prompt, re.S).group(1)


class FakeModel:
    """Deterministic stand-in keyed on the prompt content; counts calls."""

    def __init__(self):
        self.calls = 0

    async def __call__(self, system: str, user: str, max_tokens: int) -> str:
        self.calls += 1
        msg = user_message(user)
        if system.startswith("You extract"):
            if "If I can get 6 done" in msg:
                c = {"kind": "duration", "subject_key": "days_behind", "new_subject": True,
                     "description": "days behind", "value": "6", "unit": "days", "modality": "asserted",
                     "quote": "then I am only 6 days behind"}
            elif "12 days i think" in msg:
                c = {"kind": "duration", "subject_key": "sprint_span", "new_subject": True,
                     "description": "sprint length", "value": "12", "unit": "days", "modality": "estimate",
                     "quote": "12 days i think?"}
            elif "fifteen days" in msg:
                c = {"kind": "duration", "subject_key": "sprint_span", "new_subject": False,
                     "description": "sprint length", "value": "15", "unit": "days", "modality": "asserted",
                     "quote": "about fifteen days of work in total"}
            else:
                return json.dumps({"claims": []})
            return json.dumps({"claims": [c]})
        if "roughly twelve days" in msg:
            f = {"subject_key": "sprint_span", "kind": "duration", "value": "12", "unit": "days",
                 "modality": "estimate", "quote": "roughly twelve days"}
        elif "six days behind" in msg:
            f = {"subject_key": "days_behind", "kind": "duration", "value": "6", "unit": "days",
                 "modality": "asserted", "quote": "after being six days behind"}
        else:
            return json.dumps({"findings": []})
        return json.dumps({"findings": [f]})


def factory_for(model):
    return lambda _name: model


def run_args(corpus, out, *extra):
    return ["--corpus", str(corpus), "--out-dir", str(out), "--run", "--model", "fake", "--max-usd", "5",
            *extra]


def test_dry_run_makes_zero_model_calls_and_prints_the_estimate(tmp_path, capsys):
    corpus = write_corpus(tmp_path / "corpus.json")

    def exploding_factory(_name):
        raise AssertionError("a dry run must not construct or call a model")

    rc = base.main(["--corpus", str(corpus), "--price-in", "0.15", "--price-out", "0.60"],
                   call_factory=exploding_factory)
    out = capsys.readouterr().out
    assert rc == 0
    assert "DRY RUN" in out and "turns in window: 5" in out
    assert "phase (a) extraction" in out and "phase (b) reply check" in out
    assert "TOTAL estimated cost" in out and "chars/4" in out


def test_run_with_fake_model_writes_ledger_findings_and_report(tmp_path, capsys):
    corpus = write_corpus(tmp_path / "corpus.json")
    out = tmp_path / "out"
    model = FakeModel()
    assert base.main(run_args(corpus, out), call_factory=factory_for(model)) == 0
    assert {p.name for p in out.iterdir()} >= {"ledger.json", "findings.jsonl", "report.md", "progress.json"}
    ledger = json.loads((out / "ledger.json").read_text())
    by_value = {(c["subject_key"], c["value"]): c for c in ledger["claims"]}
    assert by_value[("days_behind", "6")]["modality"] == "conditional"          # parse overrode "asserted"
    assert by_value[("sprint_span", "12")]["superseded_by"] == by_value[("sprint_span", "15")]["id"]
    rows = [json.loads(line) for line in (out / "findings.jsonl").read_text().splitlines()]
    assert sorted(r["classification"] for r in rows) == ["conditional_as_fact", "stale"]
    report = (out / "report.md").read_text()
    assert "stale" in report and "conditional_as_fact" in report and "supersessions: 1" in report
    assert "by modality: asserted=1, conditional=1, estimate=1" in report
    assert model.calls > 0


def test_max_usd_stops_before_exceeding_and_resume_finishes(tmp_path, capsys):
    corpus = write_corpus(tmp_path / "corpus.json")
    out = tmp_path / "out"
    model = FakeModel()
    tiny = ["--corpus", str(corpus), "--out-dir", str(out), "--run", "--model", "fake", "--max-usd", "0.0003",
            "--price-in", "1.0", "--price-out", "1.0", "--checkpoint-every", "1"]
    rc = base.main(tiny, call_factory=factory_for(model))
    assert rc == base.EXIT_BUDGET
    assert "STOPPED" in capsys.readouterr().err
    progress = json.loads((out / "progress.json").read_text())
    assert progress["next_turn"] < len(TURNS)
    assert progress["stats"]["spent_usd"] <= 0.0003 + 1e-9
    # resume with a real budget: completes, no duplicated findings
    rc = base.main(run_args(corpus, out, "--resume", "--price-in", "0.15", "--price-out", "0.6"),
                   call_factory=factory_for(FakeModel()))
    assert rc == 0
    rows = (out / "findings.jsonl").read_text().splitlines()
    assert len(rows) == len(set(rows)) == 2
    assert json.loads((out / "progress.json").read_text())["next_turn"] == len(TURNS)


def test_credit_error_stops_immediately_with_a_clear_message(tmp_path, capsys):
    corpus = write_corpus(tmp_path / "corpus.json")
    out = tmp_path / "out"
    inner = FakeModel()
    state = {"n": 0}

    async def flaky(system, user, max_tokens):
        state["n"] += 1
        if state["n"] == 3:
            return "[CREDITS EXHAUSTED] The provider reports insufficient credits for this request (HTTP 402)."
        return await inner(system, user, max_tokens)

    rc = base.main(run_args(corpus, out), call_factory=factory_for(flaky))
    assert rc == base.EXIT_CREDITS
    assert state["n"] == 3                                   # no call after the credit failure
    assert "credit exhausted" in capsys.readouterr().err
    assert (out / "report.md").exists() and "STOPPED EARLY" in (out / "report.md").read_text()


def test_http_402_exception_also_stops(tmp_path, capsys):
    corpus = write_corpus(tmp_path / "corpus.json")

    class Boom(Exception):
        status_code = 402

    async def raises(system, user, max_tokens):
        raise Boom("This request requires more credits")

    rc = base.main(run_args(corpus, tmp_path / "out"), call_factory=factory_for(raises))
    assert rc == base.EXIT_CREDITS


def test_refuses_an_out_dir_under_data(tmp_path, capsys):
    corpus = write_corpus(tmp_path / "corpus.json")
    bad = tmp_path / "data" / "baseline"
    for extra in ([], ["--run", "--max-usd", "1"]):
        rc = base.main(["--corpus", str(corpus), "--out-dir", str(bad), *extra],
                       call_factory=lambda _n: pytest.fail("no model may be built"))
        assert rc == base.EXIT_USAGE
    assert not bad.exists()
    assert "refusing --out-dir" in capsys.readouterr().err


def test_run_requires_max_usd_and_out_dir(tmp_path, capsys):
    corpus = write_corpus(tmp_path / "corpus.json")
    assert base.main(["--corpus", str(corpus), "--run"], call_factory=factory_for(FakeModel())) == base.EXIT_USAGE


def test_existing_run_dir_needs_resume(tmp_path):
    corpus = write_corpus(tmp_path / "corpus.json")
    out = tmp_path / "out"
    assert base.main(run_args(corpus, out), call_factory=factory_for(FakeModel())) == 0
    assert base.main(run_args(corpus, out), call_factory=factory_for(FakeModel())) == base.EXIT_USAGE


# --- AMEND1 A7 (2026-10-20 trial): subjects table + drop examples in report.md --------------------------


def label_corpus(path: Path, n_label_turns: int):
    rows = [("2026-10-17T10:00:00", "Quick update. If I can get 6 done today then I am only 6 days behind.",
             "Sounds like a plan.")]
    for i in range(n_label_turns):
        rows.append((f"2026-10-18T0{i % 9 + 1}:00:00", f"Done with HW {i + 1} and it went fine.", "Nice work."))
    path.write_text(json.dumps([{"query": q, "response": r, "timestamp": ts} for ts, q, r in rows]),
                    encoding="utf-8")
    return path


class LabelModel:
    """Misbehaving extractor: reports every 'HW n' label as a count."""

    async def __call__(self, system, user, max_tokens):
        msg = user_message(user)
        if not system.startswith("You extract"):
            return json.dumps({"findings": []})
        m = re.search(r"HW (\d+)", msg)
        if m:
            return json.dumps({"claims": [{"kind": "count", "subject_key": "assignments_finished",
                                           "new_subject": True, "description": "items finished",
                                           "value": m.group(1), "unit": "assignments", "modality": "asserted",
                                           "quote": f"HW {m.group(1)}"}]})
        if "If I can get 6 done" in msg:
            return json.dumps({"claims": [{"kind": "duration", "subject_key": "days_behind", "new_subject": True,
                                           "description": "days behind", "value": "6", "unit": "days",
                                           "modality": "asserted", "quote": "then I am only 6 days behind"}]})
        return json.dumps({"claims": []})


def drop_example_lines(report: str, reason: str):
    section = report.split("## Drop examples", 1)[1].split("### reply", 1)[0]
    block = section.split(f"- {reason} (", 1)[1].splitlines()[1:]
    out = []
    for line in block:
        if not line.startswith("  - "):
            break
        out.append(line[4:])
    return out


def test_a7_report_has_a_subjects_table_and_capped_drop_examples(tmp_path):
    corpus = label_corpus(tmp_path / "c.json", 7)
    out = tmp_path / "out"
    assert base.main(run_args(corpus, out), call_factory=factory_for(LabelModel())) == 0
    report = (out / "report.md").read_text()
    assert "## Subjects" in report
    assert "| days_behind | duration | day | 1 | none (conditional only: 6 2026-10-17) | - |" in report
    assert "identifier_number=7" in report                                   # existing totals kept
    examples = drop_example_lines(report, "identifier_number")
    assert len(examples) == 5 and all(len(e) <= 120 for e in examples)       # capped at 5 per reason
    assert all(e.startswith("HW ") for e in examples)
    assert "- identifier_number (7 total)" in report


def test_a7_fewer_drops_than_the_cap_lists_them_all(tmp_path):
    corpus = label_corpus(tmp_path / "c.json", 3)
    out = tmp_path / "out"
    assert base.main(run_args(corpus, out), call_factory=factory_for(LabelModel())) == 0
    examples = drop_example_lines((out / "report.md").read_text(), "identifier_number")
    assert examples == ["HW 1", "HW 2", "HW 3"]


def test_a7_subjects_table_shows_current_and_superseded_values(tmp_path):
    corpus = write_corpus(tmp_path / "corpus.json")
    out = tmp_path / "out"
    assert base.main(run_args(corpus, out), call_factory=factory_for(FakeModel())) == 0
    report = (out / "report.md").read_text()
    assert "| sprint_span | duration | day | 2 | 15 (asserted, 2026-10-20) | 12 (estimate, 2026-10-18) |" in report


def test_b3_report_lists_differs_from_estimate_in_its_own_section(tmp_path):
    corpus = tmp_path / "c.json"
    rows = [("2026-10-18T09:00:00", "Covered 7 weeks of material over the last like. 12 days i think?", "Nice."),
            ("2026-10-20T09:00:00", "Anything to line up?", "At this point it is fifteen days of work.")]
    corpus.write_text(json.dumps([{"query": q, "response": r, "timestamp": ts} for ts, q, r in rows]),
                      encoding="utf-8")

    class M:
        async def __call__(self, system, user, max_tokens):
            msg = user_message(user)
            if system.startswith("You extract"):
                if "12 days i think" in msg:
                    return json.dumps({"claims": [{"kind": "duration", "subject_key": "sprint_span",
                        "new_subject": True, "description": "sprint length", "value": "12", "unit": "days",
                        "modality": "estimate", "quote": "12 days i think?"}]})
                return json.dumps({"claims": []})
            return json.dumps({"findings": [{"subject_key": "sprint_span", "kind": "duration", "value": "15",
                "unit": "days", "modality": "asserted", "quote": "fifteen days of work"}]})

    out = tmp_path / "out"
    assert base.main(run_args(corpus, out), call_factory=factory_for(M())) == 0
    report = (out / "report.md").read_text()
    assert "by class: differs_from_estimate=1" in report
    assert "### top differs_from_estimate (1 shown)" in report and "### top stale (0 shown)" in report
