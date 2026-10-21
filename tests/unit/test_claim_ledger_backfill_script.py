"""Backfill script for the user-claim ledger (scripts/backfill_claim_ledger.py).

Incident 2026-10-20 (restatement drift). Classes: BC-75, BC-52, BC-87 (script
guards fail closed: a store-writing script refuses while the Daemon runs).
Dry run is the default and never calls a model; every model here is a local
fake and every corpus is a tmp file.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import backfill_claim_ledger as bf  # noqa: E402

TODAY = date(2026, 10, 21)
TURNS = [
    ("2026-08-01T10:00:00", "Old note from long ago. I am 40 pages in.", "Ok."),
    ("2026-10-10T10:00:00", "Update. I am only 9 days behind on the reading.", "Noted."),
    ("2026-10-18T09:00:00", "Covered 7 weeks of the material over the last like. 12 days i think?", "Nice."),
    ("2026-10-20T02:47:00", "Worked it out properly: so about fifteen days of work in total.", "Great."),
]


def write_corpus(path: Path) -> Path:
    path.write_text(json.dumps([{"query": q, "response": r, "timestamp": ts} for ts, q, r in TURNS]),
                    encoding="utf-8")
    return path


def user_message(prompt: str) -> str:
    return re.search(r"<<<\n(.*?)\n>>>", prompt, re.S).group(1)


class FakeModel:
    def __init__(self):
        self.calls = 0

    async def __call__(self, system: str, user: str, max_tokens: int) -> str:
        self.calls += 1
        msg = user_message(user)
        if "9 days behind" in msg:
            c = {"kind": "duration", "subject_key": "days_behind", "new_subject": True,
                 "description": "days behind", "value": "9", "unit": "days", "modality": "asserted",
                 "scope": "standing", "quote": "I am only 9 days behind"}
        elif "12 days i think" in msg:
            c = {"kind": "duration", "subject_key": "sprint_span", "new_subject": True,
                 "description": "sprint length", "value": "12", "unit": "days", "modality": "estimate",
                 "scope": "standing", "quote": "12 days i think?"}
        elif "fifteen days" in msg:
            c = {"kind": "duration", "subject_key": "sprint_span", "new_subject": False,
                 "description": "sprint length", "value": "15", "unit": "days", "modality": "asserted",
                 "scope": "standing", "quote": "about fifteen days of work in total"}
        else:
            return json.dumps({"claims": []})
        return json.dumps({"claims": [c]})


def factory(model):
    return lambda _name: model


@pytest.fixture()
def no_daemon(monkeypatch):
    monkeypatch.setattr(bf, "daemon_running", lambda: False)


def apply_args(corpus, store, *extra):
    return ["--corpus", str(corpus), "--apply", "--max-usd", "5", "--store", str(store), "--model", "fake", *extra]


def test_dry_run_makes_zero_calls_writes_nothing_and_prints_turns_and_estimate(tmp_path, capsys):
    corpus, store = write_corpus(tmp_path / "corpus.json"), tmp_path / "ledger.json"

    def exploding(_name):
        raise AssertionError("a dry run must not construct or call a model")

    rc = bf.main(["--corpus", str(corpus), "--store", str(store)], call_factory=exploding, today=TODAY)
    out = capsys.readouterr().out
    assert rc == 0 and not store.exists()
    assert "DRY RUN" in out and "turns: 3" in out          # the 08-01 turn is outside the 28-day window
    assert "extraction: 3 calls" in out and "chars/4" in out and "absent/empty" in out


def test_dry_run_reports_a_non_empty_target_store(tmp_path, capsys):
    corpus, store = write_corpus(tmp_path / "corpus.json"), tmp_path / "ledger.json"
    store.write_text('{"schema_version": 1, "claims": []}')
    bf.main(["--corpus", str(corpus), "--store", str(store)], call_factory=lambda _n: None, today=TODAY)
    assert "needs --replace" in capsys.readouterr().out


def test_default_window_is_28_days_and_days_flag_widens_it(tmp_path, capsys):
    corpus = write_corpus(tmp_path / "corpus.json")
    bf.main(["--corpus", str(corpus), "--days", "200"], call_factory=lambda _n: None, today=TODAY)
    assert "turns: 4" in capsys.readouterr().out


def test_apply_refuses_while_the_daemon_runs_before_any_call_or_write(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bf, "daemon_running", lambda: True)
    corpus, store = write_corpus(tmp_path / "corpus.json"), tmp_path / "ledger.json"
    model = FakeModel()
    rc = bf.main(apply_args(corpus, store), call_factory=factory(model), today=TODAY)
    assert rc == 1 and model.calls == 0 and not store.exists()
    assert "live Daemon" in capsys.readouterr().err


def test_apply_requires_max_usd(tmp_path, no_daemon, capsys):
    rc = bf.main(["--corpus", str(write_corpus(tmp_path / "c.json")), "--apply", "--store", str(tmp_path / "l.json")],
                 call_factory=factory(FakeModel()), today=TODAY)
    assert rc == 2 and "--max-usd" in capsys.readouterr().err


def test_apply_refuses_an_existing_non_empty_store_without_replace(tmp_path, no_daemon, capsys):
    corpus, store = write_corpus(tmp_path / "corpus.json"), tmp_path / "ledger.json"
    store.write_text('{"schema_version": 1, "claims": []}')
    model = FakeModel()
    rc = bf.main(apply_args(corpus, store), call_factory=factory(model), today=TODAY)
    assert rc == 1 and model.calls == 0
    assert store.read_text() == '{"schema_version": 1, "claims": []}'
    assert "--replace" in capsys.readouterr().err


def test_apply_writes_the_ledger_with_supersession_and_scope(tmp_path, no_daemon):
    corpus, store = write_corpus(tmp_path / "corpus.json"), tmp_path / "ledger.json"
    model = FakeModel()
    assert bf.main(apply_args(corpus, store), call_factory=factory(model), today=TODAY) == 0
    saved = json.loads(store.read_text())
    by = {(c["subject_key"], c["value"]): c for c in saved["claims"]}
    assert saved["schema_version"] == 1 and len(saved["claims"]) == 3
    assert by[("sprint_span", "12")]["superseded_by"] == by[("sprint_span", "15")]["id"]
    assert {c["scope"] for c in saved["claims"]} == {"standing"}
    assert model.calls == 3                       # extraction only: one call per in-window turn, no reply check


def test_replace_copies_the_existing_store_aside_before_writing(tmp_path, no_daemon):
    corpus, store = write_corpus(tmp_path / "corpus.json"), tmp_path / "ledger.json"
    old = '{"schema_version": 1, "claims": []}'
    store.write_text(old)
    assert bf.main(apply_args(corpus, store, "--replace"), call_factory=factory(FakeModel()), today=TODAY) == 0
    (aside,) = [p for p in tmp_path.iterdir() if ".pre-backfill-" in p.name]
    assert aside.read_text() == old and aside.name.startswith("ledger.json.pre-backfill-")
    assert len(json.loads(store.read_text())["claims"]) == 3


def test_a_budget_stop_writes_nothing(tmp_path, no_daemon, capsys):
    corpus, store = write_corpus(tmp_path / "corpus.json"), tmp_path / "ledger.json"
    rc = bf.main(["--corpus", str(corpus), "--apply", "--max-usd", "0.0000001", "--store", str(store),
                  "--model", "fake"], call_factory=factory(FakeModel()), today=TODAY)
    assert rc == 3 and not store.exists()
    assert "nothing was written" in capsys.readouterr().err


def test_a_402_stops_the_run_and_writes_nothing(tmp_path, no_daemon, capsys):
    corpus, store = write_corpus(tmp_path / "corpus.json"), tmp_path / "ledger.json"

    async def paywalled(system, user, max_tokens):
        return "[CREDITS EXHAUSTED] OpenRouter balance is empty."

    rc = bf.main(apply_args(corpus, store), call_factory=lambda _n: paywalled, today=TODAY)
    assert rc == 4 and not store.exists()


def test_the_script_reuses_the_baseline_helpers_rather_than_copying_them():
    import claim_ledger_baseline as base
    assert bf.base.__file__ == base.__file__ or bf.base.__file__.endswith("claim_ledger_baseline.py")
    for name in ("select_turns", "load_corpus", "estimate_plan", "Spend", "_wrap_call", "default_call_factory"):
        assert getattr(bf.base, name).__module__.endswith("claim_ledger_baseline"), name
    assert not hasattr(bf, "select_turns") and not hasattr(bf, "Spend")      # nothing was copied in
