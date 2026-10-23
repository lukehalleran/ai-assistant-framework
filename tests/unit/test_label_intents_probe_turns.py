"""Intent-labeling scripts skip [test] probe turns.

Incident 2026-10-23, class: BC-58. label_intents.py / auto_label_intents.py
labeled every turn_records.jsonl row, so [test]...[/test] probe turns entered
data/intent_labels.jsonl and the confusion matrix. Drives the deployed loaders
(label_intents.load_turn_records / report, auto_label_intents.main).
"""
import asyncio
import json

import pytest

import scripts.auto_label_intents as auto
import scripts.label_intents as li

CLEAN = "[test]probe about the dentist[/test]"
WRAPPED = "   [test]\n    probe about the dentist\n   [/test]"


def _rows(probe_query):
    return [
        {"ts": "t1", "query": "what is my cat's name", "intent": "factual_recall", "test_probe": False},
        {"ts": "t2", "query": "probe text", "intent": "general", "test_probe": True},
        {"ts": "t3", "query": probe_query, "intent": "general"},          # legacy probe
        {"ts": "t4", "query": "plain legacy question", "intent": "general"},  # legacy real
    ]


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


@pytest.mark.parametrize("q", [CLEAN, WRAPPED])
def test_loader_skips_probes(tmp_path, q):
    f = tmp_path / "turn_records.jsonl"
    _write(f, _rows(q))
    recs, skipped = li.load_turn_records(f)
    assert [r["ts"] for r in recs] == ["t1", "t4"]
    assert skipped == 2


def test_test_probe_key_wins_over_query_text(tmp_path):
    f = tmp_path / "r.jsonl"
    _write(f, [{"ts": "a", "query": CLEAN, "test_probe": False}])
    recs, skipped = li.load_turn_records(f)
    assert len(recs) == 1 and skipped == 0


@pytest.mark.parametrize("q", [CLEAN, WRAPPED])
def test_interactive_loader_and_summary(tmp_path, monkeypatch, capsys, q):
    f = tmp_path / "r.jsonl"
    _write(f, _rows(q))
    monkeypatch.setattr(li, "RECORDS", f)
    recs = li._load_records()
    assert [r["ts"] for r in recs] == ["t1", "t4"]
    assert "skipped 2 probe turns" in capsys.readouterr().out


@pytest.mark.parametrize("q", [CLEAN, WRAPPED])
def test_report_drops_probe_labels(tmp_path, monkeypatch, capsys, q):
    f = tmp_path / "r.jsonl"
    _write(f, _rows(q))
    lab = tmp_path / "labels.jsonl"
    _write(lab, [
        {"ts": "t1", "query": "x", "predicted": "factual_recall", "true": "factual_recall"},
        {"ts": "t2", "query": "probe text", "predicted": "general", "true": "general"},
        {"ts": "t3", "query": q, "predicted": "general", "true": "general"},
        {"ts": "t4", "query": "plain legacy question", "predicted": "general", "true": "general"},
    ])
    monkeypatch.setattr(li, "RECORDS", f)
    monkeypatch.setattr(li, "LABELS", lab)
    li.report()
    out = capsys.readouterr().out
    assert "2 labeled" in out and "skipped 2 probe turns" in out


@pytest.mark.parametrize("q", [CLEAN, WRAPPED])
def test_auto_labeler_never_labels_probes(tmp_path, monkeypatch, capsys, q):
    f = tmp_path / "r.jsonl"
    _write(f, _rows(q))
    lab = tmp_path / "labels.jsonl"
    seen = []

    class FakeMM:
        async def generate_once(self, prompt, **kw):
            seen.append(prompt)
            return "general"

    import models.model_manager as mmod
    monkeypatch.setattr(mmod, "ModelManager", FakeMM)
    monkeypatch.setattr(auto, "RECORDS", f)
    monkeypatch.setattr(auto, "LABELS", lab)
    monkeypatch.setattr("sys.argv", ["auto_label_intents.py"])
    assert asyncio.run(auto.main()) == 0
    written = [json.loads(x)["ts"] for x in lab.read_text().splitlines()]
    assert written == ["t1", "t4"]
    assert len(seen) == 2
    assert "skipped 2 probe turns" in capsys.readouterr().out
