"""Ratchet semantics of the content-anchored bug-class baseline.

The gate has to fail in BOTH directions.  A new instance of a class must go
red, and so must a baseline entry whose finding is gone — otherwise fixed debt
is laundered by a row nobody deletes, and the file slowly stops describing the
tree.  Empty input is a failure too (BC-64: a scanner that processed zero
files has not proved anything), and a scanner that raises exits 2 rather than
reporting a clean pass (BC-47: "could not check" is not "nothing wrong").
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import check_bug_classes
from bug_class_guards import baseline as baseline_mod
from bug_class_guards.scanners import SCANNERS

from fixtures import (
    DM01_GREEN_ORDINARY,
    DM01_RED,
    DM17_GREEN_SCRIPT,
    DM17_GREEN_TEST,
    DM18_GREEN_NARROW,
    DM29_WIDENING_LINE,
    build_tree,
    catalog_doc,
    changelog,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_BASELINE = REPO_ROOT / baseline_mod.DEFAULT_BASELINE



def _mini_repo(tmp_path: Path, extra: dict[str, str] | None = None) -> Path:
    """A tree with at least one file in EVERY gated scanner's roots.

    Leaving any of them empty is itself a gate failure (BC-64), so the green
    controls double as proof that the empty-input rule is not accidentally
    satisfied by the fixture.
    """
    files = {
        "core/tone.py": DM01_RED,
        "memory/store.py": DM18_GREEN_NARROW,
        "scripts/purge_things.py": DM17_GREEN_SCRIPT,
        "tests/unit/test_store.py": DM17_GREEN_TEST,
        "docs/BUG_CLASSES.md": catalog_doc([("BC-01", "closed")]),
        "CLAUDE_CHANGELOG.md": changelog([("2026-09-10", "nothing widened here.")]),
    }
    files.update(extra or {})
    return build_tree(tmp_path, files)


def _scan(tmp_path: Path, *args: str) -> int:
    baseline = tmp_path / "baseline.json"
    return check_bug_classes.main(
        ["scan", "--root", str(tmp_path), "--baseline", str(baseline), *args]
    )


class TestRatchet:
    def test_write_baseline_round_trips_and_then_scans_clean(self, tmp_path, capsys):
        _mini_repo(tmp_path)
        assert _scan(tmp_path, "--write-baseline") == 0
        capsys.readouterr()
        stored = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
        assert stored["schema"] == baseline_mod.SCHEMA
        assert {entry["scanner"] for entry in stored["entries"]} == {"dm01_raw_substring"}
        assert _scan(tmp_path) == 0
        assert "bug-class scan: OK" in capsys.readouterr().out

    def test_a_new_finding_fails(self, tmp_path, capsys):
        _mini_repo(tmp_path)
        assert _scan(tmp_path, "--write-baseline") == 0
        capsys.readouterr()
        build_tree(tmp_path, {"memory/other.py": DM01_RED})
        assert _scan(tmp_path) == 1
        out = capsys.readouterr().out
        assert "NEW findings (2)" in out
        assert "memory/other.py" in out

    def test_a_stale_baseline_entry_fails(self, tmp_path, capsys):
        _mini_repo(tmp_path)
        assert _scan(tmp_path, "--write-baseline") == 0
        capsys.readouterr()
        build_tree(tmp_path, {"core/tone.py": DM01_GREEN_ORDINARY})
        assert _scan(tmp_path) == 1
        out = capsys.readouterr().out
        assert "STALE baseline entries (2)" in out
        assert "delete the entry" in out

    def test_pure_line_number_drift_still_matches(self, tmp_path, capsys):
        _mini_repo(tmp_path)
        assert _scan(tmp_path, "--write-baseline") == 0
        capsys.readouterr()
        drifted = DM01_RED.replace(
            "def is_heavy(text):\n    text_lower = text.lower()",
            "EXTRA_A = 1\nEXTRA_B = 2\n\n\ndef is_heavy(text):\n"
            "    text = text.strip()\n    text_lower = text.lower()",
        )
        assert drifted != DM01_RED
        build_tree(tmp_path, {"core/tone.py": drifted})
        assert _scan(tmp_path) == 0
        assert "bug-class scan: OK" in capsys.readouterr().out

    def test_moving_the_line_into_another_function_goes_stale(self, tmp_path, capsys):
        _mini_repo(tmp_path)
        assert _scan(tmp_path, "--write-baseline") == 0
        capsys.readouterr()
        moved = DM01_RED.replace("def is_heavy(text):", "def is_heavy_renamed(text):")
        build_tree(tmp_path, {"core/tone.py": moved})
        assert _scan(tmp_path) == 1
        out = capsys.readouterr().out
        assert "NEW findings" in out and "STALE baseline entries" in out


class TestFailClosed:
    def test_zero_processed_files_is_a_failure_not_a_pass(self, tmp_path, capsys):
        (tmp_path / "baseline.json").write_text(
            json.dumps({"schema": 1, "entries": []}), encoding="utf-8"
        )
        assert _scan(tmp_path, "--scanner", "dm01_raw_substring") == 1
        assert "EMPTY INPUT" in capsys.readouterr().out

    def test_scanner_error_exits_two(self, tmp_path, capsys):
        (tmp_path / "baseline.json").write_text(
            json.dumps({"schema": 1, "entries": []}), encoding="utf-8"
        )
        assert _scan(tmp_path, "--scanner", "catalog") == 2
        assert "scanner error" in capsys.readouterr().err

    def test_corrupt_baseline_exits_two(self, tmp_path, capsys):
        _mini_repo(tmp_path)
        (tmp_path / "baseline.json").write_text("{not json", encoding="utf-8")
        assert _scan(tmp_path) == 2
        assert "baseline error" in capsys.readouterr().err

    def test_unsupported_baseline_schema_exits_two(self, tmp_path, capsys):
        _mini_repo(tmp_path)
        (tmp_path / "baseline.json").write_text(
            json.dumps({"schema": 99, "entries": []}), encoding="utf-8"
        )
        assert _scan(tmp_path) == 2
        assert "baseline error" in capsys.readouterr().err

    def test_unknown_scanner_is_rejected(self, tmp_path):
        with pytest.raises(SystemExit):
            _scan(tmp_path, "--scanner", "dm99_nope")


class TestReportOnly:
    def test_report_only_findings_print_but_never_fail_the_gate(self, tmp_path, capsys):
        noisy = changelog(
            [
                ("2026-08-15", DM29_WIDENING_LINE),
                ("2026-09-07", DM29_WIDENING_LINE),
                ("2026-09-10", DM29_WIDENING_LINE),
            ]
        )
        _mini_repo(tmp_path, {"CLAUDE_CHANGELOG.md": noisy})
        assert _scan(tmp_path, "--write-baseline") == 0
        out = capsys.readouterr().out
        assert "report-only dm29_phrase_append_signature" in out
        assert "_INFO_SEEKING_CUES" in out
        stored = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
        assert all(
            entry["scanner"] != "dm29_phrase_append_signature"
            for entry in stored["entries"]
        )

    def test_report_only_scanner_with_zero_files_does_not_fail(self, tmp_path, capsys):
        _mini_repo(tmp_path)
        (tmp_path / "CLAUDE_CHANGELOG.md").unlink()
        assert _scan(tmp_path, "--write-baseline") == 0
        assert "bug-class scan: OK" in capsys.readouterr().out


class TestJsonOutput:
    def test_json_carries_findings_counts_and_exit(self, tmp_path, capsys):
        _mini_repo(tmp_path)
        assert _scan(tmp_path, "--write-baseline", "--json") == 0
        captured = capsys.readouterr()
        # --json makes stdout PURE JSON; the human table goes to stderr.
        payload = json.loads(captured.out)
        assert "scanner" in captured.err
        assert payload["exit"] == 0
        assert payload["new"] == [] and payload["stale"] == []
        assert set(payload["scanners"]) == set(SCANNERS)
        dm01 = payload["scanners"]["dm01_raw_substring"]
        assert dm01["mode"] == "gate" and dm01["class_ids"] == ["BC-01", "BC-02"]
        assert len(dm01["findings"]) == 2


class TestLiveTree:
    """Parity smoke — this is the test that goes red when the baseline drifts.

    It runs the real scanners over the real repository with the COMMITTED
    baseline.  A failure here means either a new instance of a bug class was
    introduced (fix it, or record it deliberately with ``--write-baseline``)
    or debt was fixed and its baseline entry must be deleted.
    """

    def test_committed_baseline_matches_the_live_tree(self, capsys):
        assert COMMITTED_BASELINE.is_file(), "committed baseline is missing"
        code = check_bug_classes.main(["scan", "--root", str(REPO_ROOT)])
        out = capsys.readouterr().out
        assert code == 0, out
        assert "bug-class scan: OK" in out

    def test_no_gated_scanner_walks_an_empty_tree(self, capsys):
        check_bug_classes.main(["scan", "--root", str(REPO_ROOT), "--json"])
        payload = json.loads(capsys.readouterr().out)
        for name, data in payload["scanners"].items():
            if SCANNERS[name].mode == "gate":
                assert data["files_processed"] > 0, name

    def test_committed_baseline_is_sorted_and_relative(self):
        payload = json.loads(COMMITTED_BASELINE.read_text(encoding="utf-8"))
        keys = [
            (e["scanner"], e["path"], e["symbol"], e["text"]) for e in payload["entries"]
        ]
        assert keys == sorted(keys)
        assert all(not key[1].startswith("/") for key in keys)
