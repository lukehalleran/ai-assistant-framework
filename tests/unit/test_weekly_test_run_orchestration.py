"""scripts/weekly_test_run.py: weekly run of the test groups pushes skip (owner decision 2026-10-28).

Bug class: BC-63 (coverage: slow/semantic/benchmark never ran), BC-58.  Git use is THROWAWAY tmp_path repos only
(tests/unit/test_no_git_state_in_tests.py); pytest is never really run - run_fn is a fake.
"""
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("weekly_test_run", ROOT / "scripts" / "weekly_test_run.py")
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)

GIT = ["git", "-c", "user.email=t@example.invalid", "-c", "user.name=Tamsin"]


def git(repo, *a):
    return subprocess.run([*GIT, "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def live(tmp_path):
    r = tmp_path / "live"
    r.mkdir()
    git(r, "init", "-q")
    (r / "a.txt").write_text("1")
    git(r, "add", ".")
    git(r, "commit", "-qm", "one")
    return r


def ok_result(**kw):
    base = {"rc": 0, "status": "ok", "passed": 5, "failed": 0, "skipped": 1, "error": 0, "duration_s": 1.0,
            "failures": [], "tail": ["5 passed, 1 skipped in 1.00s"]}
    base.update(kw)
    return base


def args(tmp_path, live, *extra):
    return ["--live-repo", str(live), "--clone", str(tmp_path / "clone"), "--report-dir", str(tmp_path / "rep"),
            "--run-lock", str(tmp_path / "run.lock"), "--pytest-lock", str(tmp_path / "p.lock"),
            "--real-benchmark", str(tmp_path / "no_such_benchmark.yaml"), *extra]


def test_skips_when_daemon_running(tmp_path, live, capsys):
    called = []
    rc = w.main(args(tmp_path, live), daemon_running=lambda r: True, mem_fn=lambda: 99999,
                run_fn=lambda *a: called.append(a))
    assert rc == 0 and not called and "Daemon is running" in capsys.readouterr().out
    assert not (tmp_path / "clone").exists()


def test_skips_on_low_memory(tmp_path, live, capsys):
    rc = w.main(args(tmp_path, live), daemon_running=lambda r: False, mem_fn=lambda: 8999, run_fn=None)
    assert rc == 0 and "MemAvailable 8999" in capsys.readouterr().out
    assert not (tmp_path / "clone").exists()


def test_skips_when_lock_held(tmp_path, live, capsys):
    import fcntl
    fh = open(tmp_path / "run.lock", "w")
    fcntl.flock(fh, fcntl.LOCK_EX)
    try:
        rc = w.main(args(tmp_path, live), daemon_running=lambda r: False, mem_fn=lambda: 99999,
                    run_fn=lambda *a: pytest.fail("must not run"))
    finally:
        fh.close()
    assert rc == 0 and "holds the lock" in capsys.readouterr().out


def test_clone_created_then_detached_to_given_sha(tmp_path, live):
    sha1 = git(live, "rev-parse", "HEAD")
    clone = tmp_path / "clone"
    assert w.prepare_clone(live, clone, sha1) == (True, "ok")
    (live / "a.txt").write_text("2")
    git(live, "commit", "-qam", "two")
    sha2 = git(live, "rev-parse", "HEAD")
    assert w.prepare_clone(live, clone, sha2)[0]
    assert git(clone, "rev-parse", "HEAD") == sha2
    assert subprocess.run(["git", "-C", str(clone), "symbolic-ref", "-q", "HEAD"], capture_output=True).returncode != 0


def test_dirty_clone_refused_untouched(tmp_path, live, capsys):
    sha = git(live, "rev-parse", "HEAD")
    clone = tmp_path / "clone"
    w.prepare_clone(live, clone, sha)
    (clone / "a.txt").write_text("local edit")
    (clone / "new.txt").write_text("x")
    rc = w.main(args(tmp_path, live), daemon_running=lambda r: False, mem_fn=lambda: 99999,
                run_fn=lambda *a: pytest.fail("must not run"))
    assert rc == 2 and "local changes" in capsys.readouterr().out
    assert (clone / "a.txt").read_text() == "local edit" and (clone / "new.txt").exists()


def test_report_summary_and_exit_codes(tmp_path, live):
    calls = []

    def run_ok(cmd, clone, t):
        calls.append(cmd)
        return ok_result()
    rc = w.main(args(tmp_path, live), daemon_running=lambda r: False, mem_fn=lambda: 99999, run_fn=run_ok)
    assert rc == 0 and len(calls) == 2
    rows = [json.loads(x) for x in (tmp_path / "rep" / "summary.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["status"] == "ok" and rows[0]["sha"] == git(live, "rev-parse", "HEAD")
    assert set(rows[0]["passes"]) == {"B_unit", "B_remainder"}
    assert rows[0]["passes"]["B_unit"]["passed"] == 5
    rep = next((tmp_path / "rep").glob("????-??-??.txt"))
    assert "status=ok" in rep.read_text()

    def run_bad(cmd, clone, t):
        return ok_result(rc=1, status="failed", failed=2, failures=["FAILED tests/x.py::t - boom"])
    rc = w.main(args(tmp_path, live), daemon_running=lambda r: False, mem_fn=lambda: 99999, run_fn=run_bad)
    assert rc == 1
    last = json.loads((tmp_path / "rep" / "summary.jsonl").read_text().splitlines()[-1])
    assert last["status"] == "failed"
    assert "FAILED tests/x.py::t - boom" in rep.read_text()


def test_daemon_start_between_passes_aborts(tmp_path, live):
    seq = iter([False, False, True])  # entry check, before pass 1, before pass 2
    ran = []
    rc = w.main(args(tmp_path, live), daemon_running=lambda r: next(seq), mem_fn=lambda: 99999,
                run_fn=lambda *a: ran.append(1) or ok_result())
    assert rc == 1 and len(ran) == 1
    last = json.loads((tmp_path / "rep" / "summary.jsonl").read_text().splitlines()[-1])
    assert last["status"].startswith("aborted")


def test_dry_run_prints_plan_runs_nothing(tmp_path, live, capsys):
    rc = w.main(args(tmp_path, live, "--dry-run"), daemon_running=lambda r: pytest.fail("no checks"),
                mem_fn=lambda: pytest.fail("no checks"), run_fn=lambda *a: pytest.fail("no run"))
    out = capsys.readouterr().out
    assert rc == 0 and "not slow" not in out and "slow or" not in out and "-rs" in out and "MemoryMax=8G" in out and "MemoryMax=6G" in out
    assert "PYTHONPATH=" in out and "DAEMON_TEST_MODE=1" in out and "HF_HUB_OFFLINE=1" in out
    assert not (tmp_path / "clone").exists() and not (tmp_path / "rep").exists()


def test_parse_summary():
    assert w.parse_summary("x\n=== 3 failed, 10 passed, 2 skipped in 5.00s ===")["failed"] == 3
    assert w.parse_summary("5 passed, 1 error in 1s")["error"] == 1
    assert w.parse_summary("nothing") == {"passed": 0, "failed": 0, "skipped": 0, "error": 0}


def test_timer_is_monday_0330_not_persistent():
    t = (ROOT / "scripts" / "systemd" / "daemon-weekly-tests.timer").read_text()
    assert re.search(r"^OnCalendar=Mon \*-\*-\* 03:30:00$", t, re.M)
    assert re.search(r"^Persistent=false$", t, re.M)
    s = (ROOT / "scripts" / "systemd" / "daemon-weekly-tests.service").read_text()
    assert "OnFailure=daemon-weekly-tests-failed.service" in s
    assert (ROOT / "scripts" / "systemd" / "daemon-weekly-tests-failed.service").exists()


def test_only_one_pass_family_and_no_marker_filter():
    names = [p["name"] for p in w.plan_passes()]
    assert names == ["B_unit", "B_remainder"]
    assert all("-m" not in p["args"] for p in w.plan_passes())


def test_real_benchmark_copied_when_present_and_clone_stays_clean(tmp_path, live):
    src = tmp_path / "private" / "bench.yaml"
    src.parent.mkdir()
    src.write_text("cases: []\n")
    (live / ".gitignore").write_text(f"{w.REAL_BENCHMARK_REL.parent}/\n")
    git(live, "add", ".")
    git(live, "commit", "-qm", "ignore")
    rc = w.main(args(tmp_path, live)[:-2] + ["--real-benchmark", str(src)], daemon_running=lambda r: False,
                mem_fn=lambda: 99999, run_fn=lambda *a: ok_result())
    assert rc == 0
    assert (tmp_path / "clone" / w.REAL_BENCHMARK_REL).read_text() == "cases: []\n"
    assert git(tmp_path / "clone", "status", "--porcelain") == ""


def test_real_benchmark_absent_is_fine(tmp_path, live, capsys):
    rc = w.main(args(tmp_path, live),
                daemon_running=lambda r: False, mem_fn=lambda: 99999, run_fn=lambda *a: ok_result())
    assert rc == 0 and "real benchmark: absent" in capsys.readouterr().out
    assert not (tmp_path / "clone" / w.REAL_BENCHMARK_REL).exists()
