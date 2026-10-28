#!/usr/bin/env python3
"""Weekly full run (no marker filter) including the slow, semantic and benchmark groups no push runs.

Module Contract:
  Purpose: the pre-push hook selects ``-m "not slow and not benchmark and not semantic"``; this job runs
           what pushes skip, in a dedicated scratch clone, Monday 03:30 (scripts/systemd/daemon-weekly-tests.*).
  Reads:   the live checkout's HEAD sha only (git rev-parse); never its data/ or any live store.
  Writes:  the scratch clone (checkout --detach), <live>/logs/weekly_tests/<date>.txt and summary.jsonl.
  Exit:    0 = success or skip; 1 = a pass failed/timed out/aborted; 2 = refused (dirty scratch clone, git error).

Skips (exit 0, one logged line): live Daemon running, MemAvailable below the floor, another run holds the lock.
Never resets, cleans or deletes anything.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HOME = Path.home()
DEFAULT_LIVE = HOME / "Daemon_v1"
DEFAULT_CLONE = HOME / "daemon_exec" / "weekly_tests_clone"
DEFAULT_PYTEST_LOCK = HOME / "daemon_exec" / ".pytest.lock"
DEFAULT_RUN_LOCK = HOME / "daemon_exec" / ".weekly_tests.lock"
DEFAULT_PYTHON = HOME / ".pyenv" / "versions" / "3.11.8" / "bin" / "python"
MIN_MEM_MB = 9000
PASS_TIMEOUT_MIN = 90
REAL_BENCHMARK_REL = Path("data/benchmarks_private/retrieval_benchmarks_real.yaml")
DEFAULT_REAL_BENCHMARK = DEFAULT_LIVE / REAL_BENCHMARK_REL

_COUNT_RE = re.compile(r"(\d+) (passed|failed|skipped|error|errors|deselected|xfailed|xpassed)")


def log(msg: str) -> None:
    print(f"[weekly-tests] {msg}", flush=True)


def mem_available_mb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 0


def parse_summary(text: str) -> dict:
    """Counts from the LAST pytest summary line (``== 3 failed, 10 passed in 5s ==`` or the -q form)."""
    counts = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    for line in reversed(text.splitlines()):
        if " in " in line and _COUNT_RE.search(line):
            for n, key in _COUNT_RE.findall(line):
                key = "error" if key == "errors" else key
                if key in counts:
                    counts[key] = int(n)
            break
    return counts


def failure_lines(text: str, limit: int = 200) -> list[str]:
    keep = [ln for ln in text.splitlines() if ln.startswith(("FAILED ", "ERROR "))]
    return keep[:limit]


def copy_real_benchmark(src: Path, clone: Path) -> str:
    """Copy the ONE private benchmark YAML into the clone (gitignored there); never opens anything else."""
    if not src.is_file():
        return "absent"
    dst = clone / REAL_BENCHMARK_REL
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    return "copied"


def skip_lines(text: str, limit: int = 60) -> list[str]:
    return [ln for ln in text.splitlines() if ln.startswith("SKIPPED ")][:limit]


def plan_passes(extra_selection: list[str] | None = None) -> list[dict]:
    if extra_selection:
        return [{"name": "selection", "args": list(extra_selection), "mem": "8G"}]
    return [
        {"name": "B_unit", "args": ["tests/unit"], "mem": "6G"},
        {"name": "B_remainder", "args": ["tests", "--ignore=tests/unit"], "mem": "8G"},
    ]


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def live_head_sha(live: Path) -> str:
    r = _git(live, "rev-parse", "HEAD")
    if r.returncode != 0:
        raise RuntimeError(f"cannot read HEAD of {live}: {r.stderr.strip()}")
    return r.stdout.strip()


def prepare_clone(live: Path, clone: Path, sha: str) -> tuple[bool, str]:
    """Create or refresh the scratch clone, detached at ``sha``. (False, reason) = refuse; nothing is reset."""
    if not clone.exists():
        clone.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["git", "clone", "-q", str(live), str(clone)], capture_output=True, text=True)
        if r.returncode != 0:
            return False, f"clone failed: {r.stderr.strip()}"
    st = _git(clone, "status", "--porcelain")
    if st.returncode != 0:
        return False, f"git status failed in scratch clone: {st.stderr.strip()}"
    if st.stdout.strip():
        return False, "scratch clone has local changes; refusing (nothing was touched)"
    if _git(clone, "cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
        _git(clone, "fetch", "-q", str(live), "+refs/heads/*:refs/remotes/live/*")
        if _git(clone, "cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
            _git(clone, "fetch", "-q", str(live), sha)
    co = _git(clone, "checkout", "-q", "--detach", sha)
    if co.returncode != 0:
        return False, f"checkout --detach {sha[:10]} failed: {co.stderr.strip()}"
    return True, "ok"


def build_command(python: str, clone: Path, pytest_lock: Path, p: dict) -> list[str]:
    env = ["env", "-u", "PYTHONPATH", f"PYTHONPATH={clone}/scripts/bin", "DAEMON_TEST_MODE=1",
           "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1"]
    inner = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={p['mem']}", "-p", "MemorySwapMax=512M",
             *env, python, "-s", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=short", "--color=no", "-rs", *p["args"]]
    return ["flock", str(pytest_lock), *inner]


def run_pass(cmd: list[str], clone: Path, timeout_s: int) -> dict:
    t0 = _dt.datetime.now()
    try:
        r = subprocess.run(cmd, cwd=str(clone), capture_output=True, text=True, timeout=timeout_s)
        out, rc, status = (r.stdout or "") + (r.stderr or ""), r.returncode, None
    except subprocess.TimeoutExpired as e:
        raw = e.stdout or b""
        out = raw.decode(errors="replace") if isinstance(raw, bytes) else raw
        rc, status = -1, "timeout"
    counts = parse_summary(out)
    if status is None:
        status = "ok" if rc == 0 else "failed"
    return {"rc": rc, "status": status, **counts, "duration_s": round((_dt.datetime.now() - t0).total_seconds(), 1),
            "failures": failure_lines(out), "skips": skip_lines(out), "tail": out.splitlines()[-8:]}


def write_report(report_dir: Path, day: str, sha: str, results: dict, status: str) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"weekly test run {day}  sha={sha}  status={status}", ""]
    for name, r in results.items():
        lines.append(f"== {name}: {r['status']} passed={r['passed']} failed={r['failed']} "
                     f"skipped={r['skipped']} error={r['error']} duration={r['duration_s']}s rc={r['rc']}")
        lines += [f"  {x}" for x in r["failures"]]
        lines += [f"  skip: {x}" for x in r.get("skips", [])]
        lines += [f"  | {x}" for x in r["tail"]]
        lines.append("")
    path = report_dir / f"{day}.txt"
    path.write_text("\n".join(lines))
    row = {"date": day, "sha": sha, "status": status,
           "passes": {n: {k: r[k] for k in ("status", "passed", "failed", "skipped", "error", "duration_s")}
                      for n, r in results.items()}}
    with open(report_dir / "summary.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    return path


def main(argv: list[str] | None = None, *, daemon_running=None, mem_fn=None, run_fn=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live-repo", type=Path, default=DEFAULT_LIVE)
    ap.add_argument("--clone", type=Path, default=DEFAULT_CLONE)
    ap.add_argument("--report-dir", type=Path, default=None)
    ap.add_argument("--pytest-lock", type=Path, default=DEFAULT_PYTEST_LOCK)
    ap.add_argument("--run-lock", type=Path, default=DEFAULT_RUN_LOCK)
    ap.add_argument("--real-benchmark", type=Path, default=DEFAULT_REAL_BENCHMARK,
                    help="private real-data benchmark YAML copied into the scratch clone when present")
    ap.add_argument("--python", default=str(DEFAULT_PYTHON))
    ap.add_argument("--timeout-min", type=int, default=PASS_TIMEOUT_MIN)
    ap.add_argument("--min-mem-mb", type=int, default=MIN_MEM_MB)
    ap.add_argument("--selection", nargs=argparse.REMAINDER, default=None,
                    help="pytest args replacing the three passes (testing); must be last")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    report_dir = a.report_dir or (a.live_repo / "logs" / "weekly_tests")
    if daemon_running is None:
        from utils.daemon_guard import daemon_running as daemon_running  # lazy import: only needed at run time
    mem_fn = mem_fn or mem_available_mb
    run_fn = run_fn or run_pass
    passes = plan_passes(a.selection)

    if a.dry_run:
        log(f"dry-run: live={a.live_repo} clone={a.clone} report_dir={report_dir} min_mem={a.min_mem_mb}MB")
        for p in passes:
            log(f"dry-run: pass {p['name']}: " + shlex.join(build_command(a.python, a.clone, a.pytest_lock, p)))
        return 0

    if daemon_running(a.live_repo):
        log("skip: the live Daemon is running")
        return 0
    mem = mem_fn()
    if mem < a.min_mem_mb:
        log(f"skip: MemAvailable {mem} MB < {a.min_mem_mb} MB")
        return 0
    a.run_lock.parent.mkdir(parents=True, exist_ok=True)
    lock_fh = open(a.run_lock, "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log("skip: another weekly run holds the lock")
        return 0
    try:
        try:
            sha = live_head_sha(a.live_repo)
        except RuntimeError as e:
            log(f"refused: {e}")
            return 2
        ok, why = prepare_clone(a.live_repo, a.clone, sha)
        if not ok:
            log(f"refused: {why}")
            return 2
        log(f"real benchmark: {copy_real_benchmark(a.real_benchmark, a.clone)}")
        day = _dt.date.today().isoformat()
        results: dict = {}
        status = "ok"
        for p in passes:
            if daemon_running(a.live_repo):
                log("aborted: Daemon started")
                status = "aborted: Daemon started"
                break
            log(f"pass {p['name']} at {sha[:10]}")
            r = run_fn(build_command(a.python, a.clone, a.pytest_lock, p), a.clone, a.timeout_min * 60)
            results[p["name"]] = r
            log(f"pass {p['name']}: {r['status']} passed={r['passed']} failed={r['failed']}")
            if r["status"] != "ok":
                status = "failed"
        rep = write_report(report_dir, day, sha, results, status)
        log(f"report: {rep}")
        return 0 if status == "ok" else 1
    finally:
        lock_fh.close()


if __name__ == "__main__":
    sys.exit(main())
