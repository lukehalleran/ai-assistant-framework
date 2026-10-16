"""2026-10-15 (class: BC-37): running the benchmarks rewrote a TRACKED file.

tests/benchmarks/conftest.py wrote data/benchmark_per_case.csv at the end of
every run; scripts/generate_doc_metrics.py reads that file for the public
retrieval figures. The default target is now the untracked logs/ path; the
tracked file is written only on `--benchmark-csv PATH`.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFTEST = ROOT / "tests" / "benchmarks" / "conftest.py"


def _load():
    spec = importlib.util.spec_from_file_location("_bench_conftest_under_test", CONFTEST)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_is_untracked_logs_path(tmp_path):
    p = _load().benchmark_csv_path(None, tmp_path)
    assert p == tmp_path / "logs" / "benchmark_per_case.csv"
    assert _load().benchmark_csv_path("", tmp_path) == p


def test_explicit_option_wins(tmp_path):
    m = _load()
    assert m.benchmark_csv_path("out/bench.csv", tmp_path) == tmp_path / "out" / "bench.csv"
    abs_p = tmp_path / "elsewhere" / "x.csv"
    assert m.benchmark_csv_path(str(abs_p), tmp_path) == abs_p


def test_conftest_no_longer_hardcodes_tracked_target():
    src = CONFTEST.read_text()
    assert '"data" / "benchmark_per_case.csv"' not in src
    assert "benchmark_csv_path(" in src


def test_option_registered_in_root_conftest(pytestconfig):
    # same-instance: the live pytest config knows the option (root conftest hook)
    assert pytestconfig.getoption("benchmark_csv", default="MISSING") is None


def test_logs_is_gitignored():
    assert "logs/" in (ROOT / ".gitignore").read_text().split()
