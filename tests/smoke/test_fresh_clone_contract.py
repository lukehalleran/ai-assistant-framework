"""tests/smoke/test_fresh_clone_contract.py

Lane 1 Subgoal C: proves, from the current checkout and a newly created,
isolated data directory, the start -> wizard -> 3 turns -> ASGI lifespan
shutdown (real backup) -> fresh process restart -> persisted-state read-back
contract. This is a Linux Python runtime contract, not clone/install proof.

Each boot runs as a genuine OS subprocess (tests/smoke/fresh_clone_boot.py)
with an EXPLICIT environment dict -- never os.environ.copy() -- so
DAEMON_TEST_MODE (set by tests/conftest.py for this pytest process) is
absent in the child and the real utils.backup_manager.run_shutdown_backup
path executes instead of its "skipped: test_mode" branch. Two separate
processes also means "boot 2 reads what boot 1 wrote" is a real claim: no
Python module state (main.py's own globals, loaded model singletons, ...)
survives between them, only what actually landed on disk.

Slow (real local embedder loads + two full orchestrator boots); not part of
the default unit lane. The required CI lane provisions the exact model
artifacts first and sets DAEMON_SMOKE_REQUIRED=1. For a local optional run,
the test skips when the offline cache is missing. Run memory-capped:
    flock ~/daemon_exec/.pytest.lock systemd-run --user --scope -p MemoryMax=6G \\
        python -m pytest tests/smoke/test_fresh_clone_contract.py -x -q -p no:cacheprovider
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

CLONE_ROOT = Path(__file__).resolve().parents[2]
BOOT_SCRIPT = Path(__file__).resolve().parent / "fresh_clone_boot.py"

# The boot script exits 4 specifically when HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE
# find no cached embedder -- an actionable "provision this machine" skip, not
# a failure of the contract itself.
_MISSING_MODEL_CACHE_EXIT = 4

_BOOT_TIMEOUT_S = 600


def _boot_env(run_dir: Path) -> dict:
    """Explicit env dict for one boot subprocess.

    Never os.environ.copy(): the pytest parent process has DAEMON_TEST_MODE=1
    set by tests/conftest.py (collection-time os.environ.setdefault), and
    inheriting it here would route utils.backup_manager.run_shutdown_backup
    into its no-op "skipped: test_mode" branch -- exactly the path that never
    proves a real backup ran (C_agentA_pathmap.md section 4 / gap 9).
    """
    data_root = run_dir / "data"
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "DISABLE_FS_GUARD": "1",
        "USER_PROFILE_PATH": str(data_root / "user_profile.json"),
        "CHROMA_PATH": str(data_root / "chroma"),
        "CORPUS_FILE": str(data_root / "corpus.json"),
        "KNOWLEDGE_GRAPH_PERSIST_PATH": str(data_root / "knowledge_graph.json"),
        "DAEMON_BACKUP_DIR": str(data_root / "backups"),
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        # Defense in depth (C_fable_decisions.md decision 7): DAEMON_TEST_MODE
        # is deliberately absent below, which re-arms location_resolver's
        # background IP-geolocation thread unless independently suppressed.
        "LOCATION_IP_LOOKUP_ENABLED": "0",
        # Keep thread pools bounded under the memory-capped systemd-run scope
        # (same hygiene scripts/audit_runtime_smoke.py applies).
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false",
        # NO DAEMON_TEST_MODE. NO API keys (OPENAI_API_KEY / TAVILY_API_KEY / ...).
    }
    return env


def _run_boot(run_dir: Path, boot: int, report_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-s",
            str(BOOT_SCRIPT),
            "--run-dir",
            str(run_dir),
            "--boot",
            str(boot),
            "--report",
            str(report_path),
        ],
        env=_boot_env(run_dir),
        capture_output=True,
        text=True,
        timeout=_BOOT_TIMEOUT_S,
    )


def _assert_no_traceback(proc: subprocess.CompletedProcess, run_dir: Path, label: str) -> None:
    assert "Traceback" not in proc.stdout, f"{label} stdout contained a traceback:\n{proc.stdout}"
    assert "Traceback" not in proc.stderr, f"{label} stderr contained a traceback:\n{proc.stderr}"
    for log_path in run_dir.rglob("*.log"):
        try:
            text = log_path.read_text(errors="replace")
        except OSError:
            continue
        assert "Traceback" not in text, f"{label}: {log_path} contained a traceback"


@pytest.mark.slow
def test_isolated_runtime_two_boot_cycle(tmp_path):
    run_dir = tmp_path / "daemon-fresh-clone"
    run_dir.mkdir()

    report1_path = tmp_path / "boot1_report.json"
    proc1 = _run_boot(run_dir, 1, report1_path)
    if proc1.returncode == _MISSING_MODEL_CACHE_EXIT:
        _handle_missing_model_cache(proc1, "boot 1")
    assert proc1.returncode == 0, (
        f"boot 1 failed (exit {proc1.returncode})\n"
        f"--- stdout ---\n{proc1.stdout}\n--- stderr ---\n{proc1.stderr}"
    )
    _assert_no_traceback(proc1, run_dir, "boot 1")

    assert report1_path.is_file(), "boot 1 did not write its report JSON"
    report1 = json.loads(report1_path.read_text())

    result1 = report1["result"]
    assert result1["wizard"]["final_step"] == "complete"
    assert result1["turns"]["debug_hit_pixel"] is True
    assert result1["backup"]["found"] is True
    assert result1["backup"]["has_profile"] is True
    assert result1["backup"]["has_corpus"] is True
    assert not result1["backup"].get("manifest", {}).get("skipped_reason")

    report2_path = tmp_path / "boot2_report.json"
    proc2 = _run_boot(run_dir, 2, report2_path)
    if proc2.returncode == _MISSING_MODEL_CACHE_EXIT:
        _handle_missing_model_cache(proc2, "boot 2")
    assert proc2.returncode == 0, (
        f"boot 2 failed (exit {proc2.returncode})\n"
        f"--- stdout ---\n{proc2.stdout}\n--- stderr ---\n{proc2.stderr}"
    )
    _assert_no_traceback(proc2, run_dir, "boot 2")

    assert report2_path.is_file(), "boot 2 did not write its report JSON"
    report2 = json.loads(report2_path.read_text())

    result2 = report2["result"]
    assert result2["corpus_count"] >= 3
    assert result2["identity_name"] == "Sam Fresh"
    assert result2["turns"]["debug_hit_pixel"] is True
    assert result2["backup"]["found"] is True

    # bootstrap.get_user_profile_path() resolved identically across the
    # restart -- the persisted-state read-back this contract is about
    # (report["resolved_paths"][2], per _isolation_check's fixed ordering).
    profile_path_1 = report1["resolved_paths"][2]
    profile_path_2 = report2["resolved_paths"][2]
    assert profile_path_1 == profile_path_2, (
        f"boot 1 and boot 2 resolved different profile paths: "
        f"{profile_path_1!r} != {profile_path_2!r}"
    )


def _handle_missing_model_cache(proc: subprocess.CompletedProcess, label: str) -> None:
    reason = (
        f"HF embedder cache is missing during {label} (HF_HUB_OFFLINE=1). "
        f"stderr:\n{proc.stderr[-2000:]}"
    )
    if os.environ.get("DAEMON_SMOKE_REQUIRED") == "1":
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)
