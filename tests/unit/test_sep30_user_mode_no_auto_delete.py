"""2026-09-30: user mode set CROSS_DEDUP_AUTO_EXECUTE=True — every shutdown would
execute cross-collection dedup deletions unreviewed (CLAUDE.md: NEVER auto-delete
user data). Latent while the committed default mode was dev; the default is now
user, so both must hold."""
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def _constant_under(mode: str) -> str:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(DAEMON_MODE=mode, DISABLE_FS_GUARD="1", DAEMON_TEST_MODE="1")
    out = subprocess.run(
        [sys.executable, "-s", "-c",
         "import config.app_config as c; print(c.CROSS_DEDUP_AUTO_EXECUTE)"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


def test_user_mode_never_auto_executes_dedup():
    assert _constant_under("user") == "False"


def test_dev_mode_never_auto_executes_dedup():
    assert _constant_under("dev") == "False"


def test_committed_default_mode_is_user():
    cfg = yaml.safe_load((REPO / "config" / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["daemon"]["mode"] == "user"
