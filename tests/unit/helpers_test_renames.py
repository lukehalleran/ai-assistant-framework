"""Shared synthetic-repo builder for the test-rename tooling tests (2026-10-16).

Never touches the real tests/ tree: everything is built in a pytest tmp_path with its own throwaway git repo.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

TEST_LANES_PARAGRAPH = (
    "## 6. Naming new tests\n\n"
    "New test files are named by module and behaviour:\n"
    "`tests/unit/test_<module>_<behaviour>.py`. The incident date and the bug\n"
    "class (`BC-nn`) go in the module docstring, not the file name. Existing\n"
    "dated files (`test_sep*.py`, `test_oct*.py`, ...) are NOT renamed: they are\n"
    "cited by path across `docs/BUG_CLASSES.md`, the changelog, disposition\n"
    "records and CGR files.\n"
)

GUARD = '''"""synthetic guard"""
APPROVED_EXCEPTIONS = frozenset({"tests/unit/test_sep10_calendar_forced_action_loop.py"})
# BEGIN PRE-MIGRATION ALLOWLIST (temporary)
PRE_MIGRATION_ALLOWLIST = frozenset({
    "tests/unit/test_sep09_engine_widget.py",
})
# END PRE-MIGRATION ALLOWLIST
'''

FILES: dict[str, str] = {
    "core/engine.py": "def run(x):\n    return x\n\n\n# regression: tests/unit/test_sep09_engine_widget.py:@@LOC@@\n",
    "core/valence.py": "def v():\n    return 1\n",
    "pkg_a/alpha.py": "def a():\n    return 1\n",
    "pkg_b/alpha.py": "def a():\n    return 2\n",
    "scripts/tool.py": "def main():\n    return 0\n",
    "api/routes/chat.py": "def handler():\n    return 1\n",
    "tests/__init__.py": "",
    "tests/unit/__init__.py": "",
    "tests/conftest.py": (
        "import re as _re\n\n_SAFETY_TEST_RE = _re.compile(\n    r\"(tone_detect|escalation|valence)\"\n)\n"
    ),
    "tests/RENAMES.tsv": "kind\told_path\tnew_path\tsubject_module\tother_subjects\tconfidence\tevidence\tflags\treason\n",
    "tests/unit/test_test_file_naming_guard.py": GUARD,
    "tests/unit/test_sep09_engine_widget.py": (
        '"""2026-09-09 engine widget regression (class: BC-01)."""\n\n'
        "from core.engine import run\n\n\n"
        "def test_run():\n"
        "    assert run(1) == 1\n"
    ),
    "tests/unit/test_sep10_tool_dry_run.py": (
        '"""2026-09-10 tool dry run."""\n\n'
        "import importlib.util\nfrom pathlib import Path\n\nfrom core.engine import run  # noqa: F401\n\n"
        'SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "tool.py"\n\n\n'
        "def _load():\n"
        '    spec = importlib.util.spec_from_file_location("tool_under_test", SCRIPT)\n'
        "    mod = importlib.util.module_from_spec(spec)\n"
        "    spec.loader.exec_module(mod)\n"
        "    return mod\n\n\n"
        "def test_main():\n"
        "    assert _load().main() == 0\n"
    ),
    "tests/unit/test_oct04_static_dominates.py": (
        "import importlib.util\nfrom pathlib import Path\n\nfrom core import engine\n\n"
        'SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "tool.py"\n\n\n'
        "def _load():\n"
        '    spec = importlib.util.spec_from_file_location("tool_under_test", SCRIPT)\n'
        "    return importlib.util.module_from_spec(spec)\n\n\n"
        "def test_many():\n"
        + "".join(f"    assert engine.run({i}) == {i}\n" for i in range(10))
    ),
    "tests/unit/test_oct01_thing.py": "from pkg_b.alpha import a\n\n\ndef test_a():\n    assert a() == 2\n",
    "tests/unit/test_oct02_mixed.py": (
        "from core.engine import run\nfrom pkg_a.alpha import a\n\n\n"
        "def test_both():\n    assert run(a()) == 1\n"
    ),
    "tests/unit/test_sep11_B4.py": "from core.engine import run\n\n\ndef test_x():\n    assert run(2) == 2\n",
    "tests/unit/test_oct05_flush.py": "from core.engine import run\n\n\ndef test_x():\n    assert run(3) == 3\n",
    "tests/unit/test_sep12_flush.py": "from core.engine import run\n\n\ndef test_y():\n    assert run(4) == 4\n",
    "tests/unit/test_oct06_misc.py": "from core.valence import v\n\n\ndef test_v():\n    assert v() == 1\n",
    "tests/unit/test_sep27b_B5.py": "from core.engine import run\n\n\ndef test_b5():\n    assert run(5) == 5\n",
    "tests/unit/test_oct03_user_of_helper.py": (
        "from api.routes.chat import handler\n"
        "from tests.unit.test_sep09_engine_widget import run\n\n\n"
        "def test_u():\n    assert run(handler()) == 1\n"
    ),
    "tests/unit/test_sep10_calendar_forced_action_loop.py": (
        '"""exempt anchor"""\nfrom core.engine import run\n\n\ndef test_anchor():\n    assert run(6) == 6\n'
    ),
    "docs/TEST_LANES.md": "# Lanes\n\n" + TEST_LANES_PARAGRAPH + "\n## 7. Other\n",
    "docs/BUG_CLASSES.md": (
        "# Classes\n\n"
        "- Closure: the guard is pinned by `tests/unit/test_sep09_engine_widget.py`.\n"
        "- Incidents: 2026-09-09 the first regression is test_sep09_engine_widget.py in the narrative.\n"
        "| DM-04 | probe | test pattern, `test_oct01_thing.py` | BC-11 |\n"
        "- Find: run `pytest tests/unit/test_sep27b_B5.py -q`.\n"
    ),
    "docs/PLAN_20261001_history.md": "History: see test_sep09_engine_widget.py (never rewritten).\n",
    "docs/notes.md": "Upper-case mention TEST_SEP09_ENGINE_WIDGET.py is a case variant.\n",
    "CLAUDE.md": "Project note: tests/unit/test_oct02_mixed.py covers the mix.\n",
}


# pinned commit dates: two synthetic repos built in different seconds must still get the same base
# SHA, or the sealed map (which records it) and so tests/RENAMES.tsv differ between them
_GIT_DATES = {"GIT_AUTHOR_DATE": "2026-10-16T12:00:00+0000", "GIT_COMMITTER_DATE": "2026-10-16T12:00:00+0000"}


def git(cwd: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-c", "user.name=T", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", *args],
        cwd=str(cwd), check=True, capture_output=True, text=True, env={**os.environ, **_GIT_DATES},
    )
    return out.stdout


def make_repo(root: Path, extra: dict[str, str] | None = None, drop: tuple[str, ...] = ()) -> Path:
    """Create the synthetic repo (committed) at ``root`` and return it."""
    files = dict(FILES)
    files.update(extra or {})
    for k in drop:
        files.pop(k, None)
    # locator line: the `assert run(1) == 1` line of the widget test
    widget = files["tests/unit/test_sep09_engine_widget.py"].splitlines()
    loc = next(i for i, ln in enumerate(widget, 1) if "assert run(1)" in ln)
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text.replace("@@LOC@@", str(loc)), encoding="utf-8")
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    return root


def snapshot(root: Path) -> dict[str, str]:
    """path -> sha256 of every file outside .git."""
    out: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for name in filenames:
            p = Path(dirpath) / name
            out[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out
