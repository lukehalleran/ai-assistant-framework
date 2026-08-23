"""
Dual-module shutdown bug (2026-08-22).

Launched as `python main.py`, the script runs as module `__main__`. api/app.py's
lifespan does `import main as main_mod` (and gui/handlers pokes
`main.update_activity_timestamp` the same way) — without an alias that import
RE-EXECUTES main.py as a second module instance with its own globals. Observed
2026-08-22 13:16-13:18: the FastAPI lifespan shutdown set `_shutdown_requested`
on the copy, the running instance's finally-block read False, and the full
shutdown sequence (reflection, LLM fact extraction, graph save) ran TWICE —
the second pass AFTER the backup, racing closing HTTP transports.

The fix: `sys.modules["main"] = sys.modules[__name__]` at the top of the
entry block, so every later `import main` resolves to the running instance.
"""

import re
from pathlib import Path


def _entry_block():
    src = Path("main.py").read_text()
    m = re.search(r'^if __name__ == "__main__":\n', src, re.MULTILINE)
    assert m, "main.py entry block missing"
    return src[m.end(): m.end() + 1500]


def test_entry_aliases_running_module_as_main():
    block = _entry_block()
    assert 'modules["main"]' in block, (
        "main.py entry must alias the running module as 'main' — otherwise "
        "api/app.py's `import main` creates a second module instance and the "
        "shutdown double-run guard is split across two copies of the global"
    )


def test_alias_precedes_gui_launch():
    # the alias must be the FIRST thing in the entry block, before anything
    # that could trigger an `import main` (uvicorn/lifespan/gui launch)
    block = _entry_block()
    alias_pos = block.find('modules["main"]')
    for launcher in ("launch_gui", "uvicorn", "create_app"):
        pos = block.find(launcher)
        if pos != -1:
            assert alias_pos < pos


def test_api_lifespan_still_imports_main():
    # the alias strategy only works while api/app.py resolves shutdown via
    # `import main` — if that changes, this file's premise needs revisiting
    src = Path("api/app.py").read_text()
    assert "import main as main_mod" in src
    assert "run_shutdown_tasks_async" in src
