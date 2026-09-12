"""Isolation boundary for pure bug-class catalog and scanner tests.

These tests run under ``--confcutdir=tests/bug_class_guards`` so the root
``tests/conftest.py`` (which imports the application, and with it torch) is
never loaded: the gate they cover runs in ``hooks/pre-push`` and in CI before
any application import, and the lane must stay stdlib-only and instant.

Two explicit path insertions, both deterministic:
  * ``scripts/`` so ``bug_class_guards`` resolves exactly as it does for
    ``python scripts/check_bug_classes.py`` (which gets it from sys.path[0]);
  * this directory, so ``fixtures`` resolves to THIS lane's snippets and can
    never be shadowed by ``tests/fixtures/`` as a namespace package.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPTS = ROOT / "scripts"

for entry in (str(SCRIPTS), str(HERE)):
    if entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)
