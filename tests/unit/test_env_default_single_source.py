"""
Guard against the dead/contradictory env-default class found 2026-09-19 in
core/prompt/context_gatherer.py and core/prompt/summarizer.py: the same
env var name was read in two different places with two different literal
defaults, and one of the readers had no callers at all (a dead
re-derivation of a threshold owned elsewhere), while the other's default
silently disagreed with its only caller's own gating constant.

This is a PINNED set, not a repo-wide policy: an `rg` sweep at the time
this test was written found roughly a dozen more env names with differing
literal defaults across the tree (e.g. HEAVY_TOPIC_MAX_TOKENS, which names
two unrelated things in utils/query_checker.py and config/app_config.py
with no shared reader) — those are recorded for a later, broader pass
(plan P2) and are deliberately NOT asserted on here. Only the two names
verified to be a genuine single-concept collision are pinned below.

Scope is core/ + processing/ (where both known incidents live), matching
os.getenv(NAME, DEFAULT) and os.environ.get(NAME, DEFAULT) calls whose
name and default are both literal constants (dynamic names/defaults can't
be compared here and are skipped, not flagged).
"""
import ast
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = ("core", "processing")

# Names known (2026-09-19) to be a genuine single-concept collision: the
# same env var controls one thing but disagreed on its default across two
# reader sites. Growing this set is a deliberate, reviewed decision — see
# the module docstring for why the rest of the repo's differing defaults
# are out of scope here.
PINNED_NAMES = frozenset({"GATE_COSINE_THRESHOLD", "REFLECTIONS_ON_DEMAND"})


def _literal(node):
    """The Python value of an ast.Constant, else a sentinel meaning
    "not a literal we can compare" (dynamic default, f-string, name, ...)."""
    if isinstance(node, ast.Constant):
        return node.value
    return _NOT_LITERAL


_NOT_LITERAL = object()


def _is_getenv_call(node) -> bool:
    """os.getenv(...) or os.environ.get(...) — by attribute shape, not by
    which module actually bound the name (matches the codebase's own
    `import os` convention throughout)."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr == "getenv" and isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
        return True
    if (node.func.attr == "get" and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "environ"
            and isinstance(node.func.value.value, ast.Name) and node.func.value.value.id == "os"):
        return True
    return False


def _find_env_defaults_in_source(source: str, rel: str):
    """[(name, default_literal, rel, lineno), ...] for every getenv-shaped
    call with a literal name and a literal (or absent) default."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    found = []
    for node in ast.walk(tree):
        if not _is_getenv_call(node):
            continue
        if len(node.args) < 1:
            continue
        name = _literal(node.args[0])
        if not isinstance(name, str):
            continue
        if len(node.args) >= 2:
            default = _literal(node.args[1])
        else:
            default = None  # os.getenv(NAME) with no default -> None
        if default is _NOT_LITERAL:
            continue  # dynamic default (e.g. str(some_var)) — can't compare
        found.append((name, default, rel, node.lineno))
    return found


def _iter_py_files():
    for d in SCAN_DIRS:
        base = REPO_ROOT / d
        if base.exists():
            yield from base.rglob("*.py")


def _collect_defaults_by_name():
    by_name = defaultdict(list)  # name -> [(default, rel, lineno), ...]
    for path in _iter_py_files():
        rel = str(path.relative_to(REPO_ROOT))
        try:
            source = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for name, default, rel_, lineno in _find_env_defaults_in_source(source, rel):
            by_name[name].append((default, rel_, lineno))
    return by_name


class TestPinnedEnvDefaultsAgree:
    def test_pinned_names_have_a_single_literal_default(self):
        by_name = _collect_defaults_by_name()
        conflicts = []
        for name in sorted(PINNED_NAMES):
            sites = by_name.get(name, [])
            distinct = sorted({default for default, _rel, _lineno in sites}, key=repr)
            if len(distinct) > 1:
                where = "; ".join(f"{rel}:{lineno}={default!r}" for default, rel, lineno in sites)
                conflicts.append(f"{name}: defaults {distinct!r} disagree ({where})")
        assert not conflicts, (
            "Pinned env var(s) read with more than one literal default — "
            "pick one owner and align the rest:\n" + "\n".join(conflicts)
        )

    def test_pinned_names_are_still_read_somewhere(self):
        """Sanity: if a pinned name's only reader were ever deleted, the
        first test above would pass vacuously (zero sites -> zero
        conflicts). Catch that silently-vacuous case explicitly."""
        by_name = _collect_defaults_by_name()
        missing = sorted(name for name in PINNED_NAMES if not by_name.get(name))
        assert not missing, (
            f"Pinned env var(s) {missing} have no getenv-shaped reader left "
            "in core/ or processing/ — remove them from PINNED_NAMES "
            "(the collision they guarded against no longer exists)."
        )


class TestDetectorSelfTest:
    """Proves the scanner actually flags a two-defaults collision and
    passes a single-default (or absent) case, using synthetic source —
    independent of whatever the real tree currently looks like."""

    def test_two_different_literal_defaults_is_flagged(self):
        a = 'import os\nX = float(os.getenv("FOO_THRESH", "0.45"))\n'
        b = 'import os\nX = float(os.getenv("FOO_THRESH", "0.50"))\n'
        found = _find_env_defaults_in_source(a, "a.py") + _find_env_defaults_in_source(b, "b.py")
        distinct = {default for _name, default, _rel, _lineno in found if default is not None}
        assert distinct == {"0.45", "0.50"}

    def test_matching_literal_defaults_is_not_flagged(self):
        a = 'import os\nX = os.getenv("FOO_THRESH", "0")\n'
        b = 'import os\nY = os.environ.get("FOO_THRESH", "0")\n'
        found = _find_env_defaults_in_source(a, "a.py") + _find_env_defaults_in_source(b, "b.py")
        distinct = {default for _name, default, _rel, _lineno in found}
        assert distinct == {"0"}

    def test_dynamic_default_is_skipped_not_flagged(self):
        src = 'import os\nfallback = "1"\nX = os.getenv("FOO_THRESH", fallback)\n'
        assert _find_env_defaults_in_source(src, "a.py") == []

    def test_no_default_argument_yields_none(self):
        src = 'import os\nX = os.getenv("FOO_THRESH")\n'
        found = _find_env_defaults_in_source(src, "a.py")
        assert found == [("FOO_THRESH", None, "a.py", 2)]

    def test_unparseable_file_yields_nothing(self):
        assert _find_env_defaults_in_source("def broken(:\n", "a.py") == []
