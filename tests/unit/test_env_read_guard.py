"""
Repo-wide guard for BC-88 (setting read straight from the environment, outside
the config pipeline): a call to ``os.getenv`` / ``os.environ.get`` /
``os.environ[NAME]`` whose name is not an ops/test/credential switch has no
YAML default, no ``config.local`` override, no schema, and no Settings
reach -- the read itself IS the whole "config surface" for that value. Two
independent readers of the same name can (and did) silently disagree on the
default (2026-09-19 env-read inventory; docs/BUG_CLASSES.md BC-88).

Mode (2026-09-20): CEILING RATCHET, same shape as
``tests/unit/test_import_hygiene_guard.py`` (read that file first -- this one
mirrors its two-test policy: one test fails when the count GROWS past the
ceiling, the second fails when the ceiling is left slack after a batch that
lowered the real count). This wave installs the GUARD and the CATALOG ENTRY
only; migrating any of these reads to a chokepoint is a separate, later batch
(the design of that chokepoint is an owner/Codex decision -- see BC-88 in
docs/BUG_CLASSES.md).

A second, narrower guard below (``KNOWN_CONFLICTING_DEFAULTS``) tracks env
names read with more than one distinct LITERAL default. It is a shrink-only
allowlist, the mirror image of the growth ratchet above: a name entering the
set for the first time means the guard caught something new (fail loudly); a
name leaving the set means it was fixed (must be removed by hand, never left
stale) -- same discipline as ``PINNED_NAMES`` in
tests/unit/test_env_default_single_source.py, which this file does not
modify and does not duplicate (that file's two pinned names,
``GATE_COSINE_THRESHOLD`` and ``REFLECTIONS_ON_DEMAND``, were fixed to a
single default on 2026-09-19 and are intentionally NOT re-asserted here).

The scan is self-contained (``ast`` over the working tree; no application
import -- importing ``config.app_config`` creates directories on disk, which
a test must never do -- and no git state, per
tests/unit/test_no_git_state_in_tests.py).
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Universe: the app packages that read config, EXCLUDING config/ itself (a
# read inside config/app_config.py or config/schema.py IS the chokepoint, not
# a bypass of it) -- ported from the P2 Phase 0 inventory's file universe.
GATED_DIRS = (
    "core", "memory", "knowledge", "utils", "gui", "api", "models",
    "processing", "agent_branch", "eval",
)
GATED_FILES = ("main.py",)
SKIP_PARTS = {"__pycache__", "node_modules", "integration.bak"}

# ---------------------------------------------------------------------------
# Bucket (b): ops/test switches + credentials, ported verbatim (same
# prefixes/exacts/suffixes) from P2_phase0/env_read_inventory.py's
# `classify_ops_or_test`. These are legitimately environment-only knobs
# (process wiring, CI, offline-mode, secrets) and are out of scope for BC-88.
# ---------------------------------------------------------------------------
_OPS_PREFIXES = (
    "DAEMON_", "PYTEST", "DISABLE_", "SKIP_", "CI", "HF_",
    "TRANSFORMERS_", "CUDA", "TOKENIZERS_", "OMP_",
)
_OPS_EXACT = {"HOME", "PATH", "USER", "DISPLAY", "TERM", "SHELL", "TMPDIR"}
_OPS_XDG_PREFIX = "XDG_"
_CRED_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD", "_CLIENT_ID")


def is_allowed_name(name: str | None) -> bool:
    """True for an ops/test switch or credential name (bucket (b)); everything
    else that reaches this function is a SETTING read (bucket (a)+(c) merged,
    per the plan -- this guard does not distinguish "has an app_config
    constant of the same name" from "has none", only "is or isn't a
    setting")."""
    if name is None:
        return False
    for suf in _CRED_SUFFIXES:
        if name.endswith(suf):
            return True
    for pre in _OPS_PREFIXES:
        if name.startswith(pre):
            return True
    if name in _OPS_EXACT:
        return True
    if name.startswith(_OPS_XDG_PREFIX):
        return True
    return False


_NOT_LITERAL = object()  # sentinel: default expression is not a literal constant


class _EnvImports:
    """Tracks how `os`, `os.environ`, `os.getenv` are bound in one module,
    including `import os as X`, `from os import environ`,
    `from os import getenv as G`, etc. -- ported from env_read_inventory.py's
    ImportTracker."""

    def __init__(self) -> None:
        self.os_aliases: set[str] = set()
        self.environ_aliases: set[str] = set()
        self.getenv_aliases: set[str] = set()

    def visit(self, tree: ast.Module) -> "_EnvImports":
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "os":
                        self.os_aliases.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module == "os":
                for alias in node.names:
                    if alias.name == "environ":
                        self.environ_aliases.add(alias.asname or alias.name)
                    elif alias.name == "getenv":
                        self.getenv_aliases.add(alias.asname or alias.name)
        return self


def _is_os_attr(node: ast.AST, tracker: _EnvImports, attr: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attr
        and isinstance(node.value, ast.Name)
        and node.value.id in tracker.os_aliases
    )


def _is_environ_expr(node: ast.AST, tracker: _EnvImports) -> bool:
    return _is_os_attr(node, tracker, "environ") or (
        isinstance(node, ast.Name) and node.id in tracker.environ_aliases
    )


def _is_getenv_func(node: ast.AST, tracker: _EnvImports) -> bool:
    return _is_os_attr(node, tracker, "getenv") or (
        isinstance(node, ast.Name) and node.id in tracker.getenv_aliases
    )


def _literal(node: ast.AST | None):
    """Python value of a literal default. An OMITTED default (no 2nd
    positional arg, no `default=`/`value=` keyword, or a bare
    `os.environ[NAME]` subscript which cannot carry one) is represented as
    the Python value `None` -- matching
    tests/unit/test_env_default_single_source.py's convention, so an implicit
    "returns None when missing" reader can still be flagged as conflicting
    with an explicit literal default elsewhere (this is exactly the
    OPENAI_API_KEY shape: five call sites with no default, one with `''`).
    Anything computed (a Name, Call, IfExp, f-string, BinOp, ...) returns the
    `_NOT_LITERAL` sentinel and is never compared -- "computed defaults are
    ignored" per the plan."""
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return node.value
    return _NOT_LITERAL


def _default_node(call: ast.Call) -> ast.AST | None:
    if len(call.args) >= 2:
        return call.args[1]
    for kw in call.keywords:
        if kw.arg in ("default", "value"):
            return kw.value
    return None


class EnvSite:
    __slots__ = ("name", "dynamic", "default", "path", "line")

    def __init__(self, name, dynamic, default, path, line):
        self.name = name
        self.dynamic = dynamic
        self.default = default
        self.path = path
        self.line = line


def _sites_from_tree(tree: ast.Module, rel: str) -> list[EnvSite]:
    """Every READ-shaped env access: `os.getenv(...)`, `os.environ.get(...)`,
    `os.environ[NAME]` in a Load context. Writes (`setdefault`, `pop`,
    `os.environ[NAME] = ...`, `del os.environ[NAME]`) and membership tests
    (`NAME in os.environ`) are deliberately out of scope -- the plan's port
    instruction names only these three read shapes."""
    tracker = _EnvImports().visit(tree)
    sites: list[EnvSite] = []
    for node in ast.walk(tree):
        name_node = None
        default_node = None
        if isinstance(node, ast.Call) and _is_getenv_func(node.func, tracker):
            name_node = node.args[0] if node.args else None
            default_node = _default_node(node)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and _is_environ_expr(node.func.value, tracker)
        ):
            name_node = node.args[0] if node.args else None
            default_node = _default_node(node)
        elif (
            isinstance(node, ast.Subscript)
            and _is_environ_expr(node.value, tracker)
            and isinstance(node.ctx, ast.Load)
        ):
            name_node = node.slice  # py3.9+: no ast.Index wrapper
            default_node = None  # a bare subscript can never carry a default
        else:
            continue

        if isinstance(name_node, ast.Constant) and isinstance(name_node.value, str):
            name = name_node.value
            dynamic = False
        else:
            name = None
            dynamic = True
        sites.append(EnvSite(name, dynamic, _literal(default_node), rel, node.lineno))
    return sites


def _sites_from_source(source: str, rel: str = "<memory>") -> list[EnvSite]:
    return _sites_from_tree(ast.parse(source), rel)


def _python_files(root: Path, dirs: tuple[str, ...], files: tuple[str, ...] = ()):
    for d in dirs:
        base = root / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.py")):
            if SKIP_PARTS.isdisjoint(p.parts):
                yield p
    for f in files:
        p = root / f
        if p.is_file():
            yield p


def _scan_repo(root: Path) -> list[EnvSite]:
    sites: list[EnvSite] = []
    for path in _python_files(root, GATED_DIRS, GATED_FILES):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue
        rel = path.relative_to(root).as_posix()
        sites.extend(_sites_from_tree(tree, rel))
    return sites


def _setting_sites(sites: list[EnvSite]) -> list[EnvSite]:
    """Sites that are neither dynamic-named (can't be classified, reported
    separately, never gated) nor an allowed ops/test/credential switch."""
    return [s for s in sites if not s.dynamic and not is_allowed_name(s.name)]


def _conflicting_defaults(sites: list[EnvSite]) -> dict[str, list[str]]:
    """name -> sorted distinct literal defaults (as `repr`), for names with
    more than one. Dynamic-named sites and computed (`_NOT_LITERAL`) defaults
    never enter the comparison. Writes are never in `sites` to begin with."""
    by_name: dict[str, set[str]] = defaultdict(set)
    for s in sites:
        if s.dynamic or s.default is _NOT_LITERAL:
            continue
        by_name[s.name].add(repr(s.default))
    return {name: sorted(vals) for name, vals in by_name.items() if len(vals) > 1}


# ---------------------------------------------------------------------------
# Measured ceilings (2026-09-20, base 3d28858). Lowered by every guard batch
# that removes/reroutes a setting read; raised by nobody. See module
# docstring for the ratchet policy this mirrors from test_import_hygiene_guard.py.
# ---------------------------------------------------------------------------
MAX_SETTING_ENV_READ_SITES = 220
CEILING_SLACK = 25

# Names currently read (this file's read-only, LITERAL-default-only detector)
# with more than one distinct literal default. Measured directly on this
# clone -- NOT copied from the 14-name table in P2_phase0/SUMMARY.md, which
# used unparsed source text (not Python literal values) and included write
# sites: most of those 14 names' "second form" is a *computed* default
# (`str(x)`, `os.path.join(...)`, an `IfExp` ternary, a module-level
# fallback constant) which this narrower literal-only comparison deliberately
# ignores per the plan ("computed defaults are ignored"). `GATE_COSINE_THRESHOLD`
# and `REFLECTIONS_ON_DEMAND` were already fixed to a single default on
# 2026-09-19 and are pinned separately by test_env_default_single_source.py
# (not re-asserted here -- do not add them back without re-measuring).
KNOWN_CONFLICTING_DEFAULTS = frozenset({
    "APPDATA", "OPENAI_API_KEY", "WIKI_BUDGET_S", "WORKER_OBJECTIVE",
})


def _report(setting_sites: list[EnvSite]) -> str:
    per_file = Counter(s.path for s in setting_sites)
    top = "\n".join(f"  {n:4d}  {rel}" for rel, n in per_file.most_common(15))
    return f"setting-shaped env reads: {len(setting_sites)} in {len(per_file)} files\n{top}"


def test_setting_env_read_sites_do_not_grow():
    sites = _scan_repo(REPO_ROOT)
    setting = _setting_sites(sites)
    dynamic = [s for s in sites if s.dynamic]
    print(_report(setting))
    print(f"dynamic-name sites (ignored, not gated, printed only): {len(dynamic)}")
    for s in dynamic:
        print(f"  {s.path}:{s.line}")
    offenders = "\n".join(f"  {s.path}:{s.line}  name={s.name!r}" for s in setting[:40])
    assert len(setting) <= MAX_SETTING_ENV_READ_SITES, (
        f"{len(setting)} setting-shaped os.getenv/os.environ.get/os.environ[] reads outside "
        f"config/, ceiling is {MAX_SETTING_ENV_READ_SITES}. A setting read straight from the "
        "environment has no YAML default, no config.local override, no schema, and no Settings "
        "reach (BC-88, docs/BUG_CLASSES.md). Route the new read through config/app_config.py "
        f"instead of adding another direct os.getenv call; never raise the ceiling.\nFirst "
        f"offenders:\n{offenders}"
    )


def test_ceiling_is_lowered_after_each_batch():
    sites = _scan_repo(REPO_ROOT)
    setting = _setting_sites(sites)
    slack = MAX_SETTING_ENV_READ_SITES - len(setting)
    assert slack <= CEILING_SLACK, (
        f"MAX_SETTING_ENV_READ_SITES ({MAX_SETTING_ENV_READ_SITES}) is {slack} above the measured "
        f"count ({len(setting)}); lower it to {len(setting)} in this batch so the ratchet only "
        "ever tightens."
    )


def test_known_conflicting_defaults_matches_reality():
    sites = _scan_repo(REPO_ROOT)
    measured = set(_conflicting_defaults(sites))
    new_conflicts = measured - KNOWN_CONFLICTING_DEFAULTS
    assert not new_conflicts, (
        f"New env name(s) with >1 distinct literal default, not yet in "
        f"KNOWN_CONFLICTING_DEFAULTS: {sorted(new_conflicts)}. Either fix the drift (align the "
        "defaults -- BC-88) or, if it is a reviewed, deliberate difference, add the name to "
        "KNOWN_CONFLICTING_DEFAULTS with a comment explaining why."
    )
    resolved = KNOWN_CONFLICTING_DEFAULTS - measured
    assert not resolved, (
        f"{sorted(resolved)} no longer have conflicting literal defaults -- remove from "
        "KNOWN_CONFLICTING_DEFAULTS by hand (the set only ever shrinks, never grows back "
        "automatically)."
    )


class TestDetectorSelfTest:
    """Proves the AST detector above actually finds/ignores what the plan
    specifies, using synthetic in-memory sources -- independent of whatever
    the real tree currently looks like."""

    def test_getenv_with_literal_default_is_a_setting_site(self):
        sites = _sites_from_source('import os\nX = os.getenv("MY_SETTING", "5")\n')
        assert len(sites) == 1
        s = sites[0]
        assert s.name == "MY_SETTING" and not s.dynamic and s.default == "5"
        assert not is_allowed_name(s.name)
        assert _setting_sites(sites) == sites

    def test_environ_get_is_detected(self):
        sites = _sites_from_source('import os\nX = os.environ.get("MY_SETTING", "5")\n')
        assert [s.name for s in sites] == ["MY_SETTING"]

    def test_environ_subscript_load_is_detected_with_no_default(self):
        sites = _sites_from_source('import os\nX = os.environ["MY_SETTING"]\n')
        assert len(sites) == 1
        assert sites[0].name == "MY_SETTING" and sites[0].default is None

    def test_environ_subscript_store_is_not_a_read(self):
        sites = _sites_from_source('import os\nos.environ["MY_SETTING"] = "1"\n')
        assert sites == []

    def test_environ_setdefault_and_pop_are_not_reads(self):
        sites = _sites_from_source(
            'import os\n'
            'os.environ.setdefault("MY_SETTING", "1")\n'
            'os.environ.pop("MY_SETTING", None)\n'
        )
        assert sites == []

    def test_del_environ_is_not_a_read(self):
        sites = _sites_from_source('import os\ndel os.environ["MY_SETTING"]\n')
        assert sites == []

    def test_membership_test_is_not_a_read(self):
        sites = _sites_from_source('import os\nif "MY_SETTING" in os.environ:\n    pass\n')
        assert sites == []

    def test_dynamic_name_is_ignored_but_countable_separately(self):
        sites = _sites_from_source('import os\nkey = "X"\nos.getenv(key, "1")\n')
        assert len(sites) == 1 and sites[0].dynamic
        assert _setting_sites(sites) == []  # a dynamic name never counts as a setting site
        assert _conflicting_defaults(sites) == {}  # nor does it enter the conflict comparison

    def test_ops_and_test_and_credential_names_are_allowed_not_settings(self):
        src = (
            'import os\n'
            'os.getenv("DAEMON_TEST_MODE", "0")\n'
            'os.getenv("PYTEST_CURRENT_TEST")\n'
            'os.getenv("HOME")\n'
            'os.getenv("XDG_CONFIG_HOME")\n'
            'os.getenv("OPENAI_API_KEY")\n'
        )
        sites = _sites_from_source(src)
        assert len(sites) == 5
        assert _setting_sites(sites) == []

    def test_from_os_import_getenv_and_environ_are_tracked(self):
        src = (
            'from os import getenv, environ\n'
            'getenv("MY_SETTING", "1")\n'
            'environ.get("MY_SETTING2", "2")\n'
            'environ["MY_SETTING3"]\n'
        )
        sites = _sites_from_source(src)
        assert sorted(s.name for s in sites) == ["MY_SETTING", "MY_SETTING2", "MY_SETTING3"]

    def test_aliased_os_import_is_tracked(self):
        sites = _sites_from_source('import os as _os\nX = _os.getenv("MY_SETTING", "5")\n')
        assert [s.name for s in sites] == ["MY_SETTING"]

    def test_two_literal_defaults_conflict(self):
        a = _sites_from_source('import os\nos.getenv("FOO_THRESH", "0.45")\n', "a.py")
        b = _sites_from_source('import os\nos.getenv("FOO_THRESH", "0.50")\n', "b.py")
        conflicts = _conflicting_defaults(a + b)
        assert "FOO_THRESH" in conflicts
        assert conflicts["FOO_THRESH"] == sorted([repr("0.45"), repr("0.50")])

    def test_matching_literal_defaults_do_not_conflict(self):
        a = _sites_from_source('import os\nos.getenv("FOO_THRESH", "0")\n', "a.py")
        b = _sites_from_source('import os\nos.environ.get("FOO_THRESH", "0")\n', "b.py")
        assert _conflicting_defaults(a + b) == {}

    def test_computed_default_is_ignored_not_compared(self):
        a = _sites_from_source('import os\nfallback = "1"\nos.getenv("FOO_THRESH", fallback)\n', "a.py")
        b = _sites_from_source('import os\nos.getenv("FOO_THRESH", "1")\n', "b.py")
        # Only ONE literal default exists across both sites (the computed one
        # in `a.py` is ignored, not compared) -> no conflict.
        assert _conflicting_defaults(a + b) == {}

    def test_omitted_default_compares_as_none(self):
        a = _sites_from_source('import os\nos.getenv("FOO_THRESH")\n', "a.py")
        b = _sites_from_source('import os\nos.getenv("FOO_THRESH", "1")\n', "b.py")
        conflicts = _conflicting_defaults(a + b)
        assert "FOO_THRESH" in conflicts

    def test_single_site_never_conflicts(self):
        a = _sites_from_source('import os\nos.getenv("SOLO_SETTING", "1")\n', "a.py")
        assert _conflicting_defaults(a) == {}

    def test_unparseable_source_is_handled_by_the_repo_scan(self, tmp_path):
        (tmp_path / "core").mkdir()
        (tmp_path / "core" / "broken.py").write_text("def broken(:\n")
        assert _scan_repo(tmp_path) == []


class TestConflictGuardGeneralizes:
    """Demonstrates -- via a throwaway tmp tree, never a repo file -- that
    `test_known_conflicting_defaults_matches_reality`'s mechanism actually
    catches a brand-new drift. This is the proof that the guard has teeth:
    were `SYNTH_NEW_SETTING` a real name in the real tree, that test would
    compute `new_conflicts == {"SYNTH_NEW_SETTING"}` and fail exactly as
    designed."""

    def test_new_conflict_on_a_synthetic_tree_would_fail_the_real_test(self, tmp_path):
        core = tmp_path / "core"
        core.mkdir()
        (core / "a.py").write_text('import os\nX = float(os.getenv("SYNTH_NEW_SETTING", "0.1"))\n')
        (core / "b.py").write_text('import os\nY = float(os.getenv("SYNTH_NEW_SETTING", "0.2"))\n')

        sites = _scan_repo(tmp_path)
        measured = set(_conflicting_defaults(sites))

        assert "SYNTH_NEW_SETTING" in measured
        assert "SYNTH_NEW_SETTING" not in KNOWN_CONFLICTING_DEFAULTS
        would_fail = measured - KNOWN_CONFLICTING_DEFAULTS
        assert would_fail == {"SYNTH_NEW_SETTING"}

    def test_a_name_matching_the_known_set_on_a_synthetic_tree_passes(self, tmp_path):
        """Sanity check on the other arm: a synthetic tree whose only
        conflict IS already in KNOWN_CONFLICTING_DEFAULTS produces no new
        finding (proves the test isn't just always failing)."""
        core = tmp_path / "core"
        core.mkdir()
        (core / "a.py").write_text('import os\nos.getenv("APPDATA", "")\n')
        (core / "b.py").write_text('import os\nos.getenv("APPDATA", ".")\n')

        sites = _scan_repo(tmp_path)
        measured = set(_conflicting_defaults(sites))
        assert measured - KNOWN_CONFLICTING_DEFAULTS == set()
