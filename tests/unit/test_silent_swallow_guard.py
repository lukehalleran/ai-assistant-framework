"""
Repo-wide guard for BC-69 (silent ops failure) / BC-70 (log severity misdescribes control
flow): a broad `except` (bare / `Exception` / `BaseException` / a tuple containing one of
those) whose body does nothing but pass/continue/break, return an empty/None/falsy literal,
log at debug or info, or set at most one literal/empty-container variable is a SILENT
SWALLOW -- the caller (and the owner reading the logs) has no way to know something failed.
The 2026-09-20 P1 inventory (a stdlib AST inventory kept with the batch's run artifacts, outside the repo)
found 472 of these in app code (352 free-file, 120 debt-file), 8 carrying any comment about
what is lost . Live cost seen 2026-09-19: an `AttributeError` raised
inside such a handler surfaced only as a test's "mock called 0 times".

Doctrine this guard enforces (docs/BUG_CLASSES.md BC-69/BC-70, DM-35): every swallow names its
degradation with a `# degrades: <what the user/owner loses>` comment (>= 3 words) on the
`except` line or anywhere in the handler body. This guard does not require raising log
severity or changing control flow -- a later batch does that (see PLAN_W4.md executors A2/A3);
this one only stops the UNMARKED count from growing.

Mode: CEILING RATCHET, same shape as `tests/unit/test_import_hygiene_guard.py`.
`MAX_UNMARKED_SWALLOWS` is the count measured on the PRISTINE base clone
(a pristine clone of commit 3d28858) BEFORE the first marking batch of this wave
landed -- 472, i.e. at that point no handler in the tree carried a `# degrades:` comment
anywhere, so unmarked == total swallow count. The first test fails when the unmarked count
GROWS past the ceiling; the second fails when the ceiling is left slack after a marking batch
(lower it to the newly measured count -- `CEILING_SLACK` is 0 here, unlike the import-hygiene
guard's accrued 25, because this is this guard's first measurement: there is no reason to ship
slack on day one).

The detector below is a self-contained re-implementation of that inventory's
definition (same handler-type test, same allowed-body-statement set) -- it does NOT import that
tool (stdlib only, no application import). The file universe is a filesystem walk, not
`git ls-files` (this test may not read git state -- see `test_no_git_state_in_tests.py`); there
were no untracked `.py` files under the gated dirs at the time this guard was written, so the
two universes coincide.
"""
from __future__ import annotations

import ast
import io
import tokenize
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GATED_DIRS = (
    "core", "memory", "knowledge", "utils", "gui", "api", "models",
    "processing", "config", "agent_branch", "eval",
)
GATED_FILES = ("main.py",)
SKIP_PARTS = {"__pycache__", "node_modules", "integration.bak"}

MIN_MARKER_WORDS = 3

# Lowered by every swallow-marking batch; raised by nobody. See module docstring.
MAX_UNMARKED_SWALLOWS = 419  # 2026-09-20: 472 on master 3d28858; first two file-complete batches (8 files, 53 records: 53 markers, 24 debug->warning raises that leave the swallow definition)
CEILING_SLACK = 10  # an unrelated change that removes a swallow must not fail this test; a batch lowers the ceiling to the measured count


# --------------------------------------------------------------------------
# File universe (filesystem walk -- no git state; see test_no_git_state_in_tests.py)
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Swallow detector (ported from the 2026-09-20 inventory -- same definition)
# --------------------------------------------------------------------------

def _is_broad_name(node: ast.AST) -> bool:
    if isinstance(node, ast.Name) and node.id in ("Exception", "BaseException"):
        return True
    if isinstance(node, ast.Attribute) and node.attr in ("Exception", "BaseException"):
        return True
    return False


def _matches_broad_exception(type_node) -> bool:
    if type_node is None:
        return True
    if _is_broad_name(type_node):
        return True
    if isinstance(type_node, ast.Tuple):
        return any(_is_broad_name(e) for e in type_node.elts)
    return False


_LOG_LEVELS = {"debug", "info"}


def _strip_docstring_exprs(body: list[ast.stmt]) -> list[ast.stmt]:
    out = []
    for s in body:
        if (
            isinstance(s, ast.Expr)
            and isinstance(s.value, ast.Constant)
            and isinstance(s.value.value, str)
        ):
            continue
        out.append(s)
    return out


def _is_literal_or_empty(value: ast.expr) -> bool:
    if isinstance(value, ast.Constant):
        return True
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)) and len(value.elts) == 0:
        return True
    if isinstance(value, ast.Dict) and len(value.keys) == 0:
        return True
    return False


def _classify_stmt(stmt: ast.stmt) -> str | None:
    """Return a shape label if stmt is an allowed swallow-body statement, else None."""
    if isinstance(stmt, ast.Pass):
        return "pass"
    if isinstance(stmt, ast.Continue):
        return "continue"
    if isinstance(stmt, ast.Break):
        return "break"
    if isinstance(stmt, ast.Return):
        v = stmt.value
        if v is None:
            return "return_none"
        if isinstance(v, ast.Constant):
            if v.value is None:
                return "return_none"
            if v.value is False or v.value == 0 or v.value == "":
                return "return_empty"
            return None
        if isinstance(v, (ast.List, ast.Tuple, ast.Set)) and len(v.elts) == 0:
            return "return_empty"
        if isinstance(v, ast.Dict) and len(v.keys) == 0:
            return "return_empty"
        return None
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        func = stmt.value.func
        if isinstance(func, ast.Attribute) and func.attr in _LOG_LEVELS:
            return f"log_{func.attr}"
        return None
    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) == 1 and _is_literal_or_empty(stmt.value):
            return "assign_empty"
        return None
    return None


def _handler_shape(handler: ast.ExceptHandler) -> list[str] | None:
    """None when the handler does not qualify as a swallow; else its statement-shape list."""
    if not _matches_broad_exception(handler.type):
        return None
    body = _strip_docstring_exprs(handler.body)
    shapes: list[str] = []
    assign_count = 0
    for stmt in body:
        kind = _classify_stmt(stmt)
        if kind is None:
            return None
        if kind == "assign_empty":
            assign_count += 1
            if assign_count > 1:
                return None
        shapes.append(kind)
    return shapes


def _handler_span(handler: ast.ExceptHandler) -> tuple[int, int]:
    if handler.body:
        end_line = max(getattr(s, "end_lineno", s.lineno) or s.lineno for s in handler.body)
    else:
        end_line = handler.lineno
    return handler.lineno, end_line


def find_swallow_handlers(tree: ast.AST) -> list[tuple[int, int]]:
    """(except_lineno, handler_end_lineno) for every qualifying swallow handler in tree."""
    out: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                if _handler_shape(handler) is not None:
                    out.append(_handler_span(handler))
    return out


# --------------------------------------------------------------------------
# Marker detection (comment scan via tokenize)
# --------------------------------------------------------------------------

def _collect_comments(source: str) -> dict[int, list[str]]:
    comments: dict[int, list[str]] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                comments.setdefault(tok.start[0], []).append(tok.string)
    except Exception:
        # Measurement helper over arbitrary source text; a tokenize failure on a comment-only
        # pass just means no markers are found for this file, not that the scan should abort.
        pass
    return comments


def _marker_word_count(comment: str) -> int:
    """Word count following a case-insensitive "degrades:" in comment; -1 if absent."""
    low = comment.lower()
    idx = low.find("degrades:")
    if idx == -1:
        return -1
    tail = comment[idx + len("degrades:"):]
    return len([w for w in tail.split() if w])


def _is_marked(comments_by_line: dict[int, list[str]], start: int, end: int) -> bool:
    for ln in range(start, end + 1):
        for c in comments_by_line.get(ln, ()):
            if _marker_word_count(c) >= MIN_MARKER_WORDS:
                return True
    return False


# --------------------------------------------------------------------------
# Repo scan
# --------------------------------------------------------------------------

def scan(root: Path, dirs: tuple[str, ...] = GATED_DIRS, files: tuple[str, ...] = GATED_FILES):
    """Return (unmarked, marked_count, total) -- unmarked is a list of (relpath, lineno)."""
    unmarked: list[tuple[str, int]] = []
    marked = 0
    total = 0
    for path in _python_files(root, dirs, files):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue
        handlers = find_swallow_handlers(tree)
        if not handlers:
            continue
        comments_by_line = _collect_comments(source)
        rel = path.relative_to(root).as_posix()
        for start, end in handlers:
            total += 1
            if _is_marked(comments_by_line, start, end):
                marked += 1
            else:
                unmarked.append((rel, start))
    return unmarked, marked, total


def _scan_source(source: str) -> tuple[int, int, int]:
    """(unmarked, marked, total) for an in-memory source string (self-tests only)."""
    tree = ast.parse(source)
    handlers = find_swallow_handlers(tree)
    comments_by_line = _collect_comments(source)
    total = len(handlers)
    marked = sum(1 for start, end in handlers if _is_marked(comments_by_line, start, end))
    return total - marked, marked, total


def _report(unmarked: list[tuple[str, int]]) -> str:
    per_file = Counter(rel for rel, _ in unmarked)
    top = "\n".join(f"  {n:4d}  {rel}" for rel, n in per_file.most_common(15))
    return f"unmarked silent swallows: {len(unmarked)} in {len(per_file)} files\n{top}"


# --------------------------------------------------------------------------
# Ceiling-ratchet tests
# --------------------------------------------------------------------------

def test_unmarked_silent_swallows_do_not_grow():
    unmarked, marked, total = scan(REPO_ROOT)
    print(_report(unmarked))
    print(f"marked: {marked} / total swallow handlers: {total}")
    offenders = "\n".join(f"  {rel}:{ln}" for rel, ln in unmarked[:40])
    assert len(unmarked) <= MAX_UNMARKED_SWALLOWS, (
        f"{len(unmarked)} unmarked silent-swallow handlers in app code, ceiling is "
        f"{MAX_UNMARKED_SWALLOWS}. Add `# degrades: <what is lost>` (>= 3 words) on the "
        "except line or in the handler body, or handle the error properly; never raise the "
        f"ceiling. First offenders:\n{offenders}"
    )


def test_ceiling_is_lowered_after_each_batch():
    unmarked, _marked, _total = scan(REPO_ROOT)
    slack = MAX_UNMARKED_SWALLOWS - len(unmarked)
    assert slack <= CEILING_SLACK, (
        f"MAX_UNMARKED_SWALLOWS ({MAX_UNMARKED_SWALLOWS}) is {slack} above the measured "
        f"unmarked count ({len(unmarked)}); lower it to {len(unmarked)} in this batch so the "
        "ratchet only ever tightens."
    )


# --------------------------------------------------------------------------
# Detector self-tests (in-memory sources)
# --------------------------------------------------------------------------

class TestDetectorFlagsSwallows:
    def test_flags_except_exception_pass(self):
        assert _scan_source(
            "try:\n    x = 1\nexcept Exception:\n    pass\n"
        ) == (1, 0, 1)

    def test_flags_debug_log_then_return_none(self):
        src = (
            "try:\n    x = 1\nexcept Exception as e:\n"
            "    logger.debug(str(e))\n    return None\n"
        )
        assert _scan_source(src) == (1, 0, 1)

    def test_flags_bare_except(self):
        assert _scan_source(
            "try:\n    x = 1\nexcept:\n    pass\n"
        ) == (1, 0, 1)

    def test_flags_info_log_only(self):
        src = "try:\n    x = 1\nexcept Exception:\n    logger.info('skipped')\n"
        assert _scan_source(src) == (1, 0, 1)


class TestDetectorDoesNotFlagRealHandling:
    def test_does_not_flag_warning_level_log(self):
        src = (
            "try:\n    x = 1\nexcept Exception as e:\n"
            "    logger.warning(str(e))\n    return None\n"
        )
        _unmarked, _marked, total = _scan_source(src)
        assert total == 0

    def test_does_not_flag_error_level_log(self):
        src = "try:\n    x = 1\nexcept Exception:\n    logger.error('boom')\n"
        _unmarked, _marked, total = _scan_source(src)
        assert total == 0

    def test_does_not_flag_reraise(self):
        src = "try:\n    x = 1\nexcept Exception:\n    raise\n"
        _unmarked, _marked, total = _scan_source(src)
        assert total == 0

    def test_does_not_flag_function_call(self):
        src = "try:\n    x = 1\nexcept Exception:\n    handle_error()\n"
        _unmarked, _marked, total = _scan_source(src)
        assert total == 0

    def test_does_not_flag_narrow_exception_type(self):
        src = "try:\n    x = 1\nexcept ValueError:\n    pass\n"
        _unmarked, _marked, total = _scan_source(src)
        assert total == 0


class TestDetectorHonoursTheMarker:
    def test_honours_marker_on_except_line(self):
        src = (
            "try:\n    x = 1\n"
            "except Exception:  # degrades: web search falls back to keyword heuristics\n"
            "    pass\n"
        )
        assert _scan_source(src) == (0, 1, 1)

    def test_honours_marker_in_body(self):
        src = (
            "try:\n    x = 1\nexcept Exception:\n"
            "    # degrades: profile fact write is skipped this turn\n"
            "    pass\n"
        )
        assert _scan_source(src) == (0, 1, 1)

    def test_marker_needs_at_least_three_words(self):
        src = (
            "try:\n    x = 1\n"
            "except Exception:  # degrades: two words\n"
            "    pass\n"
        )
        assert _scan_source(src) == (1, 0, 1)

    def test_marker_is_case_insensitive(self):
        src = (
            "try:\n    x = 1\n"
            "except Exception:  # Degrades: caches stay cold for this request\n"
            "    pass\n"
        )
        assert _scan_source(src) == (0, 1, 1)

    def test_unrelated_comment_does_not_count_as_a_marker(self):
        src = (
            "try:\n    x = 1\n"
            "except Exception:  # best effort, ignore failures here please\n"
            "    pass\n"
        )
        assert _scan_source(src) == (1, 0, 1)
