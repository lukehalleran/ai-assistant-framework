"""Token budget must meter the sections the formatter RENDERS (2026-08-14).

Live incident: a turn's true prompt hit 17.5K tokens against the 10K budget.
Root cause: the formatter renders the SPLIT summary/reflection keys
(recent_summaries / semantic_summaries / recent_reflections /
semantic_reflections) but PRIORITY_ORDER metered the combined
"summaries"/"reflections" keys — which nothing renders. So the four rendered
sections were invisible to the budget AND untrimmable, and the builder's
floors were topping up the dead keys with retrieval calls whose output never
reached the prompt.

Strengthened 2026-09-13: rendered keys come from the formatter's syntax tree,
not a ``context.get("…")`` regex.  Single quotes, subscripts, membership
tests, keys iterated from a literal tuple and keys passed through a local
helper (``_count("graph_context")`` → ``context.get(key)``) were all
invisible to the regex.  Each exception now carries its reason, and red
controls prove a new rendered key fails until it is metered or excepted.
"""

import ast
from pathlib import Path

import pytest

from core.prompt.token_manager import (
    PRIORITY_ORDER,
    UNRENDERED_CONTEXT_KEYS,
    TokenManager,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
# Every module that renders context sections into the prompt text.
FORMATTER_PATHS = ("core/prompt/formatter.py",)
CONTEXT_NAMES = frozenset({"context", "ctx"})

# Keys the formatter reads that the budget deliberately does not meter.
METERING_EXCEPTIONS = {
    "note_images": "image payloads for multimodal calls; not prompt text",
    "visual_memories": "image references rendered as attachments, not budgeted text",
    "stm_summary": "short-term-memory state; also in UNRENDERED_CONTEXT_KEYS",
    "memory_id_map": "citation id map; metadata, never rendered",
    **{key: "structured metadata listed in UNRENDERED_CONTEXT_KEYS" for key in UNRENDERED_CONTEXT_KEYS},
}

_PRIORITY_NAMES = {name for name, _ in PRIORITY_ORDER}


def _is_context(node) -> bool:
    return isinstance(node, ast.Name) and node.id in CONTEXT_NAMES


def _string(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def rendered_context_keys(source: str) -> dict:
    """{key: sorted access forms} for every context key a formatter source reads."""
    tree = ast.parse(source)
    found: dict = {}

    def add(key, form):
        if key is not None:
            found.setdefault(key, set()).add(form)

    # A function whose parameter is used as the key of context.get/context[...]
    # makes every call with a literal at that position a read of that key.
    helpers = {}
    loop_keys = {}
    for func in ast.walk(tree):
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = [arg.arg for arg in func.args.args]
            for node in ast.walk(func):
                key_node = None
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" and _is_context(node.func.value) and node.args:
                    key_node = node.args[0]
                elif isinstance(node, ast.Subscript) and _is_context(node.value) and isinstance(node.ctx, ast.Load):
                    key_node = node.slice
                if isinstance(key_node, ast.Name) and key_node.id in params:
                    helpers[func.name] = params.index(key_node.id)
        if isinstance(func, ast.For) and isinstance(func.target, ast.Name) and isinstance(func.iter, (ast.Tuple, ast.List)):
            loop_keys[func.target.id] = [_string(elt) for elt in func.iter.elts]

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" and _is_context(node.func.value) and node.args:
            key = node.args[0]
            if _string(key) is not None:
                add(_string(key), "get")
            elif isinstance(key, ast.Name):
                for literal in loop_keys.get(key.id, []):
                    add(literal, "loop")
        elif isinstance(node, ast.Subscript) and _is_context(node.value) and isinstance(node.ctx, ast.Load):
            if _string(node.slice) is not None:
                add(_string(node.slice), "subscript")
            elif isinstance(node.slice, ast.Name):
                for literal in loop_keys.get(node.slice.id, []):
                    add(literal, "loop")
        elif isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)) and _is_context(node.comparators[0]):
            add(_string(node.left), "membership")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in helpers:
            index = helpers[node.func.id]
            if len(node.args) > index:
                add(_string(node.args[index]), f"helper:{node.func.id}")
    return {key: sorted(forms) for key, forms in found.items()}


def unmetered(keys, priority_names, exceptions) -> list:
    return sorted(set(keys) - set(priority_names) - set(exceptions))


class TestPriorityOrderCoversRenderedKeys:
    def test_split_summary_reflection_keys_are_metered(self):
        for key in (
            "recent_summaries",
            "semantic_summaries",
            "recent_reflections",
            "semantic_reflections",
        ):
            assert key in _PRIORITY_NAMES, f"rendered key {key} missing from PRIORITY_ORDER"

    def test_dead_combined_keys_are_not_metered(self):
        # The combined keys are inputs nothing renders — metering them would
        # spend budget on invisible content.
        assert "summaries" not in _PRIORITY_NAMES
        assert "reflections" not in _PRIORITY_NAMES
        assert "summaries" in UNRENDERED_CONTEXT_KEYS
        assert "reflections" in UNRENDERED_CONTEXT_KEYS

    def test_calendar_and_schedule_are_metered(self):
        assert "google_calendar" in _PRIORITY_NAMES
        assert "upcoming_schedule" in _PRIORITY_NAMES

    def test_formatter_rendered_keys_parity(self):
        """Every context key any formatter path reads must be metered (or be a reasoned
        exception). Fails loudly when a new rendered section is added without a
        PRIORITY_ORDER row — the silent-unmetered failure mode behind the 17.5K prompt."""
        problems = {}
        for rel in FORMATTER_PATHS:
            keys = rendered_context_keys((REPO_ROOT / rel).read_text(encoding="utf-8"))
            assert keys, f"{rel}: no rendered context keys discovered — the extractor is blind"
            missing = unmetered(keys, _PRIORITY_NAMES, METERING_EXCEPTIONS)
            if missing:
                problems[rel] = {key: keys[key] for key in missing}
        assert not problems, (
            f"Formatter renders unmetered context keys: {problems} — "
            "add PRIORITY_ORDER rows (or a reasoned METERING_EXCEPTIONS entry here)"
        )

    def test_every_exception_is_still_read_or_structurally_unrendered(self):
        """A stale exception would silently excuse a future key of the same name."""
        keys = set()
        for rel in FORMATTER_PATHS:
            keys |= set(rendered_context_keys((REPO_ROOT / rel).read_text(encoding="utf-8")))
        stale = sorted(
            key for key in METERING_EXCEPTIONS
            if key not in keys and key not in UNRENDERED_CONTEXT_KEYS
        )
        assert not stale, f"exceptions no formatter path reads any more: {stale}"


class TestRenderedKeyExtraction:
    """Red controls: each spelling of a new rendered key is seen and fails parity."""

    @pytest.mark.parametrize(
        "source, form",
        [
            ('def render(context):\n    return context.get("new_section", [])\n', "get"),
            ("def render(context):\n    return context.get('new_section')\n", "get"),
            ('def render(ctx):\n    return ctx["new_section"]\n', "subscript"),
            ('def render(context):\n    if "new_section" in context:\n        return 1\n', "membership"),
            (
                'def render(context):\n'
                '    def _section(key, fallback=None):\n'
                '        return context.get(key, fallback)\n'
                '    return _section("new_section")\n',
                "helper:_section",
            ),
            (
                'def render(context):\n'
                '    for name in ("new_section", "recent_summaries"):\n'
                '        context.get(name)\n',
                "loop",
            ),
        ],
    )
    def test_new_rendered_key_is_extracted_and_fails_until_metered(self, source, form):
        keys = rendered_context_keys(source)
        assert form in keys.get("new_section", []), keys
        assert unmetered(keys, _PRIORITY_NAMES, METERING_EXCEPTIONS) == ["new_section"]
        assert unmetered(keys, _PRIORITY_NAMES | {"new_section"}, METERING_EXCEPTIONS) == []
        assert unmetered(keys, _PRIORITY_NAMES, {**METERING_EXCEPTIONS, "new_section": "reason"}) == []

    def test_writes_to_the_context_are_not_reads(self):
        source = 'def render(context, images):\n    context["note_images"] = images\n'
        assert rendered_context_keys(source) == {}

    def test_live_formatter_reads_known_rendered_sections(self):
        keys = rendered_context_keys((REPO_ROOT / FORMATTER_PATHS[0]).read_text(encoding="utf-8"))
        for key in ("recent_summaries", "semantic_reflections", "user_uploads", "narrative_state"):
            assert key in keys


class _FakeModelManager:
    def get_active_model_name(self):
        return "test-model"


class _FakeTokenizerManager:
    def count_tokens(self, text, model_name=None):
        return len((text or "").split())


def _mk_items(n, words_per_item=50, tag="item"):
    return [{"content": " ".join([f"{tag}{i}"] * words_per_item)} for i in range(n)]


class TestBudgetTrimsRenderedSections:
    def _manager(self, budget):
        return TokenManager(_FakeModelManager(), _FakeTokenizerManager(), budget)

    def test_over_budget_context_trims_semantic_reflections(self):
        # Lowest-priority rendered content must actually shrink now that the
        # split keys are metered (pre-fix: untouched at any budget).
        ctx = {
            "recent_conversations": _mk_items(4, tag="conv"),
            "recent_summaries": _mk_items(5, tag="rsum"),
            "semantic_summaries": _mk_items(5, tag="ssum"),
            "recent_reflections": _mk_items(4, tag="rref"),
            "semantic_reflections": _mk_items(4, tag="sref"),
        }
        tm = self._manager(budget=600)
        trimmed = tm._manage_token_budget(ctx)
        total = sum(
            len(it["content"].split())
            for key in ctx
            for it in trimmed.get(key, [])
        )
        assert total <= 600
        kept_refl = len(trimmed.get("recent_reflections", [])) + len(
            trimmed.get("semantic_reflections", [])
        )
        assert kept_refl < 8, "reflections were not trimmed at all"

    def test_under_budget_context_untouched(self):
        ctx = {
            "recent_conversations": _mk_items(2, words_per_item=10),
            "recent_summaries": _mk_items(2, words_per_item=10),
        }
        tm = self._manager(budget=10_000)
        trimmed = tm._manage_token_budget(ctx)
        assert len(trimmed["recent_conversations"]) == 2
        assert len(trimmed["recent_summaries"]) == 2

    def test_dead_combined_keys_do_not_consume_budget(self):
        # A fat combined "summaries" key (unrendered input) must not push
        # rendered content out of the budget.
        ctx = {
            "summaries": _mk_items(50, words_per_item=100, tag="dead"),
            "recent_conversations": _mk_items(3, words_per_item=50, tag="conv"),
        }
        tm = self._manager(budget=200)
        trimmed = tm._manage_token_budget(ctx)
        assert len(trimmed["recent_conversations"]) == 3
        # The dead key passes through untouched (back-compat), just unmetered.
        assert len(trimmed["summaries"]) == 50
