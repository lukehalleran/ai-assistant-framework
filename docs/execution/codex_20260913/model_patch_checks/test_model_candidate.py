"""Standalone checks for the proposed model patch; no repo imports or pytest."""

import ast
import re
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[4]
PATCH_PATH = ROOT / "docs/execution/codex_20260913/model_integration.patch"
SOURCE_PATH = ROOT / "models/model_manager.py"


def _fake_response():
    message = SimpleNamespace(content="answer", tool_calls=None)
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=message)])


class _FakeSyncCompletions:
    def __init__(self, captured):
        self.captured = captured

    def create(self, **kwargs):
        self.captured.append(kwargs)
        return _fake_response()


class _FakeAsyncCompletions:
    def __init__(self, captured):
        self.captured = captured

    async def create(self, **kwargs):
        self.captured.append(kwargs)
        return _fake_response()


class _FakeClient:
    def __init__(self, captured, asynchronous=False):
        completions = (
            _FakeAsyncCompletions(captured)
            if asynchronous
            else _FakeSyncCompletions(captured)
        )
        self.chat = SimpleNamespace(completions=completions)


def apply_unified_diff_in_memory(original_text, patch_text, target):
    """Apply one unified-diff file section to text without touching the source."""
    patch_lines = patch_text.splitlines(keepends=True)
    original_lines = original_text.splitlines(keepends=True)
    i = 0
    while i < len(patch_lines):
        if not patch_lines[i].startswith("--- "):
            i += 1
            continue
        if i + 1 >= len(patch_lines) or not patch_lines[i + 1].startswith("+++ "):
            raise AssertionError("unified diff is missing its +++ header")
        target_name = patch_lines[i + 1][4:].strip()
        i += 2
        result = []
        cursor = 0
        found = target_name == target
        if not found:
            while i < len(patch_lines) and not patch_lines[i].startswith("--- "):
                i += 1
            continue
        while i < len(patch_lines) and not patch_lines[i].startswith("--- "):
            line = patch_lines[i]
            if line.startswith("@@ "):
                match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
                if not match:
                    raise AssertionError(f"bad hunk header: {line.rstrip()}")
                old_start = int(match.group(1)) - 1
                if old_start < cursor or old_start > len(original_lines):
                    raise AssertionError("hunk source offset is outside original file")
                result.extend(original_lines[cursor:old_start])
                cursor = old_start
                i += 1
                while i < len(patch_lines):
                    body = patch_lines[i]
                    if body.startswith("@@ ") or body.startswith("--- "):
                        break
                    if body.startswith("\\"):
                        i += 1
                        continue
                    if not body or body[0] not in " +-":
                        break
                    content = body[1:]
                    if body[0] == "+":
                        result.append(content)
                    else:
                        if cursor >= len(original_lines):
                            raise AssertionError("hunk consumes beyond original file")
                        expected = original_lines[cursor].rstrip("\r\n")
                        actual = content.rstrip("\r\n")
                        if expected != actual:
                            raise AssertionError(
                                f"hunk context mismatch at source line {cursor + 1}: "
                                f"{expected!r} != {actual!r}"
                            )
                        if body[0] == " ":
                            result.append(original_lines[cursor])
                        cursor += 1
                    i += 1
                continue
            i += 1
        result.extend(original_lines[cursor:])
        return "".join(result)
    raise AssertionError(f"diff has no section for {target}")


def candidate_namespace(candidate_text):
    """Evaluate only pure registry/helper nodes from candidate AST source."""
    tree = ast.parse(candidate_text, filename="models/model_manager.py")
    wanted_assignments = {
        "API_MODEL_ALIASES",
        "_CLAUDE",
        "_GPT",
        "MODEL_CAPABILITIES",
        "DEFAULT_API_CONTEXT_LIMIT",
        "MODEL_CONTEXT_LIMITS",
        "_OPENROUTER_UNSUPPORTED_PARAMETERS",
        "_ALWAYS_ON_REASONING_MODELS",
        "_SYNC_REASONING_MODELS",
    }
    wanted_functions = {
        "_slug_supports_reasoning",
        "_slug_supports_vision",
        "_slug_supports_tools",
        "_slug_supports_prompt_caching",
        "_filter_openrouter_request_params",
        "_slug_supports_openrouter_parameter",
        "_unsupported_tool_choice_error",
        "_reasoning_request_config",
    }
    namespace = {"__builtins__": __builtins__}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if names & wanted_assignments:
                exec(compile(ast.Module(body=[node], type_ignores=[]), "candidate", "exec"), namespace)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in wanted_assignments:
                exec(compile(ast.Module(body=[node], type_ignores=[]), "candidate", "exec"), namespace)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_functions:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "candidate", "exec"), namespace)
    return namespace


class ModelCandidateChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        patch_text = PATCH_PATH.read_text(encoding="utf-8")
        source_text = SOURCE_PATH.read_text(encoding="utf-8")
        cls.candidate = apply_unified_diff_in_memory(
            source_text, patch_text, "b/models/model_manager.py"
        )
        cls.tree = ast.parse(cls.candidate, filename="candidate_model_manager.py")
        cls.namespace = candidate_namespace(cls.candidate)

    def test_patch_applies_and_candidate_parses(self):
        self.assertNotEqual(self.candidate, SOURCE_PATH.read_text(encoding="utf-8"))

    def test_aliases_preserve_v4_flash_and_add_exact_new_slugs(self):
        aliases = self.namespace["API_MODEL_ALIASES"]
        self.assertEqual(aliases["deepseek-v4-flash"], "deepseek/deepseek-v4-flash")
        self.assertEqual(aliases["deepseek-v4.1-flash"], "deepseek/deepseek-v4.1-flash")
        self.assertEqual(aliases["claude-fable-5.1"], "anthropic/claude-fable-5.1")
        self.assertEqual(aliases["fable-5.1"], "anthropic/claude-fable-5.1")
        self.assertEqual(aliases["gpt-6-astra"], "openai/gpt-6-astra")

    def test_registry_capabilities_and_classifier_parity(self):
        ns = self.namespace
        caps = ns["MODEL_CAPABILITIES"]
        expected = {
            "deepseek/deepseek-v4.1-flash": (True, True, True, None),
            "anthropic/claude-fable-5.1": (True, True, True, "explicit"),
            "openai/gpt-6-astra": (True, True, True, "implicit"),
        }
        classifiers = (
            ns["_slug_supports_reasoning"],
            ns["_slug_supports_vision"],
            ns["_slug_supports_tools"],
        )
        for slug, truth in expected.items():
            row = caps[slug]
            self.assertEqual(
                (row["reasoning"], row["vision"], row["tools"], row["caching"]), truth
            )
            self.assertEqual(tuple(fn(slug) for fn in classifiers), truth[:3])
            self.assertEqual(
                ns["_slug_supports_prompt_caching"](slug), truth[3] == "explicit"
            )
        self.assertFalse(ns["_slug_supports_vision"]("deepseek/deepseek-v4-flash"))

    def test_context_limits_match_public_catalog(self):
        limits = self.namespace["MODEL_CONTEXT_LIMITS"]
        self.assertEqual(limits["deepseek/deepseek-v4.1-flash"], 1_048_576)
        self.assertEqual(limits["anthropic/claude-fable-5.1"], 1_000_000)
        self.assertEqual(limits["openai/gpt-6-astra"], 1_050_000)

    def test_openrouter_parameter_filters_cover_all_three_routes(self):
        filter_params = self.namespace["_filter_openrouter_request_params"]
        params = {
            "temperature": 0.2,
            "top_p": 0.8,
            "stop": ["END"],
            "tool_choice": "auto",
            "tools": [{"type": "function"}],
        }
        deepseek = filter_params("deepseek/deepseek-v4.1-flash", params)
        self.assertEqual(deepseek, params)
        fable = filter_params("anthropic/claude-fable-5.1", params)
        self.assertNotIn("temperature", fable)
        self.assertNotIn("top_p", fable)
        self.assertNotIn("tool_choice", fable)
        self.assertIn("stop", fable)
        self.assertIn("tools", fable)
        astra = filter_params("openai/gpt-6-astra", params)
        for key in ("temperature", "top_p", "stop"):
            self.assertNotIn(key, astra)
        for key in ("tools", "tool_choice"):
            self.assertIn(key, astra)

    def test_tool_choice_and_mandatory_reasoning_recovery(self):
        ns = self.namespace
        error = ns["_unsupported_tool_choice_error"]
        fable = "anthropic/claude-fable-5.1"
        self.assertEqual(error(fable, "auto"), "")
        self.assertEqual(error(fable, "none"), "")
        self.assertTrue(error(fable, {"type": "function", "function": {"name": "x"}}).startswith("[MODEL NOT SUPPORTED]"))
        self.assertEqual(error("openai/gpt-6-astra", {"type": "function"}), "")
        reasoning = ns["_reasoning_request_config"]
        self.assertEqual(reasoning("deepseek/deepseek-v4.1-flash", True), {"enabled": False})
        self.assertEqual(reasoning("deepseek/deepseek-v4.1-flash"), {"effort": "low"})
        self.assertEqual(reasoning(fable, True), {"effort": "low", "exclude": True})
        self.assertEqual(reasoning("openai/gpt-6-astra", True), {"effort": "low", "exclude": True})

    def test_sync_generation_request_boundary(self):
        method = self._candidate_method("generate_with_openai")
        captured = []
        namespace = self._runtime_namespace()
        exec(compile(ast.Module(body=[method], type_ignores=[]), "candidate_method", "exec"), namespace)

        class Manager:
            default_max_tokens = 64
            default_temperature = 0.4

            def __init__(self, full_slug):
                self.api_models = {full_slug: full_slug}
                self.client = _FakeClient(captured)

            def resolve_top_p(self, model_name, top_p):
                return 0.9 if top_p is None else top_p

            def supports_prompt_caching(self, _model_name):
                return False

            def _strip_cache_breakpoint(self, prompt):
                return prompt

            def _log_cache_usage(self, *args, **kwargs):
                return None

        for slug, expected_reasoning, unsupported in (
            ("deepseek/deepseek-v4.1-flash", {"effort": "low"}, ()),
            ("anthropic/claude-fable-5.1", {"effort": "medium"}, ("temperature", "top_p")),
            ("openai/gpt-6-astra", {"effort": "medium"}, ("temperature", "top_p", "stop")),
        ):
            captured.clear()
            self.assertEqual(namespace["generate_with_openai"](
                Manager(slug), "prompt", slug, system_prompt="system"
            ), "answer")
            params = captured[-1]
            for key in unsupported:
                self.assertNotIn(key, params)
            self.assertEqual(params["extra_body"]["reasoning"], expected_reasoning)
            self.assertEqual(params["max_tokens"], 64)

        # Existing sync models keep their former request shape.
        captured.clear()
        old = Manager("openai/gpt-5.5")
        namespace["generate_with_openai"](old, "prompt", "openai/gpt-5.5")
        self.assertNotIn("reasoning", captured[-1]["extra_body"])

    def test_tool_request_boundary_and_fable_explicit_choice(self):
        method = self._candidate_method("generate_once_with_tools")
        captured = []
        namespace = self._runtime_namespace()
        exec(compile(ast.Module(body=[method], type_ignores=[]), "candidate_method", "exec"), namespace)
        tool = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]

        class Manager:
            default_max_tokens = 64
            default_temperature = 0.4
            models = {}

            def __init__(self, full_slug):
                self.api_models = {full_slug: full_slug}
                self.async_client = _FakeClient(captured, asynchronous=True)

            def _strip_cache_breakpoint(self, prompt):
                return prompt

            def supports_reasoning(self, _model_name):
                return True

        async def call(slug, **kwargs):
            return await namespace["generate_once_with_tools"](
                Manager(slug), "prompt", model_name=slug, tools=tool, **kwargs
            )

        import asyncio
        fable = "anthropic/claude-fable-5.1"
        captured.clear()
        result = asyncio.run(call(fable, tool_choice="auto"))
        self.assertEqual(getattr(result, "content", None), "answer")
        params = captured[-1]
        self.assertIn("tools", params)
        self.assertNotIn("tool_choice", params)
        self.assertNotIn("temperature", params)

        captured.clear()
        result = asyncio.run(call(fable, tool_choice={"type": "function", "function": {"name": "lookup"}}))
        self.assertTrue(result["content"].startswith("[MODEL NOT SUPPORTED]"))
        self.assertEqual(captured, [])

        captured.clear()
        asyncio.run(call(fable, tool_choice="auto", disable_reasoning=True))
        self.assertEqual(captured[-1]["extra_body"]["reasoning"], {"effort": "low", "exclude": True})

        astra = "openai/gpt-6-astra"
        captured.clear()
        asyncio.run(call(astra, tool_choice="auto", disable_reasoning=True))
        params = captured[-1]
        self.assertIn("tools", params)
        self.assertEqual(params["tool_choice"], "auto")
        self.assertNotIn("temperature", params)
        self.assertEqual(params["extra_body"]["reasoning"], {"effort": "low", "exclude": True})

        deepseek = "deepseek/deepseek-v4.1-flash"
        captured.clear()
        asyncio.run(call(deepseek, tool_choice="auto", disable_reasoning=True))
        self.assertEqual(captured[-1]["extra_body"]["reasoning"], {"enabled": False})

    def _candidate_method(self, method_name):
        manager = next(
            node for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ModelManager"
        )
        method = next(
            node for node in manager.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
        )
        method.decorator_list = []
        return method

    def _runtime_namespace(self):
        class Logger:
            def __getattr__(self, _name):
                return lambda *args, **kwargs: None

        ns = dict(self.namespace)
        ns.update({
            "SYSTEM_PROMPT": "system",
            "logger": Logger(),
            "_classify_api_error": lambda exc: str(exc),
        })
        return ns

    def test_each_api_generation_path_filters_route_parameters(self):
        manager = next(
            node for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ModelManager"
        )
        methods = {
            node.name: node
            for node in manager.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for method_name in (
            "generate_with_openai",
            "generate_once",
            "generate_once_with_tools",
            "generate_async",
        ):
            calls = {
                node.func.id
                for node in ast.walk(methods[method_name])
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            self.assertIn("_filter_openrouter_request_params", calls, method_name)
        for method_name in ("generate_once", "generate_once_with_tools", "generate_async"):
            calls = {
                node.func.id
                for node in ast.walk(methods[method_name])
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            self.assertIn("_reasoning_request_config", calls, method_name)

    def test_gui_and_api_selectors_are_registry_driven(self):
        api = (ROOT / "api/routes/models.py").read_text(encoding="utf-8")
        gui = (ROOT / "gui/launch.py").read_text(encoding="utf-8")
        self.assertIn('getattr(mm, "api_models", {}).keys()', api)
        self.assertIn("getattr(_mm, 'api_models', {}).keys()", gui)


if __name__ == "__main__":
    unittest.main(verbosity=2)
