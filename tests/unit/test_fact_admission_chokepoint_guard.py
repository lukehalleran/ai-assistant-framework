"""
Repo-wide guard: every write to the Chroma ``facts`` collection goes through
the fact-admission chokepoint (BC-58, incident 2026-10-18).

At the 10-18 19:56 shutdown the REGEX shutdown path re-mined the session's
homework pastes and stored six course-material sentences as facts; the same
text on the per-turn path had been dropped by the paste guard two minutes
earlier. The guards lived only on the first-observed path.

Scans ``core/, memory/, knowledge/, utils/, gui/, api/, models/, processing/,
agent_branch/, eval/`` plus ``main.py`` (never ``git ls-files``) for a call
``<...>chroma_store.add_fact(...)`` / ``<...>store.add_fact(...)``. A site is
SAFE when its enclosing function also calls ``admit_extracted_facts`` (the
shared admission function in ``memory/memory_storage.py``); otherwise it must
be on the reviewed ALLOWLIST below, keyed by CONTENT anchor
``(path, enclosing function, stripped source line)`` so line drift is
harmless (same anchoring as ``test_gather_results_guard.py``).
"""
from __future__ import annotations

import ast
from collections import Counter, namedtuple
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = ("core", "memory", "knowledge", "utils", "gui", "api", "models",
             "processing", "agent_branch", "eval")
EXTRA_FILES = ("main.py",)
_CHOKEPOINT = "admit_extracted_facts"

Hit = namedtuple("Hit", "rel lineno scope text")

ALLOWLIST = (
    (("memory/shutdown_processor.py", "ShutdownProcessor._extract_llm_facts",
      "result = self.chroma_store.add_fact("),
     "LLM shutdown path is deliberately different: provenance-based. Every "
     "triple already had to join back to a USER-authored span (memory/"
     "fact_source.py) and the per-evidence paste guard in llm_fact_extractor "
     "shares THE deployed is_paste_sized/subject_is_user_anchored tests."),
    (("gui/wizard.py", "_finalize_wizard",
      "chroma_store.add_fact(fact=fact_text, source=source_dict)"),
     "onboarding facts come from the user's own background text, extracted "
     "by LLMFactExtractor.extract_triples fed a user-role message dict, so "
     "every triple already passed THE fact_source user-span join "
     "(gui/wizard.py ~783); no pasted document text and no Daemon response "
     "can reach this write."),
    (("gui/wizard.py", "_finalize_wizard", "chroma_store.add_fact("),
     "onboarding facts come from the user's own background text, extracted "
     "by LLMFactExtractor.extract_triples fed a user-role message dict, so "
     "every triple already passed THE fact_source user-span join "
     "(gui/wizard.py ~783); no pasted document text and no Daemon response "
     "can reach this write."),
)

_ALLOWED = Counter(key for key, _r in ALLOWLIST)


def _receiver_text(call: ast.Call) -> str:
    fn = call.func
    if isinstance(fn, ast.Attribute) and fn.attr == "add_fact":
        try:
            return ast.unparse(fn.value)
        except Exception:
            return ""
    return ""


def _is_facts_store_add(node) -> bool:
    if not isinstance(node, ast.Call):
        return False
    recv = _receiver_text(node)
    return bool(recv) and (recv == "store" or recv.endswith("chroma_store"))


def _spans(tree):
    out = []

    def walk(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + ch.name
                out.append((ch.lineno, ch.end_lineno or ch.lineno, name, ch))
                walk(ch, name + ".")
            elif isinstance(ch, ast.ClassDef):
                walk(ch, prefix + ch.name + ".")
            else:
                walk(ch, prefix)

    walk(tree, "")
    return out


def _innermost(spans, lineno):
    best = None
    for s, e, name, node in spans:
        if s <= lineno <= e and (best is None or e - s < best[1] - best[0]):
            best = (s, e, name, node)
    return best


def _calls_chokepoint(func_node) -> bool:
    for n in ast.walk(func_node):
        if isinstance(n, ast.Call):
            fn = n.func
            nm = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if nm == _CHOKEPOINT:
                return True
    return False


def find_violations(source: str, rel: str):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [Hit(rel, 0, "<unparsed>", "<file does not parse>")]
    lines = source.splitlines()
    spans = _spans(tree)
    out = []
    for node in ast.walk(tree):
        if not _is_facts_store_add(node):
            continue
        sc = _innermost(spans, node.lineno)
        scope = sc[2] if sc else "<module>"
        text = lines[node.lineno - 1].strip()
        if sc is not None and _calls_chokepoint(sc[3]):
            continue
        out.append(Hit(rel, node.lineno, scope, text))
    return out


def _iter_files():
    for d in SCAN_DIRS:
        base = REPO_ROOT / d
        if base.exists():
            yield from sorted(base.rglob("*.py"))
    for n in EXTRA_FILES:
        p = REPO_ROOT / n
        if p.exists():
            yield p


def _all_violations():
    out = []
    for p in _iter_files():
        try:
            src = p.read_text(encoding="utf-8")
        except Exception:
            continue
        out.extend(find_violations(src, str(p.relative_to(REPO_ROOT))))
    return out


def _norm(hit: Hit):
    return (hit.rel, hit.scope, hit.text)


def test_every_facts_write_goes_through_the_admission_chokepoint():
    seen = Counter()
    bad = []
    for h in _all_violations():
        k = _norm(h)
        seen[k] += 1
        if seen[k] > _ALLOWED.get(k, 0):
            bad.append(h)
    assert not bad, (
        "chroma_store.add_fact(...) outside the fact-admission chokepoint "
        "(call memory.memory_storage.admit_extracted_facts in the same "
        "function, or add a reviewed allowlist entry):\n"
        + "\n".join(f"  {h.rel}:{h.lineno} [{h.scope}] {h.text!r}" for h in bad))


def test_allowlist_has_no_stale_entries():
    found = Counter()
    for h in _all_violations():
        found[_norm(h)] += 1
    stale = [k for k in _ALLOWED if k not in found]
    assert not stale, f"stale allowlist entries: {stale}"


def test_every_allowlist_entry_has_a_reason():
    assert all(len(r) > 20 for _k, r in ALLOWLIST)


class TestDetectorSelfTests:
    def test_flags_a_direct_write_with_no_chokepoint(self):
        src = ("async def f(self, facts):\n"
               "    for x in facts:\n"
               "        self.chroma_store.add_fact(fact=x)\n")
        assert len(find_violations(src, "m.py")) == 1

    def test_passes_when_function_calls_the_chokepoint(self):
        src = ("async def f(self, q, facts):\n"
               "    facts = admit_extracted_facts(q, facts)\n"
               "    for x in facts:\n"
               "        self.chroma_store.add_fact(fact=x)\n")
        assert find_violations(src, "m.py") == []

    def test_other_receivers_are_not_facts_collection_writes(self):
        src = ("def f(profile):\n    profile.add_fact('a', 'b')\n")
        assert find_violations(src, "m.py") == []
