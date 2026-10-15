"""DM-41 / BC-95 ratchet: a positional index is never a durable identity.

Mechanism: an f-string that bakes a loop/arithmetic index (``block_n:{N}:{b}``)
into a ``tags`` entry, an id, or a doc_id.  In a mutable/rolling collection the
position shifts under trims, so the "identity" later names different content
(the 2026-10 summary re-summarization incident).  Identity must come from a
durable content anchor (timestamp range, content hash, stable doc id).

Scan: stdlib ``ast`` over production dirs.  Flags f-strings in identity slots
(``tags``/``id``/``doc_id``/``*_id`` dict keys or keywords, or an element of a
``tags`` list) that interpolate a name which is a range()/enumerate() loop
target, is assigned from arithmetic on one, or is an index-named parameter.
Ceiling = findings after the lane-SUM fix (print with ``pytest -s``).
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = ("core", "memory", "knowledge", "utils", "api", "gui", "models", "processing")
IDENTITY_KEYS = {"id", "doc_id", "tags"}
INDEX_PARAMS = {"b", "i", "idx", "index", "block_idx", "block_index", "pos", "position"}
ALLOWLIST = {
    # Per-document citation labels (WEB_1, DOC_2): ids are scoped to ONE generated
    # document's source list, rebuilt each run; nothing persists or is joined later.
    ("knowledge/document_generator.py", "_search_web"): "transient per-document citation label",
    ("knowledge/document_generator.py", "_search_collection"): "transient per-document citation label",
}
CEILING = 0


def _is_identity_key(name):
    return isinstance(name, str) and (name in IDENTITY_KEYS or name.endswith("_id"))


def _loop_names(fn):
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call) \
                and isinstance(node.iter.func, ast.Name) and node.iter.func.id in ("range", "enumerate"):
            tgt = node.target
            if node.iter.func.id == "enumerate" and isinstance(tgt, ast.Tuple) and tgt.elts:
                tgt = tgt.elts[0]
            if isinstance(tgt, ast.Name):
                names.add(tgt.id)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp):
                used = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
                if used & names:
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id not in names:
                            names.add(t.id)
                            changed = True
    args = fn.args
    for a in args.args + args.kwonlyargs:
        if a.arg in INDEX_PARAMS:
            names.add(a.arg)
    return names


def _fstring_names(node):
    return {n.id for fv in node.values if isinstance(fv, ast.FormattedValue)
            for n in ast.walk(fv.value) if isinstance(n, ast.Name)}


def _identity_fstrings(fn):
    """Yield JoinedStr nodes sitting in an identity slot."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and _is_identity_key(k.value):
                    yield from _slot(v)
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg and _is_identity_key(kw.arg):
                    yield from _slot(kw.value)


def _slot(value):
    if isinstance(value, ast.JoinedStr):
        yield value
    elif isinstance(value, (ast.List, ast.Tuple)):
        for e in value.elts:
            if isinstance(e, ast.JoinedStr):
                yield e


def scan_source(source, relpath="<fixture>"):
    tree = ast.parse(source)
    hits = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if (relpath, fn.name) in ALLOWLIST:
            continue
        idx = _loop_names(fn)
        for js in _identity_fstrings(fn):
            if _fstring_names(js) & idx:
                hits.append(f"{relpath}:{fn.name}:{js.lineno}")
    return sorted(set(hits))


def _scan_repo():
    hits = []
    for d in SCAN_DIRS:
        for path in sorted((ROOT / d).rglob("*.py")):
            if "tests" in path.parts:
                continue
            hits += scan_source(path.read_text(encoding="utf-8"), str(path.relative_to(ROOT)))
    return hits


def test_positional_identity_ceiling():
    hits = _scan_repo()
    print("\n".join(hits))
    assert len(hits) <= CEILING, f"positional index used as durable identity (BC-95): {hits}"


def test_sensitivity_old_shutdown_processor_shape_is_flagged():
    old = '''
def gen(selected, N):
    for b in range(len(selected)):
        start = b * N
        end = start + N
        store({"tags": f"summary,block_n:{N}:{b},block_span_n:{N}:{start}-{end - 1}"})
        add(tags=[f"block_n:{N}:{b}"])
'''
    assert len(scan_source(old)) >= 2


def test_sensitivity_stable_ids_are_not_flagged():
    ok = '''
def gen(doc_id, chunks):
    for i, c in enumerate(chunks):
        add(content=f"chunk {i}", id=doc_id)
        add(tags=[f"source:{doc_id}"])
'''
    assert scan_source(ok) == []
