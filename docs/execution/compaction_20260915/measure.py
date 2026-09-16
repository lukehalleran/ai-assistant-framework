"""Read tracked files without importing the app; write compaction audit evidence.

Run from the repository root: python docs/execution/compaction_20260915/measure.py
Requires locally cached tiktoken cl100k_base. Outputs exclude this audit's files.
Fold counts are candidates, not recommendations or behavior-equivalence proofs.
"""
import ast
import csv
import hashlib
import io
import json
import subprocess
import tokenize
from collections import defaultdict
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ENC = tiktoken.get_encoding("cl100k_base")
WIDTHS = (88, 100, 110, 120)
AUDIT_NAMES = ("COMPACTION_AUDIT_20260915", "PLAN_20260915_codebase_compaction")


def tokens(text):
    return len(ENC.encode(text, disallowed_special=()))


def folding(source):
    """Join comment-free logical lines, then compare full location-free ASTs."""
    lines = source.splitlines(keepends=True)
    candidates, statement = [], []
    tree = ast.dump(ast.parse(source), include_attributes=False)
    ignored = {tokenize.INDENT, tokenize.DEDENT, tokenize.NL, tokenize.ENDMARKER}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.NEWLINE:
            significant = [t for t in statement if t.type not in ignored]
            if significant and not any(t.type == tokenize.COMMENT for t in significant):
                start, end = significant[0].start[0], significant[-1].end[0]
                multiline_string = any(t.type == tokenize.STRING and t.start[0] != t.end[0] for t in significant)
                if end > start and not multiline_string:
                    chunk = lines[start - 1:end]
                    joined = chunk[0].rstrip()
                    for continuation in chunk[1:]:
                        piece = continuation.strip()
                        if piece:
                            separator = "" if joined.endswith(("(", "[", "{")) or piece.startswith((")", "]", "}")) else " "
                            joined += separator + piece
                    if "\\" not in "".join(chunk):
                        candidates.append((start, end, joined))
            statement = []
        elif token.type == tokenize.NL and not statement:
            continue
        elif token.type not in {tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}:
            statement.append(token)
            if token.type == tokenize.COMMENT and len(statement) == 1:
                statement = []
    counts = {}
    chosen = []
    for width in WIDTHS:
        selected = [row for row in candidates if len(row[2]) <= width]
        transformed = list(lines)
        for start, end, joined in reversed(selected):
            transformed[start - 1:end] = [joined + "\n"]
        compacted = "".join(transformed)
        equal = ast.dump(ast.parse(compacted), include_attributes=False) == tree
        if not equal:
            raise ValueError("AST differs after candidate folding")
        counts[f"fold{width}_statements"] = len(selected)
        counts[f"fold{width}_lines"] = sum(end - start for start, end, _ in selected)
        counts[f"fold{width}_tokens"] = tokens(source) - tokens(compacted)
        if width == 100:
            chosen = selected
    return counts, chosen


def category(path):
    if path.endswith(".py"):
        return "python"
    if Path(path).suffix in {".ts", ".tsx", ".css", ".html"}:
        return "frontend"
    if path.endswith(".md"):
        return "markdown"
    if "lock" in Path(path).name or path.startswith("data/") and not path.startswith("data/pipeline/"):
        return "generated_or_data"
    return "other_text"


paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")[:-1]
paths = [p for p in paths if not p.startswith("docs/execution/compaction_20260915/") and not any(n in p for n in AUDIT_NAMES)]
rows, candidate_rows, errors = [], [], []
for relative in paths:
    path = ROOT / relative
    raw = path.read_bytes()
    row = dict(path=relative, root=relative.split("/")[0] if "/" in relative else "[root]",
               category="binary", bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    try:
        source = raw.decode("utf-8")
        if "\0" in source:
            raise ValueError("binary NUL")
        lines = source.splitlines()
        row.update(category=category(relative), lines=len(lines), blank=sum(not line.strip() for line in lines),
                   tokens_cl100k=tokens(source), long100=sum(len(line) > 100 for line in lines),
                   long160=sum(len(line) > 160 for line in lines), scan="text_metrics")
        if relative.endswith(".py"):
            stream = list(tokenize.generate_tokens(io.StringIO(source).readline))
            row["comment_lines"] = len({t.start[0] for t in stream if t.type == tokenize.COMMENT and not lines[t.start[0] - 1][:t.start[1]].strip()})
            tree = ast.parse(source)
            doclines = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
                    first = node.body[0]
                    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                        doclines.update(range(first.lineno, first.end_lineno + 1))
            row["docstring_lines"] = len(doclines)
            counts, selected = folding(source)
            row.update(counts, scan="tokenize_ast_metrics")
            candidate_rows.extend(dict(path=relative, start=start, end=end, removable_lines=end-start, width=len(joined)) for start, end, joined in selected)
    except (UnicodeError, ValueError, SyntaxError, tokenize.TokenError) as exc:
        if row["category"] != "binary":
            errors.append(dict(path=relative, error=f"{type(exc).__name__}: {exc}"))
            row["scan"] = "metrics_error"
        else:
            row["scan"] = "binary_excluded"
    rows.append(row)


def write_tsv(name, data):
    keys = list(dict.fromkeys(key for row in data for key in row))
    with (OUT / name).open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def aggregate(key):
    totals = defaultdict(lambda: defaultdict(int))
    for row in rows:
        dest = totals[row[key]]
        dest["files"] += 1
        for field, value in row.items():
            if isinstance(value, int):
                dest[field] += value
    return dict(sorted(totals.items()))


supplemental = []
for name in ("CLAUDE.md", "CLAUDE_CHANGELOG.md", "docs/PLAN_20260912_session_defects.md", "docs/SOURCE_DOCUMENT_TIER_DESIGN.md"):
    path = ROOT / name
    if path.exists() and name not in paths:
        raw = path.read_bytes()
        source = raw.decode()
        supplemental.append(dict(path=name, bytes=len(raw), lines=len(source.splitlines()), tokens_cl100k=tokens(source), sha256=hashlib.sha256(raw).hexdigest()))
summary = dict(head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
               tracked_files=len(rows), tokenizer="cl100k_base (comparative proxy, not claimed Astra tokenizer)",
               widths=WIDTHS, by_root=aggregate("root"), by_category=aggregate("category"), errors=errors,
               supplemental_not_in_tracked_totals=supplemental)
write_tsv("inventory.tsv", rows)
write_tsv("fold_candidates.tsv", candidate_rows)
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({key: summary[key] for key in ("head", "tracked_files", "by_category", "errors", "supplemental_not_in_tracked_totals")}, indent=2))
