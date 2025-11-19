#!/usr/bin/env python3
"""
Quick, read-only repo scan for topic categorization logic and model usage.

Outputs:
- Likely files that define a Topic Manager or topic categorization.
- Lines that indicate LLM/model usage related to topics or default model config.

This script is intended to run via PYTHONSTARTUP so that `python` executes it
and exits immediately. It prints concise matches with small context.
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path.cwd()
EXCLUDE_DIRS = {'.git', '.venv', 'data', 'chroma_db', 'embedded_parquet', '__pycache__'}
INCLUDE_EXTS = {'.py', '.md', '.yaml', '.yml'}

topic_patterns = [
    re.compile(r'\bTopicManager\b'),
    re.compile(r'\btopic_manager\b'),
    re.compile(r'topic\s*[:=]'),
    re.compile(r'topic\w*', re.IGNORECASE),
    re.compile(r'categor\w*', re.IGNORECASE),
]

model_patterns = [
    re.compile(r'\bmodel\s*[:=]\s*[\'\"]?([\w\-:.]+)[\'\"]?', re.IGNORECASE),
    re.compile(r'\bOPENAI_?MODEL\b', re.IGNORECASE),
    re.compile(r'\bMODEL_NAME\b', re.IGNORECASE),
    re.compile(r'\bCHAT_MODEL\b', re.IGNORECASE),
    re.compile(r'\bLLM\b'),
    re.compile(r'openai', re.IGNORECASE),
]

def should_skip_dir(d: str) -> bool:
    return d in EXCLUDE_DIRS or d.startswith('.') and d not in {'.', '..'}

def scan_file(path: Path):
    try:
        text = path.read_text(encoding='utf-8', errors='ignore').splitlines()
    except Exception:
        return []
    matches = []
    for i, line in enumerate(text, start=1):
        if any(p.search(line) for p in topic_patterns + model_patterns):
            matches.append((i, line))
    return matches

def walk_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded dirs in-place
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in INCLUDE_EXTS:
                yield Path(dirpath) / fn

def print_context(path: Path, matches, ctx: int = 1):
    try:
        lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()
    except Exception:
        return
    printed = set()
    for ln, _ in matches[:50]:  # cap per file
        if ln in printed:
            continue
        start = max(1, ln - ctx)
        end = min(len(lines), ln + ctx)
        print(f"\n==> {path}#{ln}")
        for i in range(start, end + 1):
            prefix = '>' if i == ln else ' '
            snippet = lines[i - 1]
            print(f"{prefix} {i:5d}: {snippet}")
        printed.add(ln)

def main():
    topic_hits = []
    model_hits = []
    for p in walk_files(ROOT):
        ms = scan_file(p)
        if not ms:
            continue
        # Separate topic vs model-related files by heuristic
        if any(re.search(r'topic|categor', line, re.IGNORECASE) for _, line in ms):
            topic_hits.append((p, ms))
        if any(any(mp.search(line) for mp in model_patterns) for _, line in ms):
            model_hits.append((p, ms))

    print("[Topic-related matches]")
    for p, ms in topic_hits[:20]:
        print_context(p, ms)

    print("\n[Model-related matches]")
    for p, ms in model_hits[:20]:
        print_context(p, ms)

    # Ensure we exit so `python` does not drop to REPL
    sys.exit(0)

if __name__ == '__main__':
    main()
