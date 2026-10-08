"""BC-92 ratchet: streamed text is never split on its separators and re-joined
by guessing (2026-10-08).

`generate_streaming_response` used to `buffer.split(" ")` and yield bare words;
three consumers re-inserted the spaces three different ways (`smart_join`'s
punctuation guess, `chunk + " "` twice), deleting the space before every
opening quote and collapsing indentation. Chunks now carry their own
separators and every consumer joins through `core.response_generator.smart_join`.

This guard fails when a production module re-introduces either half:
  * a separator-discarding split in the streaming generator, or
  * a consumer accumulating stream chunks with a hand-inserted space.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = ("core", "gui", "api", "models", "utils", "knowledge", "memory")

# `acc += chunk + " "`, `acc = acc + " " + chunk`, `" ".join(chunks)` over a stream.
_SPACE_REJOIN_RE = re.compile(
    r"""\w*chunk\w*\s*\+\s*["'] ["']|["'] ["']\s*\+\s*\w*chunk\w*|["'] ["']\.join\(\s*\w*chunk"""
)
_SEPARATOR_SPLIT_RE = re.compile(r"""\bbuffer\.split\(\s*["'] ["']\s*\)""")


def _production_files():
    for d in SCAN_DIRS:
        for path in (ROOT / d).rglob("*.py"):
            if "tests" in path.parts:
                continue
            yield path


def test_no_consumer_rejoins_stream_chunks_with_a_guessed_space():
    hits = []
    for path in _production_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if _SPACE_REJOIN_RE.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not hits, (
        "Stream chunks carry their own whitespace; join them with "
        "core.response_generator.smart_join, never a hand-inserted space (BC-92):\n"
        + "\n".join(hits)
    )


def test_streaming_generator_keeps_separators():
    src = (ROOT / "core" / "response_generator.py").read_text(encoding="utf-8")
    assert not _SEPARATOR_SPLIT_RE.search(src), (
        "generate_streaming_response must cut with split_at_last_whitespace "
        "(separator kept on the chunk), not buffer.split(' ') (BC-92)"
    )
    assert "split_at_last_whitespace(buffer)" in src


def test_guard_patterns_catch_the_original_shapes():
    # Sensitivity: the exact pre-fix lines must trip the guard.
    assert _SPACE_REJOIN_RE.search('full_response += (chunk + " ")')
    assert _SPACE_REJOIN_RE.search("acc = acc + ' ' + chunk")
    assert _SEPARATOR_SPLIT_RE.search('words = buffer.split(" ")')
    assert not _SPACE_REJOIN_RE.search("final_output = smart_join(final_output, chunk)")
