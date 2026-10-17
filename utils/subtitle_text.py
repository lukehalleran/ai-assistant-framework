"""
# utils/subtitle_text.py

Module Contract
- Purpose: turn .srt / .vtt subtitle files (lecture transcripts) into running prose with
  coarse `[mm:ss]` paragraph anchors, dropping cue indexes, timing lines, headers and tags.
- Public: subtitle_to_text(raw) -> str. Pure, never raises; unparseable text passes through
  as plain lines.
"""
import re

PARAGRAPH_SECONDS = 60.0

_TIMING_RE = re.compile(
    r"^\s*(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})\s*-->\s*"
    r"(?:(?:\d+:)?\d{1,2}:\d{2}[.,]\d{1,3})\b.*$"
)
_INDEX_RE = re.compile(r"^\s*\d+\s*$")
_TAG_RE = re.compile(r"<[^>\n]*>|\{\\[^}\n]*\}")
_BLOCK_START_RE = re.compile(r"^(?:NOTE|STYLE|REGION)(?:\s|$)")
_WS_RE = re.compile(r"\s+")


def _start_seconds(match: "re.Match") -> float:
    hours = int(match.group(1) or 0)
    return hours * 3600 + int(match.group(2)) * 60 + int(match.group(3)) + int(match.group(4).ljust(3, "0")[:3]) / 1000.0


def _stamp(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"[{h}:{m:02d}:{s:02d}]" if h else f"[{m:02d}:{s:02d}]"


def _next_is_timing(lines, idx: int) -> bool:
    """True when the next non-empty line after `idx` is a cue timing line."""
    for nxt in lines[idx + 1:]:
        if nxt.strip():
            return bool(_TIMING_RE.match(nxt))
    return False


def subtitle_to_text(raw: str) -> str:
    """Convert SRT/VTT text to prose paragraphs prefixed with their start time."""
    try:
        text = (raw or "").lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
        paragraphs = []          # [start_seconds | None, [lines]]
        skipping_block = False
        in_cue = False
        cue_start = None
        last_line = None
        saw_timing = False
        lines_in = text.split("\n")
        for idx, line in enumerate(lines_in):
            stripped = line.strip()
            if not stripped:
                skipping_block = False
                in_cue = False
                continue
            if skipping_block:
                continue
            if stripped.startswith("WEBVTT") and not saw_timing and not paragraphs:
                continue
            if _BLOCK_START_RE.match(stripped) and not in_cue:
                skipping_block = True
                continue
            timing = _TIMING_RE.match(line)
            if timing:
                saw_timing = True
                in_cue = True
                cue_start = _start_seconds(timing)
                continue
            if _INDEX_RE.match(stripped) and (
                not in_cue or _next_is_timing(lines_in, idx)
            ):
                continue
            cleaned = _WS_RE.sub(" ", _TAG_RE.sub("", stripped)).strip()
            if not cleaned:
                continue
            if cleaned == last_line:
                continue
            last_line = cleaned
            if not paragraphs or (
                cue_start is not None and paragraphs[-1][0] is not None
                and cue_start - paragraphs[-1][0] >= PARAGRAPH_SECONDS
            ):
                paragraphs.append([cue_start, []])
            elif paragraphs[-1][0] is None and cue_start is not None:
                paragraphs[-1][0] = cue_start
            paragraphs[-1][1].append(cleaned)
        if not saw_timing:
            return "\n".join(l.strip() for l in text.split("\n") if l.strip())
        out = []
        for start, lines in paragraphs:
            body = " ".join(lines)
            out.append(f"{_stamp(start)} {body}" if start is not None else body)
        return "\n\n".join(out)
    except Exception:  # degrades: unparseable subtitle text is returned verbatim
        return raw or ""
