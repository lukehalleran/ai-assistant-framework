"""
# core/citation_drift.py

Module Contract
- Purpose: DETERMINISTIC, LOG-ONLY citation-drift check (2026-10-12, X2).
  A sentence that cites [WEB_n] should draw its key numbers from the cited
  source(s). Live miss: "21 US troops have reportedly died" was cited to a
  Reuters IMF piece; the number came from other sources and "US" was added.
- Inputs: response text + {WEB_n: source_text}.
- Outputs: list of drift items (dicts). Never raises, never edits the response,
  no LLM, no I/O. Callers record the items in the turn's telemetry / debug
  record; nothing user-visible changes.
- Rule: a drift item is recorded for a cited sentence when a number in it is
  absent from EVERY cited source. Unsupported TitleCase entities ride along on
  recorded items (context) but never create one on their own (too noisy).
"""
import calendar
import re
from typing import Dict, List

from core.citation_extractor import extract_web_citation_ids

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+|\n+')
_MARKER_RE = re.compile(r'\[[^\]\n]{1,60}\]')
_NUM_RE = re.compile(r'(?<![\w.])\$?(\d[\d,]*(?:\.\d+)?)')
_ENTITY_RE = re.compile(r"\b([A-Z][a-zA-Z]{1,}|[A-Z]{2,6})\b")
_BLOCK_HEAD = re.compile(r'(?m)^\[(WEB_\d+)\]\s')

# Categorical (not topical) capitalized words that are never entities.
_CALENDAR_WORDS = {w.lower() for w in list(calendar.month_name) + list(calendar.day_name) if w}


def normalize_number(raw: str) -> str:
    """'1,200' -> '1200', '2.60' -> '2.6', '5.0' -> '5'."""
    s = raw.replace(",", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def numbers_in(text: str) -> List[str]:
    return [normalize_number(m) for m in _NUM_RE.findall(text or "")]


def parse_web_blocks(text: str) -> Dict[str, str]:
    """Split rendered '[WEB_n] **title** (url)\\ncontent' blocks into {id: text}."""
    out: Dict[str, str] = {}
    if not text:
        return out
    heads = list(_BLOCK_HEAD.finditer(text))
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        out[m.group(1)] = out.get(m.group(1), "") + text[m.start():end]
    return out


def _cited_sentences(response: str):
    """Yield (sentence_without_markers, [WEB ids]); marker-only sentences
    (text. [WEB_5]) attach to the previous sentence."""
    pieces = [p for p in _SENT_SPLIT.split(response or "") if p and p.strip()]
    merged: List[str] = []
    for p in pieces:
        if not _MARKER_RE.sub("", p).strip(" .,;:-*") and merged:
            merged[-1] += " " + p
        else:
            merged.append(p)
    for sent in merged:
        ids = extract_web_citation_ids(sent)
        if ids:
            yield _MARKER_RE.sub(" ", sent), ids


def _entities(sentence: str) -> List[str]:
    found = []
    for m in _ENTITY_RE.finditer(sentence):
        if m.start() == 0 or not sentence[:m.start()].strip(" *-•#>\"'("):
            continue  # sentence-initial capital carries no entity signal
        if m.group(1).lower() in _CALENDAR_WORDS:
            continue
        if m.group(1) not in found:
            found.append(m.group(1))
    return found


def find_citation_drift(response: str, sources: Dict[str, str]) -> List[dict]:
    """Return one item per cited sentence whose numbers are missing from every
    cited source. Citations to ids absent from ``sources`` are skipped (nothing
    to compare against); a sentence whose cited sources are ALL unknown is not
    flagged."""
    items: List[dict] = []
    try:
        for sent, ids in _cited_sentences(response):
            known = [i for i in ids if sources.get(i)]
            if not known:
                continue
            nums = list(dict.fromkeys(numbers_in(sent)))
            if not nums:
                continue
            src_text = "\n".join(sources[i] for i in known)
            src_nums = set(numbers_in(src_text))
            missing = [n for n in nums if n not in src_nums]
            if not missing:
                continue
            src_words = {w.lower() for w in re.findall(r"[A-Za-z]+", src_text)}
            ents = [e for e in _entities(sent) if e.lower() not in src_words]
            items.append({
                "sentence": sent.strip()[:240],
                "cited": ids,
                "missing_numbers": missing,
                "missing_entities": ents,
            })
    except Exception:  # degrades: drift receipt lost for this turn (log-only check)
        return items
    return items
