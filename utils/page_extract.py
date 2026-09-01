"""Local HTML content extraction for the fetch_url tool.

Layer 1 of the layered URL fetch (2026-08-29): most pages yield their content
to a plain visible-text pass, but JS-rendered SPAs (ChatGPT share links, many
React/Next apps) serve an HTML shell whose real content lives in embedded
JSON payloads. When the visible text is thin, salvage human-readable strings
from those payloads instead of reporting a blank page:

  - <script id="__NEXT_DATA__"> / <script type="application/json"> blocks
  - react-router "turbo-stream" literals: streamController.enqueue("...")
    (the live 2026-08-29 chatgpt.com/share failure — the whole conversation
    is present as long strings in a flat reference array)
  - <script type="application/ld+json"> structured data

The salvage walk keeps document order, dedupes, and filters non-prose strings
(URLs, base64/hash runs, css/js-shaped text). Extraction never raises — any
failure returns whatever lesser layer succeeded.
"""

import json
import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Visible text below this → treat the page as an SPA shell and try JSON salvage.
MIN_VISIBLE_CHARS = 600
# Salvaged strings shorter than this are ids/labels, not prose.
MIN_SALVAGE_STRING_CHARS = 40
# Hard cap on extracted output (callers apply their own max_content_chars too).
MAX_EXTRACT_CHARS = 24000

# String literals pushed through a react-router/remix stream controller.
# The literal is matched as a real JS string (escaped chars consumed) — the
# old lazy ".*?" ended at the first '")' INSIDE the payload (audit F35
# 2026-08-31), truncating the literal and dropping the whole block.
_ENQUEUE_RE = re.compile(r'\.enqueue\(("(?:[^"\\]|\\.)*")\)', re.DOTALL)
_JSON_SCRIPT_TYPES = ("application/json", "application/ld+json")

_URLISH_RE = re.compile(r"^\s*(?:https?://|//|www\.)\S+\s*$", re.IGNORECASE)
# Long unbroken token runs (base64 blobs, hashes, minified identifiers).
_BLOBISH_RE = re.compile(r"^[A-Za-z0-9+/=_\-.$]{60,}$")
_CSSJS_HINT_RE = re.compile(r"[{};]\s*\S")
_WS_RE = re.compile(r"[ \t]{2,}")
_NEWLINES_RE = re.compile(r"\n{3,}")


def _looks_like_prose(s: str) -> bool:
    """Keep strings that read like human text, drop machine artifacts."""
    s = s.strip()
    if len(s) < MIN_SALVAGE_STRING_CHARS:
        return False
    if " " not in s:
        return False
    if _URLISH_RE.match(s) or _BLOBISH_RE.match(s):
        return False
    # css/js-shaped: several structural chars and low space density
    if _CSSJS_HINT_RE.search(s) and s.count(" ") < len(s) / 20:
        return False
    return True


def _walk_json_strings(node, out: List[str], depth: int = 0) -> None:
    if depth > 80:
        return
    if isinstance(node, str):
        if _looks_like_prose(node):
            out.append(node.strip())
    elif isinstance(node, dict):
        for v in node.values():
            _walk_json_strings(v, out, depth + 1)
    elif isinstance(node, list):
        for v in node:
            _walk_json_strings(v, out, depth + 1)


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    kept = []
    for item in items:
        key = item.strip()
        if key and key not in seen:
            seen.add(key)
            kept.append(item)
    return kept


def _salvage_embedded_json(html: str, soup) -> str:
    """Pull prose strings out of embedded JSON payloads, document order."""
    collected: List[str] = []

    for script in soup.find_all("script"):
        stype = (script.get("type") or "").lower()
        sid = (script.get("id") or "").lower()
        body = script.string or script.get_text() or ""
        if not body.strip():
            continue
        if stype in _JSON_SCRIPT_TYPES or sid == "__next_data__":
            try:
                _walk_json_strings(json.loads(body), collected)
            except (ValueError, TypeError):
                continue

    # Stream-controller literals live in inline scripts as JS string literals:
    # json.loads on the quoted literal unescapes it; the result is usually
    # itself a JSON document (turbo-stream flat array or object).
    for match in _ENQUEUE_RE.finditer(html):
        try:
            unescaped = json.loads(match.group(1))
        except (ValueError, TypeError):
            continue
        if not isinstance(unescaped, str) or len(unescaped) < 200:
            continue
        try:
            _walk_json_strings(json.loads(unescaped), collected)
        except (ValueError, TypeError):
            if _looks_like_prose(unescaped):
                collected.append(unescaped.strip())

    return "\n\n".join(_dedupe_keep_order(collected))


def extract_page_text(html: str, url: str = "") -> Tuple[str, str, str]:
    """Extract (title, text, method) from raw HTML.

    method is "visible" | "embedded_json" | "none" — surfaced in logs so a
    salvage-path regression is visible in telemetry, not just output quality.
    """
    if not html or not html.strip():
        return "", "", "none"
    try:
        from bs4 import BeautifulSoup  # lazy import: startup cost
        soup = BeautifulSoup(html, "html.parser")
    except Exception as e:
        logger.warning(f"[PageExtract] HTML parse failed for {url}: {e}")
        return "", "", "none"

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    working = BeautifulSoup(str(soup), "html.parser")
    for tag in working(["script", "style", "noscript", "template", "svg", "head"]):
        tag.decompose()
    visible = working.get_text("\n", strip=True)
    visible = _NEWLINES_RE.sub("\n\n", _WS_RE.sub(" ", visible))

    if len(visible) >= MIN_VISIBLE_CHARS:
        return title, visible[:MAX_EXTRACT_CHARS], "visible"

    salvaged = _salvage_embedded_json(html, soup)
    if len(salvaged) > len(visible):
        logger.info(
            f"[PageExtract] SPA shell ({len(visible)} visible chars) — "
            f"salvaged {len(salvaged)} chars from embedded JSON for {url}"
        )
        return title, salvaged[:MAX_EXTRACT_CHARS], "embedded_json"
    if visible:
        return title, visible[:MAX_EXTRACT_CHARS], "visible"
    return title, "", "none"
