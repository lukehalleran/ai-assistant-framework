"""Measured, deterministic text clipping shared by prompt and tool contexts."""

from typing import Callable


def fit_text_to_tokens(text: str, max_tokens: int, count: Callable[[str], int],
                       head_ratio: float = 0.6) -> str:
    """Retain a measured head/tail excerpt, including its omission marker.

    Tokenizer boundaries need not be monotonic: only measured fitting
    candidates are returned. A tiny budget may omit the marker itself.
    """
    if max_tokens <= 0:
        return ""
    if count(text) <= max_tokens:
        return text

    def candidate(keep: int, marker: bool = True) -> str:
        head = int(keep * head_ratio)
        tail = keep - head
        snip = f"\n… [middle-out snipped {len(text) - keep} chars] …\n" if marker else ""
        return text[:head] + snip + (text[-tail:] if tail else "")

    use_marker = count(candidate(0)) <= max_tokens
    low, high = 0, min(len(text) - 1, max_tokens * 4)
    best = ""
    while low <= high:
        keep = (low + high) // 2
        result = candidate(keep, use_marker)
        if len(result) < len(text) and count(result) <= max_tokens:
            best = result
            low = keep + 1
        else:
            high = keep - 1
    return best
