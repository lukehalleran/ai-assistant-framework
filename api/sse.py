"""SSE wire formatting for the /api/chat stream.

format_sse() turns a ChatEvent into a spec-compliant `event:`/`data:` block;
KEEPALIVE is a comment frame proxies can't buffer into a stall.
"""

import json

from api.schemas import ChatEvent

KEEPALIVE = ": keepalive\n\n"

# Response headers that keep intermediaries from buffering the stream.
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def format_sse(event: ChatEvent) -> str:
    """Serialize a ChatEvent as one SSE message (single-line JSON payload)."""
    payload = json.dumps(event.data, ensure_ascii=False, default=str)
    return f"event: {event.event}\ndata: {payload}\n\n"
