#!/usr/bin/env python3
"""Probe the DECLARED request-shape constraints against the live OpenRouter routes.

The public model catalog cannot answer these questions: on 2026-09-21 it listed
`tool_choice` and `reasoning` as supported for anthropic/claude-fable-5.1 while
the route answered HTTP 400 to a named tool_choice and to
`reasoning: {"enabled": false}`. Only a request shows it. This script sends, per
registered model, the two request shapes Daemon actually uses and compares the
outcome with the model's MODEL_CAPABILITIES row:

  off-switch   reasoning={"enabled": false}, max_tokens=16
               row reasoning_mandatory=True  -> the route must REFUSE (HTTP 400)
               otherwise                     -> the route must ACCEPT (HTTP 200)
  forced tool  tool_choice={"type":"function","function":{"name":...}}
               row forced_tool_choice=False  -> the route must REFUSE
               otherwise                     -> the route must ACCEPT

and, for a model whose row declares a constraint, the shape Daemon sends instead
(THE deployed reasoning_request_config / resolve_tool_choice output) must be
accepted. Exit 1 on any disagreement: a row is missing a constraint (the next
disable_reasoning call will 400) or carries a stale one.

THIS SPENDS CREDITS (a few tokens per request; two or three requests per model).
Default is a dry run that prints the plan. `--run` sends the requests.

    python scripts/probe_model_request_shapes.py                       # plan only
    python scripts/probe_model_request_shapes.py --run --models gpt-6-astra fable-5.1
    python scripts/probe_model_request_shapes.py --run --all

Run it when a model is added, and again when a provider changes a route.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.model_manager import (  # noqa: E402
    API_MODEL_ALIASES,
    MODEL_CAPABILITIES,
    reasoning_request_config,
    resolve_tool_choice,
)

URL = "https://openrouter.ai/api/v1/chat/completions"
TOOL_NAME = "propose_action"
TOOLS = [{
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Propose an action.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
}]
FORCED = {"type": "function", "function": {"name": TOOL_NAME}}
PING = [{"role": "user", "content": "Reply with the single word: pong"}]
NOTE = [{"role": "user", "content": "Save a note that says: buy milk."}]


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        try:
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).resolve().parent.parent / ".env")
            key = os.environ.get("OPENROUTER_API_KEY", "")
        except ImportError:
            pass
    return key


def _post(key: str, body: dict):
    request = urllib.request.Request(
        URL, data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, {"error_body": error.read().decode("utf-8", "replace")[:300]}


def _requests_for(slug: str) -> list:
    """(label, body, must_be_accepted) for one model, from its declared row."""
    row = MODEL_CAPABILITIES[slug]
    plan = []
    if row.get("reasoning"):
        mandatory = bool(row.get("reasoning_mandatory"))
        plan.append(("off-switch enabled=false",
                     {"messages": PING, "max_tokens": 16, "reasoning": {"enabled": False}},
                     not mandatory))
        if mandatory:
            plan.append(("deployed off-switch substitute",
                         {"messages": PING, "max_tokens": 16,
                          "reasoning": reasoning_request_config(slug, disable_reasoning=True)},
                         True))
    if row.get("tools"):
        refuses = row.get("forced_tool_choice") is False
        plan.append(("forced tool_choice",
                     {"messages": NOTE, "max_tokens": 200, "tools": TOOLS, "tool_choice": FORCED},
                     not refuses))
        if refuses:
            plan.append(("deployed forced-round substitute",
                         {"messages": NOTE, "max_tokens": 200, "tools": TOOLS,
                          "tool_choice": resolve_tool_choice(slug, FORCED)},
                         True))
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", action="store_true", help="send the requests (spends credits)")
    parser.add_argument("--all", action="store_true", help="every registered model")
    parser.add_argument("--models", nargs="*", default=[], help="aliases or full slugs")
    args = parser.parse_args()

    if args.all:
        slugs = sorted(set(API_MODEL_ALIASES.values()))
    else:
        slugs = []
        for name in args.models:
            slug = API_MODEL_ALIASES.get(name, name)
            if slug not in MODEL_CAPABILITIES:
                print(f"ERROR: {name!r} is not a registered model")
                return 2
            slugs.append(slug)
    if not slugs:
        print("No models selected. Use --models <alias...> or --all.")
        return 2

    plans = {slug: _requests_for(slug) for slug in slugs}
    total = sum(len(plan) for plan in plans.values())
    if not args.run:
        print(f"DRY RUN — {total} request(s) across {len(slugs)} model(s); --run sends them.")
        for slug, plan in plans.items():
            for label, _body, accepted in plan:
                print(f"  {slug:<36} {label:<34} expect {'ACCEPT' if accepted else 'REFUSE'}")
        return 0

    key = _api_key()
    if not key:
        print("ERROR: OPENROUTER_API_KEY is not set")
        return 2

    disagreements = []
    for slug, plan in plans.items():
        for label, body, must_accept in plan:
            status, data = _post(key, {**body, "model": slug})
            if status == 402:
                print(f"  {slug:<36} {label:<34} HTTP 402 — out of credits; stopping")
                return 2
            accepted = status == 200 and "choices" in data
            verdict = "ok" if accepted == must_accept else "DISAGREES WITH ROW"
            detail = "" if accepted else " " + json.dumps(data)[:160]
            print(f"  {slug:<36} {label:<34} HTTP {status} {verdict}{detail}")
            if accepted != must_accept:
                disagreements.append((slug, label, status))

    if disagreements:
        print(f"\n{len(disagreements)} row(s) disagree with the live route:")
        for slug, label, status in disagreements:
            print(f"    {slug}: {label} -> HTTP {status}")
        return 1
    print("\nEvery probed row matches its live route.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
