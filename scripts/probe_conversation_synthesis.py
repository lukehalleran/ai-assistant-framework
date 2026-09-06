#!/usr/bin/env python3
"""Read-only synthetic probe of deployed insight synthesis; no Daemon stores.

Default prints generated prompts. --live sends three bounded calls to the
selected model and prints the responses for human review. Results are not
stored in memory. Run from the repo: python -m scripts.probe_conversation_synthesis
"""

import argparse
import asyncio
import json

from core.insight.provenance import label_evidence
from core.insight.synthesizer import (
    build_synthesis_prompts, recent_conversation_context, synthesize_stream,
)
from core.insight.types import EvidenceItem, InsightIntent


async def run(args):
    history = recent_conversation_context([
        ("I took my prescribed stimulant at 10 AM yesterday, then went out in the evening.",
         "I was picturing a late-night dose."),
        ("10 AM, to clarify. Today I am resting and feel good, even without being productive.",
         "I had the timing wrong. It is useful that rest feels good today."),
    ])
    evidence = label_evidence([
        EvidenceItem(doc_id="observation", collection="corpus", speaker="user", date="2026-08-10",
                     text="Several days without my medication were hard for me."),
        EvidenceItem(doc_id="interpretation", collection="summaries", date="2026-08-10",
                     text="The assistant suggested that resting had reset the user's dopamine."),
    ])
    research = EvidenceItem(
        doc_id="NICE-NG87", collection="research", stance_label="external-research",
        date="2018-03-14",
        text=("NICE NG87 recommendations 1.10.1–3 support individualized review of benefits, "
              "adverse effects and the effects of missed doses or periods without treatment, "
              "and consideration of trial stopping or dose reduction when appropriate. "
              "The rationale notes limited evidence of possible worsening ADHD symptoms "
              "but reduced adverse effects after withdrawal. This does not establish a "
              "one-day-per-month regimen for an adult. "
              "https://www.nice.org.uk/guidance/ng87/chapter/recommendations#review-of-medication-and-discontinuation "
              "https://www.nice.org.uk/guidance/ng87/chapter/Rationale-and-impact#review-of-medication-and-discontinuation"),
    )
    cases = [
        ("failed_analysis", "Does my history support scheduling occasional rest days?", evidence, False),
        ("balanced_decision", "Could occasional stimulant breaks make sense for rest, given my history? "
         "Weigh the evidence on both sides.", evidence + [research], False),
        ("requested_report", "Give me a detailed analysis in a table of what my record can establish "
         "about rest days and medication gaps.", evidence, True),
    ]
    manager = None
    if args.live:
        # dotenv reads credentials without logging their contents. ModelManager
        # is the deployed provider path, not a separate API implementation.
        from dotenv import load_dotenv
        from models.model_manager import ModelManager
        load_dotenv()
        manager = ModelManager()
        if manager.async_client is None:
            raise RuntimeError("No configured provider client; live probe unavailable")
        # A bare ModelManager has no active model; generate_async ignores the
        # per-call model_name and routes on active_model_name.
        if not manager.is_api_model(args.model):
            raise RuntimeError(f"{args.model!r} is not a registered API model")
        manager.switch_model(args.model)
    try:
        for name, query, items, report in cases:
            intent = InsightIntent(kind="pattern_temporal", theme="rest days and medication gaps",
                                   raw_query=query, wants_document=report)
            manifest = {"status": "insufficient", "limitations": ["phase specification failed"],
                        "channels": [{"channel": "pattern", "status": "insufficient",
                                      "attempted": False, "count": 0}]}
            if args.live:
                async def collect():
                    return "".join([part async for part in synthesize_stream(
                        intent, items, None, model_manager=manager, model_name=args.model,
                        conversation_context=history, deliberation_manifest=manifest,
                        disable_reasoning=True,
                    )])
                reply = await asyncio.wait_for(collect(), timeout=55)
                print(json.dumps({"case": name, "response": reply}), flush=True)
            else:
                system, prompt = build_synthesis_prompts(
                    intent, items, None, conversation_context=history, deliberation_manifest=manifest,
                )
                print(json.dumps({"case": name, "system": system, "prompt": prompt}), flush=True)
    finally:
        if manager is not None:
            if manager.async_client is not None:
                await manager.async_client.close()
            if manager.client is not None:
                manager.client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default="kimi-3")
    asyncio.run(run(parser.parse_args()))
