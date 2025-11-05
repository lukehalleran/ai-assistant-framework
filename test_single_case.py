#!/usr/bin/env python3
"""Debug single test case"""

import asyncio
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

from memory.fact_extractor import FactExtractor


async def test():
    extractor = FactExtractor(use_rebel=False, use_regex=False)  # spaCy only

    query = "My cat's name is Flapjack"
    print(f"\nTesting: {query}\n")

    facts = await extractor.extract_facts(
        query=query,
        response="",
        conversation_context=[]
    )

    print(f"\n{'='*80}")
    print(f"Extracted {len(facts)} facts:")
    for fact in facts:
        meta = fact.metadata or {}
        s = meta.get("subject", "?")
        r = meta.get("relation", "?")
        o = meta.get("object", "?")
        m = meta.get("method", "?")
        c = meta.get("confidence", 0)
        print(f"  {s} | {r} | {o} (conf={c:.2f}, method={m})")


if __name__ == "__main__":
    asyncio.run(test())
