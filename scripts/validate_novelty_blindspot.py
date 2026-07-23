#!/usr/bin/env python3
"""
Validate the discovery world-novelty BLIND SPOT end-to-end, and measure it as a number.

The per-term probe (probe_top_discovery_novelty.py) showed the wiki oracle's only "known"
signals are embedding-proximity (cos(A,B)>0.45 + verbatim-claim FAISS>0.88) — structurally
blind to DOCUMENTED but embedding-distant cross-domain identities. This script confirms that
against a REAL literature corpus (arXiv, no auth): for each candidate the wiki oracle scored
as novel/accept, it searches arXiv for the SPECIFIC connection and an LLM judges, FROM THE
RETRIEVED PAPERS, whether the connection is already established/named in the literature.

Output: per-candidate {wiki_oracle verdict, arXiv-lit verdict (DOCUMENTED/NOVEL), name} and
the ORACLE FALSE-NOVELTY RATE = fraction of wiki-"novel" ACCEPTs the literature finds
documented. A high rate on the top accepts == the blind spot, quantified. (This arXiv->LLM
check is also a prototype of the claim-level novelty oracle that should replace cos(A,B).)

Reads a grading-batch CSV. ~1 arXiv call (free) + 1 LLM call per candidate.
Usage: python scripts/validate_novelty_blindspot.py <batch.csv> [--top 5 --bottom 5]
"""
import argparse
import asyncio
import csv
import os
import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

JUDGE_SYSTEM = (
    "You assess whether a proposed cross-domain connection is ALREADY ESTABLISHED in the "
    "scientific literature. Judge primarily from the retrieved papers (does the literature "
    "already treat this specific correspondence as known?), and you may note when it is a "
    "named theorem or textbook-standard identity. Be strict and honest: DOCUMENTED if the "
    "SPECIFIC identity/correspondence is a known/named/standard result; NOVEL only if the "
    "specific connection is genuinely not established."
)
JUDGE_PROMPT = """Candidate cross-domain "discovery":
  A: {a}
  B: {b}
  CLAIM: {claim}

Retrieved arXiv papers (titles + abstracts) for these concepts:
{papers}

Is the SPECIFIC structural connection asserted in the CLAIM an established, documented result
(a named theorem, a standard textbook identity, or a well-known correspondence) rather than a
genuinely novel/non-obvious connection?

Respond EXACTLY:
RATING: DOCUMENTED   or   NOVEL   or   UNCLEAR
NAME: <the established name/theorem, or "none">
EVIDENCE: <one line: which retrieved paper or standard reference establishes it>"""


async def arxiv_search(a, b):
    import httpx
    q = f"{a} {b}"
    url = (f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(q)}"
           f"&start=0&max_results=5&sortBy=relevance&sortOrder=descending")
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(url)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)
        out = []
        for e in root.findall("atom:entry", ns)[:5]:
            t = (e.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
            s = (e.findtext("atom:summary", "", ns) or "").strip().replace("\n", " ")[:280]
            out.append(f"- {t}\n  {s}")
        return "\n".join(out) if out else "(no arXiv results)"
    except Exception as ex:
        return f"(arXiv search failed: {ex})"


def _field(raw, key, default=""):
    m = re.search(rf"{key}:\s*(.+)", raw or "", re.IGNORECASE)
    return m.group(1).strip() if m else default


async def assess(mm, sem, row):
    from config.app_config import SYNTHESIS_COHERENCE_MODEL
    a, b, claim = row["concept_a"], row["concept_b"], row["connection_claim"]
    papers = await arxiv_search(a, b)
    async with sem:
        raw = await mm.generate_once(
            JUDGE_PROMPT.format(a=a, b=b, claim=claim, papers=papers[:2200]),
            model_name=SYNTHESIS_COHERENCE_MODEL, system_prompt=JUDGE_SYSTEM,
            max_tokens=200, temperature=0.0, disable_reasoning=True)
    rating = _field(raw, "RATING").upper()
    rating = "DOCUMENTED" if "DOCUMENTED" in rating else ("NOVEL" if "NOVEL" in rating else "UNCLEAR")
    return {"a": a, "b": b, "wiki_status": row["status"], "composite": row["composite"],
            "lit_rating": rating, "name": _field(raw, "NAME"), "evidence": _field(raw, "EVIDENCE")}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--bottom", type=int, default=5)
    args = ap.parse_args()
    path = args.csv if os.path.isabs(args.csv) else os.path.join(_REPO, "data", "synthesis_discovery", args.csv)
    rows = sorted(csv.DictReader(open(path)), key=lambda r: -float(r["composite"]))
    pick = rows[: args.top] + rows[-args.bottom:]

    from models.model_manager import ModelManager
    mm = ModelManager()
    sem = asyncio.Semaphore(4)
    print(f"Literature (arXiv) documented-ness check on {len(pick)} candidates "
          f"(top {args.top} + bottom {args.bottom})...\n", flush=True)
    res = await asyncio.gather(*(assess(mm, sem, r) for r in pick))

    print(f"{'pair':<46}{'wiki':>7}{'comp':>6}  {'LITERATURE':<11} name")
    print("-" * 100)
    for r in res:
        print(f"{r['a'][:22]+' <-> '+r['b'][:18]:<46}{r['wiki_status']:>7}{float(r['composite']):>6.3f}  "
              f"{r['lit_rating']:<11} {r['name'][:34]}")
    print("-" * 100)
    accepts = [r for r in res if r["wiki_status"] == "ACCEPT"]
    doc_acc = [r for r in accepts if r["lit_rating"] == "DOCUMENTED"]
    fp = len(doc_acc) / len(accepts) if accepts else 0.0
    print(f"\nORACLE FALSE-NOVELTY among wiki-ACCEPTs: {len(doc_acc)}/{len(accepts)} = {fp:.0%} "
          f"documented in arXiv (these are the connections the oracle surfaced as 'novel').")
    qd = [r for r in res if r["wiki_status"] != "ACCEPT"]
    if qd:
        qdoc = sum(1 for r in qd if r["lit_rating"] == "DOCUMENTED")
        print(f"Queued (below threshold): {qdoc}/{len(qd)} documented — for contrast.")
    print("\nREAD: a high accept-FP confirms the wiki oracle (cos(A,B) + verbatim-claim FAISS) is")
    print("      blind to documented cross-domain identities — the corpus/claim-level fix is the repair.")


if __name__ == "__main__":
    asyncio.run(main())
