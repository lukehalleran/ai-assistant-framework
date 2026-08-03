"""
Calibration probe for the tone backstop floor + exemplar coverage.

Scores a labeled probe set through THE deployed semantic detector
(utils.tone_detector._semantic_crisis_detection — never a re-derivation) and
reports, per probe: the top distress score, the conversational score, and
whether the deterministic backstop (margin 0.08 + floor) would fire at a given
floor value.

Probe groups:
  MISS      — real live misses (2026-07-25 / 2026-08-02 logs) that MUST clear
              the backstop (distress-top with margin, score above floor)
  WEAK      — media/situational sadness that MUST NOT clear it
  GAMING    — strategy-game violence talk (HOI4) that MUST NOT clear it
  CASUAL    — plain conversation, sanity floor

Usage:
    python scripts/probe_tone_backstop.py [--floor 0.37]
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROBES = {
    "MISS": [
        # 2026-08-02: semantic high=0.298, arbiter empty → CONVERSATIONAL
        "Not everyone deserves to be alive like me. A stranger just screamed "
        "insults at me for a car and I would hurt them terribly if legal",
        # 2026-08-02: concern=0.349, arbiter literal-rubric → CONVERSATIONAL
        "I am embarrassed for how I reacted earlier. My dad came by for a bit. "
        "I am so unhappy",
        # 2026-08-02 repair turn
        "No I mean a car driving by screamed random insults. I deserve to be alive",
        # 2026-07-25: medium=0.390 — the original backstop case
        "I am back now. I keep thinking I am a stupid piece of shit. "
        "I know it's the meds but I wanna cry",
    ],
    "WEAK": [
        "That movie made me sad",
        "This song always makes me cry a little",
        "The ending of that show was so depressing honestly",
        "Kind of sad my team lost again today",
    ],
    "GAMING": [
        "I killed everyone in my last run lol",
        "Gonna destroy the Allies as Germany tonight in kaiserredux",
        "I wiped out their whole army it was brutal",
        "Time to crush France again, no mercy this time",
    ],
    "CASUAL": [
        "I'm a bit tired today",
        "Just got back from the gym, feeling alright",
        "What's up, anything new?",
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--floor", type=float, default=0.37,
                    help="Backstop floor to evaluate (default: current 0.37)")
    args = ap.parse_args()

    from utils.tone_detector import _semantic_crisis_detection

    print(f"{'group':<8} {'top-distress':>12} {'conv':>6} {'margin':>7} "
          f"{'fires@' + format(args.floor, '.2f'):>10}  probe")
    rows = []
    for group, probes in PROBES.items():
        for p in probes:
            level, conf, raw = _semantic_crisis_detection(p, None, None)
            conv = raw.get("conversational", 0.0)
            top = max(raw.get(k, 0.0) for k in ("high", "medium", "concern"))
            top_name = max(
                ("high", "medium", "concern"), key=lambda k: raw.get(k, 0.0)
            )
            fires = top > conv + 0.08 and top >= args.floor
            rows.append((group, top, conv, fires))
            print(f"{group:<8} {top:>8.3f} ({top_name[:4]}) {conv:>6.3f} "
                  f"{top - conv:>7.3f} {str(fires):>10}  {p[:60]}")

    miss_scores = [t for g, t, c, f in rows if g == "MISS"]
    other_fires = [(g, t) for g, t, c, f in rows if g != "MISS" and f]
    print(f"\nMISS top-distress range: {min(miss_scores):.3f} – {max(miss_scores):.3f}")
    print(f"Non-MISS probes that fire at floor {args.floor}: {other_fires or 'none'}")
    miss_missed = [t for g, t, c, f in rows if g == "MISS" and not f]
    if miss_missed:
        print(f"MISS probes NOT caught at this floor: {len(miss_missed)} "
              f"(scores: {[f'{t:.3f}' for t in miss_missed]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
