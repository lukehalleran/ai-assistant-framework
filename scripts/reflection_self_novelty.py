#!/usr/bin/env python3
"""
SelfNoveltyOracle — the THIRD axis of the reflection scorer (GROUNDED-TRUTH × DOMAIN-DISTANCE
× SELF-NOVELTY, R = the user's own life). Answers: "has the user already RECOGNIZED this
mechanism about themselves?" A truthful, valuable cross-domain reflection is only worth
surfacing if it's NEW to them — otherwise it's a restatement of something they already know.

Design (doc §8/§10 + the discovery-arm lesson learned 2026-06-30):
  - NO existing corpus of recognized self-mechanisms exists (the `reflections` ChromaDB
    collection is Daemon's session post-mortems about ITSELF, not the user's self-insight).
    So self-novelty V0 is a GROWABLE MECHANISM REGISTRY: recognized mechanisms accumulate as
    reflections are surfaced/acknowledged. Starts empty -> everything novel -> learns.
  - Two gates: (1) SurfacingHistory cooldown (don't re-surface the same reflection); (2) is
    the candidate's MECHANISM already in the registry?
  - CRITICAL (discovery lesson): the "same mechanism?" check is a CLAIM-LEVEL LLM judgment,
    NOT embedding-proximity. Embedding distance was blind to documented cross-domain
    identities in the discovery arm; the same trap applies here (a paraphrased
    already-recognized mechanism would slip an embedding threshold). The registry is small,
    so an LLM "is this the same underlying mechanism as any of these?" call is cheap + right.

Isolated; persists a small JSON registry. Importable (oracle) + runnable (self-test).
Usage: python scripts/reflection_self_novelty.py   # runs a built-in discrimination self-test
"""
import json
import os
import re
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from memory.surfacing_history import SurfacingHistory

DEFAULT_REGISTRY = os.path.join(_REPO, "data", "synthesis_discovery", "reflection_mechanism_registry.json")

SAME_SYSTEM = (
    "You decide whether a candidate self-insight names the SAME underlying mechanism the person "
    "has ALREADY recognized, or a genuinely NEW one. Judge by the mechanism (the tunable factor / "
    "dynamic), NOT by surface wording or which life-areas it mentions: two insights are the SAME "
    "if a change in the same underlying factor drives both, even if phrased differently or applied "
    "to different areas. They are NEW/DIFFERENT if the driving factor is genuinely distinct."
)
SAME_PROMPT = """Candidate mechanism the person might newly recognize:
  {candidate}

Mechanisms the person has ALREADY recognized about themselves:
{registry}

Is the candidate essentially the SAME underlying mechanism as any one of the already-recognized
ones (just different surface wording or life-areas)?

Respond EXACTLY:
VERDICT: SAME   or   NEW
MATCH: <the already-recognized mechanism it duplicates, or "none">"""


def _novelty_key(mechanism: str, a: str, b: str) -> str:
    pair = "|".join(sorted([a.lower().strip(), b.lower().strip()]))
    mech = re.sub(r"\s+", " ", (mechanism or "").lower().strip())[:80]
    return f"reflect::{pair}::{mech}"


class SelfNoveltyOracle:
    def __init__(self, model_manager, registry_path=DEFAULT_REGISTRY,
                 history_path=None, cooldown_hours=72):
        self.mm = model_manager
        self.registry_path = registry_path
        self.cooldown_hours = cooldown_hours
        self.history = SurfacingHistory(
            history_path or os.path.join(_REPO, "data", "synthesis_discovery", "reflection_surfacing_history.json"))
        self.registry = self._load()

    def _load(self):
        if os.path.exists(self.registry_path):
            try:
                return json.load(open(self.registry_path))
            except Exception:
                return []
        return []

    def _save(self):
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        json.dump(self.registry, open(self.registry_path, "w"), indent=2)

    async def _same_as_registry(self, mechanism):
        from config.app_config import SYNTHESIS_COHERENCE_MODEL
        reg = "\n".join(f"  - {m['mechanism']}" for m in self.registry)
        raw = await self.mm.generate_once(
            SAME_PROMPT.format(candidate=mechanism, registry=reg),
            model_name=SYNTHESIS_COHERENCE_MODEL, system_prompt=SAME_SYSTEM,
            max_tokens=120, temperature=0.0, disable_reasoning=True)
        m = re.search(r"VERDICT:\s*(SAME|NEW)", raw or "", re.IGNORECASE)
        if not m:
            # FAIL CLOSED: an unparseable judge response must not default to "NEW" —
            # fail-open here inflates measured novelty yield in instrument runs and
            # would re-surface an already-recognized mechanism. None => parse failure.
            return None, ""
        verdict = m.group(1).upper()
        mm = re.search(r"MATCH:\s*(.+)", raw or "")
        return verdict == "SAME", (mm.group(1).strip() if mm else "")

    async def score(self, mechanism, entity_a, entity_b):
        """-> {self_novel: bool, reason: str, nearest_known: str}."""
        key = _novelty_key(mechanism, entity_a, entity_b)
        if self.history.was_recently_shown(key, self.cooldown_hours):
            return {"self_novel": False, "reason": "cooldown (recently surfaced)", "nearest_known": ""}
        if not self.registry:
            return {"self_novel": True, "reason": "registry empty — novel by default", "nearest_known": ""}
        same, match = await self._same_as_registry(mechanism)
        if same is None:
            return {"self_novel": False,
                    "reason": "judge parse failure (fail-closed — exclude from measured rates)",
                    "nearest_known": ""}
        if same:
            return {"self_novel": False, "reason": "mechanism already recognized", "nearest_known": match}
        return {"self_novel": True, "reason": "new mechanism", "nearest_known": ""}

    def record_surfaced(self, mechanism, entity_a, entity_b):
        """Call when a reflection IS surfaced: cooldown + add the mechanism to the registry."""
        self.history.record_surfaced(_novelty_key(mechanism, entity_a, entity_b))
        self.registry.append({"mechanism": mechanism, "entities": [entity_a, entity_b]})
        self._save()


# ───────────────────────────── built-in discrimination self-test ─────────────────────────────
async def _selftest():
    import tempfile
    from models.model_manager import ModelManager
    mm = ModelManager()
    tmp = tempfile.mkdtemp(prefix="selfnov_")
    oracle = SelfNoveltyOracle(mm, registry_path=os.path.join(tmp, "reg.json"),
                               history_path=os.path.join(tmp, "hist.json"))
    print("SelfNoveltyOracle discrimination self-test\n")
    # 1. empty registry -> novel by default
    r0 = await oracle.score("when an authority I depend on controls a resource I need, I trade "
                            "my own judgment for security", "advisor", "dad")
    print(f"[empty registry]  self_novel={r0['self_novel']}  ({r0['reason']})")
    # seed it as 'recognized'
    oracle.record_surfaced("I defer to authority figures who control something I rely on", "advisor", "dad")
    # 2. a PARAPHRASE of the recognized mechanism (diff area/wording) -> should be NOT novel
    r1 = await oracle.score("with my dad as a teen and my boss now, the more I depend on someone "
                            "who controls what I need, the more I suppress my own view", "boss", "dad")
    print(f"[paraphrase known] self_novel={r1['self_novel']}  ({r1['reason']}; ~{r1['nearest_known'][:50]})")
    # 3. a GENUINELY different mechanism -> should be novel
    r2 = await oracle.score("I pre-script interactions to eliminate any chance of being caught "
                            "off-guard, which raises the stakes and makes me rehearse more", "shower-rehearsal", "texting")
    print(f"[different mech]   self_novel={r2['self_novel']}  ({r2['reason']})")
    ok = (r0["self_novel"] and not r1["self_novel"] and r2["self_novel"])
    print("\n=> SELF-NOVELTY DISCRIMINATES" if ok else "\n=> FAILED to discriminate — inspect")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_selftest())
