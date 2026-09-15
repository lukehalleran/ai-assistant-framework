"""Parent probe (2026-09-13, T03 review): the owner's requested evidence that a
neutral follow-up after a negated crisis phrase with news framing stays
conversational — now through the REAL, implemented ContextPipeline._detect_tone
(T01+T02+T03 candidate), not a simulation. model_manager=None; tone state is
written to a throwaway temp directory, never data/."""
import asyncio, hashlib, json, sys, tempfile
from datetime import datetime
from pathlib import Path
sys.path.insert(0, "/home/lukeh/daemon_exec/generalization")
from core.context_pipeline import ContextPipeline
from core.escalation_tracker import EscalationTracker
from core.context_pipeline import ToneLevel
from utils.tone_detector import OBSERVATIONAL_NEGATED_CRISIS_TRIGGER

ROOT = Path("/home/lukeh/daemon_exec/generalization")
for f in ("core/context_pipeline.py", "core/escalation_tracker.py", "utils/tone_detector.py"):
    print(f"{hashlib.sha256((ROOT / f).read_bytes()).hexdigest()[:16]}  {f}")

NEG = "people are suffering but I don't want to die"
MILD = "people are suffering and I feel hopeless"
FOLLOWUPS = ["the weather was grey this morning",
             "I watched a documentary about whales last night",
             "my sister is visiting next week"]

def pipe(tmp):
    p = object.__new__(ContextPipeline)
    p._TONE_STATE_PATH = str(Path(tmp) / "tone_state.json")
    p._last_tone_level, p._floor_chain, p.memory_system, p.model_manager = None, 0, None, None
    return p

def row(q, heavy):
    return {"query": q, "is_heavy_topic": heavy, "timestamp": datetime.now().isoformat()}

async def turn(p, msg, hist):
    level, ctx = await p._detect_tone(msg, hist)
    return f"{level.name}/{getattr(ctx, 'tone_trigger', '')}"

async def main():
    with tempfile.TemporaryDirectory() as tmp:
        print("\n| Turn 1 | Turn 1 result | Follow-up | history row for turn 1 | Follow-up result |")
        print("|---|---|---|---|---|")
        for t1 in (NEG, MILD):
            for f in FOLLOWUPS:
                for heavy in (False, True):
                    p = pipe(Path(tmp) / f"{hash((t1, f, heavy)) & 0xffff}")
                    Path(p._TONE_STATE_PATH).parent.mkdir(parents=True, exist_ok=True)
                    r1 = await turn(p, t1, None)
                    r2 = await turn(p, f, [row(t1, heavy)])
                    print(f"| {t1} | {r1} | {f} | {'heavy' if heavy else 'not heavy'} | {r2} |")
        p = pipe(Path(tmp) / "hold"); Path(p._TONE_STATE_PATH).parent.mkdir(parents=True, exist_ok=True)
        a = await turn(p, MILD, None)
        b = await turn(p, NEG, [row(MILD, False)])
        c = await turn(p, FOLLOWUPS[0], [row(NEG, True), row(MILD, False)])
        print(f"\nHold row: N-1 {MILD!r} -> {a}; N {NEG!r} -> {b}; N+1 {FOLLOWUPS[0]!r} -> {c}")
    t = EscalationTracker()
    t.update(ToneLevel.CONCERN, MILD, tone_trigger="observational_first_person_distress")
    t.update(ToneLevel.CONCERN, NEG, tone_trigger=OBSERVATIONAL_NEGATED_CRISIS_TRIGGER)
    print(f"Tracker: organic-mild then negated-news -> consecutive_distress_count={t.consecutive_distress_count} (held at 1)")

asyncio.run(main())
