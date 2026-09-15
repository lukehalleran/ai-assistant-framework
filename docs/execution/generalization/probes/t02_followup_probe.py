import asyncio, hashlib, re, sys
from datetime import datetime
sys.path.insert(0, "/home/lukeh/daemon_exec/generalization")
import utils.tone_detector as td
from utils.tone_detector import CrisisLevel
from utils.trigger_match import is_negated
from core.escalation_tracker import EscalationTracker
from core.context_pipeline import ToneLevel

print("tone_detector sha256:", hashlib.sha256(open(td.__file__, "rb").read()).hexdigest()[:16])
FP_ANY = td._HISTORY_FIRST_PERSON_RE
SUBJ_A = re.compile(r"\b(?:i|i'm|im|i've|ive|i'd|we)\b")
SUBJ_B = re.compile(r"\b(?:i|i'm|im|i've|ive|i'd)\b")
WIN, BREAK, STRIP = 3, re.compile(r"[.!?\n]"), ",;:\"'()"

def _prefix(low, pos):
    ends = [m.end() for m in BREAK.finditer(low, 0, pos)]
    return low[(ends[-1] if ends else 0):pos]

def _qualify(low, hit, subj):
    toks = _prefix(low, hit.start).split()
    rule = "a" if FP_ANY.search(hit.keyword) else ("b" if any(subj.fullmatch(t.strip(STRIP)) for t in toks[-WIN:]) else None)
    if not rule:
        return None, False
    last = max((i for i, t in enumerate(toks) if FP_ANY.fullmatch(t.strip(STRIP))), default=None)
    span = " ".join((toks[last + 1:] if last is not None else toks)[-WIN:])
    return rule, is_negated(span, len(span), window_tokens=WIN)

def classify(msg):
    """T02 candidate tier: 'A', 'B_mild', 'B_negated_only', or None."""
    low = msg.lower()
    if not td._check_observational_language(msg):
        return None
    neg = False
    for hit in td._HIGH_MATCHER.iter_hits(low):
        rule, n = _qualify(low, hit, SUBJ_A)
        if rule and not n:
            return "A"
        neg = neg or bool(rule and n)
    mild = list(td._MEDIUM_MATCHER.iter_hits(low)) + list(td._CONCERN_MATCHER.iter_hits(low))
    if not td._OVERWHELMED_POSITIVE_MATCHER(low):
        mild += list(td._OVERWHELMED_MATCHER.iter_hits(low))
    if any(r and not n for r, n in (_qualify(low, h, SUBJ_B) for h in mild)):
        return "B_mild"
    return "B_negated_only" if neg else None

NEG_NEWS = "people are suffering but I don't want to die"
MILD_NEWS = "people are suffering and I feel hopeless"
FOLLOWUPS = [
    "the weather was grey this morning",
    "I watched a documentary about whales last night",
    "my sister is visiting next week",
]

def row(q):
    return {"query": q, "is_heavy_topic": True, "timestamp": datetime.now().isoformat()}

async def turn2(msg, previous_tone, history):
    r = await td.detect_crisis_level(msg, conversation_history=history, model_manager=None, previous_tone=previous_tone)
    return f"{r.level.name}/{r.trigger}"

async def main():
    real_row_check = td._heavy_row_is_distress_evidence

    def designed_row_check(turn):
        text = turn.get("query") or turn.get("user") or turn.get("content") or ""
        if classify(str(text)) == "B_negated_only":
            return False
        return real_row_check(turn)

    for t1 in (NEG_NEWS, MILD_NEWS):
        kind = classify(t1)
        print(f"\n== turn 1: {t1!r} -> T02 candidate tier {kind} (CONCERN); Stage 0 fires: {td._check_observational_language(t1)}")
        for f in FOLLOWUPS:
            td._heavy_row_is_distress_evidence = real_row_check
            today_prev = await turn2(f, CrisisLevel.CONCERN, [])
            today_hist = await turn2(f, None, [row(t1)])
            if kind == "B_negated_only":
                td._heavy_row_is_distress_evidence = designed_row_check
                designed = await turn2(f, None, [row(t1)])   # pipeline holds: no previous CONCERN; row re-checked
            else:
                designed = await turn2(f, CrisisLevel.CONCERN, [row(t1)])  # other triggers keep carry-over
            td._heavy_row_is_distress_evidence = real_row_check
            print(f"  follow-up {f!r}\n     carry via previous_tone (today's rule): {today_prev}\n     carry via heavy history row (today's rule): {today_hist}\n     DESIGNED: {designed}")

    print("\n== hold semantics: genuine CONCERN at N-1, negated-news at N (held), neutral at N+1")
    td._heavy_row_is_distress_evidence = designed_row_check
    held = await turn2(FOLLOWUPS[0], CrisisLevel.CONCERN, [row(MILD_NEWS), row(NEG_NEWS)])
    td._heavy_row_is_distress_evidence = real_row_check
    print(f"  N+1 with N-1 carry preserved: {held}")

    print("\n== EscalationTracker today (real class): consecutive_distress_count after one CONCERN turn")
    for trig in ("harm_score: 4.0 (0H, 0M, 2C)", "distress_sticky_floor", "observational_negated_crisis"):
        t = EscalationTracker()
        t.update(ToneLevel.CONCERN, NEG_NEWS, tone_trigger=trig)
        print(f"  trigger={trig!r:36} -> count={t.consecutive_distress_count}")

asyncio.run(main())
