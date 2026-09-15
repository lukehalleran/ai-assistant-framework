"""H02 design probe, round 2 (parent). MUST run via probes/run_sandboxed_probe.py with
TONE_EXEMPLAR_LEARNING=0 (empty temp adaptive store; nothing read from or written to data/).
Variants over the same stress + control sets, all on the REAL detect_crisis_level(model_manager=None):
  baseline : today's code (sandboxed, fresh-user state)
  A2       : A-prime (domain anchor x first-person-experiencer strain, T02 qualifier) plus
             object-bearing strain shapes ("evict us", "kick me out"); floors today's level at CONCERN
  B        : domain-spanning CONCERN seeds appended to CRISIS_EXEMPLARS["concern"] (phrasing deliberately
             different from the probe messages), prototype cache reset
  A2+B     : both
"""
import asyncio, hashlib, re, sys
sys.path.insert(0, "/home/lukeh/daemon_exec/generalization")
import utils.tone_detector as td
from utils.trigger_match import compile_keyword_matcher
print("tone_detector sha256:", hashlib.sha256(open(td.__file__, "rb").read()).hexdigest()[:16])

DOMAIN_ANCHORS = {
 "family": ["kids", "kid", "child", "children", "son", "daughter", "mom", "mother", "dad", "father",
            "parents", "partner", "husband", "wife", "spouse", "family", "sister", "brother", "baby"],
 "caregiving": ["caring for", "taking care of", "care of", "caregiver", "looking after"],
 "health": ["pain", "diagnosis", "biopsy", "results", "symptoms", "migraines", "treatment", "surgery",
            "illness", "flare", "flared", "doctor", "hospital"],
 "money": ["rent", "bills", "payroll", "savings", "cash flow", "paycheck", "debt", "money", "mortgage", "loan"],
 "housing": ["landlord", "evict", "eviction", "apartment", "house", "lease", "homeless", "home"],
 "work": ["boss", "job", "laid off", "layoff", "fired", "shifts", "clients", "business", "manager", "novel", "project"],
 "school": ["classes", "class", "exam", "thesis", "grades", "professor", "defense", "course", "chemistry", "school"],
}
STRAIN_CUES = {
 "exhaustion": ["wearing me out", "wearing me down", "exhausting me", "running on nothing", "worn out", "wiped out"],
 "overload": ["can't keep up", "piling on", "barely keeping", "falling behind", "can't do anything"],
 "threat_loss": ["might lose", "about to lose", "trying to evict", "don't know how we'll", "don't know how i'll",
                 "can't make", "wiped out our", "about to get laid off",
                 "evict us", "evict me", "kick us out", "kick me out", "fire me", "cut my hours"],
 "fear": ["panicking", "eating me alive", "crushing me", "keeps me up", "terrifies me", "freaking out"],
}
DOMAIN_SEEDS = [
 "things at home with my family are really getting to me",
 "looking after my sick parent is taking everything out of me",
 "my health keeps getting worse and it frightens me",
 "money is so short I lie awake wondering how we'll get by",
 "we could be forced out of our place and I'm scared",
 "work has been grinding me down and I'm afraid of losing my job",
 "I'm falling so far behind in school that I feel sick about it",
 "running the business alone and the money worries never stop",
]
ANCHOR_M = compile_keyword_matcher(sorted({w for v in DOMAIN_ANCHORS.values() for w in v}))
STRAIN_M = compile_keyword_matcher(sorted({w for v in STRAIN_CUES.values() for w in v}))

def sentences(low):
    start = 0
    for m in re.finditer(r"[.!?\n]", low):
        yield low[start:m.start()]
        start = m.end()
    yield low[start:]

def a2(msg):
    low = msg.lower()
    for s in sentences(low):
        if not any(True for _ in ANCHOR_M.iter_hits(s)):
            continue
        strains = list(STRAIN_M.iter_hits(s)) + [h for m in (td._MEDIUM_MATCHER, td._CONCERN_MATCHER) for h in m.iter_hits(s)]
        for h in strains:
            q, neg = td._qualifying_first_person_hit(h.keyword, h.start, s, td._CRISIS_SUBJECT_RE)
            if q and not neg:
                return h.keyword
    return None

STRESS = [
 "my kids haven't slept in days and I'm running on nothing", "my mom and I keep fighting and it's wearing me down",
 "things with my partner are falling apart and I don't know what to do", "taking care of my dad after his stroke is wearing me out",
 "I'm the only one caring for my grandmother and I can't keep up", "juggling my sister's medical appointments is exhausting me",
 "my back pain flared up again and I can't do anything", "waiting on the biopsy results is eating me alive",
 "my migraines keep getting worse and I'm scared", "cash flow is tight and payroll is due friday",
 "I don't know how we'll make rent this month", "the car repair wiped out our savings",
 "our landlord is trying to evict us", "we might lose the apartment",
 "the house has black mold and we can't afford to move", "my boss keeps piling on more and I'm barely keeping my head up",
 "I'm about to get laid off and I can't stop thinking about it", "I've been blocked on my novel for weeks and it's crushing me",
 "I'm failing chemistry and my parents are going to be furious", "my thesis defense is next week and I'm panicking",
 "I can't keep up with my classes anymore",
]
CONTROLS = [
 "my kids have soccer on saturday", "my mom is visiting next week", "I'm taking care of my neighbor's cat this weekend",
 "my dad's physical therapy starts monday", "I have a dentist appointment tomorrow", "I started taking a new vitamin",
 "I paid my rent today", "I'm budgeting for a vacation", "we're looking at apartments downtown", "the landlord fixed the sink",
 "I have a meeting with my boss at 3", "work was fine today", "my chemistry exam is on friday", "I finished my homework",
 "the kids wore me out at the park but it was a great day", "I'm not stressed about rent anymore",
 "my coworker is getting evicted and I'm helping her move", "I can't keep up with all the good shows right now",
 "my dad's surgery went really well", "my thesis is almost done and I'm excited", "rent went up a little but we're fine",
 "the kids can't keep up with me on hikes anymore haha", "my kids wore me out at the zoo but it was so fun",
 "my sister is exhausted from her new job", "our landlord finally fixed the heater",
 "my partner is panicking about his exam, I'm trying to help", "we paid off the loan and I feel so relieved",
 "I don't know how the kids have so much energy",
 # new neutral-domain controls for seed over-escalation (B)
 "I paid the bills this morning", "work was busy but good", "my parents are coming over for dinner",
 "we moved into our new place last week", "the business had a great month", "I'm looking after my nephew this afternoon",
]
RANK = {"CONVERSATIONAL": 0, "CONCERN": 1, "MEDIUM": 2, "HIGH": 3}

async def levels():
    out = {}
    for m in STRESS + CONTROLS:
        out[m] = (await td.detect_crisis_level(m, model_manager=None)).level.name
    return out

async def main():
    base = await levels()
    td.CRISIS_EXEMPLARS["concern"] = list(td.CRISIS_EXEMPLARS["concern"]) + DOMAIN_SEEDS
    td._exemplar_embeddings_cache = None
    seeded = await levels()
    def final(level, m, use_a2):
        return "CONCERN" if (use_a2 and RANK[level] == 0 and a2(m)) else level
    variants = {
        "baseline": {m: base[m] for m in base},
        "A2": {m: final(base[m], m, True) for m in base},
        "B": {m: seeded[m] for m in seeded},
        "A2+B": {m: final(seeded[m], m, True) for m in seeded},
    }
    print("| variant | stress >= CONCERN | controls changed vs baseline |")
    print("|---|---|---|")
    for name, v in variants.items():
        s = sum(RANK[v[m]] >= 1 for m in STRESS)
        c = [m for m in CONTROLS if v[m] != base[m]]
        print(f"| {name} | {s}/{len(STRESS)} | {len(c)}/{len(CONTROLS)} |")
    print("\n== per-row (baseline / A2 / B / A2+B)")
    for m in STRESS:
        row = [variants[k][m] for k in variants]
        print(f"  stress  {' / '.join(row):58} | {m}")
    for m in CONTROLS:
        row = [variants[k][m] for k in variants]
        flag = "CHANGED" if len(set(row)) > 1 else "       "
        print(f"  control {flag} {' / '.join(row):50} | {m}")

asyncio.run(main())
