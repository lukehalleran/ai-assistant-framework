"""H02 candidate A (parent design probe): categorized domain anchors x generic strain cues,
same sentence, first-person marker in the sentence, strain cue not negated -> floor at CONCERN.
Today's detector is the real detect_crisis_level(model_manager=None); the candidate is simulated
on top of it with deployed matchers/regexes."""
import asyncio, hashlib, re, sys
sys.path.insert(0, "/home/lukeh/daemon_exec/generalization")
import utils.tone_detector as td
from utils.trigger_match import compile_keyword_matcher, is_negated
print("tone_detector sha256:", hashlib.sha256(open(td.__file__, "rb").read()).hexdigest()[:16])

DOMAIN_ANCHORS = {  # categorized-generic, one matcher at runtime
 "family": ["kids", "kid", "child", "children", "son", "daughter", "mom", "mother", "dad", "father",
            "parents", "partner", "husband", "wife", "spouse", "family", "sister", "brother", "baby"],
 "caregiving": ["caring for", "taking care of", "care of", "caregiver", "looking after"],
 "health": ["pain", "diagnosis", "biopsy", "results", "symptoms", "migraines", "treatment", "surgery",
            "illness", "flare", "flared", "doctor", "hospital"],
 "money": ["rent", "bills", "payroll", "savings", "cash flow", "paycheck", "debt", "money", "mortgage", "loan"],
 "housing": ["landlord", "evict", "eviction", "apartment", "house", "lease", "homeless"],
 "work": ["boss", "job", "laid off", "layoff", "fired", "shifts", "clients", "business", "manager", "novel", "project"],
 "school": ["classes", "class", "exam", "thesis", "grades", "professor", "defense", "course", "chemistry"],
}
STRAIN_CUES = {  # categorized generic strain shapes (domain-independent)
 "exhaustion": ["wearing me out", "wearing me down", "exhausting me", "running on nothing", "worn out", "wiped out"],
 "overload": ["can't keep up", "piling on", "barely keeping", "falling behind", "can't do anything"],
 "threat_loss": ["might lose", "about to lose", "trying to evict", "don't know how we'll", "don't know how i'll",
                 "can't make", "wiped out our", "about to get laid off"],
 "fear": ["panicking", "eating me alive", "crushing me", "keeps me up", "terrifies me", "freaking out"],
}
ANCHOR_M = compile_keyword_matcher(sorted({w for v in DOMAIN_ANCHORS.values() for w in v}))
STRAIN_M = compile_keyword_matcher(sorted({w for v in STRAIN_CUES.values() for w in v}))
EXISTING_AFFECT = [td._MEDIUM_MATCHER, td._CONCERN_MATCHER]

def sentences(low):
    start = 0
    for m in re.finditer(r"[.!?\n]", low):
        yield start, low[start:m.start()]
        start = m.end()
    yield start, low[start:]

def candidate(msg):
    """A-prime: same sentence has a domain anchor; a strain cue (new strain table OR existing
    MEDIUM/CONCERN affect entry) qualifies only when its experiencer is the user, reusing T02's
    deployed qualifier: rule (a) first-person token inside the cue, or rule (b) first-person
    SUBJECT (i, i'm, im, i've, ive, i'd, we) within 3 tokens before it; negation scoped after the
    pronoun (T02). No sentence-level first-person gate any more."""
    low = msg.lower()
    for off, s in sentences(low):
        anchors = [h.keyword for h in ANCHOR_M.iter_hits(s)]
        if not anchors:
            continue
        strains = [h for h in STRAIN_M.iter_hits(s)] + [h for m in EXISTING_AFFECT for h in m.iter_hits(s)]
        for h in strains:
            qualifies, negated = td._qualifying_first_person_hit(h.keyword, h.start, s, td._CRISIS_SUBJECT_RE)
            if qualifies and not negated:
                return f"{anchors[0]}x{h.keyword}"
    return None

STRESS = [
 "my kids haven't slept in days and I'm running on nothing",
 "my mom and I keep fighting and it's wearing me down",
 "things with my partner are falling apart and I don't know what to do",
 "taking care of my dad after his stroke is wearing me out",
 "I'm the only one caring for my grandmother and I can't keep up",
 "juggling my sister's medical appointments is exhausting me",
 "my back pain flared up again and I can't do anything",
 "waiting on the biopsy results is eating me alive",
 "my migraines keep getting worse and I'm scared",
 "cash flow is tight and payroll is due friday",
 "I don't know how we'll make rent this month",
 "the car repair wiped out our savings",
 "our landlord is trying to evict us",
 "we might lose the apartment",
 "the house has black mold and we can't afford to move",
 "my boss keeps piling on more and I'm barely keeping my head up",
 "I'm about to get laid off and I can't stop thinking about it",
 "I've been blocked on my novel for weeks and it's crushing me",
 "I'm failing chemistry and my parents are going to be furious",
 "my thesis defense is next week and I'm panicking",
 "I can't keep up with my classes anymore",
]
CONTROLS = [
 "my kids have soccer on saturday", "my mom is visiting next week",
 "I'm taking care of my neighbor's cat this weekend", "my dad's physical therapy starts monday",
 "I have a dentist appointment tomorrow", "I started taking a new vitamin",
 "I paid my rent today", "I'm budgeting for a vacation",
 "we're looking at apartments downtown", "the landlord fixed the sink",
 "I have a meeting with my boss at 3", "work was fine today",
 "my chemistry exam is on friday", "I finished my homework",
 # BC-28 over-escalation controls
 "the kids wore me out at the park but it was a great day",
 "I'm not stressed about rent anymore",
 "my coworker is getting evicted and I'm helping her move",
 "I can't keep up with all the good shows right now",
 "my dad's surgery went really well",
 "my thesis is almost done and I'm excited",
 "rent went up a little but we're fine",
 "the kids can't keep up with me on hikes anymore haha",
 "my kids wore me out at the zoo but it was so fun",
 "my sister is exhausted from her new job",
 "our landlord finally fixed the heater",
 "my partner is panicking about his exam, I'm trying to help",
 "we paid off the loan and I feel so relieved",
 "I don't know how the kids have so much energy",
]
async def main():
    rank = {"CONVERSATIONAL": 0, "CONCERN": 1, "MEDIUM": 2, "HIGH": 3}
    s_hit = s_tot = c_bad = 0
    for label, rows in (("stress", STRESS), ("control", CONTROLS)):
        print(f"== {label}")
        for m in rows:
            r = await td.detect_crisis_level(m, model_manager=None)
            cand = candidate(m)
            final = r.level.name if (cand is None or rank[r.level.name] >= 1) else "CONCERN"
            if label == "stress":
                s_tot += 1; s_hit += rank[final] >= 1
                verdict = "PASS" if rank[final] >= 1 else "MISS"
            else:
                changed = final != r.level.name
                c_bad += changed
                verdict = "OVER" if changed else "ok"
            print(f"  {verdict:4} today={r.level.name:14} cand={str(cand):32} final={final:14} | {m}")
    print(f"STRESS >= CONCERN: {s_hit}/{s_tot} (today 10/21); CONTROLS newly escalated: {c_bad}/{len(CONTROLS)}")
asyncio.run(main())
