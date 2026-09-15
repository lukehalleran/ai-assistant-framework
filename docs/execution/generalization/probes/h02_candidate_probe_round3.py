"""H02 design probe, round 3 (parent), after owner decisions D-H02-1..3 (2026-09-13).
MUST run via probes/run_sandboxed_probe.py with TONE_EXEMPLAR_LEARNING=0.
No new seeds (D-H02-1). The rule stays first-person (D-H02-3). "us" counts ONLY as the object of a
threat_loss shape (D-H02-2), composed from two categorized tables (threat lead x displacement verb).
A3 changes vs A2: (i) strain evidence = affect/strain categories only, stressor-topic nouns excluded;
(ii) threat_loss "us" composition; (iii) clause boundary at "but" (anchor and strain must share a clause).
Real detect_crisis_level(model_manager=None) is today's level; the candidate floors it at CONCERN."""
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
 "housing": ["landlord", "evict", "eviction", "apartment", "house", "lease", "homeless", "home", "place"],
 "work": ["boss", "job", "laid off", "layoff", "fired", "shifts", "clients", "business", "manager", "novel", "project"],
 "school": ["classes", "class", "exam", "thesis", "grades", "professor", "defense", "course", "chemistry", "school"],
}
STRAIN_SHAPES = {  # generic strain shapes; first-person experiencer via T02 qualifier
 "exhaustion": ["wearing me out", "wearing me down", "exhausting me", "running on nothing", "worn out", "wiped out"],
 "overload": ["can't keep up", "piling on", "barely keeping", "falling behind", "can't do anything"],
 "threat_loss": ["might lose", "about to lose", "don't know how we'll", "don't know how i'll",
                 "can't make", "wiped out our", "about to get laid off", "can't afford"],
 "fear": ["panicking", "eating me alive", "crushing me", "keeps me up", "terrifies me", "freaking out"],
}
# D-H02-2: "us" as object ONLY inside a composed threat_loss shape: <threat lead> <displacement verb> us
THREAT_LEADS = ["trying to", "going to", "gonna", "about to", "threatening to", "might", "may", "could",
                "will", "wants to", "want to", "planning to"]
DISPLACEMENT = [r"evict us", r"kick us out", r"throw us out", r"foreclose on us", r"cut us off", r"force us out"]
THREAT_US_RE = re.compile(r"\b(?:" + "|".join(map(re.escape, THREAT_LEADS)) + r")\s+(?:" + "|".join(DISPLACEMENT) + r")\b")
# Stressor-topic nouns in the existing lists (H02a will categorize them); excluded as STRAIN evidence
TOPIC_ENTRIES = {"bills", "debt", "broke", "deadline", "pressure", "money problems", "financial stress",
                 "work stress", "job stress", "school stress", "behind on bills", "behind on rent",
                 "drowning in debt", "no days off", "divorce", "breakup", "insomnia"}
ANCHOR_M = compile_keyword_matcher(sorted({w for v in DOMAIN_ANCHORS.values() for w in v}))
STRAIN_M = compile_keyword_matcher(sorted({w for v in STRAIN_SHAPES.values() for w in v}))

def clauses(low):
    for sent in re.split(r"[.!?\n]", low):
        for clause in re.split(r",?\s+but\s+", sent):
            yield clause

def a3(msg):
    low = msg.lower()
    for c in clauses(low):
        if not any(True for _ in ANCHOR_M.iter_hits(c)):
            continue
        for m in THREAT_US_RE.finditer(c):
            if not td.is_negated(c, m.start(), window_tokens=3) if hasattr(td, "is_negated") else True:
                return f"threat_us:{m.group(0)}"
        strains = list(STRAIN_M.iter_hits(c)) + [
            h for mt in (td._MEDIUM_MATCHER, td._CONCERN_MATCHER) for h in mt.iter_hits(c)
            if h.keyword not in TOPIC_ENTRIES]
        for h in strains:
            q, neg = td._qualifying_first_person_hit(h.keyword, h.start, c, td._CRISIS_SUBJECT_RE)
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
 # D-H02-2 positives
 "the landlord says he's going to kick us out of the apartment next month",
 "the bank is threatening to foreclose on us and the house is all we have",
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
 "I paid the bills this morning", "work was busy but good", "my parents are coming over for dinner",
 "we moved into our new place last week", "the business had a great month", "I'm looking after my nephew this afternoon",
 # D-H02-2 threat_loss "us" controls (must stay conversational)
 "our landlord promised he won't evict us",
 "they tried to evict us years ago but we're doing fine now",
 "we were worried they'd kick us out but the lease got renewed",
 "the bouncer is going to kick us out of the bar lol",
 "the game keeps trying to kick us out of the lobby haha",
 "the landlord is trying to evict the tenants downstairs, not us",
 "my landlord would never evict us, he's like family",
]
RANK = {"CONVERSATIONAL": 0, "CONCERN": 1, "MEDIUM": 2, "HIGH": 3}

async def main():
    s_hit = c_changed = 0
    print("== stress")
    for m in STRESS:
        lvl = (await td.detect_crisis_level(m, model_manager=None)).level.name
        cand = a3(m)
        final = "CONCERN" if (RANK[lvl] == 0 and cand) else lvl
        s_hit += RANK[final] >= 1
        print(f"  {'PASS' if RANK[final] >= 1 else 'MISS'} today={lvl:14} cand={str(cand):34} | {m}")
    print("== controls")
    for m in CONTROLS:
        lvl = (await td.detect_crisis_level(m, model_manager=None)).level.name
        cand = a3(m)
        final = "CONCERN" if (RANK[lvl] == 0 and cand) else lvl
        changed = final != lvl
        c_changed += changed
        print(f"  {'OVER' if changed else 'ok  '} today={lvl:14} cand={str(cand):34} | {m}")
    print(f"STRESS >= CONCERN: {s_hit}/{len(STRESS)}; CONTROLS changed vs today: {c_changed}/{len(CONTROLS)}")
asyncio.run(main())
