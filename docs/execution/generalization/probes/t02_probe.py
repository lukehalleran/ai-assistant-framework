import asyncio, re, sys
from types import SimpleNamespace
sys.path.insert(0, "/home/lukeh/daemon_exec/generalization")
import utils.tone_detector as td
from utils.trigger_match import is_negated

FP_ANY = td._HISTORY_FIRST_PERSON_RE                    # existing closed set, unchanged
SUBJ_A = re.compile(r"\b(?:i|i'm|im|i've|ive|i'd|we)\b")  # subject forms of that set (crisis tier)
SUBJ_B = re.compile(r"\b(?:i|i'm|im|i've|ive|i'd)\b")     # singular subject forms (mild tier)
WIN = 3
BREAK = re.compile(r"[.!?\n]")
STRIP = ",;:\"'()"

def sent_prefix(low, pos):
    ends = [m.end() for m in BREAK.finditer(low, 0, pos)]
    return low[(ends[-1] if ends else 0):pos]

def qualify(low, hit, subj_re):
    toks = sent_prefix(low, hit.start).split()
    rule = "a" if FP_ANY.search(hit.keyword) else (
        "b" if any(subj_re.fullmatch(t.strip(STRIP)) for t in toks[-WIN:]) else None)
    if rule is None:
        return None, False
    last_fp = max((i for i, t in enumerate(toks) if FP_ANY.fullmatch(t.strip(STRIP))), default=None)
    span = " ".join((toks[last_fp + 1:] if last_fp is not None else toks)[-WIN:])
    return rule, is_negated(span, len(span), window_tokens=WIN)

def t01(msg):
    low = msg.lower()
    if not td._check_observational_language(msg):
        return "n/a"
    for hit in td._HIGH_MATCHER.iter_hits(low):
        toks = sent_prefix(low, hit.start).split()
        if FP_ANY.search(hit.keyword) or any(FP_ANY.fullmatch(t.strip(STRIP)) for t in toks[-WIN:]):
            return td._check_keyword_crisis(msg)[0].name
    return "CONVERSATIONAL"

def t02(msg):
    low = msg.lower()
    if not td._check_observational_language(msg):
        return "n/a", ""
    a, neg_high, b = [], [], []
    for hit in td._HIGH_MATCHER.iter_hits(low):
        rule, neg = qualify(low, hit, SUBJ_A)
        if rule and not neg: a.append(f"{hit.keyword}/{rule}")
        elif rule and neg: neg_high.append(hit.keyword)
    if a:
        return td._check_keyword_crisis(msg)[0].name, f"A:{a}"
    mild = list(td._MEDIUM_MATCHER.iter_hits(low)) + list(td._CONCERN_MATCHER.iter_hits(low))
    if not td._OVERWHELMED_POSITIVE_MATCHER(low):
        mild += list(td._OVERWHELMED_MATCHER.iter_hits(low))
    mild += [SimpleNamespace(keyword=m.group(0), start=m.start()) for m in td._SUBSTANCE_USE_RE.finditer(low)]
    for hit in mild:
        rule, neg = qualify(low, hit, SUBJ_B)
        if rule and not neg: b.append(f"{hit.keyword}/{rule}")
    if neg_high or b:
        return "CONCERN", f"B: neg_high={neg_high} mild={b}"
    return "CONVERSATIONAL", ""

CRISIS = [
    "people are suffering and I want to die",
    "According to the news, officials say millions are struggling, and honestly I can't go on",
    "People are dying in the news and I don't want to be here anymore",
    "the news is awful, people are suffering, and I'm ready to die",
    "According to the article people are struggling and I really just want to die",
    "According to the news people are suffering and I want to kill myself",
    "people are suffering and I can't stop thinking I want to die",
    "According to the news people are struggling and I can't stop crying",
    "According to officials people are suffering and we want to die",
]
CONCERN = [
    "people are suffering and I feel hopeless",
    "According to the news, officials say millions are struggling and I feel so alone",
    "According to the news people are suffering and I'm overwhelmed",
    "According to officials people are struggling and honestly I feel worthless",
    "people are suffering and I'm losing my mind",
    "the news says people are suffering and I can't cope",
    "people are suffering but I don't want to die",
    "According to the news people are suffering, I don't want to kill myself though",
    "According to the news people are suffering and I lost my job",
]
STAY = [
    "According to the news, my students want to die",
    "According to the news, our kids want to die",
    "According to officials, suicide rates rose among young people",
    "The news article says people are suicidal after the policies",
    "I think the news coverage of suicide is irresponsible, people are struggling",
    "They say people are suffering and they can't go on",
    "According to the news, millions of people want to die in that war",
    "People are struggling with the new policies, according to officials",
    "I think we are failing as a country, according to the news",
    "According to the news I can't believe this government",
    "According to officials, people are struggling, but I don't feel alone",
    "I read that people want to die, according to the news",
]
LIMITS = [
    "According to the news people are suffering and I'm not scared",
    "I lost the link, according to officials people are struggling",
    "According to the news people are suffering and I can't stop wanting to kill myself",
    "According to the news people are suffering and I'm having a breakdown and falling apart",
    "According to the news, a student said: I want to kill myself",
    "According to the news people are suffering and I can't stop cutting myself",
    "According to the news people are suffering, I don't want to die in a war like that",
]

async def main():
    fails = 0
    for label, rows, ok in (("CRISIS (expect >= MEDIUM)", CRISIS, lambda l: l in ("MEDIUM", "HIGH")),
                            ("CONCERN (expect CONCERN)", CONCERN, lambda l: l == "CONCERN"),
                            ("STAY (expect CONVERSATIONAL)", STAY, lambda l: l == "CONVERSATIONAL"),
                            ("LIMITS (report only)", LIMITS, None)):
        print("==", label)
        for m in rows:
            obs = td._check_observational_language(m)
            today = await td.detect_crisis_level(m, model_manager=None)
            lvl, why = t02(m)
            verdict = "" if ok is None else ("PASS" if ok(lvl) and obs else "FAIL")
            fails += verdict == "FAIL"
            print(f"  {verdict:4} obs={obs!s:5} today={today.level.name:14} T01={t01(m):14} T02={lvl:14} {why}\n        | {m}")
    print("FAILS:", fails)
asyncio.run(main())
