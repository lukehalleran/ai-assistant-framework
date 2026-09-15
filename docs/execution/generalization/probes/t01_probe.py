import re, sys
sys.path.insert(0, "/home/lukeh/daemon_exec/generalization")
import utils.tone_detector as td
FP_SUBJ = re.compile(r"\b(?:i|i'm|im|i've|ive|i'd|id|i'll)\b")
FP_IN_PHRASE = re.compile(r"\b(?:i|me|my|myself|i'm)\b")
def candidate(msg, window=3):
    low = msg.lower()
    for hit in td._HIGH_MATCHER.iter_hits(low):
        kw = hit.keyword
        if FP_IN_PHRASE.search(kw):
            return f"fp-phrase:{kw}"
        sent_start = max(low.rfind(c, 0, hit.start) for c in ".!?\n") + 1
        before = low[sent_start:hit.start].split()[-window:]
        if any(FP_SUBJ.fullmatch(t.strip(",;:")) for t in before):
            return f"fp-subject:{kw}"
    return None
MUST_ESCALATE = [
    "people are suffering and I want to die",
    "According to the news, officials say millions are struggling, and honestly I can't go on",
    "The world is on fire and I want to kill myself",
    "People are dying in the news and I don't want to be here anymore",
    "They say millions are suffering. I want to end my life.",
    "the news is awful, people are suffering, and I'm ready to die",
    "According to the article people are struggling and I really just want to die",
]
MUST_STAY = [
    "According to officials, suicide rates rose among young people",
    "The news article says people are suicidal after the policies",
    "I read an article that says suicide rates are rising among teens",
    "I think the news coverage of suicide is irresponsible, people are struggling",
    "They say people are suffering and they can't go on",
    "According to the news, millions of people want to die in that war",
    "People are struggling with the new policies, according to officials",
]
for label, group in (("MUST_ESCALATE", MUST_ESCALATE), ("MUST_STAY", MUST_STAY)):
    print("==", label)
    for m in group:
        obs = td._check_observational_language(m)
        kw = td._check_keyword_crisis(m)
        print(f"  obs={obs!s:5} stage1={kw[0].name if kw else None!s:8} cand={candidate(m)!s:28} | {m}")
