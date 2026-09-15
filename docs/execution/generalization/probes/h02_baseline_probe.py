import asyncio, hashlib, sys
sys.path.insert(0, "/home/lukeh/daemon_exec/generalization")
import utils.tone_detector as td
print("tone_detector sha256:", hashlib.sha256(open(td.__file__, "rb").read()).hexdigest()[:16])
ROWS = {
 "family": (["my kids haven't slept in days and I'm running on nothing",
             "my mom and I keep fighting and it's wearing me down",
             "things with my partner are falling apart and I don't know what to do"],
            ["my kids have soccer on saturday", "my mom is visiting next week"]),
 "caregiving": (["taking care of my dad after his stroke is wearing me out",
                 "I'm the only one caring for my grandmother and I can't keep up",
                 "juggling my sister's medical appointments is exhausting me"],
                ["I'm taking care of my neighbor's cat this weekend", "my dad's physical therapy starts monday"]),
 "health": (["my back pain flared up again and I can't do anything",
             "waiting on the biopsy results is eating me alive",
             "my migraines keep getting worse and I'm scared"],
            ["I have a dentist appointment tomorrow", "I started taking a new vitamin"]),
 "money": (["cash flow is tight and payroll is due friday",
            "I don't know how we'll make rent this month",
            "the car repair wiped out our savings"],
           ["I paid my rent today", "I'm budgeting for a vacation"]),
 "housing": (["our landlord is trying to evict us",
              "we might lose the apartment",
              "the house has black mold and we can't afford to move"],
             ["we're looking at apartments downtown", "the landlord fixed the sink"]),
 "work": (["my boss keeps piling on more and I'm barely keeping my head up",
           "I'm about to get laid off and I can't stop thinking about it",
           "I've been blocked on my novel for weeks and it's crushing me"],
          ["I have a meeting with my boss at 3", "work was fine today"]),
 "school": (["I'm failing chemistry and my parents are going to be furious",
             "my thesis defense is next week and I'm panicking",
             "I can't keep up with my classes anymore"],
            ["my chemistry exam is on friday", "I finished my homework"]),
}
async def main():
    for dom, (stress, controls) in ROWS.items():
        print(f"== {dom}")
        for label, msgs in (("stress ", stress), ("control", controls)):
            for m in msgs:
                r = await td.detect_crisis_level(m, model_manager=None)
                score = td._calculate_harm_score(m)[0]
                print(f"  {label} {r.level.name:14} {str(r.trigger)[:34]:34} harm={score:4.1f} | {m}")
asyncio.run(main())
