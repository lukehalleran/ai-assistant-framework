"""H02b acceptance probe (parent, post-integration).

MUST run through probes/run_sandboxed_probe.py with TONE_EXEMPLAR_LEARNING=0.

It calls the DEPLOYED detect_crisis_level(model_manager=None) and simulates no rule. The rows come
straight from the recorded round-3 output, so the row set cannot drift:
- stress rows must reach >= CONCERN: 22/23 expected, the one accepted miss being D-H02-3's
  impersonal business row;
- control rows must keep the level recorded as `today=` in round 3 (pre-H02b deployed levels): 0 changed.
"""
import asyncio, hashlib, re, sys
sys.path.insert(0, "/home/lukeh/daemon_exec/generalization")
import utils.tone_detector as td

print("tone_detector sha256:", hashlib.sha256(open(td.__file__, "rb").read()).hexdigest()[:16])
REC = ("/home/lukeh/daemon_exec/generalization/docs/execution/generalization/probes/"
       "h02_candidate_probe_round3_output.txt")
ACCEPTED_MISS = "cash flow is tight and payroll is due friday"
RANK = {"CONVERSATIONAL": 0, "CONCERN": 1, "MEDIUM": 2, "HIGH": 3}
ROW_RE = re.compile(r"^\s+\S+\s+today=(\S+)\s+cand=.*?\|\s(.*)$")

rows = {"stress": [], "controls": []}
section = None
with open(REC, encoding="utf-8") as fh:
    for line in fh:
        if line.startswith("== stress"):
            section = "stress"
            continue
        if line.startswith("== controls"):
            section = "controls"
            continue
        m = ROW_RE.match(line.rstrip("\n"))
        if m and section:
            rows[section].append((m.group(2), m.group(1)))
assert len(rows["stress"]) == 23 and len(rows["controls"]) == 41, {k: len(v) for k, v in rows.items()}


async def main():
    hit = 0
    unexpected_miss = []
    print("== stress (deployed)")
    for msg, before in rows["stress"]:
        a = await td.detect_crisis_level(msg, model_manager=None)
        ok = RANK[a.level.name] >= 1
        hit += ok
        if not ok and msg != ACCEPTED_MISS:
            unexpected_miss.append(msg)
        print(f"  {'PASS' if ok else 'MISS'} before={before:14} now={a.level.name:14} trigger={a.trigger:36} | {msg}")
    changed = []
    print("== controls (deployed)")
    for msg, before in rows["controls"]:
        a = await td.detect_crisis_level(msg, model_manager=None)
        same = a.level.name == before
        if not same:
            changed.append(msg)
        print(f"  {'ok  ' if same else 'CHANGED'} before={before:14} now={a.level.name:14} trigger={a.trigger:36} | {msg}")
    print(f"STRESS >= CONCERN: {hit}/{len(rows['stress'])} (unexpected misses: {len(unexpected_miss)}); "
          f"CONTROLS changed vs round-3 today: {len(changed)}/{len(rows['controls'])}")
    if unexpected_miss:
        print("UNEXPECTED MISSES:", unexpected_miss)
    if changed:
        print("CHANGED CONTROLS:", changed)

asyncio.run(main())
