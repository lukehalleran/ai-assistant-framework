# Design note — prong 3: a behavioural probe harness on a test instance

*2026-09-12. Owner's idea, captured for later — not scheduled, not started.*

**2026-09-13:** the [independent generalization and CI review](GENERALIZATION_CI_REVIEW_20260913.md)
connects this design to the existing gates and supplies a 15-case starter
backlog, isolation requirements, and implementation order. The runner and
machine-readable registry remain unimplemented.

> "CI guard blocks on test fail and we are also working on a full block when
> any known bug class is found. A third prong will be needed. We need a list of
> test probes and their expected response shape. To do this, we will need to
> build a separate 'test' instance of Daemon."

## 1. Why a third prong (the evidence for it is a month of defects)

| prong | what it catches | status |
|---|---|---|
| 1. test suite | unit/integration contracts | exists; see the dated CI evidence in the linked review |
| 2. bug-class scan | selected code shapes of known classes | four gated DM scanners plus catalog consistency; baseline debt is accepted |
| **3. behavioural probes** | **what the assembled system actually does on a turn** | **missing** |

Nearly every defect this month was invisible to prongs 1 and 2 and was found
by the owner pasting a dump: the distress floor latching onto homework turns,
two contradictory web-trigger verdicts in one turn, the spurious "no card to
approve" notices, `doc`→`doctor`, the calendar forced-action loop, pasted
articles matchable only by their opening. Every one is a *composition*
failure — each unit behaved as tested; the turn did not.

The catalog already names this and assigns it to a human: **DM-23 — "Live-turn
probe + debug-dump review (designed turns, then `turn_records` and debug record
line by line)", runs as: `owner relay`** (`docs/BUG_CLASSES.md:727`). It is the
only detection method whose runtime is a person. Prong 3 is the automation of
DM-23, and `docs/DEVELOPMENT_WORKFLOW.md` §7's nightly-suite recommendation is
where it would be scheduled.

## 2. The central design constraint: assert on receipts, not on prose

Model text is nondeterministic; asserting on it produces a flaky suite that
gets disabled. But this system already emits unusually rich receipts, and the
defects above were all visible in them:

- `logs/turn_records.jsonl` — `intent`/`intent_source`, `tone_level`/
  `tone_trigger`, `gate_triggered`/`gate_modes`/`gate_reason`, `mode`,
  `web_trigger_*`, `web_budget_remaining`, `grounding_status`, `plan_points`,
  `response_plan`, `phase_timings`, `wall_elapsed_s`;
- the debug record — sections present/absent, `answer_call`,
  `decision_prompt_hash`, `visible_sources`, `omitted_sections`;
- deterministic SHAPE predicates over the reply — `[WEB_n]` present/absent,
  `NO_CARD_NOTICE` absent, no literal `<thinking>`, no stream artifact, a
  length band, a pending card created or not.

So a probe asserts **routing and receipts**, plus shape predicates over the
text. It never asserts wording. Two of this month's incidents (the 09-11 tone
latch, the double web-trigger) would have gone red on `tone_trigger` and on a
classifier-call count alone.

## 3. Probe registry shape

One file, one entry per probe, each citing the incident that motivated it:

```yaml
- id: homework_paste_is_not_distress
  class: BC-01, BC-28            # what regression this guards
  source: 2026-09-11 17:07 turn  # the incident it came from
  setup:
    seed: coursework_session      # named seeded state (§4)
    prior_turns: 2
  input: |
    sucess here? install.packages("utf8")
    … (verbatim live text)
  expect:
    receipts:
      tone_level: CONVERSATIONAL
      tone_trigger: {not: distress_sticky_floor}
      gate_triggered: false
    response:
      must_not: ["⚠️", "LIGHT SUPPORT"]
    budget:
      llm_calls: {max: 3}         # catches the duplicate-classification class
```

Two properties that make the registry worth more than its tests:

1. **Every probe names a `class:`**, so coverage is measurable — "which of the
   78 bug classes have a live probe" becomes a table, like the learning-loop
   coverage map in `docs/GENERALIZATION_AUDIT_20260901.md`.
2. **Every probe names its `source:` incident**, so a probe that starts failing
   can be traced to the behaviour it was written to protect, rather than being
   "fixed" by loosening the assertion.

Seed the registry from the last month of handoffs — each already contains the
verbatim live text and the expected routing, written up at the time:
`HANDOFF_20260908_homework_session_audit`, `HANDOFF_20260910_probe_dump`,
`PLAN_20260912_session_defects`, and the 09-11/09-12 audits. ~15 probes to
start, drawn from real incidents, is more valuable than 100 invented ones.

## 4. The test instance

**Isolation (non-negotiable).** Its own data directory, corpus, graph,
telemetry and logs, seeded from fixtures — never the owner's stores. The
precedent is a scar: `scripts/generate_test_facts.py` once wrote 48 synthetic
facts and 35 synthetic graph edges into the LIVE stores (2026-09-02), and
pytest once rotated the production logs (2026-08-28). The existing
`--sandbox-dir` pattern, `DAEMON_TEST_MODE`, and `utils/daemon_guard.py` are
the building blocks; the test instance must be sandbox-only *by construction*,
not by discipline.

**Seeded state.** Probes need a corpus with known content ("recall the article
about X" needs a ground truth). A small set of named seeds — `coursework_session`,
`news_discussion`, `distress_session`, `empty` — each a fixture bundle of
corpus + facts + graph + uploads. They double as the fixtures for prong 1.

**The LLM problem, which decides everything else.** Three options:

| | determinism | cost | catches provider drift |
|---|---|---|---|
| (a) recorded cassettes keyed by prompt hash | total | zero | no |
| (b) a small local model | high | compute | no |
| (c) live calls, shape-only assertions | low | real money | **yes** |

Recommendation: **(a) in CI** — deterministic, free, fast, and the codebase
already hashes prompts (`prompt_hash`, `decision_prompt_hash`), so cassettes
key naturally — plus **(c) nightly or pre-release**, shape-only, with a hard
budget cap. Cassettes alone would have missed a whole family this project has
actually hit: the kimi-3 trailing-`e` artifact, the `<|sep|>` leak, the
`forced_top_p` 400, reasoning-only streams. Those are provider-behaviour bugs
that only a live call sees.

**Credentials.** No Google/Tavily credentials in the CI instance; the tool
layer returns recorded fixtures, and tool-health probes assert the honest
UNAVAILABLE strings (which is itself a regression worth guarding — see BC-78).

## 5. What this does NOT replace

- Prong 1 stays: probes are slow and coarse; unit tests localize.
- Prong 2 stays: a probe suite cannot find a defect that has not yet produced
  a bad turn, which is exactly what the static scanners are for.
- The owner's dump review stays for a while: probes encode *known* shapes;
  reading a live session is still how new classes are discovered. The goal is
  that a class, once found, never needs to be found by a human twice.

## 6. Staging

1. Registry format + ~15 probes transcribed from existing handoffs (no runner
   yet — the registry alone is useful as a checklist for a manual pass).
2. Cassette layer over the provider, keyed by prompt hash.
3. Runner against a sandboxed instance; assert receipts only.
4. CI job: probes gate on routing receipts, report-only on text shape until
   the false-positive rate is measured (the same ladder the scanners used:
   report → gate).
5. Coverage table: bug classes with a live probe vs without.
6. Nightly live run against the real provider, shape-only, budget-capped.

## 7. Open questions

- Does the instance run the full FastAPI app, or drive `handle_submit`
  directly in-process? In-process is far cheaper and covers every defect
  listed in §1; the HTTP layer would only add SSE/stream regressions.
- Where do cassettes live, and what redaction do they need before being
  committed (they contain full prompts — `utils/privacy_redaction.py` exists
  for exactly this, and the CI privacy guard already scans fixtures)?
- Is a probe allowed to assert on latency (`wall_elapsed_s`)? Tempting after
  the 172 s and 106 s loops, but it makes CI machine-dependent — probably a
  nightly-only assertion.
